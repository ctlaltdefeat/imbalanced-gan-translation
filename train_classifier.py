import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from sklearn.datasets import make_blobs, make_moons
from pytorch_lightning import seed_everything
from typing import Dict, Iterator, List, Optional, Union
from torch.utils.data import Dataset, DistributedSampler, Sampler
import numpy as np
import torch
import torch.utils.data
import random
from operator import itemgetter
import torch.nn as nn
from torch.nn import Parameter, init, SELU, Sequential, Linear, ReLU, Dropout
from pytorch_lightning.core import LightningModule
import torch.nn.functional as F
from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier


class BalanceClassSampler(Sampler):
    """Allows you to create stratified sample on unbalanced classes.
    Args:
        labels: list of class label for each elem in the dataset
        mode: Strategy to balance classes.
            Must be one of [downsampling, upsampling]
    """

    def __init__(
        self, labels: List[int], mode: Union[str, int] = "downsampling"
    ):
        """Sampler initialisation."""
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum() for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode
                if isinstance(mode, int)
                else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class Classifier(LightningModule):
    def __init__(self, output_dim):
        super().__init__()
        self.save_hyperparameters()
        self.output_dim = output_dim
        self.classifier = nn.Sequential(
            nn.Linear(int(self.output_dim), 128),
            SELU(),
            Dropout(),
            nn.Linear(128, 64),
            SELU(),
            Dropout(),
            nn.Linear(64, 1),
            # nn.Sigmoid(),
        )
        # self.loss = nn.BCELoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.val_metric = torchmetrics.AveragePrecision()

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.classifier.parameters(), lr=2e-5, #weight_decay=1e-6
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.loss(self(x), y.unsqueeze(-1))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log(
            "val/loss",
            self.val_metric(self.classifier(x), y.unsqueeze(-1).float()),
        )


if __name__ == "__main__":
    seed_everything(123, workers=True)
    neigh = KNeighborsClassifier(n_neighbors=11)
    x, y, x_maj, y_maj, x_min, y_min = torch.load("ds_imba_train.pt")
    neigh.fit(x, y)
    x, y, x_maj, y_maj, x_min, y_min = torch.load("ds_train.pt")
    ds_train = TensorDataset(x.float(), y.float())
    x, y, _, _, _, _ = torch.load("ds_imba_test.pt")
    ds_eval = TensorDataset(x.float(), y.float())
    # c = Classifier(21)
    c = Classifier.load_from_checkpoint(
        r"C:\Users\Jonathan\PycharmProjects\imbalanced-gan-translation\last.ckpt",
        output_dim=21,
    )
    trainer = Trainer(
        gpus=1,
        max_epochs=20000,
        checkpoint_callback=False,
        precision=16,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=1000, mode="max")
        ],
    )
    trainer.fit(
        c,
        DataLoader(
            ds_train,
            batch_size=40024,
            sampler=BalanceClassSampler(y.tolist(), mode="downsampling"),
        ),
        DataLoader(ds_eval, batch_size=40000),
    )
    c.eval()
    trainer.save_checkpoint(
        "saved_experiments/potential_test_imbalanced_4/last.ckpt"
    )
    exp = SummaryWriter("saved_experiments/potential_test_imbalanced_4")
    exp.add_pr_curve(
        "baseline",
        y.unsqueeze(-1).int(),
        torch.sigmoid(c(x.float())),
        num_thresholds=511,
    )
