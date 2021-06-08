from models import WGANGP
from catboost.core import CatBoostClassifier
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
    neigh = KNeighborsClassifier(n_neighbors=11, n_jobs=3)
    x_train, y_train, x_maj, y_maj, x_min, y_min = torch.load("ds_imba_train.pt")
    # neigh.fit(x_train, y_train)
    print("trained nn")
    x_test, y_test, x_maj, y_maj, x_min, y_min = torch.load("ds_imba_test.pt")
    x_test, y_test = x_test.numpy(), y_test.numpy()
    # model = WGANGP.load_from_checkpoint(r"C:\Users\Jonathan\PycharmProjects\imbalanced-gan-translation\lightning_logs\GAN_distance_loss_majority_sampling_census\checkpoints\epoch=19999-step=19999.ckpt", strict=False).cuda()
    # x_gen = model(torch.Tensor([4, 4]).unsqueeze(0)).detach().numpy()
    # x_gen = model(x_maj.float().cuda()).cpu().detach()
    print("generated synthetic points")
    # x_gen = x_gen[neigh_res==1]
    # x_gen = x_gen[neigh.predict_proba(x_gen)[:, 1].argsort()[::-1].copy()]
    print("predicted nn on synthetic points")
    # print(x_gen.shape)
    
    # Adjust the number of points generated
    # x_gen = x_gen[:x_maj.shape[0]]

    # print(x_gen.shape)
    x_all = x_train
    # x_all = torch.cat([x_train, x_gen])
    # reduced = reducer.fit_transform(x_all)
    y_all = y_train
    # y_all = torch.cat([y_train, torch.ones(len(x_gen))])
    # plt.scatter(reduced[:, 0],reduced[:, 1], c=y_all)
    # plt.show()

    x_all, y_all = x_all.numpy(), y_all.numpy()

    # print(cb.fit(
    #     x_all, y_all,
    #     # cat_features=categorical_features_indices,
    #     eval_set=(x_test, y_test),
    # #     logging_level='Verbose',  # you can uncomment this for text output
    #     # plot=True
    # ))

    # x, y, x_maj, y_maj, x_min, y_min = torch.load("ds_imba_train.pt")
    # neigh.fit(x, y)
    x, y, x_maj, y_maj, x_min, y_min = torch.load("ds_imba_train.pt")
    ds_train = TensorDataset(x.float(), y.float())
    x_test, y_test, _, _, _, _ = torch.load("ds_imba_test.pt")
    ds_eval = TensorDataset(x_test.float(), y_test.float())
    c = Classifier(x.shape[1])
    # c = Classifier.load_from_checkpoint(
    #     r"C:\Users\Jonathan\PycharmProjects\imbalanced-gan-translation\last.ckpt",
    #     output_dim=21,
    # )
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
            batch_size=4000024,
            sampler=BalanceClassSampler(y.tolist(), mode="upsampling"),
        ),
        DataLoader(ds_eval, batch_size=400000),
    )
    c.eval()
    trainer.save_checkpoint(
        "saved_experiments/upsampling/last.ckpt"
    )
    exp = SummaryWriter("saved_experiments/upsampling")
    exp.add_pr_curve(
        "census",
        y_test.unsqueeze(-1).int(),
        torch.sigmoid(c(x_test.float())),
        num_thresholds=511,
    )