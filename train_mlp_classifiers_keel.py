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
from pathlib import Path


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
            nn.Linear(int(self.output_dim), 64),
            SELU(),
            Dropout(0.1),
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
            self.classifier.parameters(), lr=2e-4,  # weight_decay=1e-6
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
    for ds_path in Path("datasets_keel").iterdir():
        if 'abalone' not in ds_path.name:
            break
        neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=3)
        x_train, y_train, x_maj, y_maj, x_min, y_min = torch.load(
            ds_path / "ds_train.pt"
        )
        neigh.fit(x_train, y_train)
        print("trained nn")
        x_test, y_test, _, _, _, _ = torch.load(ds_path / "ds_test.pt")
        model = WGANGP.load_from_checkpoint(
            ds_path / "ttgan_maj.ckpt", strict=False
        ).cuda()
        # x_gen = model(torch.Tensor([4, 4]).unsqueeze(0)).detach()
        # x_gen = model(torch.randn(5*x_min.shape[0], x_min.shape[1]).cuda()).cpu().detach()
        x_gen = model(x_maj.float().cuda()).cpu().detach()
        print("generated synthetic points")
        # x_gen = x_gen[neigh_res==1]
        x_gen = x_gen[neigh.predict_proba(x_gen)[:, 1].argsort()[::-1].copy()]
        torch.save(x_gen, ds_path / "x_gen.pt")
        print("predicted nn on synthetic points")
        # print(x_gen.shape)

        # x_gen = torch.load('x_gen.pt')
        # Adjust the number of points generated
        x_gen = x_gen[: 1 * x_min.shape[0]]

        # print(x_gen.shape)
        # x_all = x_train
        x_all = torch.cat([x_train] + 2 * [x_min] + [x_gen])
        # reduced = reducer.fit_transform(x_all)
        # y_all = y_train
        y_all = torch.cat([y_train] + 2 * [y_min] + [torch.ones(len(x_gen))])
        # plt.scatter(reduced[:, 0],reduced[:, 1], c=y_all)
        # plt.show()

        ds_train = TensorDataset(x_all.float(), y_all.float())
        ds_eval = TensorDataset(x_test.float(), y_test.float())
        c = Classifier(x_all.shape[1])
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
                EarlyStopping(monitor="val/loss", patience=50, mode="max")
            ],
        )
        trainer.fit(
            c,
            DataLoader(
                ds_train,
                batch_size=40024,
                sampler=BalanceClassSampler(y_all.tolist(), mode="upsampling"),
            ),
            DataLoader(ds_eval, batch_size=400000),
        )
        c.eval()
        with (ds_path / "gan_tt_maj_1_2_mlp_ap.txt").open(
            "w", encoding="utf-8"
        ) as f:
            f.write(
                str(
                    c.val_metric(
                        c(x_test.float()), y_test.unsqueeze(-1).float()
                    ).item()
                )
            )
        # trainer.save_checkpoint(
        #     "saved_experiments/gan_vanilla_keep_5_4/celeba_balanced.ckpt"
        # )
        # exp = SummaryWriter("saved_experiments/gan_vanilla_keep_5_4")
        # exp.add_pr_curve(
        #     "celeba_balanced",
        #     y_test.unsqueeze(-1).int(),
        #     torch.sigmoid(c(x_test.float())),
        #     num_thresholds=511,
        # )
