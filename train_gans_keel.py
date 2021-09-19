from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
from torch.nn import SELU, Dropout, Linear, Parameter, ReLU, Sequential, init
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from pathlib import Path

from models import WGANGP

if __name__ == "__main__":
    seed_everything(123, workers=True)
    for ds_path in Path("datasets_other").iterdir():
        if "abalone" in ds_path.name:
            continue
        x, y, x_maj, y_maj, x_min, y_min = torch.load(ds_path / "ds_train.pt")
        ds = TensorDataset(x_min.float())
        ds_maj = TensorDataset(x_maj.float())
        # train_ds = torch.utils.data.Subset(ds, list(range(len(ds) - 800)))
        # test_ds = torch.utils.data.Subset(
        #     ds, list(range(len(ds) - 800, len(ds)))
        # )
        model = WGANGP(
            latent_dim=x.shape[1],
            output_dim=x.shape[1],
            lr=1e-4,
            x_maj=x_maj,
            vanilla=True,
            cyclic_loss=10,
            identity_loss=5,
        )
        model.load_from_checkpoint(
            r"C:\Users\Jonathan\PycharmProjects\imbalanced-gan-translation\datasets_other\celeba\test_cyclic.ckpt",
            strict=False,
        )
        trainer = Trainer(
            gpus=1, max_epochs=2000, multiple_trainloader_mode="min_size"
        )
        trainer.fit(
            model,
            train_dataloaders={
                "min": DataLoader(ds, batch_size=3000, shuffle=True),
                "maj": DataLoader(ds_maj, batch_size=3000, shuffle=True),
            },
        )
        trainer.save_checkpoint(ds_path / "just_cyclic.ckpt")
