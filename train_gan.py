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

from models import WGANGP

if __name__ == "__main__":
    seed_everything(123, workers=True)
    x, y, x_maj, y_maj, x_min, y_min = torch.load("ds_train.pt")
    ds = TensorDataset(x_min.float())
    # train_ds = torch.utils.data.Subset(ds, list(range(len(ds) - 800)))
    # test_ds = torch.utils.data.Subset(
    #     ds, list(range(len(ds) - 800, len(ds)))
    # )
    model = WGANGP(
        latent_dim=x.shape[1],
        output_dim=x.shape[1],
        lr=1e-4,
        x_maj=x_maj,
        vanilla=True
    )
    trainer = Trainer(gpus=1, max_epochs=5000)
    trainer.fit(model, DataLoader(ds, batch_size=100024))
