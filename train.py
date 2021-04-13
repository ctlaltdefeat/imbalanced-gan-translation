from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from torch.nn import SELU, Dropout, Linear, Parameter, ReLU, Sequential, init
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from models import WGANGP

if __name__ == "__main__":
    x, y, x_maj, y_maj, x_min, y_min = torch.load("ds.pt")
    ds = TensorDataset(x_min.float())
    # train_ds = torch.utils.data.Subset(ds, list(range(len(ds) - 800)))
    # test_ds = torch.utils.data.Subset(
    #     ds, list(range(len(ds) - 800, len(ds)))
    # )
    model = WGANGP(latent_dim=21, output_dim=21, lr=1e-4)
    trainer = Trainer(gpus=1, max_epochs=10000)
    trainer.fit(model, DataLoader(ds, batch_size=1024))