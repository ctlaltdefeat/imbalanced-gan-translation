import torch
from models import WGANGP
import numpy as np
import matplotlib.pyplot as plt
import umap

reducer = umap.UMAP()
x, y, x_maj, y_maj, x_min, y_min = torch.load("ds_imba_train.pt")
model = WGANGP.load_from_checkpoint(
    r"C:\Users\Jonathan\PycharmProjects\imbalanced-gan-translation\lightning_logs\version_81\checkpoints\epoch=6040-step=6040.ckpt",
    strict=False,
)
# x_gen = model(torch.Tensor([4, 4]).unsqueeze(0)).detach().numpy()
x_gen = model(x_min.float()).detach().numpy()
x_all = np.concatenate([x_min.numpy(), x_gen])
reduced = reducer.fit_transform(x_all)
y_all = np.concatenate([y_min.numpy(), [2] * len(x_gen)])
plt.scatter(reduced[:, 0], reduced[:, 1], c=y_all)
plt.show()
