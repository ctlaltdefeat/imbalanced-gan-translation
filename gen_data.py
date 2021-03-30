import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from sklearn.datasets import make_blobs, make_moons

X, y = make_moons(n_samples=1000, random_state=0, noise=0.1)
X1 = X[y == 0]
y1 = y[y == 0]
X2 = X[y == 1][:(1000//10)]
y2 = y[y == 1][:(1000//10)]
torch.save(
    [
        torch.from_numpy(np.concatenate([X1, X2])),
        torch.from_numpy(np.concatenate([y1, y2])),
        torch.from_numpy(X1),
        torch.from_numpy(y1),
        torch.from_numpy(X2),
        torch.from_numpy(y2),
    ],
    "ds.pt",
)


# X, y = make_blobs(
#     n_samples=2000, n_features=5, centers=[(0, 0, 0, 0, 0), (1, 1, 1, 1, 1)], cluster_std=10
# )
# X = X[y == 0]
# y = y[y == 0]
# X2, y2 = make_blobs(
#     n_samples=200, n_features=5, centers=[(0, 0, 0, 0, 0), (1, 1, 1, 1, 1)], cluster_std=10
# )
# X2 = X2[y2 == 1]
# y2 = y2[y2 == 1]
# torch.save(
#     [
#         torch.from_numpy(np.concatenate([X, X2])),
#         torch.from_numpy(np.concatenate([y, y2])),
#         torch.from_numpy(X),
#         torch.from_numpy(y),
#         torch.from_numpy(X2),
#         torch.from_numpy(y2),
#     ],
#     "ds_gaussian.pt",
# )
