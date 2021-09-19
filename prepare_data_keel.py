import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from pathlib import Path


def dats_to_torch(f, type_ds):
    lines = open(f, "r").readlines()
    skip = next(i for i, v in enumerate(lines) if not v.startswith("@"))
    df = pd.read_csv(f, header=None, skiprows=skip)
    df = df.rename({df.shape[1] - 1: "class"}, axis=1)
    df = df.replace({" negative": 0, " positive": 1})
    df = df.replace({"M": 0, "F": 1, "I": 2})
    print(df["class"].value_counts())
    X = StandardScaler(with_std=True).fit_transform(
        df[[c for c in df.columns if c != "class"]]
    )
    y = df["class"]
    X_maj = X[y == 0]
    y_maj = y[y == 0]
    X_min = X[y == 1]
    y_min = y[y == 1]
    X = np.concatenate([X_maj, X_min])
    y = np.concatenate([y_maj, y_min])
    torch.save(
        [
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(X_maj),
            torch.from_numpy(y_maj.to_numpy()),
            torch.from_numpy(X_min),
            torch.from_numpy(y_min.to_numpy()),
        ],
        (f.parent / "ds_{}.pt".format(type_ds)),
    )


# for ds_path in Path("datasets_keel").iterdir():
#     if "abalone" in ds_path.name:
#         for f in ds_path.iterdir():
#             if "dat" in f.suffix:
#                 if "1tra" in f.name:
#                     dats_to_torch(f, "train")
#                 if "1tst" in f.name:
#                     dats_to_torch(f, "test")


for ds_path in Path("datasets_other").iterdir():
    x_train_all, y_train_all, _, _, _, _ = torch.load(ds_path / "ds_train.pt")
    perm = torch.randperm(x_train_all.size()[0])
    x_train_all, y_train_all = x_train_all[perm], y_train_all[perm]
    train_size = int(0.8 * len(x_train_all))
    x_train, y_train = x_train_all[:train_size], y_train_all[:train_size]
    x_test, y_test = x_train_all[train_size:], y_train_all[train_size:]
    torch.save([x_train, y_train], ds_path / 'x_train_train.pt')
    torch.save([x_test, y_test], ds_path / 'x_train_val.pt')

