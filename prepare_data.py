import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch


df = pd.read_csv("KDD2014_donors_10feat_nomissing_normalised.csv")
print(df["class"].value_counts())
df = df.sample(frac=1, random_state=123).reset_index(drop=True)
num_samples = df.shape[0]
X_all = StandardScaler(with_std=True).fit_transform(
    df[[c for c in df.columns if c != "class"]]
)
y_all = df["class"]
X = X_all[: -int(0.2 * num_samples)]
y = y_all[: -int(0.2 * num_samples)]
X_maj = X[y == 0]
y_maj = y[y == 0]
# X_min = X[y==1][:len(X[y==1])//100]
# y_min = y[y==1][:len(X[y==1])//100]
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
    "ds_imba_train.pt",
)

X = X_all[int(0.2 * num_samples) :]
y = y_all[int(0.2 * num_samples) :]
X_maj = X[y == 0]
y_maj = y[y == 0]
# X_min = X[y==1][:len(X[y==1])//100]
# y_min = y[y==1][:len(X[y==1])//100]
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
    "ds_imba_test.pt",
)
