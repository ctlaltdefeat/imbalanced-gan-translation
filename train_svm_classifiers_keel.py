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
from sklearn.metrics import average_precision_score
from sklearn.svm import LinearSVC, SVC
import optuna
import pickle
import joblib


def objective(trial: optuna.Trial, ds_path):
    epochs = 3000
    translation_penalty = trial.suggest_categorical(
        "translation_penalty", [0.05, 0.08, 0.1, 0.12, 0.15]
    )
    extra_min_weight = trial.suggest_categorical("extra_min_weight", [0])
    gen_pts_kept = trial.suggest_categorical("gen_pts_kept", [0.5, 1, 2, 3, 8])
    # x_train_all, y_train_all, _, _, _, _ = torch.load(ds_path / "ds_train.pt")
    x_train, y_train = torch.load(ds_path / "x_train_train.pt")
    x_maj = x_train[y_train == 0]
    x_min = x_train[y_train == 1]
    x_test, y_test = torch.load(ds_path / "x_train_val.pt")
    # x_test, y_test, _, _, _, _ = torch.load(ds_path / "ds_test.pt")
    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=3)
    neigh.fit(x_train, y_train)

    gan_model_path = Path(
        ds_path
        / "ttgan_maj_val_{}_epochs_{}_translation_penalty.ckpt".format(
            epochs, translation_penalty
        )
    )
    if not gan_model_path.exists():
        ds = TensorDataset(x_min.float())
        model = WGANGP(
            latent_dim=x_train.shape[1],
            output_dim=x_train.shape[1],
            lr=1e-4,
            x_maj=x_maj,
            vanilla=False,
            translation_penalty=translation_penalty,
        )
        trainer = Trainer(gpus=1, max_epochs=epochs)
        trainer.fit(model, DataLoader(ds, batch_size=100024))
        trainer.save_checkpoint(gan_model_path)
        print("trained new gan")
    model = WGANGP.load_from_checkpoint(gan_model_path, strict=False).cuda()
    # x_gen = model(torch.Tensor([4, 4]).unsqueeze(0)).detach()
    # x_gen = model(torch.randn(5*x_min.shape[0], x_min.shape[1]).cuda()).cpu().detach()
    x_gen = model(x_maj.float().cuda()).cpu().detach()
    print("generated synthetic points")
    # x_gen = x_gen[neigh_res==1]
    x_gen = x_gen[neigh.predict_proba(x_gen)[:, 1].argsort()[::-1].copy()]
    # torch.save(x_gen, ds_path / "x_gen.pt")
    print("predicted nn on synthetic points")
    # print(x_gen.shape)

    # x_gen = torch.load(ds_path / "x_gen.pt")
    # Adjust the number of points generated
    x_gen = x_gen[: int(gen_pts_kept * x_min.shape[0])]

    # print(x_gen.shape)
    # x_all = x_train
    x_all = torch.cat([x_train] + extra_min_weight * [x_min] + [x_gen])
    # reduced = reducer.fit_transform(x_all)
    # y_all = y_train
    y_all = torch.cat(
        [y_train]
        + extra_min_weight * [torch.ones(len(x_min))]
        + [torch.ones(len(x_gen))]
    )
    # plt.scatter(reduced[:, 0],reduced[:, 1], c=y_all)
    # plt.show()

    x_all, y_all = x_all.float().numpy(), y_all.float().numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()
    clf = LinearSVC()
    clf = SVC(kernel="linear", probability=True)
    clf.fit(x_all, y_all)
    return average_precision_score(y_test, clf.predict_proba(x_test)[:, 1])
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


if __name__ == "__main__":
    seed_everything(123, workers=True)
    for ds_path in Path("datasets_keel").iterdir():
        # if "abalone" in ds_path.name:
        #     continue
        if "ecoli" in ds_path.name:
            continue
        if "shuttle" in ds_path.name:
            continue
        if "glass5" in ds_path.name:
            continue
        if "glass-0-1-6_vs_5" in ds_path.name:
            continue
        # if (ds_path / "study.pkl").exists():
        #     continue
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, ds_path), n_trials=45)
        joblib.dump(study, ds_path / "study.pkl")
        # study = joblib.load(ds_path / "study.pkl")
        # best_params = study.best_params
        # x_train, y_train, _, _, _, _ = torch.load(ds_path / "ds_train.pt")
        # x_maj = x_train[y_train == 0]
        # x_min = x_train[y_train == 1]
        # x_test, y_test, _, _, _, _ = torch.load(ds_path / "ds_test.pt")
        # neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=3)
        # neigh.fit(x_train, y_train)
        # gan_model_path = Path(ds_path / "ttgan_maj_optuna_best.ckpt")
        # if not gan_model_path.exists():
        #     ds = TensorDataset(x_min.float())
        #     model = WGANGP(
        #         latent_dim=x_train.shape[1],
        #         output_dim=x_train.shape[1],
        #         lr=1e-4,
        #         x_maj=x_maj,
        #         vanilla=False,
        #         translation_penalty=best_params["translation_penalty"],
        #     )
        #     trainer = Trainer(gpus=1, max_epochs=3000)
        #     trainer.fit(model, DataLoader(ds, batch_size=100024))
        #     trainer.save_checkpoint(gan_model_path)
        # model = WGANGP.load_from_checkpoint(
        #     gan_model_path, strict=False
        # ).cuda()
        # x_gen = model(x_maj.float().cuda()).cpu().detach()
        # x_gen = x_gen[neigh.predict_proba(x_gen)[:, 1].argsort()[::-1].copy()]
        # x_gen = x_gen[: int(best_params["gen_pts_kept"] * x_min.shape[0])]
        # x_all = torch.cat(
        #     [x_train] + best_params["extra_min_weight"] * [x_min] + [x_gen]
        # )
        # y_all = torch.cat(
        #     [y_train]
        #     + best_params["extra_min_weight"] * [torch.ones(len(x_min))]
        #     + [torch.ones(len(x_gen))]
        # )
        # x_all, y_all = x_all.float().numpy(), y_all.float().numpy()
        # x_test, y_test = x_test.numpy(), y_test.numpy()
        # clf = LinearSVC()
        # clf = SVC(kernel="linear", probability=True)
        # clf.fit(x_all, y_all)
        # Path(ds_path / "optuna_ap.txt").write_text(
        #     str(
        #         average_precision_score(
        #             y_test, clf.predict_proba(x_test)[:, 1]
        #         )
        #     )
        # )
        # Path(ds_path / "optuna_best_params.txt").write_text(
        #     str(
        #         best_params
        #     )
        # )
