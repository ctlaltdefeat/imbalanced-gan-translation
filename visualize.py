import torch
from models import WGANGP
import numpy as np
import matplotlib.pyplot as plt
import umap

reducer = umap.UMAP()
x, y, x_maj, y_maj, x_min, y_min = torch.load(r"C:\Users\Jonathan\PycharmProjects\imbalanced-gan-translation\datasets_other\celeba\ds_train.pt")
model = WGANGP.load_from_checkpoint(
    r"C:\Users\Jonathan\PycharmProjects\imbalanced-gan-translation\datasets_other\celeba\test_cyclic.ckpt",
    strict=False,
)
# x_gen = model(torch.Tensor([4, 4]).unsqueeze(0)).detach().numpy()
x_gen = model(x_maj[:100].float()).detach().numpy()
maj_gen = model.gen2(x_min[:200].float()).detach().numpy()
# x_gen = torch.load(r"C:\Users\Jonathan\PycharmProjects\imbalanced-gan-translation\datasets_keel\yeast6\x_gen.pt")
# x_gen = x_gen[: 2 * x_min.shape[0]]
x_all = np.concatenate([x_min[:500].numpy(), x_maj[:2000], x_gen, maj_gen])
reduced = reducer.fit_transform(x_all)
y_all = np.concatenate([y_min[:500].numpy(), [3] * len(x_maj[:2000]), [2] * len(x_gen), [4] * len(maj_gen)])
plt.scatter(reduced[:, 0], reduced[:, 1], c=y_all)
plt.show()