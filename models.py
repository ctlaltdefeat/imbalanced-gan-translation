# from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter, init, SELU, Sequential, Linear, ReLU, Dropout
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from torch.utils.data.dataset import TensorDataset


# def fc1(num_classes):
#     return torch.nn.Sequential(
#         Linear(96, 256),
#         SELU(),
#         Dropout(),
#         Linear(256, 256),
#         SELU(),
#         Dropout(),
#         Linear(256, 256),
#         SELU(),
#         Dropout(),
#         Linear(256, num_classes),
#     )


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            # if normalize:
            #     layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(SELU())
            # layers.append(Dropout())
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Linear(256, int(output_dim)),
            # nn.Tanh()
        )

    def forward(self, z):
        output = self.model(z)
        # img = img.view(img.size(0), *self.img_shape)
        return output


class Discriminator(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(output_dim), 128),
            SELU(),
            nn.Linear(128, 64),
            SELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class WGANGP(LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        output_dim: int = 10,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        vanilla=True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore='x_maj')

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.vanilla = vanilla

        # networks
        self.generator = Generator(
            latent_dim=self.latent_dim, output_dim=self.output_dim
        )
        self.discriminator = Discriminator(output_dim=self.output_dim)

        self.example_input_array = torch.zeros(2, self.latent_dim)

        if 'x_maj' in kwargs and kwargs['x_maj'] is not None:
            self.register_buffer('x_maj', kwargs['x_maj'])

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(
            self.device
        )
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = (
            torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        )
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     imgs = batch[0]

    #     # sample noise
    #     z = torch.randn(imgs.shape[0], self.latent_dim)
    #     z = z.type_as(imgs)

    #     lambda_gp = 10

    #     # train generator
    #     if optimizer_idx == 0:

    #         # generate images
    #         self.generated_imgs = self(z)

    #         # log sampled images
    #         # sample_imgs = self.generated_imgs[:6]
    #         # grid = torchvision.utils.make_grid(sample_imgs)
    #         # self.logger.experiment.add_image("generated_images", grid, 0)

    #         # ground truth result (ie: all fake)
    #         # put on GPU because we created this tensor inside training_loop
    #         valid = torch.ones(imgs.size(0), 1)
    #         valid = valid.type_as(imgs)

    #         # adversarial loss is binary cross-entropy
    #         g_loss = -torch.mean(self.discriminator(self(z)))
    #         tqdm_dict = {"g_loss": g_loss}
    #         output = OrderedDict(
    #             {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
    #         )
    #         return output

    #     # train discriminator
    #     # Measure discriminator's ability to classify real from generated samples
    #     elif optimizer_idx == 1:
    #         fake_imgs = self(z)

    #         # Real images
    #         real_validity = self.discriminator(imgs)
    #         # Fake images
    #         fake_validity = self.discriminator(fake_imgs)
    #         # Gradient penalty
    #         gradient_penalty = self.compute_gradient_penalty(
    #             imgs.data, fake_imgs.data
    #         )
    #         # Adversarial loss
    #         d_loss = (
    #             -torch.mean(real_validity)
    #             + torch.mean(fake_validity)
    #             + lambda_gp * gradient_penalty
    #         )

    #         tqdm_dict = {"d_loss": d_loss}
    #         output = OrderedDict(
    #             {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
    #         )
    #         return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch[0]

        # sample noise
        # z = torch.randn(imgs.shape[0], self.latent_dim)
        # z = (torch.rand(imgs.shape[0], self.latent_dim)-0.5)*2*3
        if self.vanilla:
            z = torch.randn(imgs.shape[0], self.latent_dim)
        else:
            z = self.x_maj[torch.randint(len(self.x_maj), (imgs.shape[0],))]
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            if self.vanilla:
                g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            else:
                g_loss = self.adversarial_loss(self.discriminator(self(z)), valid) + 0.1 * nn.L1Loss()(z, self.generated_imgs)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    # def configure_optimizers(self):
    #     n_critic = 5

    #     lr = self.lr
    #     b1 = self.b1
    #     b2 = self.b2

    #     opt_g = torch.optim.Adam(
    #         self.generator.parameters(), lr=lr, betas=(b1, b2)
    #     )
    #     opt_d = torch.optim.Adam(
    #         self.discriminator.parameters(), lr=lr, betas=(b1, b2)
    #     )
    #     return (
    #         {"optimizer": opt_g, "frequency": 1},
    #         {"optimizer": opt_d, "frequency": n_critic},
    #     )

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        return [opt_g, opt_d], []
