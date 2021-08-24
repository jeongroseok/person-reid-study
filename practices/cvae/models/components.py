import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, img_dim: tuple[int, int, int], num_classes: int, batch_norm: bool):
        super().__init__()

        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(num_classes, num_classes)

        if batch_norm:
            self.fc = nn.Sequential(
                nn.Linear(np.prod(img_dim) + num_classes, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(np.prod(img_dim) + num_classes, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_lv = nn.Linear(128, latent_dim)

    def forward(self, x, c):
        c = self.embedding(c)
        x = torch.cat((x.flatten(1), c), 1)
        x = self.fc(x)
        mu = self.fc_mu(x)
        lv = self.fc_lv(x)
        
        std = torch.exp(lv / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return p, q, z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, img_dim: tuple[int, int, int], num_classes: int, batch_norm: bool):
        super().__init__()

        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(num_classes, num_classes)

        if batch_norm:
            self.fc = nn.Sequential(
                nn.Linear(latent_dim + num_classes, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, np.prod(img_dim)),
                nn.Sigmoid(),
                nn.Unflatten(1, img_dim)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(latent_dim + num_classes, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, np.prod(img_dim)),
                nn.Sigmoid(),
                nn.Unflatten(1, img_dim)
            )

    def forward(self, z, c):
        c = self.embedding(c)
        z = torch.cat((z, c), 1)
        x_hat: torch.Tensor = self.fc(z)
        return x_hat
