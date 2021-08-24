import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple[int, int, int], num_classes: int):
        super().__init__()

        self.img_shape = img_shape
        self.num_classes = num_classes
        img_dim = np.prod(img_shape)

        self.fc = nn.Sequential(
            nn.Linear(img_dim + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        y = F.one_hot(y, self.num_classes)
        x = x.flatten(1)
        x = torch.cat((x, y), 1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: tuple[int, int, int], num_classes: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_shape[0] * img_shape[1] * img_shape[2]),
            nn.Sigmoid(),
            nn.Unflatten(1, self.img_shape)
        )

    def forward(self, z, y):
        y = F.one_hot(y, self.num_classes)
        z = torch.cat((z, y), 1)
        x_hat: torch.Tensor = self.fc(z)
        return x_hat
