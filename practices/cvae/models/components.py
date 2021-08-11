import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, img_dim: tuple[int, int, int], num_classes: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_dim = img_dim

        self.fc = nn.Sequential(
            nn.Linear(img_dim[0] * img_dim[1] *
                      img_dim[2] + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
            nn.ReLU(), # !!! 꼭 추가! 없으면 학습 안됨.
        )

    def forward(self, x, c):
        x = torch.cat((x.flatten(1), F.one_hot(c, 10)), 1)
        x = self.fc(x)
        mu = x[..., :self.latent_dim]
        lv = x[..., self.latent_dim:]

        z = self.reparameterize(mu, lv)

        return mu, lv, z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, img_dim: tuple[int, int, int], num_classes: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_dim = img_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim[0] * img_dim[1] * img_dim[2]),
            nn.Sigmoid(),
            nn.Unflatten(1, self.img_dim)
        )

    def forward(self, z, c):
        z = torch.cat((z, F.one_hot(c, 10)), 1)
        x_hat: torch.Tensor = self.fc(z)
        return x_hat
