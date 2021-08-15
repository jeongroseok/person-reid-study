from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    def __init__(
        self,
        # latent_related_dim: int,
        # latent_unrelated_dim: int,
        latent_dim: int,
        img_dim: tuple[int, int, int],
        num_classes: int,
        first_hidden_dim: int = 256,
        normalize: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize: bool = normalize):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.fc = nn.Sequential(
            nn.Flatten(),
            *block(np.prod(img_dim), first_hidden_dim, False),
            *block(first_hidden_dim, first_hidden_dim // 2),
            *block(first_hidden_dim // 2, first_hidden_dim // 4),
        )
        self.related = nn.Sequential(
            nn.Linear(first_hidden_dim // 4, first_hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(first_hidden_dim // 8, latent_dim),
        )
        self.unrelated = nn.Sequential(
            nn.Linear(first_hidden_dim // 4, first_hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(first_hidden_dim // 8, latent_dim * 2)
        )
        self.classification = nn.Linear(latent_dim, num_classes)

    def forward(self, x: Tensor):
        x_fc = self.fc(x)
        z_related = self.related(x_fc)
        x_unrelated = self.unrelated(x_fc)
        mu = x_unrelated[..., :self.latent_dim]
        lv = x_unrelated[..., self.latent_dim:]

        std = torch.exp(lv / 2)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z_unrelated = q.rsample()

        y_hat = self.classification(z_related)

        return p, q, z_related, z_unrelated, y_hat


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_dim: tuple[int, int, int],
        last_hidden_dim: int = 256,
        normalize: bool = True,
        noise_dim: Optional[int] = None
    ):
        super().__init__()

        self.noise_dim = noise_dim

        def block(in_feat, out_feat, normalize: bool = normalize):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        first_features = latent_dim * \
            2 if noise_dim is None else (latent_dim * 2) + noise_dim
        self.fc = nn.Sequential(
            *block(first_features, last_hidden_dim // 4),
            *block(last_hidden_dim // 4, last_hidden_dim // 2),
            *block(last_hidden_dim // 2, last_hidden_dim),
            nn.Linear(last_hidden_dim, np.prod(img_dim)),
            nn.Tanh(),
            nn.Unflatten(1, img_dim)
        )

    def forward(self, z_related, z_unrelated):  # TODO: noise 추가 해야함
        z = torch.cat([z_related, z_unrelated], -1)
        if self.noise_dim is not None:
            z = torch.cat(
                [z, torch.randn(z.size(0), self.noise_dim, device=z.device)], -1)
        x_hat: torch.Tensor = self.fc(z)
        return x_hat


class Discriminator(nn.Module):
    def __init__(
        self,
        img_dim: tuple[int, int, int],
        num_classes: int,
        first_hidden_dim: int = 256,
        normalize: bool = True,
    ):
        super().__init__()

        def block(in_feat, out_feat, normalize: bool = normalize):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.fc = nn.Sequential(
            nn.Flatten(),
            *block(np.prod(img_dim), first_hidden_dim, False),
            *block(first_hidden_dim, first_hidden_dim // 2),
            *block(first_hidden_dim // 2, first_hidden_dim // 4),
        )
        self.fc2_discrimination = nn.Sequential(
            nn.Linear(first_hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.fc2_classification = nn.Linear(first_hidden_dim // 4, num_classes)

    def forward(self, x):
        x = self.fc(x)
        d: Tensor = self.fc2_discrimination(x)
        y: Tensor = self.fc2_classification(x)
        return d, y
