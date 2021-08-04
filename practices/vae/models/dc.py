import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, img_channels: int):
        super().__init__()

        self.latent_dim = latent_dim

        self.layer = nn.Sequential(
            nn.Conv2d(img_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, latent_dim * 2)
        )
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(128, latent_dim * 2)

    def forward(self, x: Tensor):
        x = self.layer(x)
        # x = self.avgpool(x)
        # x = self.fc(x)
        mu = x[..., :self.latent_dim]
        lv = x[..., self.latent_dim:]

        std = torch.exp(lv / 2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return mu, lv, z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, img_height: int, img_channels: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_channels = img_channels

        self.layer = nn.Sequential(
            # (B, C) -> (B, C, H, W)
            nn.Unflatten(-1, (latent_dim, 1, 1)),
            # 1 to 4
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 to 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 to 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 to 32
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Sigmoid(),  # (B, img_channels, 32, 32),
            nn.Upsample(img_height)
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.layer(z)


class DCVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 4,
        img_dim: tuple[int, int, int] = (1, 28, 28),
        lr: float = 1e-4,
        *args: any,
        **kwargs: any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.img_dim = img_dim

        self.encoder = Encoder(latent_dim, img_dim[0])
        self.decoder = Decoder(latent_dim, img_dim[1], img_dim[0])

    def encode(self, x: Tensor):
        return self.encoder.forward(x)

    def decode(self, z: Tensor):
        return self.decoder.forward(z)

    def _step(self, x) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, lv, z = self.encode(x)
        return self.decode(z), mu, lv, z

    def forward(self, z):
        return self.decode(z)

    def training_step(self, batch: list[Tensor, Tensor], batch_idx):
        x, y = batch
        x_hat, mu, lv, z = self._step(x)

        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        loss = recon_loss + kld_loss

        self.log_dict({'recon_loss': recon_loss,
                      'kld_loss': kld_loss}, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
