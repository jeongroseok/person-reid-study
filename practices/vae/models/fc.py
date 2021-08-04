import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, img_height: int):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_height ** 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )

    def forward(self, x: Tensor):
        x = self.fc(x)
        mu = x[..., :self.latent_dim]
        lv = x[..., self.latent_dim:]

        std = torch.exp(lv / 2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return mu, lv, z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, img_height: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_height = img_height

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_height ** 2),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor):
        x_hat: Tensor = self.decoder(z)
        x_hat = x_hat.unflatten(1, (1, self.img_height, self.img_height))
        return x_hat


class FCVAE(pl.LightningModule):
    img_dim = (1, 28, 28)

    def __init__(self, latent_dim: int = 4, lr: float = 0.0005, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.encoder = Encoder(latent_dim, 28)
        self.decoder = Decoder(latent_dim, 28)

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
        
        self.log_dict({'recon_loss': recon_loss, 'kld_loss': kld_loss}, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
