import pytorch_lightning as pl
import torch
from torch.functional import Tensor
from .components import *
from torch.nn import functional as F


class VAE(pl.LightningModule):
    def __init__(
            self,
            latent_dim: int = 32,
            img_dim: tuple[int, int, int] = (1, 28, 28),
            lr: float = 1e-3,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.img_dim = img_dim

        self.encoder = Encoder(latent_dim, img_dim, 10)
        self.decoder = Decoder(latent_dim, img_dim, 10)

    def encode(self, x, c) -> torch.Tensor:
        return self.encoder(x, c)

    def decode(self, z, c) -> torch.Tensor:
        return self.decoder(z, c)

    def forward(self, z: torch.Tensor, c: torch.Tensor = None):
        if c is None:
            c = torch.zeros(z.size(0), dtype=torch.long, device=self.device)
        return self.decode(z, c)

    def configure_optimizers(self):
        lr = self.hparams.lr
        return torch.optim.Adam(self.parameters(), lr)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        x, y = batch

        mu, lv, z = self.encode(x, y)
        x_hat = self.decode(z, y)

        loss_recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
        loss_kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())

        loss = (loss_recon + loss_kld)
        self.log("loss", loss, on_step=True, on_epoch=False)

        return loss
