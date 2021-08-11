import pytorch_lightning as pl
import torch
from torch.functional import Tensor
from .components import *
from torch.nn import functional as F


class CVAE(pl.LightningModule):
    def __init__(
            self,
            latent_dim: int = 32,
            img_dim: tuple[int, int, int] = (1, 28, 28),
            batch_norm: bool = False,
            lr: float = 1e-3,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.img_dim = img_dim

        self.encoder = Encoder(latent_dim, img_dim, 10, batch_norm)
        self.decoder = Decoder(latent_dim, img_dim, 10, batch_norm)

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

        p, q, z = self.encode(x, y)
        x_hat = self.decode(z, y)

        loss_recon = F.binary_cross_entropy(x_hat, x)
        loss_kld = (q.log_prob(z) - p.log_prob(z)).mean()
        loss = loss_recon + loss_kld

        self.log('loss/recon', loss_recon, on_step=True, on_epoch=False)
        self.log('loss/kld', loss_kld, on_step=True, on_epoch=False)

        return loss
