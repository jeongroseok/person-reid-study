import pytorch_lightning as pl
import torch
from torch.functional import Tensor
from .components import *
from torch.nn import functional as F


class CGAN(pl.LightningModule):
    def __init__(
            self,
            latent_dim: int = 32,
            img_shape: tuple[int, int, int] = (1, 28, 28),
            num_classes: int = 10,
            lr: float = 1e-3,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.generator = Generator(latent_dim, img_shape, num_classes)
        self.discriminator = Discriminator(img_shape, num_classes)

    def generate(self, z, c) -> torch.Tensor:
        return self.generator(z, c)

    def forward(self, z: torch.Tensor, c: torch.Tensor = None):
        if c is None:
            c = torch.zeros(z.size(0), dtype=torch.long, device=self.device)
        return self.generate(z, c)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr)
        return [
            {
                'optimizer': opt_gen,
            },
            {
                'optimizer': opt_disc,
            },
        ]

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx, optimizer_idx):
        x, y = batch
        
        if optimizer_idx == 0:
            z = torch.randn((x.size(0), self.hparams.latent_dim), device=self.device)
            y = torch.randint_like(y, 0, self.hparams.num_classes)
            x_hat = self.generate(z, y)
            d_fake = self.discriminator(x_hat, y)

            loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))

            self.log("loss/gen", loss, on_step=True, on_epoch=False)
            
            return loss
        elif optimizer_idx == 1:
            z = torch.randn((x.size(0), self.hparams.latent_dim), device=self.device)
            x_hat = self.generate(z, y)

            d_fake = self.discriminator(x_hat, y)
            loss_fake = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))

            d_real = self.discriminator(x, y)
            loss_real = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
            loss = (loss_fake + loss_real) / 2

            self.log("loss/disc", loss, on_step=True, on_epoch=False)

            return loss

