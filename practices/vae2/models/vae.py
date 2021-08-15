import pytorch_lightning as pl
import torch
import torchmetrics.classification.accuracy
from torch import ones_like, zeros_like
from torch.nn import functional as F

from .components import *


def kl_loss(p, q, z):
    log_qz = q.log_prob(z)
    log_pz = p.log_prob(z)
    kl = log_qz - log_pz
    kl = kl.mean()
    return kl


class VAE(pl.LightningModule):
    def __init__(
            self,
            latent_dim: int = 32,
            img_dim: tuple[int, int, int] = (1, 28, 28),
            num_classes: int = 10,
            lr: float = 1e-4,
            adam_beta1: float = 0.5,
            hidden_dim: int = 256,
            normalize: bool = True,
            noise_dim: int = None,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.img_dim = img_dim
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_recon = torch.nn.L1Loss()
        self.criterion_kld = kl_loss
        self.metric_accuracy = torchmetrics.classification.accuracy.Accuracy()

        self.encoder = Encoder(latent_dim, img_dim,
                               num_classes, hidden_dim, normalize)
        self.decoder = Decoder(latent_dim, img_dim,
                               hidden_dim, normalize, noise_dim)

    def encode(self, x):
        p, q, z_related, z_unrelated, y_hat = self.encoder.forward(x)
        return p, q, z_related, z_unrelated, y_hat

    def decode(self, z_related, z_unrelated):
        x_hat = self.decoder.forward(z_related, z_unrelated)
        return x_hat

    def forward(self, z_related, z_unrelated):
        return self.decode(z_related, z_unrelated)

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.adam_beta1
        betas = (beta1, 0.999)

        return torch.optim.Adam(self.parameters(), lr=lr, betas=betas)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch

        p, q, z_rel, z_unrel, y_hat = self.encode(x)
        x_hat = self.decode(z_rel, z_unrel)
        loss_cls = self.criterion_cls(y_hat, y) * 2
        loss_recon = self.criterion_recon(x_hat, x) * 10
        loss_kld = self.criterion_kld(p, q, z_unrel) * 0.1

        loss = loss_recon + loss_kld + loss_cls

        # Logging
        self.log(f"{self.__class__.__name__}/recon", loss_recon)
        self.log(f"{self.__class__.__name__}/kld", loss_kld)
        accuracy = self.metric_accuracy(y_hat, y)
        self.log(f"{self.__class__.__name__}/acc", accuracy)

        return loss
