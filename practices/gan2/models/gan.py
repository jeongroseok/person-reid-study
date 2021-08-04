import pytorch_lightning as pl
import torch
import components
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F


class GAN(pl.LightningModule):
    def __init__(
            self,
            style_latent_dim: int = 16,
            class_latent_dim: int = 16,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.encoder = components.Encoder(style_latent_dim, class_latent_dim)
        self.decoder = components.Decoder(style_latent_dim, class_latent_dim)
        self.discriminator = components.Discriminator()

    def forward(self, x: Tensor):
        mu, log_var, label = self.encoder.forward(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder.forward(z, label)
    
    def _vae_step(self, x: Tensor):
        mu, log_var, label = self.encoder.forward(x)
        p, q, z = self.sample(mu, log_var)
        x_hat = self.decoder.forward(z, label)
        
        recon_loss = F.mse_loss(x_hat, x)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl: Tensor = log_qz - log_pz
        kl = kl.mean()
        kl *= self

        F.kl_div()



    @staticmethod
    def sample(mu: Tensor, log_var: Tensor):
        std = torch.exp(log_var / 2)
        p = Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = Normal(mu, std)
        z = q.rsample()
        return p, q, z