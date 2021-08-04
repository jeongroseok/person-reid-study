import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class LatentDimInterpolator(Callback):
    def __init__(
        self,
        interpolate_epoch_interval: int = 1,
        range_start: float = -1.0,
        range_end: float = 1.0,
        steps: int = 11,
        num_samples: int = 1,
        normalize: bool = True,
    ):
        super().__init__()
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.normalize = normalize
        self.steps = steps

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(
                pl_module,
                # type: ignore[union-attr]
                latent_dim=pl_module.hparams.latent_dim
            )
            images = torch.cat(images, dim=0)  # type: ignore[assignment]

            num_rows = self.steps
            grid = torchvision.utils.make_grid(
                images, nrow=num_rows, normalize=self.normalize)
            str_title = f'{pl_module.__class__.__name__}_latent_space'
            trainer.logger.experiment.add_image(
                str_title, grid, global_step=trainer.global_step)

    def interpolate_latent_space(self, pl_module: LightningModule, latent_dim: int) -> list[Tensor]:
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                for z2 in np.linspace(self.range_start, self.range_end, self.steps):
                    # set all dims to zero
                    z = torch.zeros(self.num_samples, latent_dim,
                                    device=pl_module.device)

                    # set the fist 2 dims to the value
                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    # sample
                    # generate images
                    img: torch.Tensor = pl_module(z)

                    if len(img.size()) == 2:
                        img = img.view(self.num_samples, *pl_module.img_dim)

                    img = img[0]
                    img = img.unsqueeze(0)
                    images.append(img)

        pl_module.train()
        return images
