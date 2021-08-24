import numpy as np
import torch
from torch._C import device
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


class PairedLatentDimInterpolator(Callback):
    def __init__(
        self,
        title: str,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        steps: int = 10,
    ):
        super().__init__()
        self.title = title
        self.steps = steps
        self.img_1 = img_1
        self.img_2 = img_2

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        imgs = []
        imgs.append(self.img_1[0].to(pl_module.device))
        with torch.no_grad():
            pl_module.eval()
            z_rel_1, z_unrel_1 = pl_module.encode(self.img_1.to(pl_module.device))
            z_rel_2, z_unrel_2 = pl_module.encode(self.img_2.to(pl_module.device))
            for i in range(self.steps - 1):
                w = i / self.steps
                z_unrel = torch.lerp(z_unrel_1, z_unrel_2, w)
                img = pl_module.decode(z_rel_1, z_unrel).squeeze_()
                imgs.append(img)
            pl_module.train()
        imgs.append(self.img_2[0].to(pl_module.device))

        imgs = torch.cat(imgs, dim=1)
        num_rows = self.steps
        grid = torchvision.utils.make_grid(imgs, nrow=num_rows, normalize=True)
        str_title = f'{pl_module.__class__.__name__}_{self.title}'
        trainer.logger.experiment.add_image(
            str_title, grid, global_step=trainer.global_step)


class CVAEImageSampler(Callback):
    def __init__(
        self,
    ):
        super().__init__()

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        imgs = []
        with torch.no_grad():
            pl_module.eval()
            for c in range(10):
                z = torch.randn((10, pl_module.hparams.latent_dim), device=pl_module.device)
                output = pl_module(z, (torch.ones((10), device=pl_module.device) * c).long())
                imgs.append(torchvision.utils.make_grid(output, 10, normalize=True))
            pl_module.train()

        grid = torchvision.utils.make_grid(imgs, 1, normalize=True)

        str_title = f'{pl_module.__class__.__name__}_samples'
        trainer.logger.experiment.add_image(
            str_title, grid, global_step=trainer.global_step)
