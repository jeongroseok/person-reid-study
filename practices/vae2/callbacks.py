import numpy as np
import torch
from torch.utils import data
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torchvision.datasets import VisionDataset
from models.vae import VAE


class LatentSpaceVisualizer(Callback):
    def __init__(
        self,
        dataset: VisionDataset,
        steps: int = 11,
    ):
        super().__init__()
        self.steps = steps

        self.points = torch.stack(
            [dataset[np.random.choice(len(dataset))][0] for _ in range(4)]
        )

    def on_epoch_end(self, trainer: Trainer, pl_module: VAE) -> None:
        num_rows = self.steps
        images = self.interpolate_latent_space(pl_module)
        images = torch.cat(images, 0)
        grid = torchvision.utils.make_grid(images, num_rows, normalize=True)

        str_title = f'{pl_module.__class__.__name__}_latent_space'

        trainer.logger.experiment.add_image(
            str_title, grid, global_step=trainer.global_step)

    def interpolate_latent_space(self, pl_module: VAE) -> list[Tensor]:
        points = self.points.to(pl_module.device)
        images = []
        with torch.no_grad():
            pl_module.eval()
            _, _, z_rel, z_unrel, y_hat = pl_module.encode(points)

            for y in np.linspace(0, 1, self.steps):
                for x in np.linspace(0, 1, self.steps):
                    z_rel_cur = torch.lerp(
                        torch.lerp(z_rel[0], z_rel[1], x),
                        torch.lerp(z_rel[2], z_rel[3], x),
                        y
                    ).unsqueeze_(0)
                    z_unrel_cur = torch.lerp(
                        torch.lerp(z_unrel[0], z_unrel[1], x),
                        torch.lerp(z_unrel[2], z_unrel[3], x),
                        y
                    ).unsqueeze_(0)

                    img = pl_module.decode(z_rel_cur, z_unrel_cur)
                    images.append(img)

        pl_module.train()
        return images


class LatentDimInterpolator(Callback):
    def __init__(
        self,
        dataset: VisionDataset,
        steps: int = 11,
    ):
        super().__init__()
        self.steps = steps

        self.points = torch.stack(
            [dataset[np.random.choice(len(dataset))][0] for _ in range(2)]
        )

    def on_epoch_end(self, trainer: Trainer, pl_module: VAE) -> None:
        num_rows = self.steps
        images_rel = self.interpolate_latent_space_related(pl_module)
        images_rel = torch.cat(images_rel, 0)
        grid_rel = torchvision.utils.make_grid(images_rel, num_rows, normalize=True)
        
        images_unrel = self.interpolate_latent_space_unrelated(pl_module)
        images_unrel = torch.cat(images_unrel, 0)
        grid_unrel = torchvision.utils.make_grid(images_unrel, num_rows, normalize=True)

        str_title = f'{pl_module.__class__.__name__}_latent_space:related'
        trainer.logger.experiment.add_image(
            str_title, grid_rel, global_step=trainer.global_step)
        
        str_title = f'{pl_module.__class__.__name__}_latent_space:unrelated'
        trainer.logger.experiment.add_image(
            str_title, grid_unrel, global_step=trainer.global_step)

    def interpolate_latent_space_related(self, pl_module: VAE) -> list[Tensor]:
        points = self.points.to(pl_module.device)
        images = []
        with torch.no_grad():
            pl_module.eval()
            _, _, z_rel, z_unrel, y_hat = pl_module.encode(points)

            for w in np.linspace(0, 1, self.steps):
                z_rel_cur = torch.lerp(z_rel[0], z_rel[1], w).unsqueeze_(0)
                img = pl_module.decode(z_rel_cur, z_unrel[0].unsqueeze(0))
                images.append(img)

        pl_module.train()
        return images

    def interpolate_latent_space_unrelated(self, pl_module: VAE) -> list[Tensor]:
        points = self.points.to(pl_module.device)
        images = []
        with torch.no_grad():
            pl_module.eval()
            _, _, z_rel, z_unrel, y_hat = pl_module.encode(points)

            for w in np.linspace(0, 1, self.steps):
                z_unrel_cur = torch.lerp(z_unrel[0], z_unrel[1], w).unsqueeze_(0)
                img = pl_module.decode(z_rel[0].unsqueeze(0), z_unrel_cur)
                images.append(img)

        pl_module.train()
        return images
