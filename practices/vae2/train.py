import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision.utils
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_examples import _DATASETS_PATH
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib
from torchvision.datasets.mnist import MNIST

from callbacks import LatentSpaceVisualizer, LatentDimInterpolator
from pl_bolts.datamodules import MNISTDataModule
from models.vae import VAE


def set_persistent_workers(data_module: VisionDataModule):
    def _data_loader(self: VisionDataModule,
                     dataset: Dataset,
                     shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
    data_module._data_loader = _data_loader


def main(args=None):
    set_persistent_workers(MNISTDataModule)

    transforms = transform_lib.Compose([
        transform_lib.ToTensor(),
        transform_lib.Normalize((0.5, ), (0.5, )),
    ])
    datamodule = MNISTDataModule(_DATASETS_PATH, num_workers=3,
                                 batch_size=256, shuffle=False, drop_last=True,
                                 train_transforms=transforms,
                                 val_transforms=transforms,
                                 test_transforms=transforms)
    model = VAE(8, datamodule.dims, lr=1e-3,
                normalize=True, hidden_dim=256, noise_dim=8)

    dataset = MNIST(_DATASETS_PATH, False, transform=transforms)
    callbacks = [
        LatentSpaceVisualizer(dataset),
        LatentDimInterpolator(dataset),
    ]

    trainer = pl.Trainer(
        gpus=-1,
        progress_bar_refresh_rate=5,
        max_epochs=1000,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
