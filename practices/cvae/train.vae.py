import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from pl_examples import _DATASETS_PATH
from torchvision import transforms as transform_lib
from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
from callbacks import LatentDimInterpolator, CVAEImageSampler
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
    dm = MNISTDataModule(_DATASETS_PATH, num_workers=2,
                         batch_size=128, shuffle=True, drop_last=True,
                         train_transforms=transforms,
                         val_transforms=transforms,
                         test_transforms=transforms)

    model = VAE(latent_dim=4)

    callbacks = [
        TensorboardGenerativeModelImageSampler(10),
        LatentDimInterpolator(),
        CVAEImageSampler()
    ]
    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=1000,
        gpus=-1,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
