import models.dc
import pytorch_lightning as pl
import torch.utils.data
from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_examples import _DATASETS_PATH
from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
from callbacks import LatentDimInterpolator


def set_persistent_workers(data_module: VisionDataModule):
    def _data_loader(self: VisionDataModule,
                     dataset: torch.utils.data.Dataset,
                     shuffle: bool = False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
    data_module._data_loader = _data_loader


def main():
    set_persistent_workers(MNISTDataModule)
    dm = MNISTDataModule(_DATASETS_PATH, num_workers=8,
                         batch_size=256, shuffle=True, drop_last=True)

    model = models.dc.DCVAE(latent_dim=2)

    callbacks = [
        TensorboardGenerativeModelImageSampler(10),
        LatentDimInterpolator()
    ]
    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=150,
        gpus=-1,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
