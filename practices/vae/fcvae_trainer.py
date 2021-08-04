import models.fc
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule
from pl_examples import _DATASETS_PATH
from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
from callbacks import LatentDimInterpolator


def main():
    dm = MNISTDataModule(_DATASETS_PATH, num_workers=1,
                         batch_size=256, shuffle=True, drop_last=True)
    model = models.fc.FCVAE(latent_dim=2)

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
