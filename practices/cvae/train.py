import pytorch_lightning as pl
import torchvision.transforms
from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
from pl_bolts.datamodules import MNISTDataModule
from pl_examples import _DATASETS_PATH

from callbacks import CVAEImageSampler, LatentDimInterpolator
from models.cvae import CVAE
from utils import set_persistent_workers


def main(args=None):
    set_persistent_workers(MNISTDataModule)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, ), (0.5, )),
    ])
    dm = MNISTDataModule(_DATASETS_PATH, num_workers=4,
                         batch_size=128, shuffle=True, drop_last=True,
                         train_transforms=transforms,
                         val_transforms=transforms,
                         test_transforms=transforms)

    model = CVAE(latent_dim=16, batch_norm=True, lr=2e-3)

    callbacks = [
        TensorboardGenerativeModelImageSampler(10),
        # LatentDimInterpolator(),
        CVAEImageSampler()
    ]
    trainer = pl.Trainer(
        progress_bar_refresh_rate=5,
        max_epochs=1500,
        gpus=-1,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
