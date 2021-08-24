import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule
from pl_examples import _DATASETS_PATH
from torchvision.datasets.mnist import MNIST

from callbacks import ClassificationVisualizer
from models.classification import Classifier
from utils import set_persistent_workers


def main(args=None):
    set_persistent_workers(MNISTDataModule)

    datamodule = MNISTDataModule(_DATASETS_PATH, num_workers=0,
                                 batch_size=512, shuffle=False, drop_last=True)
    model = Classifier(img_dim=datamodule.dims)

    # dataset = MNIST(_DATASETS_PATH, False,
    #                 transform=datamodule.default_transforms())
    callbacks = [
        ClassificationVisualizer(),
        # LatentSpaceVisualizer(),
        # LatentDimInterpolator(dataset),
    ]

    trainer = pl.Trainer(
        gpus=-1 if datamodule.num_workers > 0 else None,
        progress_bar_refresh_rate=5,
        max_epochs=1000,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
