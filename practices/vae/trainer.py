from typing import Any, Optional
import pl_bolts.models.autoencoders
from pl_examples import _DATASETS_PATH
import pytorch_lightning.utilities.cli
import pl_bolts.datamodules.mnist_datamodule
import torch.utils.data
import torchvision.datasets.mnist
torchvision.datasets.mnist.MNIST.resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

class VAE(pl_bolts.models.autoencoders.VAE):
    def __init__(self, latent_dim: int = 16):
        input_height = pl_bolts.datamodules.mnist_datamodule.MNISTDataModule.dims[-1]
        super().__init__(input_height, latent_dim=latent_dim)


class MNISTDataModule(pl_bolts.datamodules.mnist_datamodule.MNISTDataModule):
    def __init__(self, data_dir: Optional[str] = _DATASETS_PATH, *args: Any, **kwargs: Any) -> None:
        super().__init__(data_dir=data_dir, *args, **kwargs)

    def _data_loader(self, dataset: torch.utils.data.Dataset, shuffle: bool = False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )


def main():
    trainer_defaults = {'gpus': -1}
    cli = pytorch_lightning.utilities.cli.LightningCLI(VAE,
                                                       MNISTDataModule,
                                                       trainer_defaults=trainer_defaults,
                                                       seed_everything_default=0, save_config_overwrite=True)


if __name__ == "__main__":
    main()
