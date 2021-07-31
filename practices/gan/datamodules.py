from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets import MNIST
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
import torch.utils.data

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


# 다운로드 버그 방지(torchvision과 lightning의 버전 불일치 문제)
MNIST.resources = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
]


class MNISTDataModule(VisionDataModule):
    name = "mnist"
    dataset_cls = MNIST
    dims = (1, 28, 28)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        image_size: int = 64,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                'You want to use MNIST dataset loaded from `torchvision` which is not installed yet.'
            )

        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )
        self.image_size = image_size

    @property
    def num_classes(self) -> int:
        return 10

    def default_transforms(self) -> Callable:
        if self.normalize:
            mnist_transforms = transform_lib.Compose([
                transform_lib.Resize(self.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=(0.5, ), std=(0.5, ))
            ])
        else:
            mnist_transforms = transform_lib.Compose([
                transform_lib.Resize(self.image_size),
                transform_lib.ToTensor()
            ])

        return mnist_transforms

    def _data_loader(self,
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
