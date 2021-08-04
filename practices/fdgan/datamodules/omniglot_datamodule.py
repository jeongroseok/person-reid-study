from typing import Callable, Optional

import torch.utils.data
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torchvision import transforms as transform_lib
from torchvision.datasets import Omniglot


class OmniglotDataModule(VisionDataModule):
    name = "omniglot"
    dataset_cls = Omniglot
    dims = (1, 105, 105)

    def __init__(self, persistent_workers: bool = True, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.persistent_workers = persistent_workers

    def prepare_data(self, *args: any, **kwargs: any) -> None:
        self.dataset_cls(self.data_dir, background=True, download=True)
        self.dataset_cls(self.data_dir, background=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms(
            ) if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms(
            ) if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(
                self.data_dir, background=True, transform=train_transforms, **self.EXTRA_ARGS)
            dataset_val = self.dataset_cls(
                self.data_dir, background=True, transform=val_transforms, **self.EXTRA_ARGS)

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
            ) if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, background=False, transform=test_transforms, **self.EXTRA_ARGS
            )

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
            persistent_workers=self.persistent_workers
        )

    def default_transforms(self) -> Callable:
        if self.normalize:
            mnist_transforms = transform_lib.Compose([
                # transform_lib.Resize(self.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=(0.5, ), std=(0.5, ))
            ])
        else:
            mnist_transforms = transform_lib.Compose([
                # transform_lib.Resize(self.image_size),
                transform_lib.ToTensor()
            ])

        return mnist_transforms
