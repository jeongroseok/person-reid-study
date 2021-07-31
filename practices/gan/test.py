import matplotlib.pyplot as plt
import models
import datamodules
from pl_examples import _DATASETS_PATH
import torch
import pytorch_lightning

CHECKPOINT_PATH = 'lightning_logs\\version_1\\checkpoints\\epoch=7-step=11999.ckpt'


def sample_batch_from_datamodule(dm: pytorch_lightning.LightningDataModule) -> list[torch.Tensor, torch.Tensor]:
    dl = dm.train_dataloader()
    batches = list(iter(dl))
    batch = batches[0]
    return batch


def main():
    dm = datamodules.MNISTDataModule(
        _DATASETS_PATH, num_workers=1, batch_size=4)
    dm.setup()

    images, labels = sample_batch_from_datamodule(dm)
    image = images[0]
    label = labels[0]
    # plt.imshow(image.permute(1, 2, 0))  # c, h, w to h, w, c
    # plt.show()

    model = models.DCGAN.load_from_checkpoint(CHECKPOINT_PATH)

    z1 = torch.randn((1, 16))
    z2 = torch.randn((1, 16))
    model.forward(torch.randn_like((1, 16)))
    print('done')


if __name__ == "__main__":
    main()
