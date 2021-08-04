import models.fc
from pl_bolts.datamodules import MNISTDataModule
from pl_examples import _DATASETS_PATH
import matplotlib.pyplot as plt
import torchvision.utils


def main():
    dm = MNISTDataModule(_DATASETS_PATH, num_workers=1, batch_size=4)
    dm.setup()
    dl = dm.train_dataloader()
    batch = list(iter(dl))[0]
    x, y = batch

    model = models.fc.FCVAE.load_from_checkpoint(
        'lightning_logs\\version_2\\checkpoints\\epoch=108-step=20382.ckpt')
    _, _, z = model.encode(x)
    x_hat = model.decode(z)

    grid_x = torchvision.utils.make_grid(tensor=x, nrow=x.shape[0])
    grid_x_hat = torchvision.utils.make_grid(tensor=x_hat, nrow=x_hat.shape[0])
    plt.imshow(grid_x.permute(1,2,0).detach().numpy()); plt.show()
    plt.imshow(grid_x_hat.permute(1,2,0).detach().numpy()); plt.show()
    print('done')


if __name__ == "__main__":
    main()
