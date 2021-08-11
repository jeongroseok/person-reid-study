import torch

from matplotlib import pyplot as plt
from models.vae import VAE
from torchvision.utils import make_grid


def main(args=None):
    model: VAE = VAE.load_from_checkpoint(
        'lightning_logs\\version_0\\checkpoints\\epoch=149-step=56249.ckpt')
    imgs = []
    for c in range(10):
        z = torch.randn((10, model.hparams.latent_dim))
        output = model.decode(z, (torch.ones((10)) * c).long())
        imgs.append(make_grid(output, 10, normalize=True))
    
    img = make_grid(imgs, 1, normalize=True)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
