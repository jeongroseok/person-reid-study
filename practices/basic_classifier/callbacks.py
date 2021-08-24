import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.tensorboard.writer import SummaryWriter

# from models.vae import VAE


class ClassificationVisualizer(Callback):
    def __init__(
        self,
        samples: int = 10,
    ):
        super().__init__()
        self.samples = samples

    def on_epoch_end(self, trainer: Trainer, pl_module) -> None:
        writer: SummaryWriter = trainer.logger.experiment
        dataloader = trainer.train_dataloader

        x = torch.Tensor().to(pl_module.device)
        y = torch.Tensor().to(pl_module.device)
        y_hat = torch.Tensor().to(pl_module.device)
        for _x, _y in dataloader:
            _x, _y = _x.to(pl_module.device), _y.to(pl_module.device)
            remainings = max(0, self.samples - x.size(0))
            if remainings < 1:
                break

            _y_hat = pl_module(_x[:remainings]).argmax(-1)

            x = torch.cat([x, _x[:remainings]], 0)
            y = torch.cat([y, _y[:remainings]], 0)
            y_hat = torch.cat([y_hat, _y_hat], 0)

        fig = plt.figure()
        rows, cols = 1, x.size(0)
        for i in range(cols):
            torch.Tensor()
            ndarr = x[i][0].to('cpu', torch.uint8).numpy()
            img = Image.fromarray(ndarr)

            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.set_title(f'{y[i]}, {y_hat[i]}')
            ax.axis('off')

        str_title = f'{pl_module.__class__.__name__}_results'
        writer.add_figure(str_title, fig, global_step=trainer.global_step)
