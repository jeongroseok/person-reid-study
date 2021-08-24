import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
from torchmetrics.classification.accuracy import Accuracy


class Classifier(pl.LightningModule):
    def __init__(
            self,
            num_classes: int = 10,
            img_dim: tuple[int, int, int] = (1, 28, 28),
            lr: float = 1e-3,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(img_dim), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        lr = self.hparams.lr

        return torch.optim.Adam(self.parameters(), lr=lr)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        accuracy = self.metric(y_hat, y)

        self.log(f"{self.__class__.__name__}/loss", loss)
        self.log(f"{self.__class__.__name__}/accuracy", accuracy)

        return loss
