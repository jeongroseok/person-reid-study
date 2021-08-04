import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Encoder, self).__init__()

        self.conv_model = nn.Sequential(
            self._make_conv_block(1, 16),
            self._make_conv_block(16, 32),
            self._make_conv_block(32, 64),
            # nn.Flatten()
        )

        # Style embeddings
        self.style_mu = nn.Linear(
            in_features=256, out_features=style_dim, bias=True)
        self.style_logvar = nn.Linear(
            in_features=256, out_features=style_dim, bias=True)

        # Class embeddings
        self.class_output = nn.Linear(
            in_features=256, out_features=class_dim, bias=True)

    @staticmethod
    def _make_conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 1
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=True),
            nn.InstanceNorm2d(num_features=out_channels,
                              track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv_model(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        style_embeddings_mu = self.style_mu(x)
        style_embeddings_logvar = self.style_logvar(x)
        class_embeddings = self.class_output(x)

        return style_embeddings_mu, style_embeddings_logvar, class_embeddings


class Decoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Decoder, self).__init__()

        # Style embeddings input
        self.style_input = nn.Linear(
            in_features=style_dim, out_features=256, bias=True)

        # Class embeddings input
        self.class_input = nn.Linear(
            in_features=class_dim, out_features=256, bias=True)

        self.deconv_model = nn.Sequential(
            self._make_deconv_block(128, 32),
            self._make_deconv_block(32, 16),
            self._make_deconv_block(16, 1, padding=1, last_block=True),
        )

    @ staticmethod
    def _make_deconv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 0,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=True),
                nn.InstanceNorm2d(num_features=out_channels,
                                  track_running_stats=True),
                nn.LeakyReLU(inplace=True)
            )
        else:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=True),
                nn.Sigmoid()
            )
        return block

    def forward(self, style_embeddings, class_embeddings):
        style_embeddings = F.leaky_relu_(
            self.style_input(style_embeddings), negative_slope=0.2)
        class_embeddings = F.leaky_relu_(
            self.class_input(class_embeddings), negative_slope=0.2)

        x = torch.cat((style_embeddings, class_embeddings), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self.deconv_model(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_model = nn.Sequential(
            self._make_disc_block(2, 32),
            self._make_disc_block(32, 64),
            self._make_disc_block(64, 128)
        )

        self.fully_connected_model = nn.Sequential(
            nn.Linear(in_features=512, out_features=2, bias=True))

    @staticmethod
    def _make_disc_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=True),
            nn.InstanceNorm2d(num_features=out_channels,
                              track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
        x = torch.cat((image_1, image_2), dim=1)
        x: torch.Tensor = self.conv_model(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fully_connected_model(x)

        return x


class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=num_classes, bias=True)
        )

    def forward(self, z):
        x = self.fc_model(z)

        return x
