from pytorch_lightning.utilities.cli import LightningCLI
import models
import datamodules
import pl_bolts.callbacks


def main():
    callbacks = [
        pl_bolts.callbacks.TensorboardGenerativeModelImageSampler(
            num_samples=5),
        pl_bolts.callbacks.LatentDimInterpolator(interpolate_epoch_interval=5),
    ]
    trainer_defaults = {'gpus': -1, 'callbacks': callbacks}
    cli = LightningCLI(models.DCGAN,
                       datamodules.MNISTDataModule,
                       trainer_defaults=trainer_defaults,
                       seed_everything_default=0, save_config_overwrite=True)


if __name__ == "__main__":
    main()
