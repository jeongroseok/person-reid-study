from pytorch_lightning.utilities.cli import LightningCLI
import models
import datamodules.omniglot_datamodule
import pl_bolts.callbacks
import pl_examples


def main():
    callbacks = [
        pl_bolts.callbacks.TensorboardGenerativeModelImageSampler(
            num_samples=5),
        pl_bolts.callbacks.LatentDimInterpolator(interpolate_epoch_interval=5),
    ]

    dm = datamodules.omniglot_datamodule.OmniglotDataModule(
        data_dir=pl_examples._DATASETS_PATH)
    dm.setup()

    dl = dm.train_dataloader()

    trainer_defaults = {'gpus': -1, 'callbacks': callbacks}
    cli = LightningCLI(models.FDGAN,
                       datamodules.omniglot_datamodule.OmniglotDataModule,
                       trainer_defaults=trainer_defaults,
                       seed_everything_default=0, save_config_overwrite=True)


if __name__ == "__main__":
    main()
