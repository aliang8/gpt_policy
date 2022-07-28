import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_lightning.utilities.cli import LightningCLI
from model.lb_mm_transformer import LB_MM_Transformer
from data.kitchen import KitchenDataset


class MyLightingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")


if __name__ == "__main__":
    cli = LightningCLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"},
        run=False,
    )
    # why is model not put on device?
    cli.trainer.fit(cli.model.cuda(), cli.datamodule)
