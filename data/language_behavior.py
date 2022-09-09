import os
import gym
import tqdm
import d4rl
import math
import json
import torch
import numpy as np
import logging
from itertools import starmap
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Dict, List, Any
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.trainer.supporters import CombinedLoader
from utils.data_utils import padded_tensor, padded_3d, collate_fn

from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset


@DATAMODULE_REGISTRY
class LanguageBehaviorDataModule(pl.LightningDataModule):
    """
    Language-Behavior dataset
    """

    def __init__(self, data_conf, **kwargs):
        super().__init__()

        # saves parameters into hparams attribute
        self.save_hyperparameters(data_conf)

        self.logger = logging.getLogger("data")
        self.logger.setLevel(logging.DEBUG)

    def split_tr_and_val(self, data: List):
        num_examples = len(data)
        num_train_ex = math.ceil(num_examples * self.hparams.split["train"])
        num_val_ex = num_examples - num_train_ex
        return num_train_ex, num_val_ex

    def prepare_data(self):
        if "kitchen" in self.hparams.env_name:
            env = gym.make(self.hparams.env_name)

            # get dataset
            # dictionary of state, actions
            self.dataset = env.get_dataset()
        else:
            self.dataset = None

    def setup(self, stage: Optional[str] = None):
        self.datasets = {}

        if "language" in self.hparams.modalities:
            # merge multiple language datasets
            l_datasets = []
            for dataset in self.hparams.language_datasets:
                self.hparams["language_dataset_cls"]["_target_"] = dataset
                l_cfg = OmegaConf.create(self.hparams["language_dataset_cls"])
                l_datasets.append(instantiate(l_cfg, _recursive_=False))

            self.lang_dataset = ConcatDataset(l_datasets)
            self.datasets["language"] = self.lang_dataset

        for modality in ["paired", "behavior"]:
            if modality in self.hparams.modalities:
                cfg = OmegaConf.create(self.hparams[f"{modality}_dataset_cls"])
                self.datasets[modality] = instantiate(
                    cfg, dataset=self.dataset, _recursive_=False
                )
                self.datasets[modality].prepare_data()

        # Assign train/val datasets for use in dataloaders
        generator = torch.Generator().manual_seed(self.hparams.seed)

        if stage == "fit" or stage is None:
            for k, dataset in self.datasets.items():
                num_tr, num_val = self.split_tr_and_val(dataset)
                train_split, val_split = random_split(
                    dataset, [num_tr, num_val], generator=generator
                )
                self.datasets[f"train/{k}"] = train_split
                self.datasets[f"val/{k}"] = val_split

                self.logger.info(f"{k} dataset train size: {num_tr}")
                self.logger.info(f"{k} dataset val size: {num_val}")

    def train_dataloader(self, phase="train"):
        cfg = OmegaConf.create(self.hparams["dataloader_cls"])

        loaders = {}

        for k, dataset in self.datasets.items():
            if phase in k:
                modality = k.split("/")[-1]
                batch_sampler = None
                if hasattr(dataset, "indices"):
                    batch_sampler = self.datasets[modality].get_sampler(dataset.indices)
                    extra_kwargs = dict(batch_sampler=batch_sampler)
                else:
                    extra_kwargs = dict(batch_sampler=None)

                if modality in ["paired", "behavior"]:
                    extra_kwargs["collate_fn"] = collate_fn

                data_loader = instantiate(
                    cfg,
                    dataset=dataset,
                    pin_memory=True,
                    worker_init_fn=lambda x: np.random.seed(
                        np.random.randint(65536) + x
                    ),
                    **extra_kwargs,
                )
                loaders[modality] = data_loader

        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loader

    def val_dataloader(self):
        return self.train_dataloader(phase="val")


@DATAMODULE_REGISTRY
class ALFREDLanguageBehaviorDataModule(LanguageBehaviorDataModule):
    """
    Language-Behavior dataset
    """

    def prepare_data(self):
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        self.datasets = {}

        if "language" in self.hparams.modalities:
            # merge multiple language datasets
            l_datasets = []
            for dataset in self.hparams.language_datasets:
                self.hparams["language_dataset_cls"]["_target_"] = dataset
                l_cfg = OmegaConf.create(self.hparams["language_dataset_cls"])
                l_datasets.append(instantiate(l_cfg, _recursive_=False))

            self.datasets["language"] = ConcatDataset(l_datasets)

        for modality in ["paired", "behavior"]:
            if modality in self.hparams.modalities:
                cfg = OmegaConf.create(self.hparams[f"{modality}_dataset_cls"])
                cfg.hparams.partition = "train"
                self.datasets[f"train/{modality}"] = instantiate(
                    cfg, dataset=self.dataset, _recursive_=False
                )
                self.datasets[f"train/{modality}"].prepare_data()
                cfg.hparams.partition = "valid_seen"
                self.datasets[f"val/{modality}"] = instantiate(
                    cfg, dataset=self.dataset, _recursive_=False
                )
                self.datasets[f"val/{modality}"].prepare_data()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if "language" in self.hparams.modalities:
                num_tr, num_val = self.split_tr_and_val(self.datasets["language"])
                (
                    self.datasets["train/language"],
                    self.datasets["val/language"],
                ) = random_split(
                    self.datasets["language"],
                    [num_tr, num_val],
                    generator=torch.Generator().manual_seed(self.hparams.seed),
                )
                self.logger.info(f"Language dataset train size: {num_tr}")
                self.logger.info(f"Language dataset val size: {num_val}")
