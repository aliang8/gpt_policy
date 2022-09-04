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
        if "language" in self.hparams.modalities:
            # merge multiple language datasets
            l_datasets = []
            for dataset in self.hparams.language_datasets:
                self.hparams["language_dataset_cls"]["_target_"] = dataset
                l_cfg = OmegaConf.create(self.hparams["language_dataset_cls"])
                l_datasets.append(instantiate(l_cfg, _recursive_=False))

            self.lang_dataset = ConcatDataset(l_datasets)

        if "paired" in self.hparams.modalities:
            p_cfg = OmegaConf.create(self.hparams["paired_dataset_cls"])
            self.paired_dataset = instantiate(
                p_cfg, dataset=self.dataset, _recursive_=False
            )

        if "behavior" in self.hparams.modalities:
            b_cfg = OmegaConf.create(self.hparams["behavior_dataset_cls"])
            self.behavior_dataset = instantiate(
                b_cfg, dataset=self.dataset, _recursive_=False
            )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if "language" in self.hparams.modalities:
                num_tr, num_val = self.split_tr_and_val(self.lang_dataset)
                self.lang_train, self.lang_val = random_split(
                    self.lang_dataset, [num_tr, num_val]
                )
                self.logger.info(f"Language dataset train size: {num_tr}")
                self.logger.info(f"Language dataset val size: {num_val}")

            if "paired" in self.hparams.modalities:
                num_tr, num_val = self.split_tr_and_val(self.paired_dataset)
                self.paired_train, self.paired_val = random_split(
                    self.paired_dataset, [num_tr, num_val]
                )

                self.logger.info(f"Paired dataset train size: {num_tr}")
                self.logger.info(f"Paired dataset val size: {num_val}")

            if "behavior" in self.hparams.modalities:
                num_tr, num_val = self.split_tr_and_val(self.behavior_dataset)
                self.behavior_train, self.behavior_val = random_split(
                    self.behavior_dataset, [num_tr, num_val]
                )

                self.logger.info(f"Behavior dataset train size: {num_tr}")
                self.logger.info(f"Behavior dataset val size: {num_val}")

    def train_dataloader(self):
        cfg = OmegaConf.create(self.hparams["dataloader_cls"])

        loaders = {}
        if "language" in self.hparams.modalities:
            lang_dl = instantiate(
                cfg,
                dataset=self.lang_train,
                pin_memory=True,
                worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
            )
            loaders["language"] = lang_dl

        if "behavior" in self.hparams.modalities:
            behavior_dl = instantiate(
                cfg,
                dataset=self.behavior_train,
                pin_memory=True,
                worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
            )
            loaders["behavior"] = behavior_dl

        if "paired" in self.hparams.modalities:
            batch_sampler = None
            if hasattr(self.paired_train, "indices"):
                batch_sampler = self.paired_dataset.get_sampler(
                    self.paired_train.indices
                )
            paired_dl = instantiate(
                cfg,
                dataset=self.paired_train,
                pin_memory=True,
                worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
            )
            loaders["paired"] = paired_dl

        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loader

    def val_dataloader(self):
        cfg = OmegaConf.create(self.hparams["dataloader_cls"])
        # cfg['n_repeat'] = 1

        loaders = {}
        if "language" in self.hparams.modalities:
            lang_dl = instantiate(
                cfg,
                dataset=self.lang_val,
                pin_memory=True,
                worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
            )
            if len(lang_dl) > 0:
                loaders["language"] = lang_dl

        if "behavior" in self.hparams.modalities:
            behavior_dl = instantiate(
                cfg,
                dataset=self.behavior_val,
                pin_memory=True,
                worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
            )
            loaders["behavior"] = behavior_dl

        if "paired" in self.hparams.modalities:
            batch_sampler = None
            if hasattr(self.paired_train, "indices"):
                batch_sampler = self.paired_dataset.get_sampler(
                    self.paired_train.indices
                )
            paired_dl = instantiate(
                cfg,
                dataset=self.paired_val,
                pin_memory=True,
                worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
            )
            loaders["paired"] = paired_dl

        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loader


@DATAMODULE_REGISTRY
class ALFREDLanguageBehaviorDataModule(LanguageBehaviorDataModule):
    """
    Language-Behavior dataset
    """

    def prepare_data(self):
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        if "language" in self.hparams.modalities:
            # merge multiple language datasets
            l_datasets = []
            for dataset in self.hparams.language_datasets:
                self.hparams["language_dataset_cls"]["_target_"] = dataset
                l_cfg = OmegaConf.create(self.hparams["language_dataset_cls"])
                l_datasets.append(instantiate(l_cfg, _recursive_=False))

            self.lang_dataset = ConcatDataset(l_datasets)

        if "paired" in self.hparams.modalities:
            p_cfg = OmegaConf.create(self.hparams["paired_dataset_cls"])
            p_cfg.hparams.partition = "train"
            self.paired_train = instantiate(
                p_cfg, dataset=self.dataset, _recursive_=False
            )
            p_cfg.hparams.partition = "valid_seen"
            self.paired_val = instantiate(
                p_cfg, dataset=self.dataset, _recursive_=False
            )

        if "behavior" in self.hparams.modalities:
            b_cfg = OmegaConf.create(self.hparams["behavior_dataset_cls"])
            b_cfg.hparams.partition = "train"
            self.behavior_train = instantiate(
                b_cfg, dataset=self.dataset, _recursive_=False
            )
            b_cfg.hparams.partition = "valid_seen"
            self.behavior_val = instantiate(
                b_cfg, dataset=self.dataset, _recursive_=False
            )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if "language" in self.hparams.modalities:
                num_tr, num_val = self.split_tr_and_val(self.lang_dataset)
                self.lang_train, self.lang_val = random_split(
                    self.lang_dataset, [num_tr, num_val]
                )
                self.logger.info(f"Language dataset train size: {num_tr}")
                self.logger.info(f"Language dataset val size: {num_val}")
