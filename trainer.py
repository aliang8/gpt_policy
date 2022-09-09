import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.trainer import Trainer

from utils.logger_utils import get_logger, print_cfg

import warnings

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_CACHE"] = "/misery/anthony/huggingface/transformers"
os.environ["TOKENIZERS_PARALLELISM"] = "False"


def merge_list_of_cfg(cfg_files):
    if type(cfg_files) is not str:
        cfg_files = OmegaConf.to_object(cfg_files)

    if type(cfg_files) is not list:
        return OmegaConf.load(cfg_files)

    confs = []
    for conf_f in cfg_files:
        confs.append(OmegaConf.load(conf_f))
    combined_cfg = OmegaConf.merge(*confs)
    return combined_cfg


def main():
    cfg = OmegaConf.from_cli()

    trainer_cfg = merge_list_of_cfg(cfg["trainer"])
    data_cfg = merge_list_of_cfg(cfg["data"])
    model_cfg = merge_list_of_cfg(cfg["model"])

    print_cfg(trainer_cfg)
    print_cfg(data_cfg)
    print_cfg(model_cfg)

    trainer = instantiate(trainer_cfg, _recursive_=True)
    model = instantiate(model_cfg, _recursive_=False)
    datamodule = instantiate(data_cfg, _recursive_=False)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        # ckpt_path=None,  # this is used for resume checkpoint
    )


if __name__ == "__main__":
    main()
