import torch
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.trainer.trainer import Trainer


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

    trainer = instantiate(trainer_cfg, _recursive_=True)
    model = instantiate(model_cfg, _recursive_=False)
    datamodule = instantiate(data_cfg, _recursive_=False)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=None,  # this is used for resume checkpoint
    )


if __name__ == "__main__":
    main()
