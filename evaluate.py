import os
import glob
import copy
import importlib
from model.lb_mm_decoder import LB_MM_Decoder
from argparse import ArgumentParser
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from pytorch_lightning.utilities.cli import LightningCLI


def create_param_grid():
    # get arguments from command line
    cfg = OmegaConf.from_cli()

    confs = []
    for conf_f in cfg["eval_config_files"]:
        confs.append(OmegaConf.load(conf_f))

    del cfg["eval_config_files"]

    all_configs = []
    # do some hacky thing to create a parameter grid for the exp name
    custom_cfg = OmegaConf.to_container(cfg)

    for i, exp_name in enumerate(custom_cfg["exp_name"]):
        base_configs = copy.deepcopy(confs)
        cli_cfg_single = copy.deepcopy(cfg)
        cli_cfg_single.sampler.config.exp_name = exp_name
        del cli_cfg_single.exp_name
        base_configs.append(cli_cfg_single)
        eval_cfg = OmegaConf.merge(*base_configs)
        all_configs.append(eval_cfg)
    return cfg, all_configs


def load_model_and_env_from_cfg(cfg):
    save_dir = cfg.sampler.config.save_dir
    exp_name = cfg.sampler.config.exp_name
    ckpt_dir = os.path.join(save_dir, exp_name)
    ckpts = glob.glob(ckpt_dir + "/*.ckpt")
    ckpt_path = sorted(ckpts)[-1]

    print(f"loading model from: {ckpt_path}")
    assert os.path.exists(ckpt_path)

    model_cls = importlib.import_module(cfg.sampler.config.model_target).Model
    model = model_cls.load_from_checkpoint(
        checkpoint_path=ckpt_path, training=False, strict=False
    )
    model = model.cuda()
    model.eval()

    # create environment
    env = instantiate(cfg.env)
    return model, env


def main():
    base_cfg, combined_conf = create_param_grid()[0]

    model, env = load_model_and_env_from_cfg(combined_conf)

    # create rollout helper
    rollout = instantiate(combined_conf.sampler, env=env, agent=model)

    # collect trajectories
    episodes = rollout.rollout_multi_episode()
    return episodes


if __name__ == "__main__":
    main()
