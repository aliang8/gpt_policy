import os
from model.lb_mm_decoder import LB_MM_Decoder
from argparse import ArgumentParser
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trainer-config-file", type=str, default="")
    parser.add_argument("--eval-config-file", type=str, default="")

    args = parser.parse_args()

    # load configs
    trainer_cfg = OmegaConf.load(args.trainer_config_file)
    eval_cfg = OmegaConf.load(args.eval_config_file)

    ckpt_dir = os.path.join(
        trainer_cfg.logger[0].init_args.save_dir,
        trainer_cfg.logger[0].init_args.name,
    )
    ckpt_path = os.path.join(
        ckpt_dir, "language_then_behavior-epoch=58-step=35400.ckpt"
    )

    print(f"loading model from: {ckpt_path}")
    assert os.path.exists(ckpt_path)

    # load the model
    model = LB_MM_Decoder.load_from_checkpoint(checkpoint_path=ckpt_path)
    model = model.cuda()
    model.eval()

    # create environment
    env = instantiate(eval_cfg.env)

    # create rollout helper
    eval_cfg.sampler.config.save_dir = ckpt_dir
    rollout = instantiate(eval_cfg.sampler, env=env, agent=model)

    # collect trajectories
    episode = rollout.rollout_multi_episode()
