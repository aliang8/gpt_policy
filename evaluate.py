import os
from model.lb_mm_decoder import LB_MM_Decoder
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.utils import instantiate

if __name__ == "__main__":
    parser = ArgumentParser()
    # trainer = Trainer()
    # parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--config-file", type=str, default="")
    args = parser.parse_args()

    # load configs
    cfg = OmegaConf.load(args.config_file)
    # trainer.from_argparse_args(cfg)

    ckpt_dir = os.path.join(cfg.trainer.logger[0].init_args.save_dir, cfg.trainer.logger[0].init_args.name)
    ckpt_path = os.path.join(ckpt_dir, "test-epoch=49-global_step=0.ckpt")

    assert os.path.exists(ckpt_path)

    # load the model
    model = LB_MM_Decoder.load_from_checkpoint(checkpoint_path=ckpt_path)
    model = model.cuda()
    model.eval()

    # create environment
    env = instantiate(cfg.env)

    # create rollout helper
    rollout = instantiate(cfg.sampler, env=env, agent=model)
    episode = rollout.rollout_multi_episode()
