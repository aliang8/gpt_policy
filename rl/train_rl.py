import time
import torch
import json

import collections
import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from trainer import merge_list_of_cfg
from rl.replay_buffer import ReplayBuffer
import rl.rl_utils as utils

from typing import List, Set, Dict, Tuple, Optional
from utils.logger_utils import get_logger, print_cfg

from pytorch_lightning.loggers import WandbLogger


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class RLTrainer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.console_logger = get_logger("rl_training")

        utils.set_seed_everywhere(hparams.seed)

        self.device = torch.device(hparams.device)
        self.console_logger.info(f"using device: {self.device}")
        self.env = instantiate(self.hparams.env)

        self.replay_buffer = ReplayBuffer(
            (self.hparams.env.config.obs_dim,),
            (self.hparams.env.config.act_dim,),
            int(self.hparams.replay_buffer_capacity),
            max_episode_len=280,
            device=self.device,
        )

        self.agent = instantiate(self.hparams.agent_cfg, _recursive_=False)
        self.step = 0

        if self.hparams.use_wandb:
            self.logger = WandbLogger(
                name=hparams.exp_name, project="gpt_lang_transformer", entity="clvr"
            )

    # def evaluate(self):
    #     average_episode_reward = 0
    #     for episode in range(self.hparams.num_eval_episodes):
    #         obs = self.env.reset()
    #         self.agent.reset()
    #         self.video_recorder.init(enabled=(episode == 0))
    #         done = False
    #         episode_reward = 0
    #         while not done:
    #             with utils.eval_mode(self.agent):
    #                 action = self.agent.act(obs, sample=False)
    #             obs, reward, done, _ = self.env.step(action)
    #             self.video_recorder.record(self.env)
    #             episode_reward += reward

    #         average_episode_reward += episode_reward
    #         self.video_recorder.save(f"{self.step}.mp4")
    #     average_episode_reward /= self.hparams.num_eval_episodes
    #     self.logger.log("eval/episode_reward", average_episode_reward, self.step)
    #     self.logger.dump(self.step)

    def train(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.hparams.num_train_steps:
            if done:
                self.console_logger.info(f"starting new episode: {episode + 1}")
                if self.step > 0:
                    # self.logger.log(
                    #     "train/duration", time.time() - start_time, self.step
                    # )
                    # start_time = time.time()
                    # self.logger.dump(
                    #     self.step, save=(self.step > self.hparams.num_seed_steps)
                    # )
                    pass

                # evaluate agent periodically
                # if self.step > 0 and self.step % self.hparams.eval_frequency == 0:
                # self.logger.log("eval/episode", episode, self.step)
                # self.evaluate()

                # self.logger.log("train/episode_reward", episode_reward, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                # self.logger.log("train/episode", episode, self.step)

            context = {
                "states": self.agent.actor.states,
                "actions": self.agent.actor.actions,
                "timesteps": self.agent.actor.timesteps,
                "masks": self.agent.actor.masks,
            }
            # sample action for data collection
            # if self.step < self.hparams.num_seed_steps:
            #     action = self.env.action_space.sample()
            # else:
            with eval_mode(self.agent):
                action = self.agent.act(obs, sample=True)

            # update actions
            self.agent.actor.actions[-1] = torch.from_numpy(action).to(self.device)

            # store context
            next_context = {
                "states": self.agent.actor.states,
                "actions": self.agent.actor.actions,
                "timesteps": self.agent.actor.timesteps,
                "masks": self.agent.actor.masks,
            }

            # run training update
            if self.step >= self.hparams.num_seed_steps:
                self.console_logger.info("updating")
                self.agent.update(self.replay_buffer, self.step)

            # self.console_logger.info(action.shape)
            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == 280 else done
            episode_reward += reward

            self.replay_buffer.add(
                obs, context, action, reward, next_obs, next_context, done, done_no_max
            )

            obs = next_obs
            episode_step += 1
            self.step += 1


def main():
    cfg = OmegaConf.from_cli()
    trainer_cfg = merge_list_of_cfg(cfg["trainer"])
    agent_cfg = merge_list_of_cfg(cfg["agent"])

    cfg = OmegaConf.merge(trainer_cfg, agent_cfg)

    print_cfg(cfg)
    rl_trainer = instantiate(cfg.trainer, _recursive_=False)
    rl_trainer.train()


if __name__ == "__main__":
    main()
