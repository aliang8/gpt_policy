import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict

import os
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from envs.base import BaseEnvironment
from utils.pytorch_utils import ten2ar
from utils.viz_utils import save_episode_as_video
from utils.data_utils import listdict2dictlist


class Rollout:
    """
    Handle collecting rollout episodes.
    """

    def __init__(
        self, config: DictConfig, env: BaseEnvironment, agent: pl.LightningModule
    ):
        self._env = env
        self._agent = agent
        self.config = config

        self._episode_step, self._episode_reward = 0, 0
        self.device = "cuda"

    def reset(self):
        self._episode_step, self._episode_reward = 0, 0.0
        obs = self._env.reset()
        obs = self._postprocess_obs(obs)
        return obs

    def _postprocess_obs(self, obs: np.ndarray):
        return obs

    def get_action(self, states, actions):
        with torch.no_grad():
            action_pred = self._agent.get_action(
                states=states,
                actions=actions,
            )
        return action_pred

    def rollout_multi_episode(self):
        episodes = []
        for i in tqdm.tqdm(range(self.config.num_samples)):
            episode = self.rollout_single_episode()

            if self.config.save_video:
                filename = os.path.join(self.config.save_dir, f"video_{i}.mp4")
                save_episode_as_video(episode, filename=filename, caption="")
        return episodes

    def rollout_single_episode(self):
        episode, done = [], False
        obs = self.reset()

        state_dim, action_dim = self._agent.state_dim, self._agent.action_dim
        device = self.device

        # keep track of the entire history to feed into GPT
        states = (
            torch.from_numpy(obs)
            .reshape(1, self._agent.state_dim)
            .to(device=self.device, dtype=torch.float32)
        )
        actions = torch.zeros(
            (0, self._agent.action_dim), device=self.device, dtype=torch.float32
        )

        while not done and self._episode_step < self.config.max_episode_len:
            actions = torch.cat(
                [actions, torch.zeros((1, action_dim), device=device)], dim=0
            )

            action = self.get_action(states, actions)
            actions[-1] = action
            action = ten2ar(action.squeeze())

            next_obs, reward, done, info = self._env.step(action)

            episode.append(
                AttrDict(
                    observation=obs,
                    action=action,
                    next_observation=next_obs,
                    done=done,
                    reward=reward,
                    info=info,
                )
            )
            if self.config.save_video:
                episode[-1].image = self._env.render()

            cur_state = (
                torch.from_numpy(next_obs).to(device=device).reshape(1, state_dim)
            )
            states = torch.cat([states, cur_state], dim=0)
            obs = next_obs
            self._episode_step += 1
            self._episode_reward += reward

        # set last step in episode as done
        episode[-1].done = True
        return listdict2dictlist(episode)
