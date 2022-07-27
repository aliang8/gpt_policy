import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict

import numpy as np
from omegaconf import DictConfig
from envs.base import BaseEnvironment


class Rollout:
    """
    Handle collecting rollout episodes.
    """

    def __init__(
        self, config: DictConfig, env: BaseEnvironment, agent: pl.LightningModule
    ):
        self._env = env
        self._agent = agent
        self._max_episode_len = config.max_episode_len

        self._episode_step, self._episode_reward = 0, 0

    def reset(self):
        self._episode_step, self._episode_reward = 0, 0.0
        obs = self._env.reset()
        obs = self._postprocess_obs(obs)
        return obs

    def _postprocess_obs(self, obs: np.ndarray):
        return obs

    def get_action(obs: np.ndarray):
        action = self._agent.forward()

    def rollout_single_episode(self):
        episode, done = [], False
        obs = self.reset()
        import ipdb

        ipdb.set_trace()

        while not done and self._episode_step < self._max_episode_len:
            action = self.get_action(obs)
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
            obs = next_obs
            self._episode_step += 1

        # set last step in episode as done
        episode[-1].done = True

        return listdict2dictlist(episode)
