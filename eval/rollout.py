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
from data.kitchen import LANG_BOS_TOKEN, LANG_EOS_TOKEN


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
        self.save_dir = os.path.join(config.save_dir, config.exp_name)

        self._episode_step, self._episode_reward = 0, 0
        self.device = "cuda"

        if self.config.prompt:
            self.prompt = self._agent.tokenizer(
                f"{LANG_BOS_TOKEN} {self.config.prompt} {LANG_EOS_TOKEN}",
                return_tensors="pt",
            )

    def reset(self):
        self._episode_step, self._episode_reward = 0, 0.0
        obs = self._env.reset()
        obs = self._postprocess_obs(obs)
        return obs

    def _postprocess_obs(self, obs: np.ndarray):
        return obs

    def rollout_multi_episode(self):
        episodes = []
        avg_num_completed_tasks = 0
        for i in tqdm.tqdm(range(self.config.num_samples)):
            with torch.no_grad():
                episode = self.rollout_single_episode()
            completed = episode.info[-1]["completed_tasks"]
            print(f"rollout {i}: completed_tasks: {completed}")
            avg_num_completed_tasks += len(completed)

            if self.config.save_video:
                filename = os.path.join(
                    self.save_dir, f"{self.config.video_prefix}_video_{i}.mp4"
                )
                save_episode_as_video(
                    episode, filename=filename, caption=self.config.prompt
                )
        avg_num_completed_tasks /= self.config.num_samples
        print(avg_num_completed_tasks)
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

        # get initial prompt / skill
        # prompt, lang_token_ids, token_type_ids = self._agent.get_prompt()
        # print(f"Prompt: {prompt}")
        lang_token_ids, token_type_ids = None, None

        while not done and self._episode_step < self.config.max_episode_len:
            actions = torch.cat(
                [actions, torch.zeros((1, action_dim), device=device)], dim=0
            )

            (
                action,
                lang_token_ids,
                token_type_ids,
                progress_pred,
            ) = self._agent.get_action(states, actions, lang_token_ids, token_type_ids)

            # time to start a new skill?
            # print(progress_pred[-1])
            # if progress_pred[-1].item() > 0.9:
            #     import ipdb

            #     ipdb.set_trace()

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
