import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict

import os
import numpy as np
import torch
import tqdm
from collections import Counter
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

        # prompt is a period deliminated string
        if self.config.prompt:
            # if prompt has multiple sentences, split them
            skills = self.config.prompt.split(". ")
            skills_str = [
                f"{LANG_BOS_TOKEN} {skill} {LANG_EOS_TOKEN}" for skill in skills
            ]
            self.prompts = skills
            self.num_skills = len(skills_str)
            self.curr_idx = 0
            self.prompt_tokens = self._agent.tokenizer(
                skills_str, padding=True, return_tensors="pt"
            )
        else:
            self.prompt = None

    def reset(self):
        self._episode_step, self._episode_reward = 0, 0.0
        obs = self._env.reset()
        obs = self._postprocess_obs(obs)
        self.curr_idx = 0
        return obs

    def _postprocess_obs(self, obs: np.ndarray):
        return obs

    def rollout_multi_episode(self):
        episodes = []
        tasks_completed = Counter()
        for i in tqdm.tqdm(range(self.config.num_samples)):
            with torch.no_grad():
                episode = self.rollout_single_episode()
            episodes.append(episode)
            completed = episode.info[-1]["completed_tasks"]
            tasks_completed.update(completed)
            print(tasks_completed)

            if self.config.save_video:
                filename = os.path.join(
                    self.save_dir, f"{self.config.video_prefix}_video_{i}.mp4"
                )
                caption = ""
                if self.config.prompt:
                    caption = self.config.prompt

                save_episode_as_video(episode, filename=filename, caption=caption)
        return episodes, tasks_completed

    def rollout_single_episode(self):
        """
        Rollout the model for a single episode, until max episode length or task is completed.

        Different modes of evaluation:
        1. Initial state conditioning, feed s0 and autoregressively predict future actions
        2. Explicit prompting, we directly provide the language prompt
        3. Self-prompting, model tells itself what it should do.
        """
        episode, done = [], False
        obs = self.reset()

        state_dim, action_dim = self._agent.state_dim, self._agent.action_dim
        device = self.device

        # keep track of the entire history to feed into GPT
        states = (
            torch.from_numpy(obs)
            .reshape(1, state_dim)
            .to(device=device, dtype=torch.float32)
        )
        actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        prompt, lang_token_ids, token_type_ids = "", None, None

        if hasattr(self, "prompt_tokens") and self.prompt_tokens is not None:
            prompt = self.prompts[self.curr_idx]
            lang_token_ids = self.prompt_tokens["input_ids"][self.curr_idx]
            attn = self.prompt_tokens["attention_mask"][self.curr_idx]
            lang_token_ids = lang_token_ids[attn.bool()].to(device)
            token_type_ids = torch.ones((1, lang_token_ids.shape[-1])).to(device)
            self.curr_idx += 1
        elif self.config.self_prompting:
            # get initial prompt / skill
            (
                prompt,
                new_token_ids,
                lang_token_ids,
                token_type_ids,
            ) = self._agent.get_prompt()

        if prompt:
            print(f"Initial prompt: {prompt}")

        while not done and self._episode_step < self.config.max_episode_len:
            actions = torch.cat(
                [actions, torch.zeros((1, action_dim), device=device)], dim=0
            )

            (
                action,
                lang_token_ids,
                token_type_ids,
                progress_pred,
            ) = self._agent.get_action(
                states=states,
                actions=actions,
                timesteps=timesteps,
                lang_token_ids=lang_token_ids,
                token_type_ids=token_type_ids,
            )

            # time to start a new skill?
            # print(progress_pred[-1])
            if progress_pred is not None and progress_pred[-1].item() > 0.95:
                if hasattr(self, "prompt_tokens"):
                    if self.curr_idx >= self.num_skills:
                        break

                    next_skill_tokens = self.prompt_tokens["input_ids"][self.curr_idx]
                    attn = self.prompt_tokens["attention_mask"][self.curr_idx]
                    next_skill_tokens = next_skill_tokens[attn.bool()].to(device)
                    lang_token_ids = torch.cat(
                        [lang_token_ids, next_skill_tokens], dim=-1
                    )
                    token_type_ids = torch.cat(
                        (
                            token_type_ids,
                            torch.ones((1, next_skill_tokens.shape[-1])).to(device),
                        ),
                        dim=-1,
                    )
                    print(f"Prompt: {self.prompts[self.curr_idx]}")
                    self.curr_idx += 1

                elif self.config.self_prompting:
                    # include state, action in the self-prompting
                    # prompt, new_token_ids, lang_token_ids, token_type_ids = self._agent.get_prompt(
                    #     states, actions, lang_token_ids, token_type_ids
                    # )

                    # don't include state, action and only condition on past language p(langB | langA)
                    (
                        prompt,
                        new_token_ids,
                        lang_token_ids,
                        _,
                    ) = self._agent.get_prompt(
                        lang_token_ids=lang_token_ids,
                        token_type_ids=torch.ones((1, lang_token_ids.shape[-1])).to(
                            device
                        ),
                    )
                    token_type_ids = torch.cat(
                        [
                            token_type_ids,
                            torch.ones((1, new_token_ids.shape[-1])).to(device),
                        ],
                        dim=1,
                    )
                    print(f"next prompt: {prompt}")

            actions[-1] = action
            action = ten2ar(action.squeeze())

            next_obs, reward, done, info = self._env.step(action)

            if progress_pred is not None:
                progress_pred = round(progress_pred[-1].item(), 4)

            episode.append(
                AttrDict(
                    observation=obs,
                    action=action,
                    next_observation=next_obs,
                    done=done,
                    reward=reward,
                    progress_pred=progress_pred,
                    info=info,
                )
            )
            if self.config.save_video:
                episode[-1].image = self._env.render()

            cur_state = (
                torch.from_numpy(next_obs).to(device=device).reshape(1, state_dim)
            )
            states = torch.cat([states, cur_state], dim=0)
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                    * (self._episode_step + 1),
                ],
                dim=1,
            )

            obs = next_obs
            self._episode_step += 1
            self._episode_reward += reward

        # set last step in episode as done
        episode[-1].done = True
        return listdict2dictlist(episode)
