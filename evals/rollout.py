import matplotlib
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict

import os
import numpy as np
import torch
import tqdm
import json
import pickle
import random
import matplotlib.pyplot as plt
from collections import Counter
from omegaconf import DictConfig
from envs.base import BaseEnvironment
from utils.pytorch_utils import ten2ar
from utils.viz_utils import save_episode_as_video
from utils.data_utils import listdict2dictlist
from utils.lang_utils import get_tokenizer, LANG_BOS_TOKEN, LANG_EOS_TOKEN

from model.modules.alfred_visual_enc import FeatureExtractor
from data.alfred import ALFREDDataset

from utils.logger_utils import get_logger
from utils import eval_utils
from utils.lang_utils import add_start_and_end_str

logger = get_logger("rollout")


def init_state_action_masks(state_dim, action_dim, start_timestep, device):
    states = torch.zeros((0, *state_dim), device=device, dtype=torch.float32)
    actions = torch.zeros((0, *action_dim), device=device, dtype=torch.float32)
    timesteps = torch.zeros((1, 0), device=device, dtype=torch.long)
    returns_to_go = torch.zeros((1, 0), device=device, dtype=torch.long)

    # create masks
    lang_token_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
    state_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
    action_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
    combined_state_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
    combined_action_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
    combined_rtg_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)

    lang_token_ids = torch.zeros((1, 0), device=device, dtype=torch.int)
    token_type_ids = torch.zeros((1, 0), device=device, dtype=torch.int)
    tokens = torch.zeros((1, 0), device=device, dtype=torch.int)
    masks = {
        "state_mask": state_mask,
        "action_mask": action_mask,
        "lang_token_mask": lang_token_mask,
        "tokens": tokens,
        "combined_state_mask": combined_state_mask,
        "combined_action_mask": combined_action_mask,
        "combined_rtg_mask": combined_rtg_mask,
        "token_type_ids": token_type_ids,
        "lang_token_ids": lang_token_ids,
    }
    return states, actions, timesteps, returns_to_go, masks


# def init_state_action_masks(obs, state_dim, action_dim, start_timestep, device):
#     if type(obs) == np.ndarray:
#         obs = torch.from_numpy(obs)

#     states = obs.reshape(1, state_dim).to(device=device, dtype=torch.float32)
#     actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
#     timesteps = torch.tensor(start_timestep, device=device, dtype=torch.long).reshape(
#         1, 1
#     )

#     # create masks
#     lang_token_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
#     state_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
#     action_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
#     combined_state_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)
#     combined_action_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)

#     lang_token_ids = torch.zeros((1, 0), device=device, dtype=torch.int)
#     token_type_ids = torch.zeros((1, 0), device=device, dtype=torch.int)
#     tokens = torch.zeros((1, 0), device=device, dtype=torch.int)
#     masks = {
#         "state_mask": state_mask,
#         "action_mask": action_mask,
#         "lang_token_mask": lang_token_mask,
#         "tokens": tokens,
#         "combined_state_mask": combined_state_mask,
#         "combined_action_mask": combined_action_mask,
#         "token_type_ids": token_type_ids,
#         "lang_token_ids": lang_token_ids,
#     }
#     return states, actions, timesteps, masks


class Rollout:
    """
    Handle collecting rollout episodes.
    """

    def __init__(
        self,
        config: DictConfig,
        env: BaseEnvironment,
        agent: pl.LightningModule,
        **kwargs,
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
            skills = add_start_and_end_str(skills)
            self.prompts = skills
            self.num_skills = len(skills)
            self.curr_idx = 0
        else:
            self.prompts = None

        if self.config.reset_from_demonstration_state:
            init_states = pickle.load(
                open(os.path.join(self.config.data_dir, "init_states.pkl"), "rb")
            )
            self.init_states = init_states["states"]
            self.init_timesteps = init_states["timesteps"]
            logger.info(
                f"resetting from demonstration states, {len(self.init_states)} states to choose from"
            )

    def reset(self, state=None, timestep=None):
        self._episode_step, self._episode_reward = 0, 0.0
        if timestep is not None:
            self._episode_step = timestep
            logger.info(f"resetting env to timestep {timestep}")

        if state is not None:
            # pick random start
            self._env.reset()
            n_jnts = self._env._env.N_DOF_ROBOT
            n_obj = self._env._env.N_DOF_OBJECT
            self._env._env.init_qpos = state[: n_jnts + n_obj]
            self._env._env.init_qvel = state[n_jnts + n_obj :]
            obs = self._env._env.reset_model()
        else:
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
                init_state, init_timestep = None, None
                if self.config.reset_from_demonstration_state:
                    indx = np.random.choice(len(self.init_states))
                    init_state = self.init_states[indx]
                    init_timestep = self.init_timesteps[indx]
                episode = self.rollout_single_episode(
                    init_state=init_state, init_timestep=init_timestep
                )
            episodes.append(episode)
            completed = episode.info[-1]["completed_tasks"]
            tasks_completed.update(completed)
            logger.info(tasks_completed)

            if self.config.save_video:
                filename = os.path.join(
                    self.save_dir, f"{self.config.video_prefix}_video_{i}.mp4"
                )
                caption = ""
                if self.config.prompt:
                    caption = self.config.prompt

                save_episode_as_video(episode, filename=filename, caption=caption)
                logger.info(f"saved video to: {filename}")
        return episodes, tasks_completed

    def rollout_single_episode(self, init_state=None, init_timestep=None):
        """
        Rollout the model for a single episode, until max episode length or task is completed.
        """

        episode, done = [], False
        obs = self.reset(init_state, init_timestep)

        state_dim, action_dim = self._agent.state_dim, self._agent.action_dim
        device = self.device

        start_ts = init_timestep if init_timestep else 0

        # keep track of the entire history to feed into GPT
        states, actions, timesteps, returns_to_go, masks = init_state_action_masks(
            state_dim, action_dim, start_ts, device
        )

        # DEBUG
        sigmoid_vals = []
        prev_completed_tasks = set()
        pivot_timesteps = []
        curr_skill = ""

        target_return = self.config.target_return

        while not done and self._episode_step < self.config.max_episode_len:
            states = torch.cat(
                [states, torch.from_numpy(obs).to(device).unsqueeze(0)], dim=0
            )
            actions = torch.cat(
                [actions, torch.zeros((1, action_dim), device=device)], dim=0
            )
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                    * (self._episode_step),
                ],
                dim=1,
            )
            # this changes in online case
            returns_to_go = torch.cat(
                [returns_to_go, torch.zeros((1, 1), device=device) + target_return],
                dim=1,
            )

            if hasattr(self, "prompts") and not self.curr_idx > len(self.prompts):
                break

            if not hasattr(self, "prompts") or self.curr_idx == len(self.prompts):
                curr_instr = ""
            else:
                curr_instr = self.prompts[self.curr_idx]

            action_output = self._agent.get_action(
                states=states,
                actions=actions,
                timesteps=timesteps,
                returns_to_go=returns_to_go,
                curr_instr=curr_instr,
                use_means=True,
                **masks,
                **self.config,
            )
            masks = action_output.masks

            # DEBUG
            if "info" in action_output:
                sigmoid_vals.append(action_output.info["sigmoid_val"].item())

                if "curr_skill" in action_output.info:
                    curr_skill = action_output.info["curr_skill"]

                import ipdb

                ipdb.set_trace()
                if action_output.binary_token[:, -1].item() == 1:
                    self.curr_idx += 1

            action = action_output.action_pred
            actions[-1] = action
            action = ten2ar(action.squeeze())

            next_obs, reward, done, info = self._env.step(action)

            if len(prev_completed_tasks) != len(info["completed_tasks"]):
                pivot_timesteps.append(self._episode_step)

            prev_completed_tasks = info["completed_tasks"]

            episode.append(
                AttrDict(
                    observation=obs,
                    action=action,
                    next_observation=next_obs,
                    done=done,
                    reward=reward,
                    progress_pred=None,
                    info=info,
                    **action_output,
                )
            )

            if self.config.save_video:
                episode[-1].image = self._env.render()

            obs = next_obs
            self._episode_step += 1
            self._episode_reward += reward
            target_return -= reward  # decrement target return

        # set last step in episode as done
        episode[-1].done = True

        # DEBUG
        # add plot for sigmoid values
        plt.clf()
        plt.plot(sigmoid_vals[1:])
        for t in pivot_timesteps:
            plt.axvline(x=t, color="red")
        idx = np.random.randint(0, 10000)
        plt.savefig(f"plots/sigmoid_{idx}.png")

        return listdict2dictlist(episode)


class ALFREDRollout(Rollout):
    def __init__(
        self,
        config,
        env,
        agent,
        dataset=None,
        object_predictor=None,
        extractor=None,
        **kwargs,
    ):
        super().__init__(config, env, agent, **kwargs)

        if object_predictor is None:
            self.object_predictor = eval_utils.load_object_predictor(
                config["object_predictor_args"]
            )
        else:
            self.object_predictor = object_predictor

        if dataset is None and self.config.reset_from_demonstration_state:
            self.dataset = ALFREDDataset(self.config["dataset_args"], dataset=None)
        else:
            self.dataset = dataset

        if extractor is None:
            dataset_info = os.path.join(self.config.data_dir, "params.json")
            with open(dataset_info, "r") as f:
                dataset_info = json.load(f)

            self.extractor = FeatureExtractor(
                archi=dataset_info["visual_archi"],
                device=self.device,
                checkpoint=dataset_info["visual_checkpoint"],
                compress_type=dataset_info["compress_type"],
            )
        else:
            self.extractor = extractor

        self._env.vocab_action = self._agent.vocab_action
        self._env.vocab_obj = self._agent.vocab_obj
        self.curr_instr = ""

    def rollout_multi_episode(self):
        episodes = []
        tasks_completed = Counter()
        # num_jsons = len(self.dataset.jsons_and_keys)
        for idx in range(2000, 2010):
            logger.info(idx)
            task_json, _ = self.dataset.jsons_and_keys[idx]

            trial_uid = "{}:{}".format(task_json["task"], task_json["repeat_idx"])

            with torch.no_grad():
                episode = self.rollout_single_episode()
            episodes.append(episode)

            if self.config.save_video:
                filename = os.path.join(self.save_dir, f"{trial_uid}.mp4")
                caption = ""
                if self.config.prompt:
                    caption = self.config.prompt

                save_episode_as_video(
                    episode, filename=filename, caption=caption, fps=1.0
                )
                logger.info(f"saved video to: {filename}")

        return episodes, tasks_completed

    def rollout_single_episode(self, traj_data=None, use_means=True):
        """
        Rollout the model for a single episode, until max episode length or task is completed.
        """
        episode, done = [], False
        obs = self.reset()
        device = self.device

        if self.config.reset_from_demonstration_state:
            idx = random.randint(0, len(self.dataset.jsons_and_keys))
            traj_data, traj_key = self.dataset.jsons_and_keys[idx]

            # for teacher forcing
            expert_actions = [
                [a["action"] for a in a_list]
                for a_list in traj_data["num"]["action_low"]
            ]
            expert_actions = sum(expert_actions, [])

            expert_target_objs = []
            idx = 0
            for a_list in traj_data["num"]["action_low"]:
                for action_dict in a_list:
                    if action_dict["valid_interact"] == 1:
                        action = traj_data["plan"]["low_actions"][idx]

                        obj_key = (
                            "receptacleObjectId"
                            if "receptacleObjectId" in action["api_action"]
                            else "objectId"
                        )
                        object_class = action["api_action"][obj_key].split("|")[0]
                        interact_obj_id = self._agent.vocab_obj.word2index(object_class)
                        expert_target_objs.append(interact_obj_id)
                    else:
                        expert_target_objs.append(-1)

                    idx += 1

            expert_actions = torch.tensor(expert_actions).unsqueeze(0)
            expert_target_objs = torch.tensor(expert_target_objs).unsqueeze(0)
            expert_actions = torch.cat([expert_actions, expert_target_objs]).T.to(
                device
            )

            # use expert prompts
            self.prompts = traj_data["turk_annotations"]["anns"][0]["high_descs"]
            self.prompts.append("stop")
            self.prompts = add_start_and_end_str(self.prompts)

        # setup scene
        self._env.setup_scene(traj_data, reward_type="dense")

        obs = self._env.get_observation(self.extractor)
        state_dim, action_dim = obs.shape[1:], (2,)
        start_ts = 0

        # keep track of the entire history to feed into GPT
        states, actions, timesteps, returns_to_go, masks = init_state_action_masks(
            state_dim, action_dim, start_ts, device
        )

        target_return = self.config.target_return
        return_conditioned = self.config.return_conditioned

        while not done and self._episode_step < self.config.max_episode_len:
            states = torch.cat([states, obs.to(device)], dim=0)
            actions = torch.cat(
                [actions, torch.zeros((1, *action_dim), device=device)], dim=0
            )
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                    * (self._episode_step),
                ],
                dim=1,
            )

            # this changes in online case
            rtg = torch.zeros((1, 1), device=device)

            if return_conditioned:
                rtg += target_return

            returns_to_go = torch.cat([returns_to_go, rtg], dim=1)

            frame = self._env.last_event.frame if self.config.save_video else None

            if self.config.teacher_forcing:
                actions = torch.clone(expert_actions)[: self._episode_step + 1]
                actions[-1] *= 0  # set 0s for predicting new action

            if self.prompts is not None and self.curr_idx >= len(self.prompts):
                break

            if self.prompts is not None:
                curr_instr = self.prompts[self.curr_idx]
            else:
                curr_instr = None

            action_output = self._agent.get_action(
                states=states,
                actions=actions,
                timesteps=timesteps,
                returns_to_go=returns_to_go,
                use_means=use_means,
                curr_instr=curr_instr,
                **masks,
                **self.config,
            )
            masks = action_output.masks

            if action_output.time_to_predict_lang:
                self.curr_idx += 1

            # some postprocessing
            action = action_output.action_pred
            action_str = self._agent.vocab_action.index2word(action[0])
            obj_str = self._agent.vocab_obj.index2word(action[1])

            # set obj_cls prediction to -1 if action is not interaction
            if not self._env.has_interaction(action_str):
                action[-1] = -1
                obj_str = "None"

            actions[-1] = action
            action = ten2ar(action.squeeze())

            if "curr_instr" in action_output:
                self.curr_instr = action_output["curr_instr"]
            else:
                action_output["curr_instr"] = self.curr_instr

            action_dict = {
                "action_str": action_str,
                "obj_str": obj_str,
                "action_idx": action[0],
                "obj_idx": action[1],
                **action_output,
            }

            next_obs, reward, done, info = self._env.step(
                action_dict, self.object_predictor, self.extractor
            )

            episode.append(
                AttrDict(
                    observation=obs,
                    action=action,
                    next_observation=next_obs,
                    done=done,
                    reward=reward,
                    progress_pred=None,
                    info=info,
                    image=frame,
                    rtg=target_return,
                    **action_dict,
                )
            )

            obs = next_obs
            self._episode_step += 1
            self._episode_reward += reward
            target_return -= reward

        # set last step in episode as done
        episode[-1].done = True
        return listdict2dictlist(episode)
