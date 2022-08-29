import os
import gym
import tqdm
import d4rl
import math
import json
import torch
import numpy as np
import logging
from itertools import starmap
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Dict, List, Any
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.trainer.supporters import CombinedLoader
from utils.data_utils import padded_tensor, padded_3d, collate_fn
from utils.lang_utils import get_tokenizer, LANG_BOS_TOKEN, LANG_EOS_TOKEN

from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from data.dataset import (
    BaseDataset,
    SingleSequenceDataset,
    SingleSequenceDatasetV2,
    SingleSequenceBinaryDataset,
)
from torch.utils.data import ConcatDataset


class KitchenDataset(BaseDataset):
    OBJS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
    OBS_ELEMENT_INDICES = {
        "bottom burner": np.array([11, 12]),
        "top burner": np.array([15, 16]),
        "light switch": np.array([17, 18]),
        "slide cabinet": np.array([19]),
        "hinge cabinet": np.array([20, 21]),
        "microwave": np.array([22]),
        "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
    }
    OBS_ELEMENT_GOALS = {
        "bottom burner": np.array([-0.88, -0.01]),
        "top burner": np.array([-0.92, -0.01]),
        "light switch": np.array([-0.69, -0.05]),
        "slide cabinet": np.array([0.37]),
        "hinge cabinet": np.array([0.0, 1.45]),
        "microwave": np.array([-0.75]),
        "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    }
    BONUS_THRESH = 0.3

    def __init__(self, hparams: AttrDict, dataset: List, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.dataset = dataset
        self.sequences = self._split_dataset_into_sequences()

        self.dataset_stats = self._compute_dataset_stats()
        self._add_skill_info()

        del self.sequences[508]  # too long for the model

        self.data = self.sequences

        # bin actions for each dimension
        if self.hparams.discretize_actions:
            self._discretize_actions()

        if self.hparams.load_lang:
            self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)
            self.skill_to_token_map = self._get_lang_tokens()

    def _compute_dataset_stats(self):
        # find the max and min action value for each dim
        max_actions = []
        min_actions = []
        # collect max and min for each sequence
        for seq in self.sequences:
            max_action = np.max(seq.actions, axis=0)
            min_action = np.min(seq.actions, axis=0)
            max_actions.append(max_action)
            min_actions.append(min_action)

        # compare max and min over all sequences
        max_actions = np.stack(max_actions)
        min_actions = np.stack(min_actions)
        max_value = np.max(max_actions, axis=0)
        min_value = np.min(min_actions, axis=0)

        stats = {"min_action": min_value, "max_action": max_value}
        return stats

    def _split_dataset_into_sequences(self):
        """
        Splits dataset into individual sequences. Each sequence is composed
        of several semantic skills.
        """
        seq_end_idxs = np.where(self.dataset["terminals"])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            seqs.append(
                AttrDict(
                    states=self.dataset["observations"][start : end_idx + 1],
                    actions=self.dataset["actions"][start : end_idx + 1],
                    timesteps=np.arange(end_idx + 1 - start),
                )
            )
            start = end_idx + 1
        return seqs

    def _get_lang_tokens(self):
        annotations_file = os.path.join(
            self.hparams.data_dir,
            self.hparams.annotation_file,
        )

        annotations = json.load(open(annotations_file, "r"))
        self.skill_map = annotations["skill_to_lang_ann_map"]

        # tokenize skills
        lang_anns = [
            f"{LANG_BOS_TOKEN} {v[0]} {LANG_EOS_TOKEN}"
            for k, v in self.skill_map.items()
        ]
        tokens = self.tokenizer(lang_anns, padding="longest", return_tensors="np")

        skill_to_token_map = {
            k: {
                "token_ids": tokens["input_ids"][i],
                "attention_mask": tokens["attention_mask"][i],
            }
            for i, k in enumerate(list(self.skill_map.keys()))
        }

        return skill_to_token_map

    def _add_skill_info(self):
        """
        Add information about skill id and semantic state
        """
        for seq_idx in tqdm.tqdm(range(len(self.sequences))):
            skills = -1 * np.ones_like(self.sequences[seq_idx].states[:, 0])
            semantic_states = np.zeros_like(
                self.sequences[seq_idx].states[:, : len(self.OBJS)]
            )
            last_boundary, done_objs = 0, []
            for t, state in enumerate(self.sequences[seq_idx].states):
                for obj in self.OBJS:
                    if obj in done_objs:
                        continue
                    obj_state, obj_goal = (
                        state[self.OBS_ELEMENT_INDICES[obj]],
                        self.OBS_ELEMENT_GOALS[obj],
                    )
                    dist = np.linalg.norm(obj_state - obj_goal)
                    if dist < self.BONUS_THRESH:
                        skill = self.OBJS.index(obj)
                        if "use_diff_skills" not in self.hparams:
                            skills[last_boundary:t] = skill
                            last_boundary = t
                        done_objs.append(obj)
                # fill semantic state --> set to 1 for every done object
                for done_obj in done_objs:
                    semantic_states[t, self.OBJS.index(done_obj)] = 1
                # register semantic diff skills
                if "use_diff_skills" in self.hparams and t > 0:
                    if np.any(semantic_states[t] - semantic_states[t - 1]):
                        # semantic state changed --> add diff skill for all *previous* time steps
                        skills[last_boundary:t] = self._semantic_diff_skills.index(
                            self.bin2dec(
                                np.concatenate(
                                    (semantic_states[t - 1], semantic_states[t])
                                )
                            )
                        )
                        last_boundary = t
            skills[last_boundary:] = (
                skills[last_boundary - 1]
                if "use_diff_skills" in self.hparams
                else skill
            )  # fill any trailing steps with last recognized skill

            assert np.all(skills >= 0)  # verify that we filled all steps
            assert (
                semantic_states[t].sum() <= 4
            )  # verify that all sequences change state of 4 objects
            self.sequences[seq_idx].skills = skills
            self.sequences[seq_idx].semantic_states = semantic_states

    def _add_language_annotations(self):
        for seq in self.sequences:
            split_indx = np.where(seq.skills[:-1] != seq.skills[1:])[0]
            skills = [seq.skills[indx] for indx in split_indx] + [seq.skills[-1]]
            skill_strs = [self.OBJS[int(skill_indx)] for skill_indx in skills]

            lang_token_ids, lang_attn_masks = [], []
            for skill_str in skill_strs:
                lang_token_ids.append(self.skill_to_token_map[skill_str]["token_ids"])
                lang_attn_masks.append(
                    self.skill_to_token_map[skill_str]["attention_mask"]
                )

            seq.lang_token_ids = np.stack(lang_token_ids)
            seq.lang_attn_masks = np.stack(lang_attn_masks)
            seq.num_unique_skills = [len(skills)]

    def _add_masks(self):
        """
        Create several masks to let model know which timesteps should correspond to state/action or language tokens
        for training model

        pre - (lang_tokens, s_0, a_0, s_1, a_0, ..., s_t, a_t)
        post - (s_0, a_0, s_1, a_0, ..., s_t, a_t, lang_tokens)
        """

        for seq in self.sequences:
            T, _ = seq.states.shape
            max_lang_tokens = seq.lang_token_ids.shape[0] * seq.lang_token_ids.shape[-1]
            combined_lang_mask = np.zeros((2 * T + max_lang_tokens,))
            combined_state_action_mask = np.ones((2 * T + max_lang_tokens,))
            combined_state_action_lang_mask = np.ones_like(combined_lang_mask)

            skills = np.stack((seq.skills, seq.skills)).transpose(1, 0).reshape(-1)
            split_indx = np.where(skills[:-1] != skills[1:])[0]
            if self.hparams.add_lang_before_state:
                split_indx = np.concatenate([[0], split_indx])
            else:
                split_indx = np.concatenate([split_indx, [len(skills)]])

            max_num_tokens = seq.lang_token_ids.shape[-1]
            for i, indx in enumerate(split_indx):

                # do we put language first or behavior first?
                # [lang_tokens, state, action] or [state, action, lang_tokens]
                if self.hparams.add_lang_before_state:
                    start = indx
                else:
                    start = i * max_num_tokens + indx

                num_tokens = np.sum(seq.lang_attn_masks[i])
                combined_lang_mask[start : start + num_tokens] = 1
                combined_state_action_mask[start : start + max_num_tokens] = 0
                combined_state_action_lang_mask[
                    start : start + max_num_tokens
                ] = seq.lang_attn_masks[i]

            # There are two types of mask
            # One for combined state_action_lang [2*T, max_tokens]
            # One for individual state [T] and action [T]

            seq.combined_lang_mask = combined_lang_mask
            seq.combined_state_action_lang_mask = combined_state_action_lang_mask
            seq.combined_state_action_mask = combined_state_action_mask

            seq.state_mask = np.ones((seq.states.shape[0],))
            seq.action_mask = np.ones((seq.actions.shape[0],))

    def _discretize_actions(self):
        min_v, max_v = (
            self.dataset_stats["min_action"],
            self.dataset_stats["max_action"],
        )

        # create bins for each dimension
        bins = np.linspace(min_v, max_v, num=self.hparams.num_bins, axis=1)

        for seq in self.sequences:
            action = seq.actions
            # map lambda to each action in sequence
            # for each action, we want to digitize each element
            discretized_action = np.array(
                list(map(lambda x: list(starmap(np.digitize, zip(x, bins))), action))
            )
            seq.action = discretized_action

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class SemanticSkillsKitchenDataset(KitchenDataset):
    def __init__(self, hparams: AttrDict, dataset: List, *args, **kwargs):
        super().__init__(hparams, dataset, *args, **kwargs)

    def _split_seq(self, seq, start, end):
        new_seq = AttrDict()
        for k, v in seq.items():
            new_seq[k] = v[start:end]
        return new_seq

    def _split_by_semantic_skills(self):
        """
        Split each sequence into individual semantic skills.
        Also add skill progress for training a done predictor.
        """
        all_semantic_seqs = []

        for seq in self.sequences:
            # identify the start of semantic skill
            split_indx = np.where(seq.skills[:-1] != seq.skills[1:])[0]
            split_indx = np.concatenate([split_indx, [len(seq.skills)]])

            semantic_seqs = []

            seq = self._add_done_info(seq)

            start = 0
            for end in split_indx:
                semantic_seq = self._split_seq(seq, start, end + 1)
                semantic_seq.actions = semantic_seq.actions  # TODO: remove action by 1
                semantic_seq.padding_mask = np.ones((len(semantic_seq.states)))

                if self.hparams.load_lang:
                    skill = self.OBJS[int(semantic_seq.skills[0])]
                    semantic_seq.lang_token_ids = self.skill_to_token_map[skill][
                        "token_ids"
                    ]
                    semantic_seq.lang_attention_mask = self.skill_to_token_map[skill][
                        "attention_mask"
                    ]
                semantic_seqs.append(semantic_seq)
                start = end + 1

            all_semantic_seqs.extend(semantic_seqs)

        return all_semantic_seqs

    # def collate_fn(self, data):
    #     # custom collate fn for pad on the fly and sorted examples
    #     # sort examples by sequence length so we batch together the longest sequences
    #     # also pad per batch instead of padding to max length
    #     bs = len(data)
    #     output = AttrDict()
    #     for k in list(data[0].keys()):
    #         vs = [torch.Tensor(data[i][k]) for i in range(bs)]

    #         # pad all to the same length
    #         if len(vs[0].shape) == 1:
    #             output[k], _ = padded_tensor(vs, pad_idx=0, left_padded=False)
    #         elif len(vs[0].shape) == 2:
    #             output[k] = padded_3d(vs, pad_idx=0, dtype=torch.float)
    #     return output

    # def get_sampler(self, indices):
    #     cfg = OmegaConf.create(self.hparams["dataset_sampler_cls"])

    #     lens = [self.data[i]["states"].shape[0] for i in range(len(self.data))]
    #     sampler = instantiate(
    #         cfg,
    #         indices=indices,
    #         lengths=lens,
    #         is_distributed=False,
    #         shuffle=True,
    #         drop_last=False,
    #     )
    #     return sampler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KitchenSingleSequenceDataset(SingleSequenceDataset, SemanticSkillsKitchenDataset):
    def __init__(self, hparams: Dict, dataset: List, *args, **kwargs):
        SemanticSkillsKitchenDataset.__init__(self, hparams, dataset, *args, **kwargs)
        SingleSequenceDataset.__init__(self, *args, **kwargs)


class KitchenSingleSequenceV2Dataset(
    SingleSequenceDatasetV2, SemanticSkillsKitchenDataset
):
    def __init__(self, hparams: Dict, dataset: List, *args, **kwargs):
        SemanticSkillsKitchenDataset.__init__(self, hparams, dataset, *args, **kwargs)
        SingleSequenceDatasetV2.__init__(self, *args, **kwargs)


class KitchenSingleSequenceBinaryDataset(
    SingleSequenceBinaryDataset, SemanticSkillsKitchenDataset
):
    def __init__(self, hparams: Dict, dataset: List, *args, **kwargs):
        SemanticSkillsKitchenDataset.__init__(self, hparams, dataset, *args, **kwargs)
        SingleSequenceDataset.__init__(self, *args, **kwargs)
