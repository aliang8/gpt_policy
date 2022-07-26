import os
import gym
import tqdm
import d4rl
import json
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Dict, List, Any
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.trainer.supporters import CombinedLoader
from utils.data_utils import padded_tensor, padded_3d

from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from transformers import AutoTokenizer
from data.dataset import BaseDataset


ACT_TOKEN = "<ACT>"
LANG_BOS_TOKEN = "<LBOS>"
LANG_EOS_TOKEN = "<LEOS>"
NEXT_LANG_BOS_TOKEN = "<NLBOS>"
NEXT_LANG_EOS_TOKEN = "<NLEOS>"


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

    def __init__(self, dataset: List, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.sequences = self._split_dataset_into_sequences()

        self._add_skill_info()

        if self.hparams.load_lang:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.decoder_model_cls,
                return_tensors="np",
            )

            # add special tokens
            special_tokens = {
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",
                "pad_token": "[PAD]",
                "additional_special_tokens": [
                    ACT_TOKEN,
                    LANG_BOS_TOKEN,
                    LANG_EOS_TOKEN,
                    NEXT_LANG_BOS_TOKEN,
                    NEXT_LANG_EOS_TOKEN,
                ],
            }
            self.tokenizer.add_special_tokens(special_tokens_dict=special_tokens)
            self.skill_to_token_map = self._get_lang_tokens()

            self._add_language_annotations()
            self._add_masks()

        del self.sequences[508]

        self.data = self.sequences

    def _split_dataset_into_sequences(self):
        """
        Splits dataset into individual sequences. Each sequence is composed
        of several semantic skills.
        """
        seq_end_idxs = np.where(self.dataset["terminals"])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            if end_idx + 1 - start < self.hparams.subseq_len:
                continue  # skip too short demos
            seqs.append(
                AttrDict(
                    states=self.dataset["observations"][start : end_idx + 1],
                    actions=self.dataset["actions"][start : end_idx + 1],
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
        tokens = self.tokenizer(lang_anns, padding="longest")

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
        Create several masks to let model know which timesteps should correspond to language tokens
        for training decoder model


        :return:
            state_action_lang_attn_mask: which timesteps are language tokens [2*T + num_total_tokens]
            padding_mask: which timesteps are pad tokens (don't consider them during training)
            action_mask: which timesteps are action
        """

        for seq in self.sequences:
            T, _ = seq.states.shape
            max_lang_tokens = seq.lang_token_ids.shape[0] * seq.lang_token_ids.shape[-1]
            combined_lang_mask = np.zeros((2 * T + max_lang_tokens,))
            combined_state_action_mask = np.ones((2 * T + max_lang_tokens,))
            combined_state_action_lang_mask = np.ones_like(combined_lang_mask)

            skills = np.stack((seq.skills, seq.skills)).transpose(1, 0).reshape(-1)
            split_indx = np.where(skills[:-1] != skills[1:])[0]
            split_indx = np.concatenate([split_indx, [len(skills)]])

            max_num_tokens = seq.lang_token_ids.shape[-1]
            for i, indx in enumerate(split_indx):
                start = i * max_num_tokens + indx
                num_tokens = np.sum(seq.lang_attn_masks[i])
                combined_lang_mask[start : start + num_tokens] = 1
                combined_state_action_mask[start : start + max_num_tokens] = 0
                combined_state_action_lang_mask[
                    start : start + max_num_tokens
                ] = seq.lang_attn_masks[i]

            # There are two types of mask
            # One for combined state_action_lang
            # One for individual state, action, and language

            seq.combined_lang_mask = combined_lang_mask
            seq.combined_state_action_lang_mask = combined_state_action_lang_mask
            seq.combined_state_action_mask = combined_state_action_mask

            seq.state_mask = np.ones((seq.states.shape[0],))
            seq.action_mask = np.ones((seq.actions.shape[0],))

    def collate_fn(self, data):
        # custom collate fn for pad on the fly and sorted examples
        # sort examples by sequence length so we batch together the longest sequences
        # also pad per batch instead of padding to max length
        bs = len(data)
        output = AttrDict()
        for k in list(data[0].keys()):
            vs = [torch.Tensor(data[i][k]) for i in range(bs)]

            # pad all to the same length
            if len(vs[0].shape) == 1:
                output[k], _ = padded_tensor(vs, pad_idx=0, left_padded=False)
            elif len(vs[0].shape) == 2:
                output[k] = padded_3d(vs, pad_idx=0, dtype=torch.float)
        return output

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class SemanticSkillsKitchenDataset(KitchenDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self._split_by_semantic_skills()

    def _split_seq(self, seq, start, end):
        new_seq = AttrDict()
        for k, v in seq.items():
            new_seq[k] = v[start:end]
        return new_seq

    def _add_done_info(self, seq):
        """
        Add done information to each step in the sequence. Can be percent done or binary.
        Used for learning a done predictor. Done is computed per semantic sequence.
        """
        seq.done = np.zeros((len(seq.states),))
        skill_done = (np.where(seq.skills[:-1] != seq.skills[1:]))[0]
        skill_done = np.concatenate([skill_done, np.array([len(seq.states) - 1])])

        start = 0
        for done_idx in skill_done:
            if self.hparams.load_frac_done:
                skill = seq.skills[start : done_idx + 1]
                seq.done[start : done_idx + 1] = np.cumsum(
                    np.ones((len(skill))) / len(skill)
                )
                start = done_idx + 1
            else:
                seq.done[done_idx] = 1

        return seq

    def _split_by_semantic_skills(self):
        """
        Split each sequence into individual semantic skills.
        Also add skill progress for training a done predictor.
        """

        semantic_seqs = []

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
                semantic_seqs.append(semantic_seq)
                start = end + 1

            semantic_seqs.extend(semantic_seqs)
        return semantic_seqs

    def _add_language_annotations(self):
        for i, seq in enumerate(self.data):
            skill = self.OBJS[int(seq.skills[0])]
            if i == len(self.data) - 1:
                next_seq = seq
            else:
                next_seq = self.data[i + 1]
            next_skill = self.OBJS[int(next_seq.skills[0])]
            seq.lang_token_ids = self.skill_to_token_map[skill][0]["token_ids"]
            seq.next_lang_token_ids = self.skill_to_token_map[next_skill][1][
                "token_ids"
            ]
            seq.lang_attention_mask = self.skill_to_token_map[skill][0][
                "attention_mask"
            ]
            seq.next_lang_attention_mask = self.skill_to_token_map[next_skill][1][
                "attention_mask"
            ]

    def collate_fn(self, data):
        # custom collate fn for pad on the fly and sorted examples
        # sort examples by sequence length so we batch together the longest sequences
        # also pad per batch instead of padding to max length
        bs = len(data)
        output = AttrDict()
        for k in list(data[0].keys()):
            vs = [torch.Tensor(data[i][k]) for i in range(bs)]

            # pad all to the same length
            if len(vs[0].shape) == 1:
                output[k], _ = padded_tensor(vs, pad_idx=0, left_padded=False)
            elif len(vs[0].shape) == 2:
                output[k] = padded_3d(vs, pad_idx=0, dtype=torch.float)
        return output

    def get_sampler(self, indices):
        cfg = OmegaConf.create(self.hparams["dataset_sampler_cls"])

        lens = [self.data[i]["states"].shape[0] for i in range(len(self.data))]
        sampler = instantiate(
            cfg,
            indices=indices,
            lengths=lens,
            is_distributed=False,
            shuffle=True,
            drop_last=False,
        )
        return sampler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KitchenSequenceDataset(SemanticSkillsKitchenDataset):
    """
    Combine all the data into a long sequence
    and then split it up into blocks for next token prediction training
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.hparams.load_lang:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.decoder_model_cls, return_tensors="np"
            )

        self.data, self.masks = self._create_long_sequence()

    def _create_long_sequence(self):
        data = []
        token_type_mask = []
        for seq in self.semantic_seqs:
            # T x (state_dim + action_dim)
            state_action = np.concatenate([seq["states"], seq["actions"]], axis=-1)

            # tokenize the string
            # (num_tokens, )
            lang_tokens = np.expand_dims(
                self.tokenizer(seq.lang_ann)["input_ids"], axis=0
            )

            # pad the language so that we can combine with state_action
            pad_size = max(state_action.shape[-1], lang_tokens.shape[-1])
            padded_lang_tokens = np.zeros((1, pad_size))
            padded_lang_tokens[:, : lang_tokens.shape[-1]] = lang_tokens

            # (T + 1) x pad_size
            state_action_language = np.concatenate(
                [state_action, padded_lang_tokens], axis=0
            )

            # mask to denote that the last timestep is the lang tokens
            mask = np.zeros((state_action.shape[0] + 1,))
            mask[-1] = 1.0

            data.append(state_action_language)
            token_type_mask.append(mask)

        data = np.concatenate(data, axis=0)
        token_type_mask = np.concatenate(token_type_mask, axis=0)

        # chunk the data by block size (number of tokens in each batch)
        data_chunks = []
        mask_chunks = []
        for i in range(0, data.shape[0], self.hparams.block_size):
            data_chunks.append(data[i : i + self.hparams.block_size])
            mask_chunks.append(token_type_mask[i : i + self.hparams.block_size])

        return data_chunks, mask_chunks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"sequence": self.data[idx], "mask": self.masks[idx]}


@DATAMODULE_REGISTRY
class KitchenDataModule(pl.LightningDataModule):
    """
    Base class for Kitchen dataset
    """

    def __init__(self, data_conf, **kwargs):
        super().__init__()

        # saves parameters into hparams attribute
        self.save_hyperparameters(data_conf)

    def prepare_data(self):
        env = gym.make(self.hparams.env_name)

        # get dataset
        # dictionary of state, actions
        self.dataset = env.get_dataset()

    def split_tr_and_val(self, data: List):
        num_examples = len(data)
        num_train_ex = int(num_examples * self.hparams.split["train"])
        num_val_ex = num_examples - num_train_ex
        return num_train_ex, num_val_ex

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cfg = OmegaConf.create(self.hparams["dataset_cls"])
            self.kitchen_dataset = instantiate(
                cfg, dataset=self.dataset, _recursive_=False
            )
            num_tr, num_val = self.split_tr_and_val(self.kitchen_dataset.data)
            self.train, self.val = random_split(self.kitchen_dataset, [num_tr, num_val])

    def train_dataloader(self):
        cfg = OmegaConf.create(self.hparams["dataloader_cls"])

        dataloader = instantiate(
            cfg,
            dataset=self.train,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
            batch_sampler=self.kitchen_dataset.get_sampler(self.train.indices),
            collate_fn=self.kitchen_dataset.collate_fn,
        )
        return dataloader

    def val_dataloader(self):
        cfg = OmegaConf.create(self.hparams["dataloader_cls"])
        cfg.n_repeat = 1

        dataloader = instantiate(
            cfg,
            dataset=self.train,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
            batch_sampler=self.kitchen_dataset.get_sampler(self.val.indices),
            collate_fn=self.kitchen_dataset.collate_fn,
        )
        return dataloader

    def test_dataloader(self):
        pass
        # return DataLoader(self.test, batch_size=32)

    def teardown(self, stage: Optional[str] = None):
        pass


@DATAMODULE_REGISTRY
class LanguageBehaviorDataModule(KitchenDataModule):
    """
    Language-Behavior dataset
    """

    def setup(self, stage: Optional[str] = None):
        l_cfg = OmegaConf.create(self.hparams["language_dataset_cls"])
        self.lang_dataset = instantiate(l_cfg, _recursive_=False)

        b_cfg = OmegaConf.create(self.hparams["behavior_dataset_cls"])
        self.behavior_dataset = instantiate(
            b_cfg, dataset=self.dataset, _recursive_=False
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            num_tr, num_val = self.split_tr_and_val(self.behavior_dataset.semantic_seqs)
            self.behavior_train, self.behavior_val = random_split(
                self.behavior_dataset, [num_tr, num_val]
            )
            num_tr, num_val = self.split_tr_and_val(self.lang_dataset)
            self.lang_train, self.lang_val = random_split(
                self.lang_dataset, [num_tr, num_val]
            )