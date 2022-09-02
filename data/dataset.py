import abc
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Dict, List, Any
from torch.utils.data.dataloader import default_collate
from utils.lang_utils import get_tokenizer


class BaseDataset(Dataset):
    def get_sampler(self, indices):
        return None  # use default batch sampler

    def collate_fn(self, data):
        return default_collate(data)

    @abc.abstractmethod
    def _split_dataset_into_sequences(self):
        """
        Given an initial dataset, create sequences.
        """
        return

    @abc.abstractmethod
    def _split_by_semantic_skills(self):
        """
        Given a dataset of sequences, split them into semantic skills
        """
        return

    @abc.abstractmethod
    def _tokenize_sequence(self):
        """
        Take semantic skill sequences and concatenate them together into a long
        sequence of tokens. Also create masks for each data modality.
        """
        return

    @abc.abstractmethod
    def _chunk_data(self):
        """
        Take concatenated sequence of state/action language and split them
        into chunks.
        """
        return

    def _add_done_info(self, seq):
        """
        Add done information to each step in the sequence. Can be percent done or binary.
        Used for learning a done predictor. Done is computed per semantic sequence.
        """
        done = np.zeros((len(seq.states),))
        skill_done = (np.where(seq.skills[:-1] != seq.skills[1:]))[0]
        skill_done = np.concatenate([skill_done, np.array([len(seq.states) - 1])])

        start = 0
        for done_idx in skill_done:
            if self.hparams.load_frac_done:
                skill = seq.skills[start : done_idx + 1]
                done[start : done_idx + 1] = np.cumsum(
                    np.ones((len(skill))) / len(skill)
                )
                start = done_idx + 1
            else:
                done[done_idx] = 1

        return done


class SingleSequenceDataset(BaseDataset):
    """
    Input format:
    v0: L1 | s1,a2,s2,a2,... | L2 | s1,a2,s2,a2,... | L3 | s1,a2,s2,a2,...
    v1: s1 | L1 | a1, s2, a2, s3, a3 ... | s1 | L2 | a1 ...
    """

    def __init__(self, *args, **kwargs):
        self.data = {}

        self.data_keys = [
            "states",
            "actions",
            "timesteps",
            "dones",
            "first_states",
            "valid_interact",
        ]

        self.mask_keys = [
            "tokens",
            "combined_state_mask",
            "combined_action_mask",
            "lang_token_mask",
            "combined_rtg_mask",
            "token_type_ids",
        ]

        for key in self.data_keys:
            self.data[f"all_{key}"] = []

        for key in self.mask_keys:
            self.data[key] = []

        # first split the demonstrations into individual semantic skills
        self.semantic_seqs = self._split_by_semantic_skills()

        self._concatenate_sequence()

        # tokenize whatever possible and put into a long sequence
        print("tokenizing data")
        self._tokenize_sequence()

        # split data into evenly sized chunks for training
        print("chunking data")
        self.chunks = self._chunk_data()

    def _concatenate_sequence(self):
        for seq in self.semantic_seqs:
            for key in self.data_keys:
                if key in seq:
                    self.data[f"all_{key}"].append(seq[key])

    def _tokenize_sequence(self):
        # combine every state/action/language into a long sequence
        # then split tokens into chunk for each data sample
        # create masks to index the language, state, and actions separately
        start = 0

        for seq in self.semantic_seqs:
            # ignore pads
            if self.hparams.load_lang:
                lang_tokens, lang_attn = (
                    seq["lang_token_ids"],
                    seq["lang_attention_mask"],
                )
                num_lang_tokens = sum(lang_attn)
            else:
                num_lang_tokens = 0

            T = seq["actions"].shape[0]

            # states - T, actions - T, return - T
            if self.hparams.return_conditioned:
                total_num_tokens = 3 * T + num_lang_tokens
                skip = 3
            else:
                total_num_tokens = 2 * T + num_lang_tokens
                skip = 2

            concat_sequence = np.zeros((total_num_tokens))
            lang_token_mask_ = np.zeros((total_num_tokens))
            action_mask_ = np.zeros((total_num_tokens))
            state_mask_ = np.zeros((total_num_tokens))
            rtg_mask_ = np.zeros((total_num_tokens))

            if self.hparams.input_format == "v0":
                # L1 | s1 | a1
                state_r = slice(num_lang_tokens, total_num_tokens, skip)
                action_r = slice(num_lang_tokens + 1, total_num_tokens, skip)

                if self.hparams.load_lang:
                    lang_r = slice(0, num_lang_tokens)
            elif self.hparams.input_format == "v1":
                if self.hparams.return_conditioned:
                    # R1, s1 | L1 | a1 ....
                    state_mask_[1] = 1
                    concat_sequence[1] = start

                    state_r = slice(2 + num_lang_tokens + 2, total_num_tokens, skip)
                    action_r = slice(2 + num_lang_tokens, total_num_tokens, skip)

                    if self.hparams.load_lang:
                        lang_r = slice(2, num_lang_tokens + 2)

                    rtg_mask_[0] = 1
                    rtg_r = slice(2 + num_lang_tokens + 1, total_num_tokens, skip)
                    rtg_mask_[rtg_r] = 1

                else:
                    # s1 | L1 | a1 ....
                    state_mask_[0] = 1
                    concat_sequence[0] = start

                    state_r = slice(1 + num_lang_tokens + 1, total_num_tokens, skip)
                    action_r = slice(1 + num_lang_tokens, total_num_tokens, skip)

                    if self.hparams.load_lang:
                        lang_r = slice(1, num_lang_tokens + 1)

            state_mask_[state_r] = 1
            action_mask_[action_r] = 1

            # temporary put filler tokens for the state so
            # that we can extract the actual states when chunking
            if self.hparams.input_format == "v0":
                concat_sequence[state_r] = np.arange(start, start + T)
            elif self.hparams.input_format == "v1":
                concat_sequence[state_r] = np.arange(start + 1, start + T)

            concat_sequence[action_r] = np.arange(start, start + T)

            if self.hparams.load_lang:
                lang_token_mask_[lang_r] = 1
                concat_sequence[lang_r] = lang_tokens[:num_lang_tokens]

            # add to list
            token_type_id = 0 * state_mask_ + 0 * action_mask_ + 1 * lang_token_mask_
            self.data["tokens"].append(concat_sequence)
            self.data["token_type_ids"].append(token_type_id)
            self.data["combined_state_mask"].append(state_mask_)
            self.data["combined_action_mask"].append(action_mask_)
            self.data["lang_token_mask"].append(lang_token_mask_)

            if self.hparams.return_conditioned:
                self.data["combined_rtg_mask"].append(rtg_mask_)

            start += T

        for k, v in self.data.items():
            if len(v) > 0:
                self.data[k] = np.concatenate(v)

    def _chunk_data(self):
        chunks = []
        chunk_size = self.hparams.chunk_size

        i = 0

        while i < len(self.data["tokens"]):
            chunk = {}

            if i + chunk_size > len(self.data["tokens"]):  # need to drop the last chunk
                break

            r = slice(i, i + chunk_size)

            # try to get an even number of states and actions
            j = 0
            while (
                self.data["combined_state_mask"][r].sum()
                != self.data["combined_action_mask"][r].sum()
            ):
                j += 1
                r = slice(i, i + chunk_size + j)

            i += chunk_size + j

            indices = self.data["tokens"][r][
                self.data["combined_state_mask"][r].astype(np.bool)
            ]

            for key in self.data_keys:
                if self.data[f"all_{key}"] != []:
                    chunk[key] = np.take(
                        self.data[f"all_{key}"], indices.astype(np.int), axis=0
                    )

            for key in self.mask_keys:
                if key in self.data:
                    chunk[key] = self.data[key][r]

            chunk.update(
                {
                    "state_mask": np.ones(len(chunk["states"])),
                    "action_mask": np.ones(len(chunk["actions"])),
                }
            )

            chunks.append(chunk)

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class SingleSequenceBinaryDataset(SingleSequenceDataset):
    """
    Input format: e.g. s1, <1> | L1 | a1, s2, <0>, a2, s3, <0>, a3
    """

    def _tokenize_sequence(self):
        """
        Combine every state/action/language into a long sequence
        Add a binary token after the state to denote whether to predict
        language or action next. 1 - predict language, 0 - predict action

        The last token of language predicts the first action.
        """
        start = 0
        for seq in self.semantic_seqs:
            states, actions, timesteps = seq["states"], seq["actions"], seq["timesteps"]

            # ignore pads
            if self.hparams.load_lang:
                lang_tokens, lang_attn = (
                    seq["lang_token_ids"],
                    seq["lang_attention_mask"],
                )
                num_lang_tokens = sum(lang_attn)
            else:
                num_lang_tokens = 0

            self.data["all_states"].append(states)
            self.data["all_actions"].append(actions)
            self.data["all_timesteps"].append(timesteps)
            if "done" in seq:
                self.data["all_dones"].append(seq.done)

            T = actions.shape[0]
            # states - T, actions - T, predictor token - T
            total_num_tokens = 3 * T + num_lang_tokens

            concat_sequence = np.zeros((total_num_tokens))
            lang_token_mask_ = np.zeros((total_num_tokens))
            action_mask_ = np.zeros((total_num_tokens))
            state_mask_ = np.zeros((total_num_tokens))

            # Fix first state, binary_token, action
            # s1, <1> | L1 | a1
            state_mask_[0] = 1
            concat_sequence[0] = 0
            if num_lang_tokens > 0:
                concat_sequence[
                    1
                ] = 1  # mark the binary token before the first language
            state_r = slice(2 + num_lang_tokens + 1, total_num_tokens, 3)
            action_r = slice(2 + num_lang_tokens, total_num_tokens, 3)

            state_mask_[state_r] = 1
            action_mask_[action_r] = 1

            # temporary put filler tokens for the state so
            # that we can extract the actual states when chunking
            concat_sequence[state_r] = np.arange(start + 1, start + T)
            concat_sequence[action_r] = np.arange(start, start + T)

            # insert the language tokens into the sequence
            if self.hparams.load_lang:
                lang_r = slice(2, num_lang_tokens + 2)
                lang_token_mask_[lang_r] = 1
                concat_sequence[lang_r] = lang_tokens[:num_lang_tokens]

            # add to list
            # token_type_id = 0 * state_mask_ + 0 * action_mask_ + 1 * lang_token_mask_
            token_type_id = 0 * concat_sequence + 1 * lang_token_mask_
            self.data["tokens"].append(concat_sequence)
            self.data["token_type_ids"].append(token_type_id)
            self.data["state_mask"].append(state_mask_)
            self.data["action_mask"].append(action_mask_)
            self.data["lang_token_mask"].append(lang_token_mask_)

            start += T

        for k, v in self.data.items():
            if len(v) > 0:
                self.data[k] = np.concatenate(v)


# class MultiModalSingleSequenceDataset(SingleSequenceDataset):
