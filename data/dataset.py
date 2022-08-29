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
    def _tokenize_and_concatenate_sequence(self):
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
    Input format: L1 | s1,a2,s2,a2,... | L2 | s1,a2,s2,a2,... | L3 | s1,a2,s2,a2,...
    """

    def __init__(self, *args, **kwargs):
        self.semantic_seqs = self._split_by_semantic_skills()
        self.concat_seq = self._tokenize_and_concatenate_sequence()
        self.chunks = self._chunk_data()

    def _tokenize_and_concatenate_sequence(self):
        output = {
            "full_sequence": [],
            "token_type_ids": [],
            "state_mask": [],
            "action_mask": [],
            "lang_token_mask": [],
            "all_states": [],
            "all_actions": [],
            "all_dones": [],
            "all_first_states": [],
            "all_timesteps": [],
        }

        start = 0

        # combine every state/action/language into a long sequence
        # L1 | s1,a2,s2,a2,... | L2 | s1,a2,s2,a2,... | L3 | s1,a2,s2,a2,...
        # then split tokens into chunk for each data sample
        # create masks to index the language, state, and actions separately
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

            output["all_states"].append(states)
            output["all_actions"].append(actions)
            output["all_timesteps"].append(timesteps)

            if "done" in seq:
                output["all_dones"].append(seq.done)
            
            if "first_states" in seq:
                output["all_first_states"].append(seq.first_states)

            T = actions.shape[0]
            # states - T, actions - T
            total_num_tokens = 2 * T + num_lang_tokens

            concat_sequence = np.zeros((total_num_tokens))
            lang_token_mask_ = np.zeros((total_num_tokens))
            action_mask_ = np.zeros((total_num_tokens))
            state_mask_ = np.zeros((total_num_tokens))

            state_r = slice(num_lang_tokens, total_num_tokens, 2)
            action_r = slice(num_lang_tokens + 1, total_num_tokens, 2)

            state_mask_[state_r] = 1
            action_mask_[action_r] = 1

            # temporary put filler tokens for the state so
            # that we can extract the actual states when chunking
            concat_sequence[state_r] = np.arange(start, start + T)
            concat_sequence[action_r] = np.arange(start, start + T)

            # insert the language tokens into the sequence
            if self.hparams.load_lang:
                lang_r = slice(0, num_lang_tokens)
                lang_token_mask_[lang_r] = 1
                concat_sequence[lang_r] = lang_tokens[:num_lang_tokens]

            # add to list
            token_type_id = 0 * state_mask_ + 0 * action_mask_ + 1 * lang_token_mask_
            output["full_sequence"].append(concat_sequence)
            output["token_type_ids"].append(token_type_id)
            output["state_mask"].append(state_mask_)
            output["action_mask"].append(action_mask_)
            output["lang_token_mask"].append(lang_token_mask_)

            start += T

        for k, v in output.items():
            if len(v) > 0:
                output[k] = np.concatenate(v)

        return output

    def _chunk_data(self):
        concat_seq = self.concat_seq
        chunks = []
        chunk_size = self.hparams.chunk_size

        i = 0

        while i < len(concat_seq["full_sequence"]):
            if i + chunk_size > len(
                concat_seq["full_sequence"]
            ):  # need to drop the last chunk
                break

            r = slice(i, i + chunk_size)

            # try to get an even number of states and actions
            j = 0
            while (
                concat_seq["state_mask"][r].sum() != concat_seq["action_mask"][r].sum()
            ):
                j += 1
                r = slice(i, i + chunk_size + j)

            i += chunk_size + j

            state_indices = concat_seq["full_sequence"][r][
                concat_seq["state_mask"][r].astype(np.bool)
            ]
            action_indices = concat_seq["full_sequence"][r][
                concat_seq["action_mask"][r].astype(np.bool)
            ]

            states = np.take(
                concat_seq["all_states"], state_indices.astype(np.int), axis=0
            )
            actions = np.take(
                concat_seq["all_actions"], action_indices.astype(np.int), axis=0
            )
            timesteps = np.take(
                concat_seq["all_timesteps"], state_indices.astype(np.int), axis=0
            )

            if "all_dones" in concat_seq and len(concat_seq["all_dones"]) > 0:
                dones = np.take(
                    concat_seq["all_dones"], action_indices.astype(np.int), axis=0
                )
            else:
                dones = None
                
            if "all_first_states" in concat_seq and len(concat_seq["all_first_states"]) > 0:
                first_states = np.take(
                    concat_seq["all_first_states"], action_indices.astype(np.int), axis=0
                )
            else:
                first_states = None

            chunks.append(
                {
                    "tokens": concat_seq["full_sequence"][r],
                    "states": states,
                    "actions": actions,
                    "timesteps": timesteps,
                    "dones": dones,
                    "first_states": first_states,
                    "state_mask": np.ones(len(states)),
                    "action_mask": np.ones(len(actions)),
                    "combined_state_mask": concat_seq["state_mask"][r],
                    "combined_action_mask": concat_seq["action_mask"][r],
                    "lang_token_mask": concat_seq["lang_token_mask"][r],
                    "token_type_ids": concat_seq["token_type_ids"][r],
                }
            )

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class SingleSequenceDatasetV2(SingleSequenceDataset):
    """
    Input format: s1 | L1 | a1, s2, a2, s3, a3 ... | s1 | L2 | a1 ...
    """

    def _tokenize_and_concatenate_sequence(self):
        """
        Combine every state/action/language into a long sequence
        """

        output = {
            "full_sequence": [],
            "token_type_ids": [],
            "state_mask": [],
            "action_mask": [],
            "lang_token_mask": [],
            "all_states": [],
            "all_actions": [],
            "all_dones": [],
            "all_first_states": [],
            "all_timesteps": [],
        }

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

            output["all_states"].append(states)
            output["all_actions"].append(actions)
            output["all_timesteps"].append(timesteps)

            if "done" in seq:
                output["all_dones"].append(seq.done)

            if "first_states" in seq:
                output["all_first_states"].append(seq.first_states)
                
            T = actions.shape[0]
            # states - T, actions - T
            total_num_tokens = 2 * T + num_lang_tokens

            concat_sequence = np.zeros((total_num_tokens))
            lang_token_mask_ = np.zeros((total_num_tokens))
            action_mask_ = np.zeros((total_num_tokens))
            state_mask_ = np.zeros((total_num_tokens))

            # s1 | L1 | a1 ....
            state_mask_[0] = 1
            concat_sequence[0] = start

            state_r = slice(1 + num_lang_tokens + 1, total_num_tokens, 2)
            action_r = slice(1 + num_lang_tokens, total_num_tokens, 2)

            state_mask_[state_r] = 1
            action_mask_[action_r] = 1

            # temporary put filler tokens for the state so
            # that we can extract the actual states when chunking
            concat_sequence[state_r] = np.arange(start + 1, start + T)
            concat_sequence[action_r] = np.arange(start, start + T)

            # insert the language tokens into the sequence
            if self.hparams.load_lang:
                lang_r = slice(1, num_lang_tokens + 1)
                lang_token_mask_[lang_r] = 1
                concat_sequence[lang_r] = lang_tokens[:num_lang_tokens]

            # add to list
            token_type_id = 0 * state_mask_ + 0 * action_mask_ + 1 * lang_token_mask_
            output["full_sequence"].append(concat_sequence)
            output["token_type_ids"].append(token_type_id)
            output["state_mask"].append(state_mask_)
            output["action_mask"].append(action_mask_)
            output["lang_token_mask"].append(lang_token_mask_)

            start += T

        for k, v in output.items():
            if len(v) > 0:
                output[k] = np.concatenate(v)

        return output


class SingleSequenceBinaryDataset(SingleSequenceDataset):
    """
    Input format: e.g. s1, <1> | L1 | a1, s2, <0>, a2, s3, <0>, a3
    """

    def _tokenize_and_concatenate_sequence(self):
        """
        Combine every state/action/language into a long sequence
        Add a binary token after the state to denote whether to predict
        language or action next. 1 - predict language, 0 - predict action

        The last token of language predicts the first action.
        """
        output = {
            "full_sequence": [],
            "token_type_ids": [],
            "state_mask": [],
            "action_mask": [],
            "lang_token_mask": [],
            "all_states": [],
            "all_actions": [],
            "all_dones": [],
            "all_timesteps": [],
        }

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

            output["all_states"].append(states)
            output["all_actions"].append(actions)
            output["all_timesteps"].append(timesteps)
            if "done" in seq:
                output["all_dones"].append(seq.done)

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
            output["full_sequence"].append(concat_sequence)
            output["token_type_ids"].append(token_type_id)
            output["state_mask"].append(state_mask_)
            output["action_mask"].append(action_mask_)
            output["lang_token_mask"].append(lang_token_mask_)

            start += T

        for k, v in output.items():
            if len(v) > 0:
                output[k] = np.concatenate(v)

        return output
