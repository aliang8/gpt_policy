import os
import copy
import pickle
import random
import numpy as np
from data.dataset import BaseDataset
from torch.utils.data import random_split, DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Optional, Dict, List, Any
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict

from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class LanguageSkillsDataset(Dataset):
    def __init__(self, hparams: AttrDict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams

        data_file = os.path.join(
            self.hparams.data_dir, self.hparams.language_skills_file
        )
        with open(data_file, "r+") as f:
            sents = f.readlines()

        # tokenize the sentences
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.encoder_model)

        # use collator for masking sentences
        self.collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=self.hparams.mlm,
            mlm_probability=self.hparams.mlm_probability,
            return_tensors="np",
        )

        self.data = self._tokenize_and_mask(sents)

    def _tokenize_and_mask(self, sents):
        # create next skill prediction pairs
        nsp_data, is_next = self._generate_nsp_data(sents)

        tokens = self.tokenizer(nsp_data, padding=True, truncation=True)
        self.masked_tokens = self.collator(tokens.input_ids)

        data = copy.deepcopy(tokens)

        # copy over masked input ids and labels for masked tokens
        data.data["input_ids"] = self.masked_tokens["input_ids"]
        data.data["labels"] = self.masked_tokens["labels"]
        data.data["is_next"] = is_next
        return data

    def _generate_nsp_data(self, sents):
        # TODO: need to scale this to more data
        nsp_data, is_next_data = [], []
        for i in range(len(sents) - 1):
            sent = sents[i]

            if random.random() < 0.5:
                next_sent = sents[i + 1]
                is_next = True
            else:
                # select a different sentence
                indices = np.arange(len(sents))
                indices = np.delete(indices, i + 1)
                next_sent_idx = random.choice(indices)
                next_sent = sents[next_sent_idx]
                is_next = False

            nsp_data.append([sent, next_sent])
            is_next_data.append(is_next)

        return nsp_data, is_next_data

    def get_sampler(self):
        cfg = OmegaConf.create(self.hparams["language_dataset_sampler_cls"])

        sampler = instantiate(
            cfg,
            is_distributed=False,
            shuffle=self.shuffle,
            drop_last=False,
        )
        return sampler

    def __len__(self):
        return self.data.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": np.array(self.data.input_ids[idx]),
            "token_type_ids": np.array(self.data.token_type_ids[idx]),
            "attention_mask": np.array(self.data.attention_mask[idx]),
            "labels": np.array(self.data.labels[idx]),
            "is_next": int(self.data.is_next[idx]),
        }
