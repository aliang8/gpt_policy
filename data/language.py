import os
import numpy as np
from typing import Optional, Dict, List, Any
from data.dataset import BaseDataset
from utils.lang_utils import get_tokenizer, LANG_BOS_TOKEN, LANG_EOS_TOKEN
from data.ai2_objects import ALL_AI2THOR_OBJECT_CLASSES


class LanguageDataset(BaseDataset):
    """
    Inputs a long string of text. Each sentence delimited by a period is a language
    annotation / description. Tokenize each sentence.
    """

    def __init__(self, hparams: Dict, text: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        if text is None:
            text = self.get_text()

        self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)
        input_ids, attention_mask = self.encode_text(text)

        # split up tokens into chunks
        # this is useful when the text is too long for the model
        self.chunks = []
        chunk_size = self.hparams.chunk_size
        for i in range(0, len(input_ids), chunk_size):
            self.chunks.append(
                {
                    "input_ids": input_ids[i : i + chunk_size],
                    "attention_mask": attention_mask[i : i + chunk_size],
                }
            )

        # pad to chunk size
        last_input_id = self.chunks[-1]["input_ids"]
        last_attn_mask = self.chunks[-1]["attention_mask"]
        self.chunks[-1]["input_ids"] = np.pad(
            last_input_id, pad_width=(0, chunk_size - len(last_input_id))
        )
        self.chunks[-1]["attention_mask"] = np.pad(
            last_attn_mask, pad_width=(0, chunk_size - len(last_attn_mask))
        )

    def get_text(self):
        text_file = os.path.join(
            self.hparams.data_dir, self.hparams.language_skills_file
        )
        with open(text_file, "r") as f:
            text = f.read().strip()
        return text

    def encode_text(self, text):
        # deliminate text
        sentences = text.split(". ")

        # add bos and eos tokens
        sentences = [
            f"{LANG_BOS_TOKEN} {s.strip()} {LANG_EOS_TOKEN}" for s in sentences
        ]

        # tokenize the sentences, don't pad so we can just concatenate them later
        tokens = self.tokenizer(sentences, return_tensors="np")

        input_ids = np.concatenate(tokens["input_ids"])
        attention_mask = np.concatenate(tokens["attention_mask"])
        return input_ids, attention_mask

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class WikiHowDataset(LanguageDataset):
    def get_text(self):
        base_dir = self.hparams.wikihow_data_dir
        file = self.hparams.wikihow_data_file
        with open(os.path.join(base_dir, file), "r") as f:
            text = f.read()

        return text
