import os
from typing import Optional, Dict, List, Any
from data.dataset import BaseDataset
from utils.lang_utils import get_tokenizer, LANG_BOS_TOKEN, LANG_EOS_TOKEN


class LanguageDataset(BaseDataset):
    """
    Inputs a long string of text. Each sentence delimited by a period is a language
    annotation / description. Tokenize each sentence.
    """

    def __init__(self, text: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if text:
            self.text = text
        else:
            text_file = os.path.join(
                self.hparams.data_dir, self.hparams.language_skills_file
            )
            with open(text_file, "r") as f:
                self.text = f.read()

        # deliminate text
        sentences = self.text.split(". ")

        # add bos tokens
        sentences = [f"{LANG_BOS_TOKEN} {s} {LANG_EOS_TOKEN}" for s in sentences]

        # combine
        text = " ".join(sentences)

        self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)
        tokens = self.tokenizer(text, padding="longest", return_tensors="np")

        input_ids, attention_mask = tokens["input_ids"][0], tokens["attention_mask"][0]

        # split up tokens into chunks
        # this is useful when the text is too long for the model
        self.chunks = []
        chunk_size = self.hparams.chunk_size
        for i in range(0, len(tokens), chunk_size):
            self.chunks.append(
                {
                    "input_ids": input_ids[i : i + chunk_size],
                    "attention_mask": attention_mask[i : i + chunk_size],
                }
            )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]
