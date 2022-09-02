import torch
import numpy as np
from transformers import AutoTokenizer

ACT_TOKEN = "<ACT>"
LANG_BOS_TOKEN = "<LBOS>"
LANG_EOS_TOKEN = "<LEOS>"
NEXT_LANG_BOS_TOKEN = "<NLBOS>"
NEXT_LANG_EOS_TOKEN = "<NLEOS>"


def get_tokenizer(model_cls):
    tokenizer = AutoTokenizer.from_pretrained(
        model_cls,
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
    tokenizer.add_special_tokens(special_tokens_dict=special_tokens)
    return tokenizer


def add_start_and_end_str(list_of_str):
    for i, string in enumerate(list_of_str):
        list_of_str[i] = f"{LANG_BOS_TOKEN} {string} {LANG_EOS_TOKEN}"
    return list_of_str


def add_start_and_end_token(tokenizer, lang_tokens):
    start_tok = tokenizer.vocab[LANG_BOS_TOKEN]
    end_tok = tokenizer.vocab[LANG_EOS_TOKEN]

    if torch.is_tensor(lang_tokens):
        lang_tokens = torch.cat(
            [torch.tensor([start_tok]), lang_tokens, torch.tensor(end_tok)], dim=0
        )
    elif type(lang_tokens) is np.ndarray:
        lang_tokens = np.concatenate(
            [np.array([start_tok]), lang_tokens, np.array([end_tok])], axis=0
        )
    else:
        lang_tokens = [start_tok] + lang_tokens + [end_tok]

    return lang_tokens
