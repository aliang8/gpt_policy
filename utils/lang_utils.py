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
