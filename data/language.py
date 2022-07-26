import os
import pickle
from data.dataset import BaseDataset
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Optional, Dict, List, Any
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict


class LanguageSkillsDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

        data_file = os.path.join(
            self.hparams.data_dir, self.hparams.language_skills_file
        )
        with open(data_file, "r+") as f:
            sents = f.readlines()

        vocab_file = os.path.join(self.hparams.data_dir, self.hparams.vocab_file)
        self.vocab = pickle.load(open(vocab_file, "rb"))
        self.skill_embs, self.skill_tokens = self.preprocess_text(sents)

        for skill_emb, next_skill_emb, skill_token, next_skill_token in zip(
            self.skill_embs[:-1],
            self.skill_embs[1:],
            self.skill_tokens[:-1],
            self.skill_tokens[1:],
        ):
            self.data.append(
                AttrDict(
                    skill_emb=skill_emb,
                    next_skill_emb=next_skill_emb,
                    skill_tokens=skill_token,
                    next_skill_token=next_skill_token,
                )
            )

    def split_nsp(self):
        import random

        sentence_a = []
        sentence_b = []
        label = []

        num_sentences = len(self.sentences)
        start = random.randint(0, num_sentences - 2)
        # 50/50 whether is IsNextSentence or NotNextSentence
        if random.random() >= 0.5:
            # this is IsNextSentence
            sentence_a.append(self.sentences[start])
            sentence_b.append(self.sentences[start + 1])
            label.append(0)
        else:
            index = random.randint(0, bag_size - 1)
            # this is NotNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(bag[index])
            label.append(1)

    def preprocess_text(self, sents):
        map_file = os.path.join(self.hparams.data_dir, self.hparams.mapping_file)
        maps = pickle.load(open(map_file, "rb"))
        text_to_emb_map = maps["text_to_emb"]
        text_to_lang_token_map = maps["text_to_lang_token"]

        sentence_tokens, sentence_embs = [], []

        for sent in sents:
            sent = sent.strip()
            if sent not in text_to_emb_map:
                raise KeyError(f"{sent} not in mapping")
            else:
                sentence_embs.append(text_to_emb_map[sent])
                sentence_tokens.append(text_to_lang_token_map[sent])

        return sentence_embs, sentence_tokens

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
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
