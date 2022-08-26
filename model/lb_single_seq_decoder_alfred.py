import os
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    GPT2Config,
    EncoderDecoderConfig,
)
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertOnlyNSPHead

from utils.pytorch_utils import ten2ar
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    accuracy_score,
)
import numpy as np
import more_itertools as mit
import torch.nn as nn
from transformers import AutoTokenizer
import pytorch_lightning as pl
from transformers import GPT2Model
from model.trajectory_gpt2 import TrajectoryGPT2
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertOnlyNSPHead
from utils.lang_utils import get_tokenizer, LANG_BOS_TOKEN, LANG_EOS_TOKEN

from model.modules.alfred_state_encoder import ALFREDStateEncoder
from model.lb_single_seq_decoder import Model as LBSingleSeqDecoder

import sys

# sys.path.append("/data/anthony/alfred")

# TODO: predict mask


@MODEL_REGISTRY
class Model(LBSingleSeqDecoder):
    def __init__(self, model_conf, **kwargs):
        super().__init__(model_conf, **kwargs)

        self.vocab_action = torch.load(
            os.path.join(self.hparams.data_dir, "data.vocab")
        )["action_low"]

        self.vocab_obj = torch.load(
            os.path.join(self.hparams.ET_ROOT, "files/obj_cls.vocab")
        )

        self.num_actions = len(self.vocab_action)
        self.num_objects = len(self.vocab_obj)

        with open(os.path.join(self.hparams.data_dir, "params.json"), "r") as f_params:
            self.dataset_info = json.load(f_params)

        self.embed_state = nn.Sequential(
            ALFREDStateEncoder(self.hparams.state_encoder, self.dataset_info),
            nn.Linear(self.state_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.action_dim = self.num_actions + self.num_objects
        self.predict_action = nn.Linear(self.embed_dim, self.action_dim)

        self.action_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.interact_obj_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def _compute_action_pred_loss(
        self,
        action_preds: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        # compute action prediction loss on valid actions
        action_gt = target_actions[..., 0].reshape(-1)
        target_obj_gt = target_actions[..., 1].reshape(-1)

        action_pred_logits = action_preds[..., : self.num_actions].reshape(
            -1, self.num_actions
        )
        target_obj_pred_logits = action_preds[..., self.num_actions :].reshape(
            -1, self.num_objects
        )

        # action prediction
        action_pred_loss = self.action_loss_fn(action_pred_logits, action_gt.long())
        action_pred_loss *= action_mask.flatten()
        action_pred_loss = action_pred_loss.mean()

        # target object prediction
        target_pred_loss = self.interact_obj_loss_fn(
            target_obj_pred_logits, target_obj_gt.long()
        )
        target_pred_loss *= action_mask.flatten()
        target_pred_loss = target_pred_loss.mean()

        return action_pred_loss, target_pred_loss

    def training_step(self, batch, batch_idx, phase="train"):
        losses = {}
        if "language" in batch:
            lang_input = batch["language"]
            token_ids, attention_mask = (
                lang_input["input_ids"],
                lang_input["attention_mask"],
            )

            _, lang_token_logits, _ = self.forward_state_action_lang(
                tokens=token_ids,
                lang_token_mask=attention_mask,
                token_type_ids=attention_mask,
            )

            lang_only_pred_loss = self._compute_lang_pred_loss(
                lang_token_logits=lang_token_logits,
                target_lang_tokens=lang_input["input_ids"],
                lang_token_mask=lang_input["attention_mask"].bool(),
            )
            losses["lang_only_pred_loss"] = lang_only_pred_loss

        if "behavior" in batch:
            behavior_input = batch["behavior"]

            action_preds, _, aux_pred = self.forward_state_action_lang(**behavior_input)
            action_pred_loss = self._compute_action_pred_loss(
                action_preds=action_preds,
                target_actions=behavior_input["actions"],
                action_mask=behavior_input["action_mask"].bool(),
            )
            losses["behavior_only_action_pred_loss"] = action_pred_loss

            binary_token_mask = torch.roll(
                behavior_input["combined_state_mask"], 1, dims=1
            ).bool()
            binary_token_mask[:, 0] = 0

            aux_pred_loss = self._compute_aux_pred_loss(
                aux_values=aux_pred,
                target_aux_values=behavior_input["tokens"],
                mask=binary_token_mask,
            )
            losses["aux_pred_loss"] = aux_pred_loss

        if "paired" in batch:
            paired_input = batch["paired"]
            action_preds, lang_token_logits, aux_pred = self.forward_state_action_lang(
                **paired_input
            )
            action_pred_loss, target_pred_loss = self._compute_action_pred_loss(
                action_preds=action_preds,
                target_actions=paired_input["actions"],
                action_mask=paired_input["action_mask"].bool(),
            )
            losses["paired_action_pred_loss"] = action_pred_loss
            losses["paired_target_pred_loss"] = target_pred_loss

            bos_token_mask = (paired_input["tokens"] == 50261) & paired_input[
                "lang_token_mask"
            ].bool()
            states_before_eos = torch.roll(bos_token_mask, -1, dims=1)
            states_before_eos[:, -1] = 0

            aux_pred_loss = self._compute_aux_pred_loss(
                aux_values=aux_pred,
                target_aux_values=states_before_eos.float(),
                mask=paired_input["combined_state_mask"].bool(),
            )
            losses["paired_aux_pred_loss"] = aux_pred_loss

            aux_pred_acc, aux_report = self._compute_aux_pred_stats(
                aux_values=aux_pred,
                target_aux_values=states_before_eos.float(),
                mask=paired_input["combined_state_mask"].bool(),
                phase=phase,
            )

            # log metrics for binary classification
            self.log_dict(
                {f"{phase}/binary_pred_acc": aux_pred_acc}, on_step=True, on_epoch=True
            )
            self.log_dict(aux_report, on_step=True, on_epoch=True)

        if self.hparams.get("train_paired_lang", False):
            lang_pred_loss = self._compute_lang_pred_loss(
                lang_token_logits=lang_token_logits,
                target_lang_tokens=paired_input["tokens"],
                lang_token_mask=paired_input["lang_token_mask"].bool(),
            )
            losses["paired_lang_pred_loss"] = lang_pred_loss

        # add prefix to losses and multiply by weight
        losses_with_prefix = {
            f"{phase}/{k}": v * self.hparams[f"{k}_weight"] for k, v in losses.items()
        }

        self.log_dict(
            losses_with_prefix,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # sum losses
        total_loss = sum([v for k, v in losses_with_prefix.items()])
        return total_loss
