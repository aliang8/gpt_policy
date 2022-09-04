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
            os.path.join(self.hparams.data_dir, "obj_cls.vocab")
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
        action_log_probs: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor,
        **kwargs
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

        mask = action_mask.flatten()

        if any(action_gt[mask] >= self.num_actions):
            import ipdb

            ipdb.set_trace()

        # action prediction
        action_pred_loss = self.action_loss_fn(
            action_pred_logits[mask], action_gt[mask].long()
        )
        action_pred_loss = action_pred_loss.mean()

        # target object prediction
        valid_interact = kwargs["valid_interact"].flatten()
        valid_interact = mask & valid_interact

        if any(target_obj_gt[valid_interact].long() >= self.num_objects) or any(
            target_obj_gt[valid_interact].long() == -1
        ):
            import ipdb

            ipdb.set_trace()

        target_pred_loss = self.interact_obj_loss_fn(
            target_obj_pred_logits[valid_interact], target_obj_gt[valid_interact].long()
        )
        target_pred_loss = target_pred_loss.mean()

        return {
            "action_pred_loss": action_pred_loss,
            "target_pred_loss": target_pred_loss,
        }
