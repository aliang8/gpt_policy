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

import torch.nn.functional as F
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

from utils.logger_utils import get_logger

logger = get_logger("model")


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

        state_enc = ALFREDStateEncoder(self.hparams.state_encoder, self.dataset_info)
        state_dim = state_enc.layers[-1].out_features

        self.embed_state = nn.Sequential(
            state_enc,
            nn.Linear(state_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.action_dim = self.num_actions + self.num_objects
        self.predict_action = nn.Linear(
            self.embed_dim, self.num_actions + self.num_objects
        )

        self.action_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.interact_obj_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def get_action_preds(self, model_out, masks, use_means=False, target_actions=None):
        # B x T x D
        action_preds_combined = torch.zeros(
            (*model_out.shape[:2], self.action_dim), device=self.device
        )

        action_dist, action_entropy, target_obj_entropy = None, None, None

        # make output easy for computing loss
        # B x num_action_tokens x D
        action_preds_full = torch.zeros(
            (*masks["action_mask"].shape, self.action_dim), device=self.device
        )

        # predict actions
        # only use states not before a BOS token
        state_out = model_out[
            masks["combined_state_mask"] & ~masks["states_before_bos_mask"]
        ]

        if not self.training and masks["combined_state_mask"][:, -1].any():
            state_out = state_out[:-1]

        action_preds = self.predict_action(state_out)

        # predict action from EOS token
        if masks["lang_token_mask"].sum() != 0:
            eos_token_out = model_out[masks["eos_token_mask"]].reshape(
                -1, self.embed_dim
            )

            actions_eos_mask = (
                masks["combined_action_mask"] & ~masks["actions_post_eos_mask"]
            )

            lang_act_preds = self.predict_action(eos_token_out)
            action_preds_combined[actions_eos_mask] = action_preds
            action_preds_combined[masks["actions_post_eos_mask"]] = lang_act_preds
            action_preds = action_preds_combined[masks["combined_action_mask"]]

        action_preds_full[masks["action_mask"]] = action_preds

        # predict binary token
        state_out = model_out[masks["combined_state_mask"]]
        aux_pred = self.aux_pred(state_out)

        action_log_probs, entropy = None, None

        action_pred_, target_pred_ = torch.split(
            action_preds, [self.num_actions, self.num_objects], dim=-1
        )

        action_dist = torch.distributions.Categorical(F.softmax(action_pred_, dim=-1))
        target_obj_dist = torch.distributions.Categorical(
            F.softmax(target_pred_, dim=-1)
        )

        if target_actions is not None:
            # split target actions
            target_actions, target_objs = target_actions[..., 0], target_actions[..., 1]

            action_log_probs = action_dist.log_prob(
                action_pred_.reshape(-1, self.num_actions).argmax(dim=-1)
            )

            target_obj_log_probs = target_obj_dist.log_prob(
                target_pred_.reshape(-1, self.num_objects).argmax(dim=-1)
            )
            action_entropy = action_dist.entropy()
            target_obj_entropy = target_obj_dist.entropy()

            action_log_probs = torch.cat(
                [action_log_probs, target_obj_log_probs], dim=-1
            )
            entropy = torch.cat([action_entropy, target_obj_entropy], dim=-1)

        return AttrDict(
            action_preds=action_preds_full,
            aux_pred=aux_pred,
            action_dist=action_dist,
            target_obj_dist=target_obj_dist,
            action_log_probs=action_log_probs,
            action_entropy=action_entropy,
            target_obj_entropy=target_obj_entropy,
        )

    def _compute_action_pred_loss(
        self,
        action_preds: torch.Tensor,
        action_log_probs: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor,
        **kwargs,
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
        valid_interact = kwargs["valid_interact_mask"].flatten()
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

    def train_behavior(self, behavior_input, phase, modality):
        if "image_key" in behavior_input:
            image_feats = self.get_image_features(
                behavior_input["image_key"], f"{phase}/{modality}"
            )
            behavior_input["states"] = image_feats

        model_out, masks = self.forward_state_action_lang(**behavior_input)
        output_dict = self.get_predictions(
            model_out,
            masks,
            use_means=False,
            target_actions=behavior_input["actions"],
        )

        action_losses = self._compute_action_pred_loss(
            action_preds=output_dict["action_preds"],
            action_log_probs=output_dict["action_log_probs"],
            target_actions=behavior_input["actions"],
            action_mask=masks["action_mask"].bool(),
            valid_interact_mask=masks["valid_interact_mask"].bool()
            if "valid_interact_mask" in masks
            else None,
        )

        # compute mse
        if self.hparams.continuous_actions:
            action_pred_mse = self.action_loss_fn(
                output_dict["action_preds"], behavior_input["actions"]
            )
            action_pred_mse *= masks["action_mask"].bool().unsqueeze(-1)
            action_pred_mse = action_pred_mse.mean()
        else:
            action_pred_mse = None

        # entropy term
        if self.target_entropy:
            action_losses["entropy_loss"] = (
                -torch.exp(self.log_alpha.detach())
                * output_dict["action_entropy"].mean()
            )
            log_alpha_loss = self._update_log_alpha(output_dict["action_entropy"])
        else:
            action_losses["entropy_loss"] = -output_dict["action_entropy"].mean()

        return output_dict, masks, action_losses, log_alpha_loss, action_pred_mse
