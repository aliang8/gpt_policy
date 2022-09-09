import copy
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
import collections
import numpy as np
import more_itertools as mit
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import pytorch_lightning as pl
from transformers import GPT2Model
from model.trajectory_gpt2 import TrajectoryGPT2
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertOnlyNSPHead
from utils.lang_utils import get_tokenizer, LANG_BOS_TOKEN, LANG_EOS_TOKEN
from utils.stat_utils import merge_normal_dist
from utils.data_utils import padded_first_dim

# from model.modules.alfred_state_encoder import ALFREDStateEncoder

from torch.distributions import Normal, Independent
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

from utils.logger_utils import get_logger

logger = get_logger("model")


@MODEL_REGISTRY
class Model(pl.LightningModule):
    """
    Implementation of transformer decoder model that can consume both language
    and behavior. States predict a binary token which determines whether we
    predict language next or actions next. The binary prediction is implemented
    as a separate head while predicting the action.

                                                  <0>     <0>
    <1> Open  the  microwave  <EOS>               a2      a3
    |    |     |      |         |                 |       |
    ================================ Transformer ================================
    |    |     |      |         |       |         |   |   |   |
    s1 <BOS>  Open  the      microwave <EOS>      s2  a2  s3  a3


    Model is an autoregressive decoder model.
    """

    def __init__(self, model_conf, **kwargs):
        super().__init__()

        # saves parameters into hparams attribute
        self.save_hyperparameters(model_conf)
        self.decoder_config = GPT2Config.from_pretrained(self.hparams.decoder_model_cls)

        self.embed_dim = self.hparams.hidden_dim

        self.decoder_config.update(self.hparams.decoder_config)

        # self.decoder_config.update(
        #     {
        #         "n_layer": self.hparams.n_layer,
        #         "n_head": self.hparams.n_head,
        #     }
        # )

        logger.info(f"decoder config: {self.decoder_config}")

        if self.hparams.get("load_pretrained_lm_weights", False):
            logger.info("loading pretrained GPT2")
            # load from pretrained GPT2
            trajectory_gpt2 = TrajectoryGPT2.from_pretrained(
                self.hparams.decoder_model_cls
            )
        else:
            trajectory_gpt2 = TrajectoryGPT2(self.decoder_config)
            pretrained_gpt = GPT2Model.from_pretrained(self.hparams.decoder_model_cls)

            # load the pretrained word embedding and positional embedding from pretrained gpt model
            self.wte = nn.Embedding(self.decoder_config.vocab_size, self.embed_dim)
            self.wpe = nn.Embedding(
                self.decoder_config.max_position_embeddings, self.embed_dim
            )

            if self.training:
                self.wte.load_state_dict(pretrained_gpt.wte.state_dict())
                self.wpe.load_state_dict(pretrained_gpt.wpe.state_dict())

        self.model = trajectory_gpt2

        if not self.hparams.get("load_pretrained_lm_weights", False):
            self.model.transformer.wte = self.wte
            self.model.transformer.wpe = self.wpe

        self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)
        self.vocab_size = len(self.tokenizer.vocab)
        logger.info(f"vocab size: {self.vocab_size}")

        self.model.transformer.wte = self.model.resize_token_embeddings(
            len(self.tokenizer)
        )

        self.state_dim = self.hparams.state_dim
        self.action_dim = self.hparams.action_dim
        self.output_gaussian = (
            "output_gaussian" in self.hparams and self.hparams.output_gaussian
        )

        # embedding heads each modality before inputting to transformer
        self.embed_state = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.embed_action = nn.Linear(self.action_dim, self.embed_dim)
        self.embed_return = torch.nn.Linear(1, self.embed_dim)
        self.embed_ln = nn.LayerNorm(self.embed_dim)

        self.model.embed_ln = self.embed_ln
        self.embed_timestep = nn.Embedding(self.hparams.max_ep_len, self.embed_dim)

        # output head for predicting action
        action_tanh = False
        if "discretize_actions" in self.hparams and self.hparams.discretize_actions:
            self.predict_action = nn.Linear(self.embed_dim, self.hparams.num_bins)
        elif self.hparams.stochastic:
            self.predict_action_mean = nn.Sequential(
                nn.Linear(self.embed_dim, self.action_dim)
            )
            self.predict_action_logstd = nn.Sequential(
                nn.Linear(self.embed_dim, self.action_dim)
            )
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(self.embed_dim, self.action_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )

        # loss functions
        if self.output_gaussian:
            self.action_loss_fn = torch.nn.GaussianNLLLoss(reduction="none")
        else:
            self.action_loss_fn = torch.nn.MSELoss(reduction="none")

        self.lang_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.log_alpha = torch.nn.Parameter(
            torch.zeros(1, requires_grad=True, device=self.device)
        )
        self.target_entropy = -self.action_dim

        # predicts a float value between 0 and 1
        if self.hparams.get("pred_progress", False):
            self.aux_pred = nn.Sequential(nn.Linear(self.embed_dim, 1), nn.Tanh())
            self.aux_pred_loss_fn = torch.nn.MSELoss()
        elif self.hparams.get("pred_done", False):
            # don't need sigmoid because we are doing BCEWithLogits
            self.aux_pred = nn.Linear(self.embed_dim, 1)
            if self.hparams.get("binary_pos_weight", None):
                self.aux_pred_loss_fn = torch.nn.BCEWithLogitsLoss(
                    reduction="none",
                    pos_weight=torch.tensor(self.hparams.binary_pos_weight),
                )
            else:
                self.aux_pred_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        # predict return
        self.predict_return = torch.nn.Linear(self.embed_dim, 1)

        self.automatic_optimization = False
        self.return_conditioned = True

        self.mask_keys = [
            "tokens",
            "state_mask",
            "action_mask",
            "combined_state_mask",
            "combined_action_mask",
            "lang_token_mask",
            "combined_rtg_mask",
            "token_type_ids",
            "valid_interact_mask",
            "lang_token_ids",
        ]

    def add_extra_configs(self, extra_configs):
        # language generation
        if "decode_params" in extra_configs:
            bos_token_id = self.tokenizer.vocab[LANG_BOS_TOKEN]
            eos_token_id = self.tokenizer.vocab[LANG_EOS_TOKEN]

            self.generation_kwargs = dict(
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                **extra_configs.decode_params,
            )
        self.hparams.update(extra_configs)

    def get_masks(self, kwargs):
        # a bunch of useful masks
        masks = {}
        extra_config = {}
        for k, kwarg in kwargs.items():
            if kwarg is not None and k in self.mask_keys:
                masks[k] = kwarg
                if "mask" in k:
                    masks[k] = kwarg.bool()

            if k not in self.mask_keys and "mask" not in k:
                extra_config[k] = kwarg

        tokens = masks["tokens"]

        if tokens.shape[-1] != 0:
            # add additional masks
            masks["bos_token_mask"] = (tokens == 50261) & masks["lang_token_mask"]
            masks["states_before_bos_mask"] = torch.roll(
                masks["bos_token_mask"], -1, dims=1
            )
            masks["states_before_bos_mask"][:, -1] = False
            masks["eos_token_mask"] = (tokens == 50262) & masks["lang_token_mask"]
            masks["actions_post_eos_mask"] = torch.roll(
                masks["eos_token_mask"], 1, dims=1
            )
            masks["actions_post_eos_mask"][:, 0] = False

        return masks, extra_config

    def get_stochastic_action_pred(self, model_input):
        # predict mean and logstd
        action_means = self.predict_action_mean(model_input)
        action_logstds = self.predict_action_logstd(model_input)

        # bound logstd
        action_logstds = torch.clamp(
            action_logstds, self.hparams.log_std_min, self.hparams.log_std_max
        )
        action_stds = torch.exp(action_logstds)
        return action_means, action_stds

    def get_action_preds(self, model_out, masks, use_means=False, target_actions=None):
        # B x T x D
        action_preds_combined = torch.zeros(
            (*model_out.shape[:2], self.action_dim), device=self.device
        )

        if self.hparams.stochastic:
            action_preds_means = torch.zeros_like(action_preds_combined)
            action_preds_stds = torch.zeros_like(action_preds_combined)
        else:
            action_dist = None

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

        if self.hparams.stochastic:
            # predict mean and standard deviation
            action_means, action_stds = self.get_stochastic_action_pred(state_out)

            # use eos token to predict action
            if masks["lang_token_mask"].sum() != 0:
                eos_token_out = model_out[masks["eos_token_mask"]].reshape(
                    -1, self.embed_dim
                )

                actions_mask = (
                    masks["combined_action_mask"] & ~masks["actions_post_eos_mask"]
                )

                (
                    lang_action_means,
                    lang_action_stds,
                ) = self.get_stochastic_action_pred(eos_token_out)

                action_preds_means[actions_mask] = action_means
                action_preds_stds[actions_mask] = action_stds
                action_preds_means[masks["actions_post_eos_mask"]] = lang_action_means
                action_preds_stds[masks["actions_post_eos_mask"]] = lang_action_stds

                action_preds_means = action_preds_means[masks["combined_action_mask"]]
                action_preds_stds = action_preds_stds[masks["combined_action_mask"]]
            else:
                action_preds_means = action_means
                action_preds_stds = action_stds

            action_dist = Independent(Normal(action_preds_means, action_preds_stds), 1)

            if use_means:
                action_preds = action_dist.mean
            else:
                action_preds = action_dist.sample()
        else:
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
        if self.hparams.stochastic and target_actions is not None:
            # clamp target actions to prevent nans
            # eps = torch.finfo(target_actions.dtype).eps
            # target_actions = torch.clamp(target_actions, -1 + eps, 1 - eps)
            action_log_probs = action_dist.log_prob(
                target_actions[masks["action_mask"]].reshape(-1, self.action_dim)
            )
            entropy = action_dist.entropy()

        return AttrDict(
            action_preds=action_preds_full,
            aux_pred=aux_pred,
            action_dist=action_dist,
            action_log_probs=action_log_probs,
            entropy=entropy,
        )

    def get_lang_token_preds(self, model_out, lang_token_mask):
        lang_token_logits = torch.zeros(
            (*model_out.shape[:2], self.vocab_size), device=self.device
        )
        # use previous lang to predict next lang tokens
        lang_out = model_out[lang_token_mask].reshape(-1, self.embed_dim)
        lang_token_pred = self.model.lm_head(lang_out)
        lang_token_logits[lang_token_mask] = lang_token_pred

        return AttrDict(lang_token_logits=lang_token_logits)

    def get_predictions(self, model_out, masks, use_means=False, target_actions=None):
        output_dict = AttrDict()

        # predict actions and returns
        if "state_mask" in masks and masks["state_mask"].sum() != 0:
            action_pred_dict = self.get_action_preds(
                model_out, masks, use_means=use_means, target_actions=target_actions
            )
            # predict next return given state and action
            action_out = model_out[masks["combined_action_mask"]]
            return_preds = self.predict_return(action_out)

            output_dict.update(action_pred_dict)
            output_dict["return_preds"] = return_preds

        # predict lang tokens
        if "lang_token_mask" in masks and masks["lang_token_mask"].sum() != 0:
            lang_pred_dict = self.get_lang_token_preds(
                model_out, masks["lang_token_mask"]
            )
            output_dict.update(lang_pred_dict)

        return output_dict

    def forward_state_action_lang(
        self,
        states: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        returns_to_go: torch.Tensor = None,
        actions: torch.Tensor = None,
        **kwargs,
    ):
        """
        :param Tensor tokens:
            Tensor of sequence of tokens [batch_size x total_num_tokens x 1], mainly used for the binary tokens
        :param Tensor states:
            Tensor of sequence of states [batch_size x timesteps x state_dim]
        :param Tensor actions:
            Tensor of sequence of actions [batch_size x timesteps x action_dim]
        :param Tensor timesteps:
            Tensor of tokenized skills [batch_size x timesteps x 1]
        :param Dict kwargs:
            Additional data e.g. masks

        :return:
            - action_preds_full: Tensor of size [batch_size x total_num_tokens x action_dim]
            - lang_token_logits_full: Tensor of size [batch_size x total_num_tokens x vocab_size]
            - aux_pred: Tensor of size [batch_size x ]
        """
        masks, extra_configs = self.get_masks(kwargs)
        tokens = masks["tokens"]

        # create tensor to store embeddings as a sequence
        state_action_lang = torch.zeros(
            (*tokens.shape[:2], self.embed_dim), device=self.device
        )

        attention_mask = torch.zeros_like(tokens, device=self.device).bool()

        # filter states and actions from pad
        if states is not None:
            states = states[masks["state_mask"]]
            actions = actions[masks["action_mask"]]
            timesteps = timesteps[masks["state_mask"]]
            returns_to_go = returns_to_go[masks["state_mask"]]

            state_embeddings = self.embed_state(states)
            action_embeddings = self.embed_action(actions.float())
            timestep_embeddings = self.embed_timestep(timesteps.int())

            state_embeddings = state_embeddings + timestep_embeddings
            action_embeddings = (
                action_embeddings + timestep_embeddings[: action_embeddings.shape[0]]
            )

            state_action_lang[masks["combined_state_mask"]] = state_embeddings
            state_action_lang[masks["combined_action_mask"]] = action_embeddings

            attention_mask |= (
                masks["combined_state_mask"] | masks["combined_action_mask"]
            )

            if returns_to_go is not None:
                return_embeddings = self.embed_return(returns_to_go.unsqueeze(-1))
                return_embeddings = return_embeddings + timestep_embeddings
                state_action_lang[masks["combined_rtg_mask"]] = return_embeddings

                if self.return_conditioned:
                    attention_mask |= masks["combined_rtg_mask"]

        if masks["lang_token_mask"].sum() != 0:
            lang_token_ids = tokens[masks["lang_token_mask"]]
            lang_embeddings = self.model.transformer.wte(lang_token_ids.long())

            # create position ids for language
            seq_length_per_batch = masks["lang_token_mask"].sum(-1)
            position_ids = torch.zeros(0)
            for seq_length in seq_length_per_batch:
                position_ids = torch.cat(
                    [position_ids, torch.arange(seq_length)], dim=-1
                )

            # add position id to word token embedding
            position_ids = position_ids.int().to(self.device)
            position_embeds = self.model.transformer.wpe(position_ids)
            lang_embeddings = lang_embeddings + position_embeds
            state_action_lang[masks["lang_token_mask"]] = lang_embeddings

            attention_mask |= masks["lang_token_mask"]

        state_action_lang_embs = self.embed_ln(state_action_lang)

        transformer_outputs = self.model(
            inputs_embeds=state_action_lang_embs,
            attention_mask=attention_mask,
            token_type_ids=kwargs.get("token_type_ids", None).long(),
            lang_only_input=states is None,
        )

        # reshape x so the second dimension corresponds to states (0), actions (1)
        model_out = transformer_outputs["last_hidden_state"]
        return model_out, masks

    def _compute_action_pred_loss(
        self,
        action_preds: torch.Tensor,
        action_log_probs: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor,
        **kwargs,
    ):
        # compute action prediction loss on valid actions
        if self.hparams.stochastic:
            action_pred_loss = -torch.mean(action_log_probs)  # - action log prob
        else:
            action_pred_loss = self.action_loss_fn(action_preds, target_actions)
            action_pred_loss *= action_mask.unsqueeze(-1)
            action_pred_loss = action_pred_loss.mean()
        return {"action_pred_loss": action_pred_loss}

    def _compute_lang_pred_loss(
        self,
        lang_token_logits: torch.Tensor,
        target_lang_tokens: torch.Tensor,
        lang_token_mask: torch.Tensor,
    ):
        # predict next language skill given current skill as context
        # mask out loss for padding tokens
        pred_logits = lang_token_logits[lang_token_mask][..., :-1, :].contiguous()
        gt_tokens = target_lang_tokens[lang_token_mask][..., 1:].contiguous()

        # apply mask
        lang_pred_loss = self.lang_loss_fn(
            pred_logits.view(-1, self.vocab_size),
            gt_tokens.long().view(-1),
        )

        lang_pred_loss = lang_pred_loss.mean()

        return {"lang_pred_loss": lang_pred_loss}

    def _compute_aux_pred_loss(
        self,
        aux_values: torch.Tensor,
        target_aux_values: torch.Tensor,
        mask: torch.Tensor,
    ):
        # aux_value is N x 1 and target_aux_values is B x T
        # need to mask the target first
        # TODO: clean this up
        aux_pred_loss = self.aux_pred_loss_fn(
            aux_values.squeeze(), target_aux_values[mask]
        )
        aux_pred_loss = aux_pred_loss.mean()
        return aux_pred_loss

    def _compute_aux_pred_stats(
        self,
        aux_values: torch.Tensor,
        target_aux_values: torch.Tensor,
        mask: torch.Tensor,
        phase: str,
    ):
        pred_labels = (torch.sigmoid(aux_values) > 0.5).int()
        pred_labels = ten2ar(pred_labels).squeeze()

        true_labels = ten2ar(target_aux_values[mask].int())

        acc = accuracy_score(true_labels, pred_labels)

        unique_labels = np.unique(np.concatenate([true_labels, pred_labels]))
        target_names = [f"{phase}/{label}" for label in unique_labels]
        report = classification_report(
            true_labels,
            pred_labels,
            output_dict=True,
            zero_division=0,
            target_names=target_names,
        )
        # returns precision, recall, and support
        # precision: tp / (fp + tp)
        # recall: tp / (fn + tp), we want to maximize recall of positive class

        return acc, report

    def _update_log_alpha(self, entropy):
        _, log_alpha_opt = self.optimizers()

        log_alpha_loss = (
            torch.exp(self.log_alpha) * entropy.detach().mean() - self.target_entropy
        )

        log_alpha_opt.zero_grad()
        self.manual_backward(log_alpha_loss)
        log_alpha_opt.step()

        return log_alpha_loss

    def get_image_features(self, keys, dataset_key):
        keys = keys[:, 0].int().cpu().numpy()

        all_image_feats = []

        phase, modality = dataset_key.split("/")
        for idx in keys:
            key = "{:06}".format(idx).encode("ascii")
            # ds = self.trainer.datamodule.datasets[dataset_key]
            if phase == "train":
                ds = self.trainer.train_dataloader.dataset.datasets[modality]
            elif phase == "val":
                ds = self.trainer.val_dataloader.dataset.datasets[modality]

            image_feats = ds.load_frames(key)
            all_image_feats.append(image_feats)

        image_feats = padded_first_dim(all_image_feats)
        image_feats = image_feats.to(self.device)

        return image_feats

    def train_lang(self, lang_input):
        token_ids, attention_mask = (
            lang_input["input_ids"],
            lang_input["attention_mask"],
        )

        model_out, masks = self.forward_state_action_lang(
            tokens=token_ids,
            lang_token_mask=attention_mask,
            token_type_ids=attention_mask,
        )

        output_dict = self.get_predictions(model_out, masks)

        lang_losses = self._compute_lang_pred_loss(
            lang_token_logits=output_dict["lang_token_logits"],
            target_lang_tokens=lang_input["input_ids"],
            lang_token_mask=lang_input["attention_mask"].bool(),
        )
        return lang_losses

    def train_behavior(self, behavior_input, phase, modality):
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
                -torch.exp(self.log_alpha.detach()) * output_dict["entropy"].mean()
            )
            log_alpha_loss = self._update_log_alpha(output_dict["entropy"])
        else:
            action_losses["entropy_loss"] = -output_dict["entropy"].mean()

        log_alpha_loss = self._update_log_alpha(output_dict["entropy"])
        return output_dict, masks, action_losses, log_alpha_loss, action_pred_mse

    def training_step(self, batch, batch_idx, phase="train"):
        opt, _ = self.optimizers()

        losses, info = {}, {}

        if "language" in batch:
            lang_input = batch["language"]
            lang_losses = self.train_lang(lang_input)
            for k, loss in lang_losses.items():
                losses[f"language_only/{k}"] = loss

        if "behavior" in batch:
            behavior_input = batch["behavior"]

            output_dict, masks, action_losses, log_alpha_loss, *_ = self.train_behavior(
                behavior_input, phase, "behavior"
            )
            for k, loss in action_losses.items():
                losses[f"behavior_only/{k}"] = loss

            info["behavior_only/log_alpha_loss"] = log_alpha_loss

        if "paired" in batch:
            paired_input = batch["paired"]

            output_dict, masks, action_losses, log_alpha_loss, *_ = self.train_behavior(
                paired_input, phase, "paired"
            )

            for k, loss in action_losses.items():
                losses[f"paired/{k}"] = loss

            aux_pred_loss = self._compute_aux_pred_loss(
                aux_values=output_dict["aux_pred"],
                target_aux_values=masks["states_before_bos_mask"].float(),
                mask=masks["combined_state_mask"],
            )
            losses["paired/aux_pred_loss"] = aux_pred_loss

            info["paired/log_alpha_loss"] = log_alpha_loss

            aux_pred_acc, aux_report = self._compute_aux_pred_stats(
                aux_values=output_dict["aux_pred"],
                target_aux_values=masks["states_before_bos_mask"].float(),
                mask=masks["combined_state_mask"],
                phase=phase,
            )

            # log metrics for binary classification
            self.log_dict(
                {f"{phase}/binary_pred_acc": aux_pred_acc}, on_step=True, on_epoch=True
            )
            self.log_dict(aux_report, on_step=True, on_epoch=True)

            if output_dict["lang_token_logits"] is not None and self.hparams.get(
                "train_paired_lang", False
            ):
                lang_losses = self.train_lang(lang_input)
                for k, loss in lang_losses.items():
                    losses[f"paired_{k}"] = loss

        # add prefix to losses and multiply by weight
        losses_with_prefix = {
            f"{phase}/{k}": v
            * self.hparams[k.split("/")[0]][f"{k.split('/')[1]}_weight"]
            for k, v in losses.items()
        }

        self.log_dict(
            losses_with_prefix,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(info, on_step=True, on_epoch=True)

        # sum losses
        total_loss = sum([v for k, v in losses_with_prefix.items()])

        if self.training:
            opt.zero_grad()
            self.manual_backward(total_loss)
            opt.step()
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, phase="val")
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        log_alpha_opt = torch.optim.AdamW(
            [self.log_alpha],
            lr=self.hparams.alpha_lr,
            weight_decay=self.hparams.weight_decay,
        )

        return opt, log_alpha_opt

    def tokenize(self, text: str = None):
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def get_prompt(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        returns_to_go: torch.Tensor = None,
        **kwargs,
    ):
        """
        Implements greedy decoding.
        Take weighted average between lang_only logits and paired lang logits or support filtering.
        """
        start = self.tokenize(LANG_BOS_TOKEN)

        # start with the BOS token
        curr_token = self.tokenizer.vocab[LANG_BOS_TOKEN]

        kwargs["lang_token_ids"] = torch.cat([kwargs["lang_token_ids"], start], dim=-1)

        num_tokens = 1

        while True:
            kwargs = self.update_masks(kwargs, pad_size=1)
            kwargs["tokens"][:, -1] = curr_token
            kwargs["token_type_ids"][:, -1] = 1
            kwargs["lang_token_mask"][:, -1] = 1

            if curr_token == self.tokenizer.vocab[LANG_EOS_TOKEN] or num_tokens >= 30:
                break

            # language-only conditioning
            only_lang = copy.deepcopy(kwargs)

            # mask out all previous action predictions
            only_lang["state_mask"] *= False
            only_lang["action_mask"] *= False
            only_lang["combined_state_mask"] *= False
            only_lang["combined_action_mask"] *= False
            only_lang["tokens"] = only_lang["lang_token_ids"]
            only_lang["token_type_ids"] = torch.ones_like(only_lang["tokens"])
            only_lang["lang_token_mask"] = torch.ones_like(only_lang["tokens"])

            model_out, only_lang = self.forward_state_action_lang(**only_lang)
            output_dict = self.get_predictions(model_out, only_lang)
            lang_only_next_token_logits = output_dict["lang_token_logits"][:, -1:]

            # paired lang-behavior conditioning
            model_out, paired_masks = self.forward_state_action_lang(
                states=states,
                actions=actions,
                timesteps=timesteps,
                returns_to_go=returns_to_go,
                **kwargs,
            )

            output_dict = self.get_predictions(model_out, paired_masks)
            paired_next_token_logits = output_dict["lang_token_logits"][:, -1:]

            # support filtering, mask out tokens from lang_only that are less than some threshold
            if "support_threshold" in kwargs:
                support_mask = paired_next_token_logits < kwargs["support_threshold"]
                lang_only_next_token_logits[support_mask] = -torch.inf

            lang_only_prob = F.softmax(lang_only_next_token_logits, dim=-1)
            paired_prob = F.softmax(paired_next_token_logits, dim=-1)
            weighted_prob = (
                kwargs["fixed_lang_only_prior_weight"] * lang_only_prob
                + kwargs["fixed_paired_lang_prior_weight"] * paired_prob
            )

            # merge two distributions by weighting

            # weighted_logits = (
            #     kwargs["fixed_lang_only_prior_weight"] * lang_only_next_token_logits
            #     + kwargs["fixed_paired_lang_prior_weight"] * paired_next_token_logits
            # )
            if (
                kwargs["debug"] and curr_token != self.tokenizer.vocab[LANG_EOS_TOKEN]
            ):  # DEBUG
                k = 5
                curr = kwargs["lang_token_ids"]

                print("=" * 50)
                print(f"curr sentence: {self.tokenizer.decode(curr[0])}")
                print("lang-only next token: ")
                top_k = torch.topk(lang_only_next_token_logits, k=k)
                next_token_candidates = (
                    self.tokenizer.decode(top_k.indices[0][0], skip_special_tokens=True)
                    .strip()
                    .split(" ")
                )
                for i, candidate in enumerate(next_token_candidates):
                    print(f"\t{candidate}, {top_k.values[0][0][i].item()}")

                print("paired next token: ")
                top_k = torch.topk(paired_next_token_logits, k=k)
                next_token_candidates = (
                    self.tokenizer.decode(top_k.indices[0][0], skip_special_tokens=True)
                    .strip()
                    .split(" ")
                )
                for i, candidate in enumerate(next_token_candidates):
                    print(f"\t{candidate}, {top_k.values[0][0][i].item()}")

                print("weighted next token: ")
                top_k = torch.topk(weighted_prob, k=k)
                next_token_candidates = (
                    self.tokenizer.decode(top_k.indices[0][0], skip_special_tokens=True)
                    .strip()
                    .split(" ")
                )
                for i, candidate in enumerate(next_token_candidates):
                    print(f"\t{candidate}, {top_k.values[0][0][i].item()}")
                import ipdb

                ipdb.set_trace()

            next_token = weighted_prob.argmax(-1)
            curr_token = next_token.item()

            # add next token
            kwargs["lang_token_ids"] = torch.cat(
                [kwargs["lang_token_ids"], next_token], dim=-1
            )

            num_tokens += 1

        # decode
        prompt_str = self.tokenizer.decode(
            token_ids=kwargs["lang_token_ids"][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        return {
            "lang_skill": prompt_str,
            "masks": kwargs,
        }

    def generate(self, **kwargs):
        # use previous tokens as prompting
        context = kwargs["lang_token_ids"]

        bos_token = self.tokenizer([LANG_BOS_TOKEN], return_tensors="pt")[
            "input_ids"
        ].to(self.device)

        if context.shape[-1] == 0:
            context = bos_token
        else:
            context = torch.cat([context, bos_token], dim=-1)

        # lang_token_ids = self.model.generate(context, **self.generation_kwargs)
        lang_token_ids = self.model.generate(
            context,
            **{
                "bos_token_id": 50261,
                "eos_token_id": 50262,
                "max_length": 100,
                "num_return_sequences": 1,
            },
        )

        kwargs = self.update_kwargs(lang_token_ids, kwargs)
        prompt_str = self.tokenizer.decode(lang_token_ids[0], skip_special_tokens=True)
        return prompt_str, kwargs

    def update_masks(self, masks, pad_size=1):
        mask_keys = [
            "combined_state_mask",
            "combined_action_mask",
            "combined_rtg_mask",
            "lang_token_mask",
            "token_type_ids",
            "tokens",
        ]

        zeros = torch.zeros((1, pad_size)).to(self.device)

        for mask_k in mask_keys:
            masks[mask_k] = torch.cat([masks[mask_k], zeros], dim=-1)

        return masks

    def update_masks_with_lang_tokens(self, masks, lang_token_ids):
        masks = self.update_masks(masks, pad_size=lang_token_ids.shape[-1])
        masks["tokens"][:, -lang_token_ids.shape[-1] :] = lang_token_ids
        masks["token_type_ids"][:, -lang_token_ids.shape[-1] :] = 1
        masks["lang_token_ids"] = torch.cat(
            [masks["lang_token_ids"], lang_token_ids], dim=-1
        )
        masks["lang_token_mask"][:, -lang_token_ids.shape[-1] :] = 1
        return masks

    def get_action(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        returns_to_go: torch.Tensor = None,
        curr_instr: str = "",
        **kwargs,
    ):
        """
        Run inference on model to decode a_t given s_0 ... s_t, a_0 ... a_t-1 and lang annotation.

        :param Tensor states:
            Tensor of size [T, state_dim]
        :param Tensor actions:
            Tensor of size [T, action_dim]
        :param Tensor timesteps:
            Tensor of size [1, T]
        :param Tensor returns_to_go:
            Tensor of size [T, ]
        :param Tensor lang_token_ids:
            Optional tensor of size [1, N]
        :param Tensor token_type_ids:
            tensor of size [1, 2*T+N], stores the token types of full sequence
        """

        # add batch dimension during inference time
        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)

        # run three separate forward passes and
        # take a weighted average between logits

        action_output = AttrDict()
        masks, extra_configs = self.get_masks(kwargs)

        # first given the state, predict the binary token
        # the binary token determines whether we predict
        # language or actions in the next forward pass
        # add 1 for state and 0 for action
        masks["state_mask"] = torch.cat(
            [masks["state_mask"], torch.ones((1, 1)).to(self.device)], dim=-1
        )
        masks["action_mask"] = torch.cat(
            [masks["action_mask"], torch.zeros((1, 1)).to(self.device)], dim=-1
        )

        masks = self.update_masks(masks, pad_size=2)
        masks["combined_state_mask"][:, -1] = 1
        masks["combined_rtg_mask"][:, -2] = 1

        # ============= BINARY TOKEN PREDICTION =============

        # first forward predicts the binary token
        model_out, masks = self.forward_state_action_lang(
            states=states,
            actions=actions,
            timesteps=timesteps,
            returns_to_go=returns_to_go,
            **masks,  # masks contains the tokens
        )

        output_dict = self.get_predictions(model_out, masks)
        aux_pred = output_dict["aux_pred"]
        aux_pred = (torch.sigmoid(aux_pred) > 0.5).int()[-1]

        time_to_predict_lang = (aux_pred == 1).item()

        action_output.aux_pred = aux_pred
        action_output.time_to_predict_lang = time_to_predict_lang

        if time_to_predict_lang and kwargs["predict_lang"]:
            if curr_instr is not None:
                # tokenize instr
                lang_token_ids = self.tokenize(curr_instr)

                # update masks
                masks = self.update_masks_with_lang_tokens(masks, lang_token_ids)
            else:
                # predict next language instruction first
                # and then predict action after
                prompt_output = self.get_prompt(
                    states, actions, timesteps, **masks, **extra_configs
                )
                # logger.info(f"skill: {prompt_output['lang_skill']}")

                # split into individual skills
                masks = prompt_output["masks"]
                lang_token_ids = masks["lang_token_ids"]
                action_output.full_instruction = prompt_output["lang_skill"]

            # get the current skill
            lang_skills = torch.tensor_split(
                lang_token_ids,
                torch.where(lang_token_ids == 50261)[0].detach().cpu(),
            )
            curr_instr = self.tokenizer.decode(
                lang_skills[-1][0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            action_output.curr_instr = curr_instr
            logger.info(f"curr_instr: {curr_instr}")

        # predict next action
        masks = self.update_masks(masks, pad_size=1)
        masks["action_mask"][0, -1] = 1
        masks["combined_action_mask"][0, -1] = 1

        # ============== PAIRED ACTION PREDICTION ==============
        model_out, masks = self.forward_state_action_lang(
            states=states,
            actions=actions,
            timesteps=timesteps,
            returns_to_go=returns_to_go,
            **masks,
        )
        output_dict = self.get_predictions(
            model_out, masks, use_means=kwargs["use_means"]
        )

        # 1 x T x action_dim
        if self.hparams.stochastic and self.hparams.continuous_actions:
            paired_action_pred = output_dict.action_preds
            paired_action_dist = (
                output_dict.action_dist
            )  # this will return some normal dist for stochastic policies

        if not self.hparams.continuous_actions:  # for alfred
            # paired_action_pred = output_dict.action_preds
            # paired_action_pred_, paired_target_obj_pred_ = torch.split(
            #     paired_action_pred, [self.num_actions, self.num_objects], dim=-1
            # )
            paired_action_dist, paired_target_obj_dist = (
                output_dict.action_dist,
                output_dict.target_obj_dist,
            )

        # ============== BEHAVIOR_ONLY ACTION PREDICTION ==============
        # behavior_only prediction, ignore previous language
        masks_copy = copy.deepcopy(masks)
        masks_copy["lang_token_mask"] *= False  # mask out prior language predictions

        model_out, masks_copy = self.forward_state_action_lang(
            states=states,
            actions=actions,
            timesteps=timesteps,
            returns_to_go=returns_to_go,
            **masks_copy,
        )
        output_dict = self.get_predictions(model_out, masks_copy)

        if self.hparams.stochastic and not self.hparams.continuous_actions:
            behavior_only_action_pred = output_dict.action_preds
            behavior_only_action_dist = output_dict.action_dist

        if not self.hparams.continuous_actions:  # for alfred
            beh_only_action_dist, beh_only_target_obj_dist = (
                output_dict.action_dist,
                output_dict.target_obj_dist,
            )

        if not self.hparams.continuous_actions:
            # get action
            weighted_action_probs = (
                extra_configs["fixed_paired_beh_prior_weight"]
                * paired_action_dist.probs
                + extra_configs["fixed_beh_only_prior_weight"]
                * beh_only_action_dist.probs
            )
            action_dist = torch.distributions.Categorical(weighted_action_probs)
            if kwargs["use_means"]:
                action_pred = action_dist.sample()
            else:
                action_pred = action_dist.probs.argmax(-1)

            # get target obj
            weighted_target_probs = (
                extra_configs["fixed_paired_beh_prior_weight"]
                * paired_target_obj_dist.probs
                + extra_configs["fixed_beh_only_prior_weight"]
                * beh_only_target_obj_dist.probs
            )
            target_obj_dist = torch.distributions.Categorical(weighted_target_probs)
            if kwargs["use_means"]:
                target_obj_pred = target_obj_dist.sample()  # T
            else:
                target_obj_pred = target_obj_dist.probs.argmax(-1)

            action_pred = torch.cat(
                [action_pred.unsqueeze(0), target_obj_pred.unsqueeze(0)], dim=0
            ).T

            action_output.target_obj_dist = target_obj_dist
        else:
            # if it is continuous
            if self.hparams.stochastic:
                # merge the two normal distributions
                # might need to anneal the weighting over time
                action_dist = merge_normal_dist(
                    paired_action_dist,
                    behavior_only_action_dist,
                    n1=extra_configs["fixed_paired_beh_prior_weight"],
                    n2=extra_configs["fixed_beh_only_prior_weight"],
                )

                if kwargs["use_mean"]:
                    action_pred = action_dist.mean
                else:
                    action_pred = action_dist.sample()
            else:
                # if it isn't stochastic, not sure what to do???
                import ipdb

                ipdb.set_trace()

        # should be (action_dim,)
        if len(action_pred.shape) > 1:
            action_output.action_pred = action_pred[-1]
        else:
            action_output.action_pred = action_pred

        action_output.action_dist = action_dist
        action_output.masks = masks
        return action_output
