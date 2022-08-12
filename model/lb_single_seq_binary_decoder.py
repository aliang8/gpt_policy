import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    GPT2Config,
    EncoderDecoderConfig,
)
from utils.pytorch_utils import ten2ar
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertOnlyNSPHead

from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    accuracy_score,
)
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


@MODEL_REGISTRY
class Model(pl.LightningModule):
    """
    Implementation of language behavior multi-modal transformer decoder model.

    Model is an autoregressive decoder model. Trained from scratch, we don't initialize it with
    any pretrained LM weights.
    """

    def __init__(self, model_conf, **kwargs):
        super().__init__()

        # saves parameters into hparams attribute
        self.save_hyperparameters(model_conf)
        self.decoder_config = GPT2Config.from_pretrained(self.hparams.decoder_model_cls)
        self.embed_dim = self.hparams.hidden_dim

        # load from pretrained GPT2
        trajectory_gpt2 = TrajectoryGPT2.from_pretrained(self.hparams.decoder_model_cls)
        self.model = trajectory_gpt2
        self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)

        # because we added special tokens
        self.model.transformer.wte = self.model.resize_token_embeddings(
            len(self.tokenizer)
        )

        self.state_dim = self.hparams.state_dim
        self.action_dim = self.hparams.action_dim
        self.output_gaussian = (
            "output_gaussian" in self.hparams and self.hparams.output_gaussian
        )

        # embedding heads each modality before inputting to transformer
        self.embed_state = nn.Linear(self.state_dim, self.embed_dim)
        self.embed_action = nn.Linear(self.action_dim, self.embed_dim)
        self.embed_ln = nn.LayerNorm(self.embed_dim)
        self.embed_binary = nn.Embedding(2, self.embed_dim)

        self.model.embed_ln = self.embed_ln
        self.embed_timestep = nn.Embedding(self.hparams.max_ep_len, self.embed_dim)

        # output head for predicting action
        action_tanh = False
        if "discretize_actions" in self.hparams and self.hparams.discretize_actions:
            self.predict_action = nn.Linear(self.embed_dim, self.hparams.num_bins)
        else:
            self.predict_action = nn.Sequential(
                *(
                    [
                        nn.Linear(
                            self.embed_dim,
                            self.action_dim * 2
                            if self.output_gaussian
                            else self.action_dim,
                        )
                    ]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )

        # loss functions
        if self.output_gaussian:
            self.action_loss_fn = torch.nn.GaussianNLLLoss(reduction="none")
        else:
            self.action_loss_fn = torch.nn.MSELoss(reduction="none")

        self.lang_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # do we predict language or action next?
        self.binary_predictor = nn.Linear(self.embed_dim, 1)

        # self.binary_prediction_loss_fn = torch.nn.BCEWithLogitsLoss(
        #     reduction="none", pos_weight=torch.tensor(500)
        # )
        if self.hparams.get("binary_pos_weight", None):
            self.binary_prediction_loss_fn = torch.nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=torch.tensor(self.hparams.binary_pos_weight),
            )
        else:
            self.binary_prediction_loss_fn = torch.nn.BCEWithLogitsLoss(
                reduction="none"
            )

        # self.binary_prediction_loss_fn = torch.nn.BCELoss(reduction="none")

    def forward_state_action_lang(
        self,
        tokens: torch.Tensor = None,
        states: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        actions: torch.Tensor = None,
        **kwargs,
    ):
        # a bunch of useful masks
        masks = [
            kwargs.get("lang_token_mask", None),
            kwargs.get("state_mask", None),
            kwargs.get("action_mask", None),
            kwargs.get("combined_state_mask", None),
            kwargs.get("combined_action_mask", None),
        ]

        for i, mask in enumerate(masks):
            if mask is not None:
                masks[i] = mask.bool()

        (
            lang_token_mask,
            state_mask,
            action_mask,
            combined_state_mask,
            combined_action_mask,
        ) = masks

        if states is not None:
            binary_token_mask = torch.roll(combined_state_mask, 1, dims=1)
            binary_token_mask[:, 0] = 0
        else:
            binary_token_mask = None

        # create tensor to store embeddings as a sequence
        state_action_lang = torch.zeros(
            (*tokens.shape[:2], self.embed_dim), device=self.device
        )

        attention_mask = torch.zeros_like(tokens, device=self.device).bool()

        # filter states and actions from pad
        if states is not None:
            if len(states.shape) == 2:
                states = states.reshape(1, -1, self.state_dim)
                actions = actions.reshape(1, -1, self.action_dim)

            states = states[state_mask]
            actions = actions[action_mask]
            timesteps = timesteps[state_mask]
            binary_tokens = tokens[binary_token_mask]

            state_embeddings = self.embed_state(states)
            action_embeddings = self.embed_action(actions)
            timestep_embeddings = self.embed_timestep(timesteps.int())
            binary_token_embeddings = self.embed_binary(binary_tokens.int())

            state_embeddings = state_embeddings + timestep_embeddings
            action_embeddings = (
                action_embeddings + timestep_embeddings[: action_embeddings.shape[0]]
            )
            binary_token_embeddings = binary_token_embeddings + timestep_embeddings
            state_action_lang[combined_state_mask] = state_embeddings
            state_action_lang[combined_action_mask] = action_embeddings
            state_action_lang[binary_token_mask] = binary_token_embeddings

            attention_mask |= (
                combined_state_mask | combined_action_mask | binary_token_mask
            )

        if lang_token_mask.sum() != 0:
            lang_token_ids = tokens[lang_token_mask]
            lang_embeddings = self.model.transformer.wte(lang_token_ids.long())

            # create position ids for language
            seq_length_per_batch = lang_token_mask.sum(-1)
            position_ids = torch.zeros(0)
            for seq_length in seq_length_per_batch:
                position_ids = torch.cat(
                    [position_ids, torch.arange(seq_length)], dim=-1
                )

            # add position id to word token embedding
            position_ids = position_ids.int().to(self.device)
            position_embeds = self.model.transformer.wpe(position_ids)
            lang_embeddings = lang_embeddings + position_embeds
            state_action_lang[lang_token_mask] = lang_embeddings

            attention_mask |= lang_token_mask

        state_action_lang_embs = self.embed_ln(state_action_lang)

        transformer_outputs = self.model(
            inputs_embeds=state_action_lang_embs,
            attention_mask=attention_mask,
            token_type_ids=kwargs.get("token_type_ids", None).long(),
            lang_only_input=False,
        )

        # reshape x so the second dimension corresponds to states (0), actions (1)
        model_out = transformer_outputs["last_hidden_state"]

        # split model output and create output tensors
        if state_mask is not None:  # predict actions
            state_out = model_out[combined_state_mask]
            binary_token_out = model_out[binary_token_mask]
            action_preds_combined = torch.zeros(
                (*combined_action_mask.shape, self.action_dim), device=self.device
            )
        else:
            state_out = None
            aux_pred = None
            binary_token_mask = None

        if lang_token_mask.sum() != 0:  # predict language
            lang_out = model_out[lang_token_mask].reshape(-1, self.embed_dim)
            lang_token_logits_full = torch.zeros(
                (*tokens.shape, len(self.tokenizer)), device=self.device
            )
        else:
            lang_out = None

        if state_out is not None:
            # predict action based on binary token for actions after a <0> token
            zero_binary_mask = (tokens == 0) & binary_token_mask
            action_preds = self.predict_action(model_out[zero_binary_mask])

            action_indices = torch.roll(zero_binary_mask, 1, dims=1)
            action_indices[:, 0] = False
            action_preds_combined[action_indices] = action_preds

            # predict action from EOS token
            if lang_token_mask.sum() != 0:
                eos_token_mask = tokens == 50262
                eos_token_out = model_out[eos_token_mask].reshape(-1, self.embed_dim)
                lang_act_preds = self.predict_action(eos_token_out)

                action_post_eos = torch.roll(eos_token_mask, 1, dims=1)
                action_post_eos[:, 0] = False
                action_preds_combined[action_post_eos] = lang_act_preds

            # ================ BINARY PREDICTION ================
            # predict binary token based on state embedding
            # aux_pred = torch.sigmoid(self.binary_predictor(state_out))
            aux_pred = self.binary_predictor(state_out)

            # make output easy for computing loss
            action_preds_full = torch.zeros(
                (*action_mask.shape, self.action_dim), device=self.device
            )
            action_preds_full[action_mask] = action_preds_combined[combined_action_mask]
        else:
            action_preds_full = None

        if lang_out is not None:
            # ================ LANGUAGE PREDICTION ================
            # use previous lang to predict next lang tokens
            lang_token_logits = self.model.lm_head(lang_out)
            lang_token_logits_full[lang_token_mask] = lang_token_logits

            if binary_token_mask is not None:
                # use binary token to predict first lang
                pos_binary_mask = (tokens == 1) & binary_token_mask
                pos_binary_out = model_out[pos_binary_mask]
                pos_binary_lang_logits = self.model.lm_head(pos_binary_out)
                lang_token_logits_full[pos_binary_mask] = pos_binary_lang_logits
        else:
            lang_token_logits_full = None

        return action_preds_full, lang_token_logits_full, aux_pred

    def _compute_action_pred_loss(
        self,
        action_preds: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        # compute action prediction loss on valid actions
        if self.output_gaussian:
            mean, logvar = torch.chunk(action_preds, 2, dim=-1)
            action_pred_loss = self.action_loss_fn(
                input=mean, target=target_actions, var=logvar.exp()
            )
        else:
            action_pred_loss = self.action_loss_fn(action_preds, target_actions)

        action_pred_loss *= action_mask.unsqueeze(-1)
        action_pred_loss = action_pred_loss.mean()
        return action_pred_loss

    def _compute_lang_pred_loss(
        self,
        lang_token_logits: torch.Tensor,
        target_lang_tokens: torch.Tensor,
        lang_token_mask: torch.Tensor,
    ):
        # predict next language skill given current skill as context
        # mask out loss for padding tokens
        vocab_size = lang_token_logits.size(-1)

        pred_logits = lang_token_logits[lang_token_mask][..., :-1, :].contiguous()
        gt_tokens = target_lang_tokens[lang_token_mask][..., 1:].contiguous()

        # apply mask
        lang_pred_loss = self.lang_loss_fn(
            pred_logits.view(-1, vocab_size),
            gt_tokens.long().view(-1),
        )

        lang_pred_loss = lang_pred_loss.mean()

        return lang_pred_loss

    def _compute_aux_pred_loss(
        self,
        aux_values: torch.Tensor,
        target_aux_values: torch.Tensor,
        mask: torch.Tensor,
    ):
        # aux_value is N x 1 and target_aux_values is B x T
        # need to mask the target first
        # TODO: clean this up
        aux_pred_loss = self.binary_prediction_loss_fn(
            aux_values.squeeze(), target_aux_values[mask].float()
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
        pred_labels = ten2ar(pred_labels)

        true_labels = ten2ar(target_aux_values[mask].int())

        acc = accuracy_score(true_labels, pred_labels)
        report = classification_report(
            true_labels,
            pred_labels,
            output_dict=True,
            zero_division=0,
            target_names=[f"{phase}/0", f"{phase}/1"],
        )
        # returns precision, recall, and support
        # precision: tp / (fp + tp)
        # recall: tp / (fn + tp), we want to maximize recall of positive class

        return acc, report

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
            losses[f"{phase}/lang_only_pred_loss"] = lang_only_pred_loss

        if "behavior" in batch:
            behavior_input = batch["behavior"]

            action_preds, _, aux_pred = self.forward_state_action_lang(**behavior_input)
            action_pred_loss = self._compute_action_pred_loss(
                action_preds=action_preds,
                target_actions=behavior_input["actions"],
                action_mask=behavior_input["action_mask"].bool(),
            )
            losses[f"{phase}/behavior_only_action_pred_loss"] = action_pred_loss

            binary_token_mask = torch.roll(
                behavior_input["combined_state_mask"], 1, dims=1
            ).bool()
            binary_token_mask[:, 0] = 0

            aux_pred_loss = self._compute_aux_pred_loss(
                aux_values=aux_pred,
                target_aux_values=behavior_input["tokens"],
                mask=binary_token_mask,
            )
            losses[f"{phase}/aux_pred_loss"] = aux_pred_loss

        if "paired" in batch:
            paired_input = batch["paired"]
            action_preds, lang_token_logits, aux_pred = self.forward_state_action_lang(
                **paired_input
            )

            action_pred_loss = self._compute_action_pred_loss(
                action_preds=action_preds,
                target_actions=paired_input["actions"],
                action_mask=paired_input["action_mask"].bool(),
            )
            losses[f"{phase}/paired_action_pred_loss"] = action_pred_loss

            binary_token_mask = torch.roll(
                paired_input["combined_state_mask"], 1, dims=1
            ).bool()
            binary_token_mask[:, 0] = 0

            aux_pred_loss = self._compute_aux_pred_loss(
                aux_values=aux_pred,
                target_aux_values=paired_input["tokens"],
                mask=binary_token_mask,
            )
            losses[f"{phase}/paired_aux_pred_loss"] = aux_pred_loss

            aux_pred_acc, aux_report = self._compute_aux_pred_stats(
                aux_values=aux_pred,
                target_aux_values=paired_input["tokens"],
                mask=binary_token_mask,
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
            losses[f"{phase}/paired_lang_pred_loss"] = lang_pred_loss

        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        loss = sum([v for k, v in losses.items()])
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, phase="val")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def postprocess_action(self, action):
        if self.output_gaussian:
            mean, logvar = torch.chunk(action, 2, dim=-1)
            std = torch.exp(0.5 * logvar)
            dist = torch.distributions.Normal(loc=mean, scale=std)
            action = dist.sample()
        return action[0, -1]

    def tokenize(self, text: str = None):
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def get_prompt(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        **kwargs,
    ):
        """
        Implements greedy decoding.
        """
        start = self.tokenize(LANG_BOS_TOKEN)

        # start with the BOS token
        curr_token = self.tokenizer.vocab[LANG_BOS_TOKEN]

        kwargs["lang_token_ids"] = torch.cat([kwargs["lang_token_ids"], start], dim=-1)

        kwargs = self.update_masks(kwargs, next_pred="lang")
        kwargs["tokens"][:, -1] = start

        while curr_token != self.tokenizer.vocab[LANG_EOS_TOKEN]:
            # build masks
            _, lang_token_logits, _ = self.forward_state_action_lang(
                states=states, actions=actions, timesteps=timesteps, **kwargs
            )
            next_token = lang_token_logits[:, -1:].argmax(-1)
            curr_token = next_token.item()

            kwargs = self.update_masks(kwargs, next_pred="lang")
            kwargs["tokens"][:, -1] = next_token

            # add next token
            kwargs["lang_token_ids"] = torch.cat(
                [kwargs["lang_token_ids"], next_token], dim=-1
            )

        # decode
        prompt_str = self.tokenizer.decode(
            token_ids=kwargs["lang_token_ids"][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return prompt_str, kwargs

    def update_masks(self, masks, next_pred="binary"):
        mask_keys = [
            "combined_state_mask",
            "combined_action_mask",
            "lang_token_mask",
            "token_type_ids",
            "tokens",
        ]

        if next_pred == "binary":
            extras = torch.zeros((1, 2)).to(self.device)
            affected = ["combined_state_mask"]
        elif next_pred == "action":
            extras = torch.zeros((1, 1)).to(self.device)
            affected = ["combined_action_mask"]
        elif next_pred == "lang":
            extras = torch.zeros((1, 1)).to(self.device)
            affected = ["lang_token_mask", "token_type_ids"]

        unaffected = list(set(mask_keys) - set(affected))
        for mask_k in unaffected:
            masks[mask_k] = torch.cat([masks[mask_k], extras], dim=-1)

        # add one for state and one for binary token
        if next_pred == "binary":
            extras[:, 0] = 1
        else:
            extras[:, -1] = 1
        for mask_k in affected:
            masks[mask_k] = torch.cat([masks[mask_k], extras], dim=-1)

        return masks

    def crop_input(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        **kwargs,
    ):

        if kwargs["tokens"].shape[-1] < 512:
            return states, actions, timesteps, kwargs

        # index of next state
        start_y = torch.where(kwargs["combined_state_mask"][0] == 1)
        start_y = start_y[0][1]

        # crop all the mask to max size
        import ipdb

        ipdb.set_trace()

        for k, v in kwargs.items():
            if k not in ["state_mask", "action_mask"]:
                kwargs[k] = v[:, start_y:]

        # cut off some states and actions
        kwargs["state_mask"] = kwargs["state_mask"][:, 1:]
        kwargs["action_mask"] = kwargs["action_mask"][:, 1:]
        states = states[:, 1:]
        actions = actions[:, 1:]
        timesteps = timesteps[:, 1:]
        return states, actions, timesteps, kwargs

    def get_action(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        **kwargs,
    ):
        """
        Run inference on model to decode a_t given s_0 ... s_t, a_0 ... a_t-1 and lang annotation.

        :param Tensor states:
            Tensor of size [T, state_dim]
        :param Tensor actions:
            Tensor of size [T, action_dim]
        :param Tensor timesteps:
            Tensor of size [T, ]
        :param Tensor lang_token_ids:
            Optional tensor of size [1, N]
        :param Tensor token_type_ids:
            tensor of size [1, 2*T+N], stores the token types of full sequence
        """
        if len(states.shape) == 2:
            states = states.reshape(1, -1, self.state_dim)
            actions = actions.reshape(1, -1, self.action_dim)

        # add 1 for state and 0 for action
        kwargs["state_mask"] = torch.cat(
            [kwargs["state_mask"], torch.ones((1, 1)).to(self.device)], dim=-1
        )
        kwargs["action_mask"] = torch.cat(
            [kwargs["action_mask"], torch.zeros((1, 1)).to(self.device)], dim=-1
        )

        # # crop context to 512
        # states, actions, timesteps, kwargs = self.crop_input(
        #     states, actions, timesteps, **kwargs
        # )

        kwargs = self.update_masks(kwargs, next_pred="binary")

        # first forward predicts the binary token
        _, _, aux_pred = self.forward_state_action_lang(
            states=states,
            actions=actions,
            timesteps=timesteps,
            **kwargs,  # masks contains the tokens
        )

        # print(torch.sigmoid(aux_pred)[-1].item())
        thres = 0.5
        aux_pred = (torch.sigmoid(aux_pred) > thres).int()

        kwargs["tokens"][:, -1] = aux_pred[-1]

        if aux_pred[-1].item() > thres:
            print("Predicting new language...")

            prompt_str, kwargs = self.get_prompt(
                states=states,
                actions=actions,
                timesteps=timesteps,
                **kwargs,
            )
            print(f"new prompt: {prompt_str}")

        kwargs = self.update_masks(kwargs, next_pred="action")
        kwargs["action_mask"][:, -1] = 1

        # predict next action
        action_preds, _, _ = self.forward_state_action_lang(
            states=states,
            actions=actions,
            timesteps=timesteps,
            **kwargs,
        )

        return self.postprocess_action(action_preds), aux_pred, kwargs
