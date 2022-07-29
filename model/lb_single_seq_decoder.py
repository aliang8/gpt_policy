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


import torch
import more_itertools as mit
import torch.nn as nn
from transformers import AutoTokenizer
import pytorch_lightning as pl
from transformers import GPT2Model
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertOnlyNSPHead
from utils.lang_utils import get_tokenizer


@MODEL_REGISTRY
class LB_SingleSeq_Decoder(pl.LightningModule):
    """
    Implementation of language behavior multi-modal transformer decoder model.

    Model is an autoregressive decoder model. Trained from scratch, we don't initialize it with
    any pretrained LM weights.
    """

    def __init__(self, model_conf, **kwargs):
        super().__init__()

        # saves parameters into hparams attribute
        self.save_hyperparameters(model_conf)
        self.decoder_config = GPT2Config.from_pretrained("gpt2")

        gpt_decoder = GPT2Model.from_pretrained("gpt2")

        self.model = gpt_decoder

        self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)

        # because we added special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.state_dim = self.hparams.state_dim
        self.action_dim = self.hparams.action_dim
        self.output_gaussian = (
            "output_gaussian" in self.hparams and self.hparams.output_gaussian
        )
        self.embed_dim = self.hparams.hidden_dim

        # embedding heads each modality before inputting to transformer
        self.embed_state = nn.Linear(self.state_dim, self.embed_dim)
        self.embed_action = nn.Linear(self.action_dim, self.embed_dim)
        self.embed_ln = nn.LayerNorm(self.embed_dim)

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

        # output head for next token prediction
        self.lm_head = nn.Linear(
            self.decoder_config.n_embd, len(self.tokenizer), bias=False
        )

        # loss functions
        if self.output_gaussian:
            self.action_loss_fn = torch.nn.GaussianNLLLoss(reduction="none")
        else:
            self.action_loss_fn = torch.nn.MSELoss(reduction="none")

        self.lang_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        tokens: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
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

        # filter states and actions
        states = states[state_mask]
        actions = actions[action_mask]

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        lang_token_ids = tokens[lang_token_mask]
        lang_embeddings = self.model.wte(lang_token_ids.long())

        state_action_lang = torch.zeros(
            (*tokens.shape[:2], self.embed_dim), device=self.device
        )
        state_action_lang[combined_state_mask] = state_embeddings
        state_action_lang[combined_action_mask] = action_embeddings
        state_action_lang[lang_token_mask] = lang_embeddings

        state_action_lang_embs = self.embed_ln(state_action_lang)

        transformer_outputs = self.model(
            inputs_embeds=state_action_lang_embs,
            attention_mask=(
                lang_token_mask | combined_state_mask | combined_action_mask
            ).bool(),
            token_type_ids=kwargs.get("token_type_ids", None).long(),
        )

        # reshape x so the second dimension corresponds to states (0), actions (1)
        model_out = transformer_outputs["last_hidden_state"]

        state_out = model_out[combined_state_mask]

        # use previous states to predict next states
        action_preds = self.predict_action(state_out)

        # put it back into batch form
        action_preds_full = torch.zeros(
            (*action_mask.shape, action_preds.shape[-1]), device=self.device
        )
        action_preds_full[action_mask] = action_preds

        # use previous lang to predict next lang tokens
        lang_out = model_out[lang_token_mask].reshape(-1, self.embed_dim)
        lang_token_logits = self.lm_head(lang_out)

        # put it back into batch form
        lang_token_logits_full = torch.zeros(
            (*tokens.shape, lang_token_logits.shape[-1]), device=self.device
        )
        lang_token_logits_full[lang_token_mask] = lang_token_logits
        return action_preds_full, lang_token_logits_full

    def _compute_prediction_loss(
        self,
        action_preds: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor,
        lang_token_logits: torch.Tensor,
        target_lang_tokens: torch.Tensor,
        lang_token_mask: torch.Tensor,
    ):
        # compute action prediction loss on valid actions
        if self.output_gaussian:
            mean, logvar = torch.chunk(action_preds, 2, dim=-1)
            action_pred_loss = self.action_loss_fn(
                input=mean, target=target_actions, var=logvar.exp()
            )
        else:
            action_pred_loss = self.action_loss_fn(action_preds, target_actions)

        action_pred_loss *= action_mask.unsqueeze(-1).bool()
        action_pred_loss = action_pred_loss.mean()

        # predict next language skill given current skill as context
        # mask out loss for padding tokens
        vocab_size = lang_token_logits.size(-1)

        target_lang_tokens[~lang_token_mask[..., 1:].bool()] = 0  # HACK
        lang_pred_loss = self.lang_loss_fn(
            lang_token_logits.view(-1, vocab_size),
            target_lang_tokens.long().view(-1),
        )
        lang_pred_loss *= lang_token_mask[..., :-1].contiguous().view(-1).bool()
        lang_pred_loss = lang_pred_loss.mean()

        # log losses
        losses = {
            "action_pred_loss": action_pred_loss.item(),
            "lang_pred_loss": lang_pred_loss.item(),
        }

        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return action_pred_loss + lang_pred_loss

    def training_step(self, batch, batch_idx):
        action_preds, lang_token_logits = self.forward(**batch)

        loss = self._compute_prediction_loss(
            action_preds=action_preds,
            target_actions=batch["actions"],
            action_mask=batch["action_mask"],
            lang_token_logits=lang_token_logits[..., :-1, :].contiguous(),
            target_lang_tokens=batch["tokens"][..., 1:].contiguous(),
            lang_token_mask=batch["lang_token_mask"],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        action_preds, lang_token_logits = self.forward(**batch)

        loss = self._compute_prediction_loss(
            action_preds=action_preds,
            target_actions=batch["actions"],
            action_mask=batch["action_mask"],
            lang_token_logits=lang_token_logits[..., :-1, :].contiguous(),
            target_lang_tokens=batch["tokens"][..., 1:].contiguous(),
            lang_token_mask=batch["lang_token_mask"],
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def postprocess_action(self, action):
        if self.output_gaussian:
            mean, logvar = torch.chunk(action, 2, dim=-1)
            std = torch.exp(0.5 * logvar)
            dist = torch.distributions.Normal(loc=mean, scale=std)
            action = dist.sample()
        return action[-1]

    def get_action(self, states, actions, **kwargs):
        """
        Run inference on model to decode a_t given s_0 ... s_t and a_0 ... a_t-1.
        """
        # add batch dimension
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        T = states.shape[1]
        state_mask = torch.ones((1, T)).to(self.device)
        action_mask = torch.ones((1, T)).to(self.device)

        action_preds = self.forward_state_action(
            states, actions, state_mask=state_mask, action_mask=action_mask
        )

        # get the last action predicted
        return self.postprocess_action(action_preds)

    def get_language_conditioned_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lang_token_ids: torch.Tensor,
        lang_attn_masks: torch.Tensor,
        **kwargs,
    ):
        """
        Run inference on model to decode a_t given s_0 ... s_t, a_0 ... a_t-1 and lang annotation.
        """

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        T = states.shape[1]
        state_mask = torch.ones((1, T)).to(self.device)
        action_mask = torch.ones((1, T)).to(self.device)

        num_tokens = lang_token_ids.shape[-1]
        combined_lang_mask = torch.zeros((1, num_tokens + 2 * T)).to(self.device)
        combined_lang_mask[:, :num_tokens] = 1
        combined_lang_mask = combined_lang_mask.bool()
        combined_state_action_mask = ~combined_lang_mask

        # attend to all tokens
        combined_state_action_lang_mask = torch.ones_like(combined_lang_mask)

        masks = dict(
            state_mask=state_mask,
            action_mask=action_mask,
            combined_lang_mask=combined_lang_mask,
            combined_state_action_mask=combined_state_action_mask,
            combined_state_action_lang_mask=combined_state_action_lang_mask,
        )

        lang_token_ids = lang_token_ids.unsqueeze(1).to(self.device)
        lang_attn_masks = lang_attn_masks.unsqueeze(1).to(self.device)

        action_preds, _ = self.forward_state_action_lang(
            states,
            actions,
            lang_token_ids=lang_token_ids,
            lang_attn_masks=lang_attn_masks,
            **masks,
        )
        return self.postprocess_action(action_preds)
