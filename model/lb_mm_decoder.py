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


@MODEL_REGISTRY
class LB_MM_Decoder(pl.LightningModule):
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

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", padding=True, return_tensors="np"
        )

        # add special tokens to distinguish action from text
        special_tokens = {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "additional_special_tokens": [
                "<ACT>",
                "<LBOS>",
                "<LEOS>",
                "<NLBOS>",
                "<NLEOS>",
            ],
        }
        tokenizer.add_special_tokens(special_tokens_dict=special_tokens)

        # because we added special tokens
        self.model.resize_token_embeddings(len(tokenizer))

        self.state_dim = self.hparams.state_dim
        self.action_dim = self.hparams.action_dim
        self.embed_dim = self.hparams.hidden_dim

        # embedding heads each modality before inputting to transformer
        self.embed_state = nn.Linear(self.state_dim, self.embed_dim)
        self.embed_action = nn.Linear(self.action_dim, self.embed_dim)
        self.embed_ln = nn.LayerNorm(self.embed_dim)

        # output head for predicting action
        action_tanh = True
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(self.embed_dim, self.action_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )

        # output head for next token prediction
        self.lm_head = nn.Linear(self.decoder_config.n_embd, len(tokenizer), bias=False)

        # loss functions
        self.action_loss_fn = torch.nn.MSELoss(reduction="none")
        self.lang_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def forward_state_action_lang(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lang_token_ids: torch.Tensor,
        lang_attn_masks: torch.Tensor,
        **kwargs,
    ):
        B, T, _ = actions.shape

        # a bunch of useful masks
        masks = (
            kwargs.get("combined_lang_mask", None),
            kwargs.get("combined_state_action_lang_mask", None),
            kwargs.get("combined_state_action_mask", None),
            kwargs.get("state_mask", None),
            kwargs.get("action_mask", None),
        )
        for mask in masks:
            if mask:
                mask = mask.bool()

        (
            combined_lang_mask,
            combined_state_action_lang_mask,
            combined_state_action_mask,
            state_mask,
            action_mask,
        ) = masks

        if lang_attn_masks:
            lang_mask = lang_attn_masks.bool()

        # mask out padded state, actions
        state_action_mask = (
            torch.stack([state_mask, action_mask], dim=1)
            .permute(0, 2, 1)
            .reshape(B, 2 * T)
        )

        # embed each modality with a separate head
        # B x T x HD
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        # TODO: should i use pretrained language embeddings?
        lang_embeddings = self.model.wte(lang_token_ids.long())

        # rearrange state action into a staggered sequence (s_0, a_0, s_1, a_1, etc)
        state_action = torch.stack((state_embeddings, action_embeddings), dim=1)
        state_action = state_action.permute(0, 2, 1, 3).reshape(B, 2 * T, -1)

        # create a new tensor to append the language description after state/action sequence
        # (s_0, a_0, s_1, a_1, ..., tok_1_A, tok_2_A, ...)
        state_action_lang = torch.zeros(
            (*combined_state_action_lang_mask.shape[:2], self.embed_dim),
            device=states.device,
        )

        # put the states, actions in correct spot
        state_action_lang[combined_state_action_mask] = state_action[
            state_action_mask
        ].reshape(-1, 768)

        # put lang embs in the correct spot
        state_action_lang[combined_lang_mask] = lang_embeddings[lang_mask].reshape(
            -1, 768
        )
        state_action_lang_embs = self.embed_ln(state_action_lang)

        # distinguish between two different segment types (state/action and language)
        state_act_lang_token_ids = combined_lang_mask.long()

        transformer_outputs = self.model(
            inputs_embeds=state_action_lang_embs,
            attention_mask=combined_state_action_lang_mask,
            token_type_ids=state_act_lang_token_ids,
        )

        # reshape x so the second dimension corresponds to states (0), actions (1)
        # B x total_tokens x HD
        model_out = transformer_outputs["last_hidden_state"]

        lang_out = model_out[combined_lang_mask].reshape(-1, self.embed_dim)

        state_action_out = (
            model_out[combined_state_action_mask]
            .reshape(-1, 2, self.embed_dim)
            .permute(1, 0, 2)
        )

        # use previous states to predict next states
        action_preds = self.predict_action(state_action_out[0])

        # use previous lang to predict next lang tokens
        lang_token_logits = self.lm_head(lang_out)
        return action_preds, lang_token_logits

    def forward_state_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        **kwargs,
    ):
        B, T, _ = actions.shape

        state_mask = kwargs["state_mask"].bool()
        action_mask = kwargs["action_mask"].bool()

        # mask out padded state, actions
        state_action_mask = (
            torch.stack([state_mask, action_mask], dim=1)
            .permute(0, 2, 1)
            .reshape(B, 2 * T)
        )

        # embed each modality with a separate head
        # B x T x HD
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        # rearrange state action into a staggered sequence (s_0, a_0, s_1, a_1, etc)
        state_action = torch.stack((state_embeddings, action_embeddings), dim=1)
        state_action = state_action.permute(0, 2, 1, 3).reshape(B, 2 * T, -1)

        state_action_embs = self.embed_ln(state_action)
        state_action_token_ids = torch.zeros_like(state_action_mask).long()

        transformer_outputs = self.model(
            inputs_embeds=state_action_embs,
            attention_mask=state_action_mask,
            token_type_ids=state_action_token_ids,
        )

        # reshape x so the second dimension corresponds to states (0), actions (1)
        model_out = transformer_outputs["last_hidden_state"]

        state_action_out = (
            model_out[state_action_mask].reshape(-1, 2, self.embed_dim).permute(1, 0, 2)
        )

        # use previous states to predict next states
        action_preds = self.predict_action(state_action_out[0])
        return action_preds

    def forward_lang(
        self,
        lang_token_ids: torch.Tensor = None,
        lang_attn_masks: torch.Tensor = None,
    ):
        pass

    def forward(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        lang_token_ids: torch.Tensor = None,
        lang_attn_masks: torch.Tensor = None,
        **kwargs,
    ):
        """
        Input a sequence of tokens. Calls respective function depending on
        what input is available.

        a1      a2      a3      ...         Open   the    microwave
         |       |       |                    |      |         |
        ================================ Transformer ================================
         |   |   |   |   |   |   |        |   |      |
        s1  a1  s2  a2  s3  a3  s4  ... Open the microwave   <EOS>

        :param Tensor states:
            Tensor of sequence of states [batch_size x timesteps x state_dim]
        :param Tensor actions:
            Tensor of sequence of states [batch_size x timesteps x action_dim]
        :param Tensor lang_token_ids (optional):
            Tensor of tokenized skills [batch_size x num_skills x max_token_length]
        :param Tensor lang_attn_masks (optional):
            Tensor of attention masks for language tokens [batch x num_skills x max_token_length]
        :param Dict kwargs:
            Additional data e.g. masks

        :return:
            - action_pred: Tensor of size [num_valid_actions, action_dim]
            - lang_token_logits: Tensor of size [num_valid_lang_tokens, vocab_size]
        """

        # Paired language and behavior
        if states and lang_token_ids:
            return self.forward_state_action_lang(
                states, actions, lang_token_ids, lang_attn_masks, **kwargs
            )

        # Only behavior
        if states and not lang_token_ids:
            return self.forward_state_action(states, actions, **kwargs)

        # Only language
        if not states and lang_token_ids:
            return self.forward_lang(lang_token_ids, lang_attn_masks, **kwargs)

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
        action_pred_loss = self.action_loss_fn(
            action_preds, target_actions[action_mask.bool()]
        )
        action_pred_loss = action_pred_loss.mean()

        # predict next language skill given current skill as context
        # mask out loss for padding tokens
        vocab_size = lang_token_logits.size(-1)
        lang_pred_loss = self.lang_loss_fn(
            lang_token_logits.view(-1, vocab_size),
            target_lang_tokens.long()[lang_token_mask.bool()].view(-1),
        )
        # lang_pred_loss *= lang_token_mask.unsqueeze(-1)
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
            target_lang_tokens=batch["lang_token_ids"].view(-1)[..., 1:].contiguous(),
            lang_token_mask=batch["lang_attn_masks"].view(-1)[..., 1:].contiguous(),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        action_preds, lang_token_logits = self.forward(**batch)
        loss = self._compute_prediction_loss(
            action_preds=action_preds,
            target_actions=batch["actions"],
            action_mask=batch["action_mask"],
            lang_token_logits=lang_token_logits[..., :-1, :].contiguous(),
            target_lang_tokens=batch["lang_token_ids"].view(-1)[..., 1:].contiguous(),
            lang_token_mask=batch["lang_attn_masks"].view(-1)[..., 1:].contiguous(),
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_action(self, states, actions, **kwargs):
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
        return action_preds[-1]
