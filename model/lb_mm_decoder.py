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

        self.embed_state = nn.Linear(self.state_dim, self.embed_dim)
        self.embed_action = nn.Linear(self.action_dim, self.embed_dim)
        self.embed_ln = nn.LayerNorm(self.embed_dim)

        # head for predicting action
        action_tanh = True
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(self.embed_dim, self.action_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )

        self.lm_head = nn.Linear(self.decoder_config.n_embd, len(tokenizer), bias=False)
        self.action_loss_fn = torch.nn.MSELoss(reduction="none")
        self.lang_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # update model configs
        print(self.model.config)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lang_token_ids: torch.Tensor,
        lang_attn_masks: torch.Tensor,
        **kwargs
    ):
        """
        Input a sequence of tokens.

            a1      a2      a3      ...      Open   the    microwave
        |       |       |                     |      |         |
        ================================ Transformer ================================
         |   |   |   |   |   |   |        |   |      |
        s1  a1  s2  a2  s3  a3  s4  ... Open the microwave   <EOS>

        :param Tensor states:
            Tensor of sequence of states [batch_size x timesteps x state_dim]
        :param Tensor actions:
            Tensor of sequence of states [batch_size x timesteps x action_dim]
        :param Tensor lang_token_ids:
            Tensor of tokenized skills [batch_size x num_skills x max_token_length]
        :param Tensor lang_attn_masks:
            Tensor of attention masks for language tokens [batch x num_skills x max_token_length]

        :return:
            - loss: tensor
        """
        # need a mask for timesteps that don't exist
        B, T, _ = actions.shape

        (
            combined_lang_mask,
            combined_state_action_lang_mask,
            combined_state_action_mask,
            state_mask,
            action_mask,
        ) = (
            kwargs["combined_lang_mask"].bool(),
            kwargs["combined_state_action_lang_mask"].bool(),
            kwargs["combined_state_action_mask"].bool(),
            kwargs["state_mask"].bool(),
            kwargs["action_mask"].bool(),
        )
        lang_mask = lang_attn_masks.bool()

        # embed each modality with a separate head
        # B x T x HD
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        state_action = torch.stack((state_embeddings, action_embeddings), dim=1)

        # rearrange so that it looks like (s_0, a_0, s_1, a_1, etc)
        state_action = state_action.permute(0, 2, 1, 3).reshape(B, 2 * T, -1)

        # mask out padded state, actions
        state_action_mask = (
            torch.stack([state_mask, action_mask], dim=1)
            .permute(0, 2, 1)
            .reshape(B, 2 * T)
        )

        # embed language separately
        # TODO: should i use pretrained language embeddings?
        lang_embeddings = self.model.wte(lang_token_ids.long())

        # unique_skills = kwargs["num_unique_skills"].squeeze(-1).int().tolist()
        # max_uniq_skills = max(unique_skills)

        # # mask more for sequences that have less skills than the max
        # for i in range(B):
        #     if unique_skills[i] < max_uniq_skills:
        #         import ipdb

        #         ipdb.set_trace()
        #         diff = int(max_uniq_skills - unique_skills[i])
        #         state_action_mask[i][-diff * lang_token_ids.shape[-1] :] = False

        # create a new tensor to store all state, action and language
        # (s_0, a_0, s_1, a_1, ..., tok_1_A, tok_2_A, ...)
        state_action_lang = torch.zeros(
            (*combined_state_action_lang_mask.shape[:2], self.embed_dim),
            device=states.device,
        )

        # tmp = state_action[seq_mask].reshape(-1, 768)
        # if state_action_lang[state_action_mask].shape[0] != tmp.shape[0]:
        #     import ipdb

        #     ipdb.set_trace()

        state_action_lang[combined_state_action_mask] = state_action[
            state_action_mask
        ].reshape(-1, 768)

        state_action_lang[combined_lang_mask] = lang_embeddings[lang_mask].reshape(
            -1, 768
        )
        state_action_lang_embs = self.embed_ln(state_action_lang)

        # tells model to distinguish between two different segment types
        state_act_lang_token_ids = combined_lang_mask.long()

        transformer_outputs = self.model(
            inputs_embeds=state_action_lang_embs,
            attention_mask=combined_state_action_lang_mask,
            token_type_ids=state_act_lang_token_ids,
        )

        # reshape x so the second dimension corresponds to states (0), actions (1)
        # B x total_tokens x HD
        x = transformer_outputs["last_hidden_state"]

        lang_out = x[combined_lang_mask].reshape(-1, self.embed_dim)

        state_action_out = (
            x[combined_state_action_mask]
            .reshape(-1, 2, self.embed_dim)
            .permute(1, 0, 2)
        )

        # use previous states to predict next states
        action_preds = self.predict_action(state_action_out[0])

        # use previous lang to predict next lang tokens
        lang_token_logits = self.lm_head(lang_out)
        return action_preds, lang_token_logits

    def _compute_prediction_loss(
        self,
        action_preds,
        target_actions,
        action_mask,
        lang_token_logits,
        target_lang_tokens,
        lang_token_mask,
    ):
        # predict next action given state
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

        self.log(
            "action_pred_loss",
            action_pred_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "lang_pred_loss",
            lang_pred_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
