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
from utils.lang_utils import get_tokenizer, LANG_BOS_TOKEN, LANG_EOS_TOKEN


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

        # self.tokenizer = get_tokenizer(self.hparams.decoder_model_cls)
        self.tokenizer = get_tokenizer("gpt2")

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

        # predicts a float value between 0 and 1
        if self.hparams.get("pred_progress", False):
            self.predict_progress = nn.Sequential(
                nn.Linear(self.embed_dim, 1), nn.Sigmoid()
            )
        self.progress_pred_loss_fn = torch.nn.MSELoss()

    def forward_lang(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B = input_ids.shape[0]
        lang_embs = self.model.wte(input_ids.long())
        lang_mask = attention_mask.clone()
        lang_token_type_id = attention_mask.clone()  # language has tt_id 1

        model_out = self.model(
            inputs_embeds=lang_embs,
            attention_mask=lang_mask.bool(),
            token_type_ids=lang_token_type_id.int(),
        )

        lang_out = model_out["last_hidden_state"]
        lang_token_logits = self.lm_head(lang_out)
        return lang_token_logits

    def forward_state_action_lang(
        self,
        tokens: torch.Tensor = None,
        states: torch.Tensor = None,
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

        # create tensor to store embeddings as a sequence
        state_action_lang = torch.zeros(
            (*tokens.shape[:2], self.embed_dim), device=self.device
        )

        # filter states and actions from pad
        if states is not None:
            states = states[state_mask]
            actions = actions[action_mask]

            state_embeddings = self.embed_state(states)
            action_embeddings = self.embed_action(actions)
            state_action_lang[combined_state_mask] = state_embeddings
            state_action_lang[combined_action_mask] = action_embeddings

        if lang_token_mask.sum() != 0:
            lang_token_ids = tokens[lang_token_mask]
            lang_embeddings = self.model.wte(lang_token_ids.long())
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

        aux_pred = None

        if states is not None:
            state_out = model_out[combined_state_mask]

            # ================ ACTION PREDICTION ================
            # use previous states to predict actions
            action_preds = self.predict_action(state_out)

            # put it back into batch form
            action_preds_full = torch.zeros(
                (*action_mask.shape, action_preds.shape[-1]), device=self.device
            )
            action_preds_full[action_mask] = action_preds

            # ================ DONE PREDICTION ================
            # use previous states + actions to predict skill progress
            if self.hparams.get("pred_progress", False):
                aux_pred = self.predict_progress(state_out)
        else:
            action_preds_full = None

        if lang_token_mask.sum() != 0:
            # ================ LANGUAGE PREDICTION ================
            # use previous lang to predict next lang tokens
            lang_out = model_out[lang_token_mask].reshape(-1, self.embed_dim)
            lang_token_logits = self.lm_head(lang_out)

            # put it back into batch form
            lang_token_logits_full = torch.zeros(
                (*tokens.shape, lang_token_logits.shape[-1]), device=self.device
            )
            lang_token_logits_full[lang_token_mask] = lang_token_logits
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
        aux_pred_loss = self.progress_pred_loss_fn(
            aux_values.squeeze(), target_aux_values[mask]
        )
        aux_pred_loss = aux_pred_loss.mean()
        return aux_pred_loss

    def training_step(self, batch, batch_idx):
        losses = {}
        if "language" in batch:
            lang_input = batch["language"]
            lang_token_logits = self.forward_lang(**lang_input)

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
            losses["paired_action_pred_loss"] = action_pred_loss

            if self.hparams.get("train_paired_lang", False):
                lang_pred_loss = self._compute_lang_pred_loss(
                    lang_token_logits=lang_token_logits,
                    target_lang_tokens=paired_input["tokens"],
                    lang_token_mask=paired_input["lang_token_mask"].bool(),
                )
                losses["paired_lang_pred_loss"] = lang_pred_loss

            if self.hparams.get("pred_progress", False):
                progress_pred_loss = self._compute_aux_pred_loss(
                    aux_values=aux_pred,
                    target_aux_values=paired_input["dones"],
                    mask=paired_input["action_mask"].bool(),
                )
                losses["progress_pred_loss"] = progress_pred_loss

        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        loss = sum([v for k, v in losses.items()])
        return loss

    def validation_step(self, batch, batch_idx):
        # loss = self.training_step(batch, batch_idx)
        # return loss
        return 0

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

    def build_masks(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        lang_token_ids: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        **kwargs,
    ):
        B = token_type_ids.shape[0]  # usually should be 1
        if states is not None:
            state_mask = torch.ones((B, states.shape[1])).to(self.device)
            action_mask = torch.ones((B, actions.shape[1])).to(self.device)
        else:
            state_mask = None
            action_mask = None

        tokens = torch.zeros_like(token_type_ids, dtype=torch.long)
        lang_token_mask = token_type_ids.clone()

        if lang_token_ids is not None:
            # indexing loses a dimension?
            tokens[lang_token_mask.bool()] = lang_token_ids.squeeze()

        not_lang = ~lang_token_mask.bool()

        combined_state_mask = torch.zeros_like(token_type_ids)
        indices = not_lang.nonzero()[::2]
        combined_state_mask[indices[:, 0], indices[:, 1]] = 1

        combined_action_mask = torch.zeros_like(token_type_ids)
        indices = not_lang.nonzero()[1::2]
        combined_action_mask[indices[:, 0], indices[:, 1]] = 1

        masks = {
            "state_mask": state_mask,
            "action_mask": action_mask,
            "lang_token_mask": lang_token_mask,
            "tokens": tokens,
            "combined_state_mask": combined_state_mask,
            "combined_action_mask": combined_action_mask,
            "token_type_ids": token_type_ids,
        }

        return masks

    def get_prompt(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        lang_token_ids: torch.Tensor = None,
        **kwargs,
    ):
        """
        Implements greedy decoding.
        """

        tokens = self.tokenize(LANG_BOS_TOKEN)

        # start with the BOS token
        curr_token = self.tokenizer.vocab[LANG_BOS_TOKEN]

        while curr_token != self.tokenizer.vocab[LANG_EOS_TOKEN]:
            token_type_ids = torch.ones((1, tokens.shape[-1])).to(self.device)

            # build masks
            masks = self.build_masks(states, actions, tokens, token_type_ids)
            _, lang_token_logits, _ = self.forward_state_action_lang(
                states=states, actions=actions, **masks
            )
            next_token = lang_token_logits[:, -1:].argmax(-1)
            curr_token = next_token.item()

            # add next token
            tokens = torch.cat([tokens, next_token], dim=-1)

        # one more for the last token
        token_type_ids = torch.ones((1, tokens.shape[-1])).to(self.device)

        # decode
        prompt_str = self.tokenizer.decode(
            token_ids=tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return prompt_str, tokens, token_type_ids

    def get_action(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        lang_token_ids: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        **kwargs,
    ):
        """
        Run inference on model to decode a_t given s_0 ... s_t, a_0 ... a_t-1 and lang annotation.

        :param Tensor states:
            Tensor of size [T, state_dim]
        :param Tensor actions:
            Tensor of size [T, action_dim]
        :param Tensor lang_token_ids:
            Optional tensor of size [1, N]
        :param Tensor token_type_ids:
            tensor of size [1, 2*T+N], stores the token types of full sequence
        """
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        T = states.shape[1]

        # add one for state and one for action
        sa_token_type_ids = torch.zeros((1, 2)).to(self.device)
        if token_type_ids is not None:
            token_type_ids = torch.cat([token_type_ids, sa_token_type_ids], dim=-1)
        else:
            token_type_ids = sa_token_type_ids

        masks = self.build_masks(
            states,
            actions,
            lang_token_ids=lang_token_ids,
            token_type_ids=token_type_ids,
        )

        # predicts actions and progress
        action_preds, _, aux_pred = self.forward_state_action_lang(
            states=states,
            actions=actions,
            **masks,
        )

        return (
            self.postprocess_action(action_preds),
            lang_token_ids,
            token_type_ids,
            aux_pred,
        )
