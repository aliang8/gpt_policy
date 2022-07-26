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


@MODEL_REGISTRY
class LB_MM_Transformer(pl.LightningModule):
    """
    Implementation of language behavior multi-modal transformer encoder-decoder model.

    Model is an encoder-decoder style. We use EncoderDecoder from HuggingFace and initialize the encoder
    with pretrained BERT and decoder with pretrained GPT. This is an arbitrary choice, we can
    initialize with any pretrained autoencoder and generative model.
    """

    def __init__(self, model_conf, **kwargs):
        super().__init__()

        # saves parameters into hparams attribute
        self.save_hyperparameters(model_conf)

        self.encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        self.decoder_config = GPT2Config.from_pretrained("gpt2")

        self.encoder_config.position_embedding_type = ""
        bert2gpt2 = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "bert-base-uncased",
            "gpt2",
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
            tie_encoder_decoder=True,
        )

        self.model = bert2gpt2

        self.embed_state = nn.Linear(self.hparams.state_dim, self.hparams.hidden_dim)
        self.embed_action = nn.Linear(self.hparams.action_dim, self.hparams.hidden_dim)
        self.embed_ln = nn.LayerNorm(self.hparams.hidden_dim)

        # head for predicting action
        action_tanh = True
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(self.hparams.hidden_dim, self.hparams.action_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )

        # update model configs
        print(self.model.config)

        # BERT's MLM head
        self.lm_head = BertOnlyMLMHead(self.encoder_config)
        self.lm_loss_fn = torch.nn.CrossEntropyLoss()
        self.nsp_head = BertOnlyNSPHead(self.encoder_config)
        self.nsp_loss_fn = torch.nn.CrossEntropyLoss()

    def _language_only_losses(self, lang_inputs):
        """

        Compute next skill prediction loss. Similar to next sentence prediction loss.
        Given language input A and language input B, predict likelihood that B comes after A.

        Run language input through encoder to get contextual representations
        for each token. Take CLS token representation and run through MLP head to get likelihood score.

        Also compute masked language modeling loss.
        Randomly mask some tokens in the input. Predict masked tokens as objective.


        Likelihood B follows A
          |
        FF + Softmax                          light
          |                                     |
        ====================== Encoder ======================
          |     |   |      |       |      |     |      |
        [CLS] Open the microwave [SEP] Toggle [MASK] switch

                 <Language A>            <Language B>

        :param Dict lang_inputs:
            Dictionary of inputs for language model
            keys:
            input_ids
            attention_mask
            token_type_ids
            labels
            is_next

        :return:
            - loss: tensor
        """
        encoder = self.model.encoder
        lang_inputs = AttrDict(lang_inputs)

        import ipdb

        ipdb.set_trace()

        # Input will be [CLS] Language A [SEP] Language B
        encoder_outputs = encoder(
            input_ids=lang_inputs.input_ids,
            attention_mask=lang_inputs.attention_mask,
            token_type_ids=lang_inputs.token_type_ids,
        )

        # copying from BERT NSP training
        pooled_output = encoder_outputs[1]  # B x HD
        relation_pred_scores = self.nsp_head(pooled_output)  # B x 2

        nsp_loss = self.nsp_loss_fn(
            relation_pred_scores.view(-1, 2), lang_inputs.is_next.view(-1)
        )

        # copying from BERT MLM training
        sequence_output = encoder_outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        mlm_loss = self.lm_loss_fn(
            prediction_scores.view(-1, self.encoder_config.vocab_size),
            lang_inputs.labels.view(-1),
        )

        total_loss = nsp_loss + mlm_loss
        return total_loss

    def _masked_behavior_modeling_loss(
        self, states: torch.Tensor, actions: torch.Tensor
    ):
        """
        Compute masked behavior loss.
        Given a sequence of (s,a) steps, mask out contiguous actions. Model is tasked with predicting /
        infilling the missing actions given the before and after context.

        Not sure if I should incorporate this into the decoder.**
        Other things to try: masking out non-contiguous actions? masking out the state as well?

                              a3           a4
                              |            |
        ====================== Encoder ======================
         |   |   |   |   |    |      |     |
        s1  a1  s2  a2  s3  [MASK]  s4  [MASK]  ...

        :param Tensor states:
            Tensor of shape [batch_size x timesteps x state_dim]
        :param Tensor actions:
            Tensor of shape [batch_size x timesteps x action_dim]

        :return:
            - loss: tensor
        """

        # Encode sequence
        encoder = self.model.encoder

        # Need to somehow mask some embeddings

        B, T, _ = actions.shape

        # embed each modality with a separate head
        # B x T x HD
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        stacked_inputs = torch.stack(
            (state_embeddings[:, :-1], action_embeddings), dim=1
        )

        # B x 2*T x HD
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(
            B, 2 * T, self.hparams.hidden_dim
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        device = stacked_inputs.device

        # tells the model which timesteps it can attend to
        # B x 2*T
        attention_mask = torch.ones((B, T), dtype=torch.long)
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(B, 2 * T)
        ).to(device)

        token_type_ids = torch.ones((B, T), dtype=torch.long)
        stacked_token_type_ids = (
            torch.stack((token_type_ids, token_type_ids), dim=1)
            .permute(0, 2, 1)
            .reshape(B, 2 * T)
        ).to(device)

        encoder_outputs = encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            token_type_ids=stacked_token_type_ids,
        )

        # reshape x so the second dimension corresponds to states (0), actions (1)
        x = encoder_outputs["last_hidden_state"]
        x = x.reshape(B, T, 2, self.hparams.hidden_dim).permute(0, 2, 1, 3)

        # predict masked actions given all the context
        action_preds = self.predict_action(x)
        import ipdb

        ipdb.set_trace()

    def _causal_behavior_modeling_loss(self):
        """
        Causal behavior modeling. Encode context and perform autoregressive decoding.



                                  a4  a5  a6  ....
                                   |   |   |
        ======= Encoder =======  ======= Decoder =======
         |   |   |   |   |   |     |   |   |
        s1  a1  s2  a2  s3  a3    s4  s5  s6  ....
        """
        pass

    def _paired_masked_modeling_loss(self):
        """
        Compute losses for paired input.
        1. Language-conditioned masked behavior modeling
        2. Behavior-conditioned masked language modeling
        3. Language-behavior alignment

        Are language + behavior aligned?
          |
        FF + Softmax   microwave                a1       a2
          |               |                     |        |
        ============================ Encoder ============================
          |     |   |     |     |    |  |  |    |    |   |
        [CLS] Open the [MASK] [SEP] s0 a0 s1 [MASK] s2 [MASK] ...

                <Language>                 <Skill>


        """

    def training_step(self, batch, batch_idx):
        lang_inputs, behavior_inputs = batch["language"], batch["behavior"]

        # lang-only inference
        lang_loss = self._language_only_losses(lang_inputs)
        mbm_loss = self._masked_behavior_modeling_loss(
            behavior_inputs["states"], behavior_inputs["actions"]
        )
        behavior_loss = 0.0

        paired_loss = 0.0

        loss = lang_loss + behavior_loss + paired_loss
        return None

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LB_MM_Transformer(pl.LightningModule):
    """
    Implementation of language behavior multi-modal transformer encoder-decoder model.

    Model is an encoder-decoder style. We use EncoderDecoder from HuggingFace and initialize the encoder
    with pretrained BERT and decoder with pretrained GPT. This is an arbitrary choice, we can
    initialize with any pretrained autoencoder and generative model.
    """

    def __init__(self, model_conf, **kwargs):
        super().__init__()

        # saves parameters into hparams attribute
        self.save_hyperparameters(model_conf)

        self.encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        self.decoder_config = GPT2Config.from_pretrained("gpt2")

        self.encoder_config.position_embedding_type = ""
        bert2gpt2 = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "bert-base-uncased",
            "gpt2",
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
            tie_encoder_decoder=True,
        )

        self.model = bert2gpt2

        self.embed_state = nn.Linear(self.hparams.state_dim, self.hparams.hidden_dim)
        self.embed_action = nn.Linear(self.hparams.action_dim, self.hparams.hidden_dim)
        self.embed_ln = nn.LayerNorm(self.hparams.hidden_dim)

        # head for predicting action
        action_tanh = True
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(self.hparams.hidden_dim, self.hparams.action_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )

        # update model configs
        print(self.model.config)

        # BERT's MLM head
        self.lm_head = BertOnlyMLMHead(self.encoder_config)
        self.lm_loss_fn = torch.nn.CrossEntropyLoss()
        self.nsp_head = BertOnlyNSPHead(self.encoder_config)
        self.nsp_loss_fn = torch.nn.CrossEntropyLoss()

    def _language_only_losses(self, lang_inputs):
        """

        Compute next skill prediction loss. Similar to next sentence prediction loss.
        Given language input A and language input B, predict likelihood that B comes after A.

        Run language input through encoder to get contextual representations
        for each token. Take CLS token representation and run through MLP head to get likelihood score.

        Also compute masked language modeling loss.
        Randomly mask some tokens in the input. Predict masked tokens as objective.


        Likelihood B follows A
          |
        FF + Softmax                          light
          |                                     |
        ====================== Encoder ======================
          |     |   |      |       |      |     |      |
        [CLS] Open the microwave [SEP] Toggle [MASK] switch

                 <Language A>            <Language B>

        :param Dict lang_inputs:
            Dictionary of inputs for language model
            keys:
            input_ids
            attention_mask
            token_type_ids
            labels
            is_next

        :return:
            - loss: tensor
        """
        encoder = self.model.encoder
        lang_inputs = AttrDict(lang_inputs)

        import ipdb

        ipdb.set_trace()

        # Input will be [CLS] Language A [SEP] Language B
        encoder_outputs = encoder(
            input_ids=lang_inputs.input_ids,
            attention_mask=lang_inputs.attention_mask,
            token_type_ids=lang_inputs.token_type_ids,
        )

        # copying from BERT NSP training
        pooled_output = encoder_outputs[1]  # B x HD
        relation_pred_scores = self.nsp_head(pooled_output)  # B x 2

        nsp_loss = self.nsp_loss_fn(
            relation_pred_scores.view(-1, 2), lang_inputs.is_next.view(-1)
        )

        # copying from BERT MLM training
        sequence_output = encoder_outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        mlm_loss = self.lm_loss_fn(
            prediction_scores.view(-1, self.encoder_config.vocab_size),
            lang_inputs.labels.view(-1),
        )

        total_loss = nsp_loss + mlm_loss
        return total_loss

    def _masked_behavior_modeling_loss(
        self, states: torch.Tensor, actions: torch.Tensor
    ):
        """
        Compute masked behavior loss.
        Given a sequence of (s,a) steps, mask out contiguous actions. Model is tasked with predicting /
        infilling the missing actions given the before and after context.

        Not sure if I should incorporate this into the decoder.**
        Other things to try: masking out non-contiguous actions? masking out the state as well?

                              a3           a4
                              |            |
        ====================== Encoder ======================
         |   |   |   |   |    |      |     |
        s1  a1  s2  a2  s3  [MASK]  s4  [MASK]  ...

        :param Tensor states:
            Tensor of shape [batch_size x timesteps x state_dim]
        :param Tensor actions:
            Tensor of shape [batch_size x timesteps x action_dim]

        :return:
            - loss: tensor
        """

        # Encode sequence
        encoder = self.model.encoder

        # Need to somehow mask some embeddings

        B, T, _ = actions.shape

        # embed each modality with a separate head
        # B x T x HD
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        stacked_inputs = torch.stack(
            (state_embeddings[:, :-1], action_embeddings), dim=1
        )

        # B x 2*T x HD
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(
            B, 2 * T, self.hparams.hidden_dim
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        device = stacked_inputs.device

        # tells the model which timesteps it can attend to
        # B x 2*T
        attention_mask = torch.ones((B, T), dtype=torch.long)
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(B, 2 * T)
        ).to(device)

        token_type_ids = torch.ones((B, T), dtype=torch.long)
        stacked_token_type_ids = (
            torch.stack((token_type_ids, token_type_ids), dim=1)
            .permute(0, 2, 1)
            .reshape(B, 2 * T)
        ).to(device)

        encoder_outputs = encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            token_type_ids=stacked_token_type_ids,
        )

        # reshape x so the second dimension corresponds to states (0), actions (1)
        x = encoder_outputs["last_hidden_state"]
        x = x.reshape(B, T, 2, self.hparams.hidden_dim).permute(0, 2, 1, 3)

        # predict masked actions given all the context
        action_preds = self.predict_action(x)
        import ipdb

        ipdb.set_trace()

    def _causal_behavior_modeling_loss(self):
        """
        Causal behavior modeling. Encode context and perform autoregressive decoding.



                                  a4  a5  a6  ....
                                   |   |   |
        ======= Encoder =======  ======= Decoder =======
         |   |   |   |   |   |     |   |   |
        s1  a1  s2  a2  s3  a3    s4  s5  s6  ....
        """
        pass

    def _paired_masked_modeling_loss(self):
        """
        Compute losses for paired input.
        1. Language-conditioned masked behavior modeling
        2. Behavior-conditioned masked language modeling
        3. Language-behavior alignment

        Are language + behavior aligned?
          |
        FF + Softmax   microwave                a1       a2
          |               |                     |        |
        ============================ Encoder ============================
          |     |   |     |     |    |  |  |    |    |   |
        [CLS] Open the [MASK] [SEP] s0 a0 s1 [MASK] s2 [MASK] ...

                <Language>                 <Skill>


        """

    def training_step(self, batch, batch_idx):
        lang_inputs, behavior_inputs = batch["language"], batch["behavior"]

        # lang-only inference
        lang_loss = self._language_only_losses(lang_inputs)
        mbm_loss = self._masked_behavior_modeling_loss(
            behavior_inputs["states"], behavior_inputs["actions"]
        )
        behavior_loss = 0.0

        paired_loss = 0.0

        loss = lang_loss + behavior_loss + paired_loss
        return None

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
