import math
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F


class _SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, hidden, valid):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).view(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = valid[:, None, None, :].expand(batch, 1, length, length)
        empty_rows = ~allowed.any(dim=-1, keepdim=True)
        allowed = allowed | (empty_rows & torch.eye(length, dtype=torch.bool, device=hidden.device)[None, None])
        weights = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, intermediate_size, layer_norm_eps,
                 attention_dropout=None):
        super().__init__()
        self.attention = _SelfAttention(d_model, num_heads,
                                        dropout if attention_dropout is None else attention_dropout)
        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.intermediate = nn.Linear(d_model, intermediate_size)
        self.activation = nn.GELU()
        self.output = nn.Linear(intermediate_size, d_model)
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden, valid):
        hidden = self.attention_norm(hidden + self.attention_dropout(self.attention(hidden, valid)))
        feed_forward = self.output(self.activation(self.intermediate(hidden)))
        hidden = self.output_norm(hidden + self.output_dropout(feed_forward))
        return hidden


class BERTEncoder(nn.Module):
    architecture_id = "bert-encoder"
    PRESETS = {
        "tiny": {"d_model": 128, "num_layers": 2, "num_heads": 4, "intermediate_size": 512},
        "small": {"d_model": 512, "num_layers": 6, "num_heads": 8, "intermediate_size": 2048},
        "base": {"d_model": 768, "num_layers": 12, "num_heads": 12, "intermediate_size": 3072},
    }

    def __init__(self, vocab_size=30522, context_length=512, d_model=128, num_layers=2,
                 num_heads=4, intermediate_size=None, type_vocab_size=2,
                 dropout=0.0, pad_token_id=0, initializer_range=0.02, preset=None,
                   layer_norm_eps=1e-12, use_nsp=True, use_token_type_embeddings=True,
                   padding_positions=False, normal_initializer=False, zero_padded_inputs=False,
                   num_hidden_groups=None, inner_group_num=1, embedding_size=None,
                   word_embedding_size=None, use_mlm=True):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError("preset must be tiny, small, or base")
            selected = self.PRESETS[preset]
            d_model = selected["d_model"]
            num_layers = selected["num_layers"]
            num_heads = selected["num_heads"]
            intermediate_size = selected["intermediate_size"]
        if intermediate_size is None:
            intermediate_size = 4 * d_model
        if embedding_size is None:
            embedding_size = d_model
        if word_embedding_size is None:
            word_embedding_size = embedding_size
        if num_hidden_groups is None:
            num_hidden_groups = num_layers
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length),
                            ("d_model", d_model), ("num_layers", num_layers),
                            ("num_heads", num_heads), ("intermediate_size", intermediate_size),
                            ("embedding_size", embedding_size),
                            ("word_embedding_size", word_embedding_size),
                            ("type_vocab_size", type_vocab_size), ("num_hidden_groups", num_hidden_groups),
                            ("inner_group_num", inner_group_num)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads:
            raise ValueError("num_heads must divide d_model")
        if num_hidden_groups > num_layers:
            raise ValueError("num_hidden_groups must not exceed num_layers")
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be an integer in [0, vocab_size)")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (isinstance(initializer_range, bool)
                or not isinstance(initializer_range, (int, float))
                or not math.isfinite(initializer_range) or initializer_range <= 0):
            raise ValueError("initializer_range must be positive")
        if (isinstance(layer_norm_eps, bool) or not isinstance(layer_norm_eps, (int, float))
                or not math.isfinite(layer_norm_eps) or layer_norm_eps <= 0):
            raise ValueError("layer_norm_eps must be positive and finite")
        for name, value in (("use_nsp", use_nsp), ("use_token_type_embeddings", use_token_type_embeddings),
                            ("padding_positions", padding_positions),
                            ("normal_initializer", normal_initializer),
                            ("zero_padded_inputs", zero_padded_inputs), ("use_mlm", use_mlm)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.embedding_size = embedding_size
        self.word_embedding_size = word_embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_hidden_groups = num_hidden_groups
        self.inner_group_num = inner_group_num
        self.type_vocab_size = type_vocab_size
        self.pad_token_id = pad_token_id
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.use_nsp = use_nsp
        self.use_mlm = use_mlm
        self.normal_initializer = normal_initializer
        self.zero_padded_inputs = zero_padded_inputs
        self.position_padding_idx = pad_token_id if padding_positions else None
        word_padding_idx = pad_token_id if zero_padded_inputs else None
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_size, padding_idx=word_padding_idx)
        position_count = context_length + (pad_token_id + 1 if padding_positions else 0)
        self.position_embeddings = nn.Embedding(position_count, embedding_size, padding_idx=self.position_padding_idx)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size) if use_token_type_embeddings else None
        self.embedding_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.word_embedding_projection = (nn.Linear(word_embedding_size, embedding_size, bias=False)
                                          if word_embedding_size != embedding_size else nn.Identity())
        self.embedding_projection = nn.Linear(embedding_size, d_model) if embedding_size != d_model else nn.Identity()
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _EncoderLayer(d_model, num_heads, dropout, intermediate_size, layer_norm_eps)
            for _ in range(num_hidden_groups * inner_group_num)
        ])
        self.pooler = nn.Linear(d_model, d_model) if use_nsp else None
        self.mlm_transform = nn.Linear(d_model, word_embedding_size) if use_mlm else None
        self.mlm_activation = nn.GELU() if use_mlm else None
        self.mlm_norm = nn.LayerNorm(word_embedding_size, eps=layer_norm_eps) if use_mlm else None
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size)) if use_mlm else None
        self.nsp_classifier = nn.Linear(d_model, 2) if use_nsp else None
        self.apply(self._initialize)

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if self.normal_initializer:
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            else:
                nn.init.trunc_normal_(module.weight, mean=0.0, std=self.initializer_range,
                                      a=-2 * self.initializer_range,
                                      b=2 * self.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate(self, tokens, mask, token_type_ids):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a LongTensor shaped [batch, time]")
        batch, length = tokens.shape
        if batch <= 0 or length <= 0 or length > self.context_length:
            raise ValueError("tokens must have non-empty batch/time within context_length")
        if tokens.device.type != "meta" and tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("token IDs must be within [0, vocab_size)")
        if mask is None:
            valid = tokens.ne(self.pad_token_id)
        else:
            if not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape or mask.device != tokens.device:
                raise ValueError("mask must match tokens shape and device")
            if mask.dtype != torch.bool and not mask.is_floating_point() and mask.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
            ):
                raise ValueError("mask must contain boolean or numeric 0/1 values")
            if mask.device.type != "meta" and not bool(torch.all((mask == 0) | (mask == 1))):
                raise ValueError("mask values must be 0/1 (1 means valid)")
            valid = mask.bool()
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(tokens)
        elif (not isinstance(token_type_ids, torch.Tensor) or token_type_ids.shape != tokens.shape
              or token_type_ids.device != tokens.device or token_type_ids.dtype != torch.long):
            raise ValueError("token_type_ids must be a LongTensor matching tokens shape and device")
        if token_type_ids.device.type != "meta" and token_type_ids.numel() and (
            token_type_ids.min() < 0 or token_type_ids.max() >= self.type_vocab_size
        ):
            raise ValueError("token type IDs must be within [0, type_vocab_size)")
        if (self.token_type_embeddings is None and token_type_ids.device.type != "meta"
                and bool(token_type_ids.ne(0).any())):
            raise ValueError("this configuration has no segment embeddings; token_type_ids must be zero")
        return valid, token_type_ids

    def position_ids(self, tokens):
        batch, length = tokens.shape
        if self.position_padding_idx is None:
            return torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, length)
        non_padding = tokens.ne(self.pad_token_id).long()
        return non_padding.cumsum(-1) * non_padding + self.position_padding_idx

    def encode(self, tokens, mask=None, token_type_ids=None):
        valid, token_type_ids = self._validate(tokens, mask, token_type_ids)
        positions = self.position_ids(tokens)
        hidden = self.word_embedding_projection(self.word_embeddings(tokens)) + self.position_embeddings(positions)
        if self.token_type_embeddings is not None:
            hidden = hidden + self.token_type_embeddings(token_type_ids)
        hidden = self.embedding_norm(hidden)
        hidden = self.embedding_dropout(hidden)
        hidden = self.embedding_projection(hidden)
        if self.zero_padded_inputs:
            hidden = hidden.masked_fill(tokens.eq(self.pad_token_id).unsqueeze(-1), 0)
        for layer_index in range(self.num_layers):
            group_index = (layer_index * self.num_hidden_groups // self.num_layers) * self.inner_group_num
            for inner_index in range(self.inner_group_num):
                hidden = self.layers[group_index + inner_index](hidden, valid)
        pooled = torch.tanh(self.pooler(hidden[:, 0])) if self.pooler is not None else None
        return hidden, pooled

    def forward_pretraining(self, tokens, mask=None, token_type_ids=None):
        if not self.use_mlm:
            raise ValueError("MLM head is disabled for this encoder")
        hidden, pooled = self.encode(tokens, mask, token_type_ids)
        mlm_hidden = self.mlm_norm(self.mlm_activation(self.mlm_transform(hidden)))
        mlm_logits = F.linear(mlm_hidden, self.word_embeddings.weight, self.mlm_bias)
        nsp_logits = self.nsp_classifier(pooled) if self.nsp_classifier is not None else None
        return {"logits": mlm_logits, "mlm_logits": mlm_logits,
                "nsp_logits": nsp_logits, "last_hidden_state": hidden,
                "pooler_output": pooled}

    def forward(self, tokens, mask=None, token_type_ids=None, return_nsp=False):
        outputs = self.forward_pretraining(tokens, mask, token_type_ids)
        if return_nsp:
            if not self.use_nsp:
                raise ValueError("next-sentence prediction is disabled for this configuration")
            return outputs["mlm_logits"], outputs["nsp_logits"]
        return outputs["mlm_logits"]


class RoBERTa(nn.Module):
    architecture_id = "roberta"
    PRESETS = BERTEncoder.PRESETS
    BUDGET_KNOBS = {"d_model": range(8, 513, 4)}

    def __init__(self, vocab_size=50265, context_length=512, d_model=128, num_layers=2,
                 num_heads=4, intermediate_size=None, dropout=0.0, pad_token_id=1,
                 initializer_range=0.02, layer_norm_eps=1e-5, preset=None):
        super().__init__()
        self.encoder = BERTEncoder(vocab_size=vocab_size, context_length=context_length, d_model=d_model,
                                   num_layers=num_layers, num_heads=num_heads,
                                   intermediate_size=intermediate_size, type_vocab_size=1,
                                   dropout=dropout, pad_token_id=pad_token_id,
                                   initializer_range=initializer_range, preset=preset,
                                   layer_norm_eps=layer_norm_eps, use_nsp=False,
                                   use_token_type_embeddings=False, padding_positions=True,
                                   normal_initializer=True, zero_padded_inputs=True)
        self.vocab_size = self.encoder.vocab_size
        self.context_length = self.encoder.context_length
        self.d_model = self.encoder.d_model
        self.num_layers = self.encoder.num_layers
        self.num_heads = self.encoder.num_heads
        self.pad_token_id = self.encoder.pad_token_id

    @property
    def layers(self):
        return self.encoder.layers

    @property
    def word_embeddings(self):
        return self.encoder.word_embeddings

    @property
    def position_embeddings(self):
        return self.encoder.position_embeddings

    @property
    def embedding_norm(self):
        return self.encoder.embedding_norm

    @property
    def mlm_norm(self):
        return self.encoder.mlm_norm

    @property
    def mlm_transform(self):
        return self.encoder.mlm_transform

    @property
    def mlm_activation(self):
        return self.encoder.mlm_activation

    @property
    def mlm_bias(self):
        return self.encoder.mlm_bias

    @property
    def pooler(self):
        return self.encoder.pooler

    @property
    def nsp_classifier(self):
        return self.encoder.nsp_classifier

    @property
    def token_type_embeddings(self):
        return self.encoder.token_type_embeddings

    def position_ids(self, tokens):
        return self.encoder.position_ids(tokens)

    def encode(self, tokens, mask=None, token_type_ids=None):
        return self.encoder.encode(tokens, mask, token_type_ids)

    def forward_pretraining(self, tokens, mask=None, token_type_ids=None):
        return self.encoder.forward_pretraining(tokens, mask, token_type_ids)

    def forward(self, tokens, mask=None, token_type_ids=None, return_nsp=False):
        return self.encoder(tokens, mask, token_type_ids, return_nsp=return_nsp)


class ALBERT(nn.Module):
    architecture_id = "albert"
    PRESETS = {
        "tiny": {"d_model": 128, "embedding_size": 64, "num_layers": 4, "num_heads": 4, "intermediate_size": 512, "num_hidden_groups": 1},
        "small": {"d_model": 256, "embedding_size": 128, "num_layers": 12, "num_heads": 4, "intermediate_size": 1024, "num_hidden_groups": 1},
        "base": {"d_model": 768, "embedding_size": 128, "num_layers": 12, "num_heads": 12, "intermediate_size": 3072, "num_hidden_groups": 1},
    }
    BUDGET_KNOBS = {"d_model": range(16, 513, 16)}

    def __init__(self, vocab_size=30000, context_length=512, d_model=128, embedding_size=64,
                 num_layers=4, num_heads=4, intermediate_size=None, num_hidden_groups=1,
                 inner_group_num=1, type_vocab_size=2, dropout=0.0, attention_dropout=None, pad_token_id=0,
                 initializer_range=0.02, layer_norm_eps=1e-12, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError("preset must be tiny, small, or base")
            chosen = self.PRESETS[preset]
            d_model, embedding_size, num_layers, num_heads = (chosen[k] for k in ("d_model", "embedding_size", "num_layers", "num_heads"))
            intermediate_size, num_hidden_groups = chosen["intermediate_size"], chosen["num_hidden_groups"]
        if intermediate_size is None:
            intermediate_size = 4 * d_model
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length),
                            ("d_model", d_model), ("embedding_size", embedding_size),
                            ("num_layers", num_layers), ("num_heads", num_heads),
                            ("intermediate_size", intermediate_size), ("num_hidden_groups", num_hidden_groups),
                            ("inner_group_num", inner_group_num), ("type_vocab_size", type_vocab_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads or num_hidden_groups > num_layers:
            raise ValueError("num_heads must divide d_model and num_hidden_groups must not exceed num_layers")
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be an integer in [0, vocab_size)")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if attention_dropout is None:
            attention_dropout = dropout
        if isinstance(attention_dropout, bool) or not isinstance(attention_dropout, (int, float)) or not math.isfinite(attention_dropout) or not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be finite and in [0, 1)")
        if isinstance(initializer_range, bool) or not isinstance(initializer_range, (int, float)) or not math.isfinite(initializer_range) or initializer_range <= 0:
            raise ValueError("initializer_range must be positive and finite")
        if isinstance(layer_norm_eps, bool) or not isinstance(layer_norm_eps, (int, float)) or not math.isfinite(layer_norm_eps) or layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive and finite")
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.embedding_size, self.num_layers, self.num_heads = embedding_size, num_layers, num_heads
        self.num_hidden_groups, self.inner_group_num = num_hidden_groups, inner_group_num
        self.pad_token_id, self.initializer_range = pad_token_id, float(initializer_range)
        self.layer_norm_eps, self.type_vocab_size = float(layer_norm_eps), type_vocab_size
        self.dropout, self.attention_dropout = float(dropout), float(attention_dropout)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(context_length, embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size)
        self.embedding_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.embedding_projection = nn.Linear(embedding_size, d_model, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([_EncoderLayer(d_model, num_heads, dropout, intermediate_size, layer_norm_eps,
            attention_dropout)
            for _ in range(num_hidden_groups * inner_group_num)])
        self.mlm_transform = nn.Linear(d_model, embedding_size)
        self.mlm_activation = nn.GELU()
        self.mlm_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.pooler = nn.Linear(d_model, d_model)
        self.sop_classifier = nn.Linear(d_model, 2)
        self.apply(self._initialize_albert)

    def _initialize_albert(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.initializer_range,
                                  a=-2 * self.initializer_range, b=2 * self.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def position_ids(self, tokens):
        return torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand_as(tokens)

    def encode(self, tokens, mask=None, token_type_ids=None):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a LongTensor shaped [batch, time]")
        if not tokens.shape[0] or not tokens.shape[1] or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have non-empty batch/time within context_length")
        if tokens.device.type != "meta" and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("token IDs must be within [0, vocab_size)")
        if mask is None:
            valid = tokens.ne(self.pad_token_id)
        else:
            if not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape or mask.device != tokens.device:
                raise ValueError("mask must match tokens shape and device")
            if mask.dtype != torch.bool and not mask.is_floating_point() and mask.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("mask must contain boolean or numeric 0/1 values")
            if mask.device.type != "meta" and not bool(torch.all((mask == 0) | (mask == 1))):
                raise ValueError("mask values must be finite 0/1 (1 means valid)")
            valid = mask.bool()
        if token_type_ids is None:
            types = torch.zeros_like(tokens)
        elif (not isinstance(token_type_ids, torch.Tensor) or token_type_ids.shape != tokens.shape
              or token_type_ids.dtype != torch.long or token_type_ids.device != tokens.device):
            raise ValueError("token_type_ids must be a matching LongTensor")
        else:
            types = token_type_ids
        if types.device.type != "meta" and types.numel() and (types.min() < 0 or types.max() >= self.type_vocab_size):
            raise ValueError("token type IDs must be within [0, type_vocab_size)")
        hidden = self.word_embeddings(tokens) + self.position_embeddings(self.position_ids(tokens)) + self.token_type_embeddings(types)
        hidden = self.embedding_dropout(self.embedding_norm(hidden))
        hidden = self.embedding_projection(hidden)
        for layer_index in range(self.num_layers):
            group_index = (layer_index * self.num_hidden_groups // self.num_layers) * self.inner_group_num
            for inner_index in range(self.inner_group_num):
                hidden = self.layers[group_index + inner_index](hidden, valid)
        return hidden, torch.tanh(self.pooler(hidden[:, 0]))

    def forward_pretraining(self, tokens, mask=None, token_type_ids=None):
        hidden, pooled = self.encode(tokens, mask, token_type_ids)
        transformed = self.mlm_norm(self.mlm_activation(self.mlm_transform(hidden)))
        logits = F.linear(transformed, self.word_embeddings.weight, self.mlm_bias)
        return {"logits": logits, "mlm_logits": logits, "sop_logits": self.sop_classifier(pooled),
                "last_hidden_state": hidden, "pooler_output": pooled}

    def forward(self, tokens, mask=None, token_type_ids=None):
        return self.forward_pretraining(tokens, mask, token_type_ids)["mlm_logits"]

    @staticmethod
    def pretraining_loss(outputs, mlm_labels, sop_labels, sop_weight=1.0):
        if bool(mlm_labels.ne(-100).any()):
            mlm = F.cross_entropy(outputs["mlm_logits"].reshape(-1, outputs["mlm_logits"].shape[-1]), mlm_labels.reshape(-1), ignore_index=-100)
        else:
            mlm = outputs["mlm_logits"].sum() * 0
        sop = F.cross_entropy(outputs["sop_logits"], sop_labels)
        return mlm + sop_weight * sop


class DistilBERT(nn.Module):
    architecture_id = "distilbert-encoder"
    PRESETS = {
        "tiny": {"d_model": 128, "num_layers": 2, "num_heads": 4, "intermediate_size": 512},
        "small": {"d_model": 512, "num_layers": 6, "num_heads": 8, "intermediate_size": 2048},
        "base": {"d_model": 768, "num_layers": 6, "num_heads": 12, "intermediate_size": 3072},
    }
    BUDGET_KNOBS = {"d_model": range(12, 769, 12)}

    def __init__(self, vocab_size=30522, context_length=512, d_model=768, num_layers=6,
                 num_heads=12, intermediate_size=None, dropout=0.0, pad_token_id=0,
                 initializer_range=0.02, layer_norm_eps=1e-12):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * d_model
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length),
                            ("d_model", d_model), ("num_layers", num_layers),
                            ("num_heads", num_heads), ("intermediate_size", intermediate_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads:
            raise ValueError("num_heads must divide d_model")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.vocab_size, self.context_length = vocab_size, context_length
        self.d_model, self.num_layers, self.num_heads = d_model, num_layers, num_heads
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.encoder = BERTEncoder(vocab_size=vocab_size, context_length=context_length,
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            intermediate_size=intermediate_size, type_vocab_size=1, dropout=dropout,
            pad_token_id=pad_token_id, initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps, use_nsp=False, use_token_type_embeddings=False)
        self.encoder.word_embeddings.padding_idx = pad_token_id
        with torch.no_grad():
            self.encoder.word_embeddings.weight[pad_token_id].zero_()

    def initialize_from_teacher(self, teacher):
        required = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads",
                    "word_embeddings", "position_embeddings", "layers", "embedding_norm",
                    "mlm_transform", "mlm_norm", "mlm_bias")
        if any(not hasattr(teacher, name) for name in required):
            raise ValueError("teacher must expose compatible BERT encoder embeddings, layers, and MLM head")
        if (getattr(teacher, "position_padding_idx", None) is not None
                or not isinstance(getattr(teacher, "mlm_activation", None), nn.GELU)
                or getattr(teacher, "layer_norm_eps", None) != self.encoder.layer_norm_eps):
            raise ValueError("teacher must use BERT learned positions, GELU MLM head, and matching LayerNorm epsilon")
        if (teacher.vocab_size != self.vocab_size or teacher.context_length < self.context_length
                or teacher.d_model != self.d_model or teacher.num_heads != self.num_heads
                or teacher.num_layers != 2 * self.num_layers):
            raise ValueError("teacher must have matching vocabulary/width/heads and exactly twice student depth")
        for name in ("word_embeddings", "embedding_norm", "mlm_transform", "mlm_norm"):
            source, target = getattr(teacher, name), getattr(self.encoder, name)
            if source.state_dict().keys() != target.state_dict().keys() or any(
                    source.state_dict()[key].shape != target.state_dict()[key].shape
                    for key in source.state_dict()):
                raise ValueError(f"teacher {name} geometry is incompatible with the student")
        if teacher.position_embeddings.weight.shape[0] < self.context_length:
            raise ValueError("teacher position table is shorter than student context_length")
        if teacher.mlm_bias.shape != self.encoder.mlm_bias.shape:
            raise ValueError("teacher MLM vocabulary bias is incompatible with the student")
        for index, layer in enumerate(self.encoder.layers):
            source = teacher.layers[index * 2].state_dict()
            target = layer.state_dict()
            if source.keys() != target.keys() or any(source[key].shape != target[key].shape for key in source):
                raise ValueError(f"teacher layer {index * 2} geometry is incompatible with student layer {index}")
        with torch.no_grad():
            for name in ("word_embeddings", "embedding_norm",
                         "mlm_transform", "mlm_norm"):
                getattr(self.encoder, name).load_state_dict(getattr(teacher, name).state_dict())
            self.encoder.position_embeddings.weight.copy_(
                teacher.position_embeddings.weight[:self.context_length])
            self.encoder.mlm_bias.copy_(teacher.mlm_bias)
            for index, layer in enumerate(self.encoder.layers):
                layer.load_state_dict(teacher.layers[index * 2].state_dict())

    def encode(self, tokens, mask=None):
        return self.encoder.encode(tokens, mask, torch.zeros_like(tokens))[0]

    def forward_pretraining(self, tokens, mask=None):
        outputs = self.encoder.forward_pretraining(tokens, mask)
        return {"logits": outputs["mlm_logits"], "mlm_logits": outputs["mlm_logits"],
                "last_hidden_state": outputs["last_hidden_state"]}

    def forward(self, tokens, mask=None, return_dict=False):
        output = self.forward_pretraining(tokens, mask)
        return output if return_dict else output["mlm_logits"]


class ELECTRA(nn.Module):
    architecture_id = "electra"
    PRESETS = {
        "tiny": {"generator_hidden_size": 16, "discriminator_hidden_size": 64, "embedding_size": 32, "generator_layers": 2, "discriminator_layers": 2, "generator_num_heads": 1, "discriminator_num_heads": 4},
        "small": {"generator_hidden_size": 64, "discriminator_hidden_size": 256, "embedding_size": 128, "generator_layers": 12, "discriminator_layers": 12, "generator_num_heads": 1, "discriminator_num_heads": 4},
        "base": {"generator_hidden_size": 256, "discriminator_hidden_size": 768, "embedding_size": 768, "generator_layers": 12, "discriminator_layers": 12, "generator_num_heads": 4, "discriminator_num_heads": 12},
    }
    BUDGET_KNOBS = {"embedding_size": range(8, 257, 8)}

    def __init__(self, vocab_size=30522, context_length=512, generator_hidden_size=64,
                 discriminator_hidden_size=128, generator_layers=2, discriminator_layers=4,
                 num_heads=4, generator_intermediate_size=None, discriminator_intermediate_size=None,
                 dropout=0.1, pad_token_id=0, mask_token_id=103, type_vocab_size=2,
                 embedding_size=None, share_embeddings=True, preset=None,
                 generator_num_heads=None, discriminator_num_heads=None, rtd_head=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError("preset must be tiny, small, or base")
            cfg = self.PRESETS[preset]
            generator_hidden_size, discriminator_hidden_size = cfg["generator_hidden_size"], cfg["discriminator_hidden_size"]
            embedding_size = cfg["embedding_size"]
            generator_layers, discriminator_layers = cfg["generator_layers"], cfg["discriminator_layers"]
            generator_num_heads = cfg["generator_num_heads"]
            discriminator_num_heads = cfg["discriminator_num_heads"]
        generator_num_heads = num_heads if generator_num_heads is None else generator_num_heads
        discriminator_num_heads = num_heads if discriminator_num_heads is None else discriminator_num_heads
        embedding_size = generator_hidden_size if embedding_size is None else embedding_size
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length),
                            ("generator_hidden_size", generator_hidden_size),
                            ("discriminator_hidden_size", discriminator_hidden_size),
                            ("generator_layers", generator_layers), ("discriminator_layers", discriminator_layers),
                            ("generator_num_heads", generator_num_heads),
                            ("discriminator_num_heads", discriminator_num_heads),
                            ("embedding_size", embedding_size),
                            ("type_vocab_size", type_vocab_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if generator_hidden_size % generator_num_heads or discriminator_hidden_size % discriminator_num_heads:
            raise ValueError("generator_num_heads and discriminator_num_heads must divide their hidden sizes")
        if not isinstance(share_embeddings, bool):
            raise TypeError("share_embeddings must be boolean")
        if rtd_head is not None and not isinstance(rtd_head, nn.Module):
            raise TypeError("rtd_head must be an nn.Module or None")
        if not 0 <= pad_token_id < vocab_size or not 0 <= mask_token_id < vocab_size or pad_token_id == mask_token_id:
            raise ValueError("pad_token_id and mask_token_id must be distinct IDs inside vocabulary")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size, self.context_length = vocab_size, context_length
        self.generator_hidden_size, self.discriminator_hidden_size = generator_hidden_size, discriminator_hidden_size
        self.embedding_size = embedding_size
        self.generator_num_heads, self.discriminator_num_heads = generator_num_heads, discriminator_num_heads
        self.pad_token_id, self.mask_token_id, self.share_embeddings = pad_token_id, mask_token_id, share_embeddings
        ge = BERTEncoder(vocab_size, context_length, generator_hidden_size, generator_layers, generator_num_heads,
                         generator_intermediate_size, type_vocab_size, dropout, pad_token_id,
                         use_nsp=False, embedding_size=embedding_size)
        de = BERTEncoder(vocab_size, context_length, discriminator_hidden_size, discriminator_layers,
                         discriminator_num_heads, discriminator_intermediate_size, type_vocab_size, dropout, pad_token_id,
                         use_nsp=False, embedding_size=embedding_size, use_mlm=False)
        self.generator_encoder = ge
        self.discriminator_encoder = de
        if share_embeddings:
            de.word_embeddings = ge.word_embeddings
            de.position_embeddings = ge.position_embeddings
            de.token_type_embeddings = ge.token_type_embeddings
            de.embedding_norm = ge.embedding_norm
        self.rtd_head = rtd_head
        if rtd_head is None:
            self.rtd_dense = nn.Linear(discriminator_hidden_size, discriminator_hidden_size)
            self.rtd_activation = nn.GELU()
            self.rtd_output = nn.Linear(discriminator_hidden_size, 1)
            for layer in (self.rtd_dense, self.rtd_output):
                nn.init.trunc_normal_(layer.weight, std=0.02, a=-0.04, b=0.04)
                nn.init.zeros_(layer.bias)

    def forward(self, tokens, mask=None, token_type_ids=None):
        generator_logits = self.generator_encoder(tokens, mask, token_type_ids)
        hidden, _ = self.discriminator_encoder.encode(tokens, mask, token_type_ids)
        if hidden.shape[-1] != self.discriminator_hidden_size:
            raise RuntimeError("discriminator encoder returned an unexpected hidden dimension")
        logits = self._rtd_logits(hidden)
        return {"generator_logits": generator_logits, "discriminator_logits": logits}

    def pretraining_forward(self, original_tokens, masked_tokens, selected, valid_mask=None,
                            token_type_ids=None, special_token_ids=(), temperature=1.0):
        valid = original_tokens.ne(self.pad_token_id) if valid_mask is None else valid_mask.bool()
        generator_logits = self.generator_encoder(masked_tokens, valid, token_type_ids)
        corrupted, replaced = self.sample_corruption(original_tokens, generator_logits,
                                                       selected, valid, special_token_ids, temperature)
        hidden, _ = self.discriminator_encoder.encode(corrupted, valid, token_type_ids)
        discriminator_logits = self._rtd_logits(hidden)
        return {"generator_logits": generator_logits,
                "discriminator_logits": discriminator_logits,
                "corrupted_tokens": corrupted, "replaced_labels": replaced}

    def _rtd_logits(self, hidden):
        if self.rtd_head is not None:
            logits = self.rtd_head(hidden)
            if logits.shape != hidden.shape[:2]:
                raise ValueError("rtd_head must return a [batch, time] tensor of logits")
            return logits
        return self.rtd_output(self.rtd_activation(self.rtd_dense(hidden))).squeeze(-1)

    @staticmethod
    def sample_corruption(original_tokens, generator_logits, selected, valid_tokens,
                          special_token_ids=(), temperature=1.0):
        if original_tokens.shape != selected.shape or original_tokens.shape != valid_tokens.shape:
            raise ValueError("tokens, selected, and valid_tokens must have matching shapes")
        if generator_logits.shape[:2] != original_tokens.shape:
            raise ValueError("generator_logits must match token batch and time dimensions")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("temperature must be positive and finite")
        eligible = selected.bool() & valid_tokens.bool()
        for token_id in special_token_ids:
            eligible &= original_tokens.ne(token_id)
        sampled = torch.zeros_like(original_tokens)
        if bool(eligible.any()):
            sampled[eligible] = torch.multinomial(
                (generator_logits[eligible].detach().float() / temperature).softmax(-1), 1
            ).squeeze(-1)
        corrupted = torch.where(eligible, sampled, original_tokens)
        labels = (corrupted != original_tokens).to(torch.float32)
        return corrupted.detach(), labels.detach()

    @staticmethod
    def pretraining_loss(outputs, mlm_labels, replaced_labels, valid_mask,
                         generator_weight=1.0, discriminator_weight=50.0):
        if generator_weight < 0 or discriminator_weight < 0:
            raise ValueError("loss weights must be non-negative")
        gen = F.cross_entropy(outputs["generator_logits"].reshape(-1, outputs["generator_logits"].shape[-1]),
                               mlm_labels.reshape(-1), ignore_index=-100)
        valid = valid_mask.bool()
        if not bool(valid.any()):
            raise ValueError("valid_mask must include at least one token")
        disc = F.binary_cross_entropy_with_logits(outputs["discriminator_logits"][valid],
                                                   replaced_labels.float()[valid])
        return generator_weight * gen + discriminator_weight * disc, {"generator_loss": gen, "discriminator_loss": disc}


class SentenceOrderCollator:
    def __init__(self, pad_token_id=0, cls_token_id=2, sep_token_id=3):
        self.pad_token_id, self.cls_token_id, self.sep_token_id = pad_token_id, cls_token_id, sep_token_id

    def __call__(self, first_sentences, second_sentences, is_swapped):
        if not (len(first_sentences) == len(second_sentences) == len(is_swapped)) or not first_sentences:
            raise ValueError("sentence batches and order labels must have the same non-zero size")
        rows, types = [], []
        for first, second in zip(first_sentences, second_sentences):
            if not first or not second:
                raise ValueError("sentence segments must be non-empty")
            rows.append([self.cls_token_id, *first, self.sep_token_id, *second, self.sep_token_id])
            types.append([0] * (len(first) + 2) + [1] * (len(second) + 1))
        tokens = nn.utils.rnn.pad_sequence([torch.tensor(row, dtype=torch.long) for row in rows], batch_first=True, padding_value=self.pad_token_id)
        token_types = nn.utils.rnn.pad_sequence([torch.tensor(row, dtype=torch.long) for row in types], batch_first=True, padding_value=0)
        return {"tokens": tokens, "token_type_ids": token_types, "mask": tokens.ne(self.pad_token_id),
                "sop_labels": torch.tensor([int(bool(label)) for label in is_swapped], dtype=torch.long)}

    def from_documents(self, documents):
        first, second, swapped = [], [], []
        for document in documents:
            if len(document) < 2:
                raise ValueError("each document must contain at least two sentences")
            for left, right in pairwise(document):
                first.extend((left, right))
                second.extend((right, left))
                swapped.extend((False, True))
        if not first:
            raise ValueError("documents must contain at least one consecutive sentence pair")
        return self(first, second, swapped)


class ALBERTPretrainingCollator:
    def __init__(self, vocab_size, mask_token_id, pad_token_id=0, cls_token_id=2,
                 sep_token_id=3, special_token_ids=(0, 1, 2, 3), mlm_probability=0.15,
                 seed=0):
        self.sentence_order = SentenceOrderCollator(pad_token_id, cls_token_id, sep_token_id)
        self.masking = DynamicMaskingCollator(vocab_size, mask_token_id, pad_token_id,
            special_token_ids, mlm_probability, seed)

    def __call__(self, documents):
        ordered = self.sentence_order.from_documents(documents)
        masked = self.masking(ordered["tokens"])
        return {**ordered, **masked}

    def wrap(self, loader):
        return _ALBERTPretrainingLoader(loader, self)


class _ALBERTPretrainingLoader:
    def __init__(self, loader, collator):
        self.loader, self.collator = loader, collator
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        for batch in self.loader:
            documents = batch["documents"] if isinstance(batch, dict) else batch
            yield self.collator(documents)

    def __len__(self):
        return len(self.loader)


class DynamicMaskingCollator:
    def __init__(self, vocab_size, mask_token_id=None, pad_token_id=1,
                 special_token_ids=(0, 1, 2, 3), probability=0.15, seed=0):
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 2:
            raise ValueError("vocab_size must be an integer of at least two")
        if mask_token_id is None:
            mask_token_id = vocab_size - 1
        special_token_ids = tuple(special_token_ids)
        for name, value in (("mask_token_id", mask_token_id), ("pad_token_id", pad_token_id),
                            *(("special_token_id", value) for value in special_token_ids)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab_size:
                raise ValueError(f"{name} must be an integer in [0, vocab_size)")
        if mask_token_id == pad_token_id:
            raise ValueError("mask_token_id must differ from pad_token_id")
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 < probability <= 1):
            raise ValueError("probability must be finite and in (0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_token_ids = tuple(sorted({*special_token_ids, mask_token_id, pad_token_id}))
        ordinary = torch.ones(vocab_size, dtype=torch.bool)
        ordinary[list(self.special_token_ids)] = False
        self.replacement_ids = ordinary.nonzero(as_tuple=True)[0]
        if not self.replacement_ids.numel():
            raise ValueError("vocabulary must contain ordinary non-special tokens")
        self.probability = float(probability)
        self.seed = seed
        self.worker_seed = None
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, tokens):
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.seed != self.worker_seed:
            self.generator.manual_seed((self.seed + worker.seed) % 2**63)
            self.worker_seed = worker.seed
        if isinstance(tokens, (list, tuple)):
            if not tokens or any(not isinstance(t, torch.Tensor) or t.ndim != 1
                                 or t.dtype != torch.long or t.device.type != "cpu"
                                 or not t.numel() for t in tokens):
                raise ValueError("samples must be non-empty one-dimensional CPU LongTensors")
            tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.pad_token_id)
        if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long
                or tokens.device.type != "cpu" or not tokens.shape[0] or not tokens.shape[1]):
            raise ValueError("tokens must be a non-empty CPU LongTensor shaped [batch, time]")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError("tokens must lie within [0, vocab_size)")
        eligible = torch.ones_like(tokens, dtype=torch.bool)
        for token_id in self.special_token_ids:
            eligible &= tokens.ne(token_id)
        if not bool(eligible.any()):
            raise ValueError("batch has no ordinary tokens available for masked-language modeling")
        selected = (torch.rand(tokens.shape, generator=self.generator) < self.probability) & eligible
        for row in range(tokens.shape[0]):
            if bool(eligible[row].any()) and not bool(selected[row].any()):
                candidates = eligible[row].nonzero(as_tuple=True)[0]
                index = torch.randint(candidates.numel(), (1,), generator=self.generator)
                selected[row, candidates[index]] = True
        labels = tokens.clone().masked_fill(~selected, -100)
        corrupt = tokens.clone()
        replacement = torch.rand(tokens.shape, generator=self.generator)
        masked = selected & (replacement < 0.8)
        random = selected & (replacement >= 0.8) & (replacement < 0.9)
        corrupt[masked] = self.mask_token_id
        indices = torch.randint(self.replacement_ids.numel(), tokens.shape, generator=self.generator)
        random_ids = self.replacement_ids[indices]
        corrupt[random] = random_ids[random]
        return {"tokens": corrupt, "labels": labels, "mask": tokens.ne(self.pad_token_id)}

    def wrap(self, loader):
        return _DynamicMaskingLoader(loader, self)


class _DynamicMaskingLoader:
    def __init__(self, loader, collator):
        self.loader = loader
        self.collator = collator
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        for batch in self.loader:
            yield self.collator(batch["tokens"] if isinstance(batch, dict) else batch)

    def __len__(self):
        return len(self.loader)
