from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(name, value, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")


class DisentangledAttention(nn.Module):
    def __init__(self, hidden_size, heads, max_relative_positions, dropout=0.0,
                 relative_position_buckets=False, num_relative_buckets=32,
                 share_position_projection=False, relative_embeddings=None):
        super().__init__()
        for name, value in (("hidden_size", hidden_size), ("heads", heads),
                            ("max_relative_positions", max_relative_positions)):
            _positive_integer(name, value)
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        if relative_position_buckets:
            _positive_integer("num_relative_buckets", num_relative_buckets)
            if num_relative_buckets < 4 or num_relative_buckets % 2:
                raise ValueError("num_relative_buckets must be even and at least four")
            if max_relative_positions <= num_relative_buckets // 2 + 1:
                raise ValueError("max_relative_positions must exceed the exact bucket range plus one")
        self.hidden_size, self.heads = hidden_size, heads
        self.head_dim = hidden_size // heads
        self.max_relative_positions = max_relative_positions
        self.relative_position_buckets = relative_position_buckets
        self.num_relative_buckets = num_relative_buckets
        self.span = num_relative_buckets if relative_position_buckets else max_relative_positions
        self.relative_embeddings = (nn.Embedding(2 * self.span, hidden_size)
                                    if relative_embeddings is None else relative_embeddings)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size, bias=share_position_projection)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.position_key = self.key if share_position_projection else nn.Linear(
            hidden_size, hidden_size, bias=False)
        self.position_query = self.query if share_position_projection else nn.Linear(
            hidden_size, hidden_size)
        self.position_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)

    def relative_indices(self, length, device, *, reverse=False):
        positions = torch.arange(length, device=device)
        relative = positions[:, None] - positions[None, :]
        if reverse:
            relative = -relative
        if self.relative_position_buckets:
            midpoint = self.num_relative_buckets // 2
            magnitude = relative.abs()
            logarithmic = midpoint + torch.ceil(
                torch.log(magnitude.clamp_min(midpoint).float() / midpoint)
                / math.log((self.max_relative_positions - 1) / midpoint) * (midpoint - 1)
            ).long()
            relative = relative.sign() * torch.where(magnitude <= midpoint, magnitude, logarithmic)
        return (relative + self.span).clamp(0, 2 * self.span - 1)

    def forward(self, hidden, mask=None, query_states=None, relative_vectors=None):
        batch, length, _ = hidden.shape
        queries = hidden if query_states is None else query_states
        if queries.shape != hidden.shape:
            raise ValueError("decoder queries must match content batch/time/hidden geometry")
        def heads(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

        query, key, value = heads(self.query(queries)), heads(self.key(hidden)), heads(self.value(hidden))
        positions = self.position_dropout(self.relative_embeddings.weight if relative_vectors is None
                                           else relative_vectors)
        position_key = self.position_key(positions).reshape(2 * self.span, self.heads, self.head_dim)
        position_query = self.position_query(positions).reshape(2 * self.span, self.heads, self.head_dim)
        indices = self.relative_indices(length, hidden.device)
        reverse = self.relative_indices(length, hidden.device, reverse=True)
        c2p = torch.einsum("bhid,rhd->bhir", query, position_key).gather(
            -1, indices[None, None].expand(batch, self.heads, -1, -1))
        p2c = torch.einsum("bhjd,rhd->bhjr", key, position_query).gather(
            -1, reverse.T[None, None].expand(batch, self.heads, -1, -1)).transpose(-1, -2)
        scores = (query @ key.transpose(-1, -2) + c2p + p2c) / math.sqrt(3 * self.head_dim)
        if mask is not None:
            mask = mask.bool()
        valid = None if mask is None else mask[:, None, :, None] & mask[:, None, None, :]
        if valid is not None:
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        if valid is not None:
            weights = weights.masked_fill(~valid, 0)
        attended = self.dropout(weights) @ value
        output = self.output(attended.transpose(1, 2).reshape(batch, length, self.hidden_size))
        return output if mask is None else output * mask.unsqueeze(-1)


class DebertaLayer(nn.Module):
    def __init__(self, hidden_size, heads, intermediate_size, max_relative_positions,
                 dropout, conv_kernel_size, relative_position_buckets,
                 num_relative_buckets, share_position_projection, relative_embeddings, norm_eps):
        super().__init__()
        self.attention = DisentangledAttention(hidden_size, heads, max_relative_positions, dropout,
            relative_position_buckets, num_relative_buckets, share_position_projection, relative_embeddings)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.conv = (nn.Conv1d(hidden_size, hidden_size, conv_kernel_size,
                              padding=conv_kernel_size // 2) if conv_kernel_size else None)
        self.conv_norm = nn.LayerNorm(hidden_size, eps=norm_eps) if conv_kernel_size else None

    def forward(self, hidden, mask, query_states=None, conv_input=None, relative_vectors=None):
        residual = hidden if query_states is None else query_states
        output = self.attention_norm(residual + self.dropout(
            self.attention(hidden, mask, query_states, relative_vectors)))
        output = self.output_norm(output + self.dropout(self.output(F.gelu(self.intermediate(output)))))
        if conv_input is not None and self.conv is not None:
            convolved = self.conv((conv_input * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
            convolved = convolved * mask.unsqueeze(-1)
            output = self.conv_norm(output + F.gelu(self.dropout(convolved)))
        return output * mask.unsqueeze(-1)


class Deberta(nn.Module):
    architecture_id = "deberta"
    PRESETS = {
        "tiny": {"hidden_size": 32, "layers": 2, "heads": 4, "intermediate_size": 64},
        "small": {"hidden_size": 256, "layers": 6, "heads": 4, "intermediate_size": 1024},
        "base": {"hidden_size": 768, "layers": 12, "heads": 12, "intermediate_size": 3072},
    }
    BUDGET_KNOBS = {"hidden_size": range(16, 513, 8)}

    def __init__(self, vocab_size=30522, context_length=512, hidden_size=128, layers=4,
                 heads=4, intermediate_size=None, max_relative_positions=512,
                 type_vocab_size=0, pad_token_id=0, dropout=0.0, conv_kernel_size=None,
                 version=2, embedding_size=None, mask_token_id=103,
                 relative_position_buckets=None, num_relative_buckets=32,
                 initializer_range=0.02, layer_norm_eps=1e-7, pretraining_head=True):
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length),
                            ("hidden_size", hidden_size), ("layers", layers), ("heads", heads),
                            ("max_relative_positions", max_relative_positions)):
            _positive_integer(name, value)
        _positive_integer("type_vocab_size", type_vocab_size, zero=True)
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        if isinstance(version, bool) or version not in (1, 2):
            raise ValueError("DeBERTa version must be 1 or 2")
        conv_kernel_size = (3 if version == 2 else 0) if conv_kernel_size is None else conv_kernel_size
        _positive_integer("conv_kernel_size", conv_kernel_size, zero=True)
        if version == 1 and conv_kernel_size:
            raise ValueError("v1 does not support the v2 convolution")
        if conv_kernel_size and conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be zero or a positive odd integer")
        for name, value in (("dropout", dropout), ("initializer_range", initializer_range),
                            ("layer_norm_eps", layer_norm_eps)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= dropout < 1 or initializer_range <= 0 or layer_norm_eps <= 0:
            raise ValueError("dropout must be in [0,1); initialization range and norm epsilon must be positive")
        for name, value in (("pad_token_id", pad_token_id), ("mask_token_id", mask_token_id)):
            _positive_integer(name, value, zero=True)
            if value >= vocab_size:
                raise ValueError(f"{name} must be a valid vocabulary ID")
        self.vocab_size, self.context_length, self.hidden_size = vocab_size, context_length, hidden_size
        self.pad_token_id, self.mask_token_id = pad_token_id, mask_token_id
        self.version, self.max_relative_positions = version, max_relative_positions
        self.pretraining_head = pretraining_head
        relative_position_buckets = version == 2 if relative_position_buckets is None else relative_position_buckets
        embedding_size = hidden_size if embedding_size is None else embedding_size
        intermediate_size = 4 * hidden_size if intermediate_size is None else intermediate_size
        _positive_integer("embedding_size", embedding_size)
        _positive_integer("intermediate_size", intermediate_size)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size) if type_vocab_size else None
        self.embedding_projection = nn.Linear(embedding_size, hidden_size, bias=False) if embedding_size != hidden_size else nn.Identity()
        self.embedding_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.embedding_dropout = nn.Dropout(dropout)
        span = num_relative_buckets if relative_position_buckets else max_relative_positions
        self.relative_embeddings = nn.Embedding(2 * span, hidden_size)
        self.relative_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if version == 2 else nn.Identity()
        self.layers = nn.ModuleList([DebertaLayer(hidden_size, heads, intermediate_size,
            max_relative_positions, dropout, conv_kernel_size if index == 0 else 0,
            relative_position_buckets, num_relative_buckets, version == 2,
            self.relative_embeddings, layer_norm_eps) for index in range(layers)])
        if pretraining_head:
            self.position_embeddings = nn.Embedding(context_length, hidden_size)
            self.mlm_transform = nn.Linear(hidden_size, embedding_size)
            self.mlm_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
            self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.apply(lambda module: self._initialize(module, initializer_range))

    @staticmethod
    def _initialize(module, std):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=std)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
            elif getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _validate_tokens(self, tokens, mask, token_type_ids):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be LongTensor[B,T]")
        if not tokens.shape[0] or not tokens.shape[1] or tokens.shape[1] > self.context_length:
            raise ValueError("sequence must be nonempty and within context_length")
        if bool(((tokens < 0) | (tokens >= self.vocab_size)).any()):
            raise ValueError("tokens must be vocabulary IDs")
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape or mask.device != tokens.device:
                raise ValueError("mask must match token batch/time shape and device")
            if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("mask must contain boolean or 0/1 validity values")
        if token_type_ids is not None:
            if (not isinstance(token_type_ids, torch.Tensor) or token_type_ids.shape != tokens.shape
                    or token_type_ids.dtype != torch.long or token_type_ids.device != tokens.device):
                raise ValueError("token_type_ids must be a matching LongTensor")
            limit = self.token_type_embeddings.num_embeddings if self.token_type_embeddings is not None else 1
            if bool(((token_type_ids < 0) | (token_type_ids >= limit)).any()):
                raise ValueError("token_type_ids exceed type vocabulary")
        return tokens.ne(self.pad_token_id) if mask is None else mask.bool() & tokens.ne(self.pad_token_id)

    def _encode(self, tokens, mask=None, token_type_ids=None, inputs_embeds=None):
        mask = self._validate_tokens(tokens, mask, token_type_ids)
        embeddings = self.word_embeddings(tokens) if inputs_embeds is None else inputs_embeds
        if self.token_type_embeddings is not None:
            types = torch.zeros_like(tokens) if token_type_ids is None else token_type_ids
            embeddings = embeddings + self.token_type_embeddings(types)
        initial = self.embedding_dropout(self.embedding_norm(self.embedding_projection(embeddings)))
        hidden = initial * mask.unsqueeze(-1)
        penultimate = hidden
        relative = self.relative_norm(self.relative_embeddings.weight)
        for index, layer in enumerate(self.layers):
            penultimate = hidden
            hidden = self._layer(layer, hidden, mask, relative, conv_input=initial if index == 0 else None)
        return hidden, mask, penultimate, relative

    @staticmethod
    def _layer(layer, hidden, mask, relative, query_states=None, conv_input=None):
        return layer(hidden, mask, query_states, conv_input, relative)

    def encode(self, tokens, mask=None, token_type_ids=None):
        hidden, valid, _, _ = self._encode(tokens, mask, token_type_ids)
        return hidden, valid

    def forward_pretraining(self, tokens, mask=None, token_type_ids=None):
        if not self.pretraining_head:
            raise ValueError("this encoder was constructed without a pretraining head")
        hidden, mask, memory, relative = self._encode(tokens, mask, token_type_ids)
        positions = self.position_embeddings(torch.arange(tokens.shape[1], device=tokens.device))
        query = (memory + positions) * mask.unsqueeze(-1)
        for _ in range(2):
            query = self._layer(self.layers[-1], memory, mask, relative, query_states=query)
        transformed = self.mlm_norm(F.gelu(self.mlm_transform(query)))
        return {"mlm_logits": F.linear(transformed, self.word_embeddings.weight, self.mlm_bias),
                "last_hidden_state": hidden, "mask": mask}

    def forward(self, tokens, mask=None, token_type_ids=None):
        return self.forward_pretraining(tokens, mask, token_type_ids)["mlm_logits"]


class DebertaV3(nn.Module):
    architecture_id = "deberta-v3"
    PRESETS = Deberta.PRESETS
    BUDGET_KNOBS = Deberta.BUDGET_KNOBS

    def __init__(self, vocab_size=30522, context_length=512, hidden_size=128, layers=4,
                 heads=4, intermediate_size=None, max_relative_positions=512,
                 type_vocab_size=0, pad_token_id=0, dropout=0.0,
                 embedding_size=None, generator_hidden_size=None, generator_layers=None,
                 generator_heads=None, mask_token_id=103, version=2):
        super().__init__()
        if version != 2:
            raise ValueError("DeBERTaV3 uses the v2 encoder")
        embedding_size = hidden_size if embedding_size is None else embedding_size
        generator_heads = heads if generator_heads is None else generator_heads
        generator_hidden_size = hidden_size if generator_hidden_size is None else generator_hidden_size
        generator_layers = max(1, layers // 2) if generator_layers is None else generator_layers
        self.vocab_size, self.context_length, self.hidden_size = vocab_size, context_length, hidden_size
        self.pad_token_id, self.mask_token_id = pad_token_id, mask_token_id
        common = dict(vocab_size=vocab_size, context_length=context_length,
                      max_relative_positions=max_relative_positions, type_vocab_size=type_vocab_size,
                      pad_token_id=pad_token_id, dropout=dropout, embedding_size=embedding_size,
                      mask_token_id=mask_token_id, version=version)
        self.generator = Deberta(hidden_size=generator_hidden_size, layers=generator_layers,
                                heads=generator_heads, **common)
        self.discriminator = Deberta(hidden_size=hidden_size, layers=layers, heads=heads,
                                    intermediate_size=intermediate_size, pretraining_head=False, **common)
        self.discriminator.word_embeddings = self.generator.word_embeddings
        self.discriminator_embedding_delta = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        nn.init.zeros_(self.discriminator_embedding_delta.weight)
        self.rtd_dense = nn.Linear(hidden_size, hidden_size)
        self.rtd_output = nn.Linear(hidden_size, 1)
        self.rtd_activation = nn.GELU()
        for module in (self.rtd_dense, self.rtd_output):
            Deberta._initialize(module, 0.02)

    def _discriminator_inputs(self, tokens):
        shared = F.embedding(tokens, self.generator.word_embeddings.weight.detach(), padding_idx=self.pad_token_id)
        return shared + self.discriminator_embedding_delta(tokens)

    def _discriminator_hidden(self, tokens, valid_tokens, token_type_ids=None):
        return self.discriminator._encode(tokens, valid_tokens, token_type_ids,
                                          self._discriminator_inputs(tokens))[0]

    def pretraining_forward(self, original_tokens, masked_tokens, selected, valid_tokens,
                            token_type_ids=None, special_token_ids=(), temperature=1.0):
        valid_tokens = self.generator._validate_tokens(original_tokens, valid_tokens, token_type_ids)
        self.generator._validate_tokens(masked_tokens, valid_tokens, token_type_ids)
        if masked_tokens.shape != original_tokens.shape or masked_tokens.device != original_tokens.device:
            raise ValueError("masked_tokens must match original geometry and device")
        if (not isinstance(selected, torch.Tensor) or selected.shape != original_tokens.shape
                or selected.dtype != torch.bool or selected.device != original_tokens.device):
            raise ValueError("selected must be a matching BoolTensor")
        if bool((selected & ~valid_tokens).any()):
            raise ValueError("selected tokens must be valid tokens")
        if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        for token in special_token_ids:
            _positive_integer("special token", token, zero=True)
            if token >= self.vocab_size:
                raise ValueError("special token IDs must be vocabulary IDs")
        if len(set(special_token_ids)) >= self.vocab_size:
            raise ValueError("sampling needs at least one ordinary token")
        generator_outputs = self.generator.forward_pretraining(masked_tokens, valid_tokens, token_type_ids)
        corrupted = original_tokens.clone()
        if bool(selected.any()):
            logits = generator_outputs["mlm_logits"][selected].detach().clone()
            if special_token_ids:
                logits[:, list(special_token_ids)] = torch.finfo(logits.dtype).min
            sampled = torch.multinomial((logits / temperature).softmax(-1), 1).squeeze(-1)
            corrupted[selected] = sampled
        replaced = (corrupted.ne(original_tokens) & valid_tokens).float()
        hidden = self._discriminator_hidden(corrupted, valid_tokens, token_type_ids)
        disc = self.rtd_output(self.rtd_activation(self.rtd_dense(hidden))).squeeze(-1)
        return {"generator_logits": generator_outputs["mlm_logits"], "discriminator_logits": disc,
                "replaced_labels": replaced, "corrupted_tokens": corrupted}

    def forward(self, tokens, mask=None, token_type_ids=None):
        valid = self.discriminator._validate_tokens(tokens, mask, token_type_ids)
        return self._discriminator_hidden(tokens, valid, token_type_ids)
