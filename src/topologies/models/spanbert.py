import math

import torch
from torch import nn
from torch.nn import functional as F


class SpanBERT(nn.Module):
    architecture_id = "spanbert"
    PRESETS = {
        "tiny": {"d_model": 128, "num_layers": 2, "num_heads": 4},
        "small": {"d_model": 512, "num_layers": 6, "num_heads": 8},
        "base": {"d_model": 768, "num_layers": 12, "num_heads": 12},
    }
    BUDGET_KNOBS = {"d_model": range(16, 513, 16)}

    def __init__(self, vocab_size=30522, context_length=512, d_model=128,
                 num_layers=2, num_heads=4, intermediate_size=None, dropout=0.0,
                 pad_token_id=0, mask_token_id=103, type_vocab_size=2,
                  max_span_length=200, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError("preset must be tiny, small, or base")
            cfg = self.PRESETS[preset]
            d_model, num_layers, num_heads = cfg["d_model"], cfg["num_layers"], cfg["num_heads"]
        intermediate_size = 4 * d_model if intermediate_size is None else intermediate_size
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length),
                            ("d_model", d_model), ("num_layers", num_layers),
                            ("num_heads", num_heads), ("intermediate_size", intermediate_size),
                            ("type_vocab_size", type_vocab_size), ("max_span_length", max_span_length)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads:
            raise ValueError("num_heads must divide d_model")
        if (isinstance(pad_token_id, bool) or isinstance(mask_token_id, bool)
                or not isinstance(pad_token_id, int) or not isinstance(mask_token_id, int)
                or not 0 <= pad_token_id < vocab_size or not 0 <= mask_token_id < vocab_size):
            raise ValueError("pad_token_id and mask_token_id must be inside the vocabulary")
        if pad_token_id == mask_token_id:
            raise ValueError("mask and padding IDs must differ")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.num_layers, self.num_heads = num_layers, num_heads
        self.pad_token_id, self.mask_token_id = pad_token_id, mask_token_id
        self.max_span_length, self.type_vocab_size = max_span_length, type_vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)
        self.embedding_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model, num_heads, intermediate_size, dropout, batch_first=True,
            norm_first=False, activation="gelu") for _ in range(num_layers)])
        self.mlm_transform = nn.Linear(d_model, d_model)
        self.mlm_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.sbo_position = nn.Embedding(max_span_length + 1, d_model, padding_idx=0)
        self.sbo_transform = nn.Linear(3 * d_model, d_model)
        self.sbo_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.sbo_output = nn.Linear(d_model, d_model)
        self.sbo_output_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.apply(self._initialize)

    def _initialize(self, module):
        if isinstance(module, nn.MultiheadAttention):
            nn.init.trunc_normal_(module.in_proj_weight, std=0.02, a=-0.04, b=0.04)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, tokens, mask=None, token_type_ids=None):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a LongTensor shaped [batch, time]")
        b, t = tokens.shape
        if not b or not t or t > self.context_length or tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError("tokens must be non-empty valid vocabulary IDs within context_length")
        if mask is None:
            valid = tokens.ne(self.pad_token_id)
        else:
            if (not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape
                    or mask.device != tokens.device):
                raise ValueError("mask must match tokens shape and device")
            if mask.dtype != torch.bool and not mask.is_floating_point() and mask.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
            ):
                raise ValueError("mask must contain boolean or numeric 0/1 values")
            if not bool(torch.all((mask == 0) | (mask == 1))):
                raise ValueError("mask values must be 0/1 (1 means valid)")
            valid = mask.bool()
        if bool((~valid.any(dim=1)).any()):
            raise ValueError("every token row must contain at least one valid token")
        types = torch.zeros_like(tokens) if token_type_ids is None else token_type_ids
        if types.shape != tokens.shape or types.dtype != torch.long or types.device != tokens.device:
            raise ValueError("token_type_ids must be a matching LongTensor")
        if types.numel() and (types.min() < 0 or types.max() >= self.type_vocab_size):
            raise ValueError("token type IDs outside configured range")
        pos = torch.arange(t, device=tokens.device).expand(b, t)
        x = self.embedding_norm(self.word_embeddings(tokens) + self.position_embeddings(pos) + self.token_type_embeddings(types))
        x = self.embedding_dropout(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~valid)
        return x

    def forward_pretraining(self, tokens, mask=None, token_type_ids=None,
                            span_starts=None, span_ends=None, span_positions=None):
        hidden = self.encode(tokens, mask, token_type_ids)
        valid = tokens.ne(self.pad_token_id) if mask is None else mask.bool()
        mlm = self.mlm_norm(F.gelu(self.mlm_transform(hidden)))
        mlm_logits = F.linear(mlm, self.word_embeddings.weight, self.mlm_bias)
        result = {"logits": mlm_logits, "mlm_logits": mlm_logits,
                  "last_hidden_state": hidden}
        if span_starts is not None or span_ends is not None or span_positions is not None:
            if span_starts is None or span_ends is None or span_positions is None:
                raise ValueError("span_starts, span_ends, and span_positions are required together")
            if span_starts.shape != span_ends.shape or span_positions.shape[:2] != span_starts.shape:
                raise ValueError("span boundaries and target positions must have matching shapes")
            for name, value in (("span_starts", span_starts), ("span_ends", span_ends),
                                ("span_positions", span_positions)):
                if value.dtype != torch.long or value.device != tokens.device:
                    raise ValueError(f"{name} must be a LongTensor on the tokens device")
            if span_positions.shape[-1] > self.max_span_length:
                raise ValueError("span target length exceeds max_span_length")
            if bool((span_starts < 0).any()) or bool((span_ends < 0).any()):
                raise ValueError("span boundaries must be non-negative; zero starts are padding slots")
            active = span_starts.gt(0)
            if bool((active & ((span_ends < span_starts) | (span_ends >= tokens.shape[1] - 1))).any()):
                raise ValueError("every span must have encoded outside-boundary tokens on both sides")
            if bool((active & (span_ends - span_starts + 1 > span_positions.shape[-1])).any()):
                raise ValueError("span bounds exceed target-position width")
            for row in range(tokens.shape[0]):
                for slot in range(span_starts.shape[1]):
                    if not bool(active[row, slot]):
                        continue
                    start, end = int(span_starts[row, slot]), int(span_ends[row, slot])
                    expected_positions = torch.arange(start, end + 1, device=tokens.device)
                    if not torch.equal(span_positions[row, slot, :expected_positions.numel()], expected_positions):
                        raise ValueError("span_positions must contain the contiguous absolute target positions")
                    if not bool(valid[row, start:end + 1].all()):
                        raise ValueError("span target positions must be valid non-padding tokens")
                    if not bool(valid[row, start - 1] & valid[row, end + 1]):
                        raise ValueError("span boundary positions must be valid non-padding tokens")
            result["sbo_logits"] = self._span_boundary_logits(
                hidden, span_starts, span_ends, span_positions)
        return result

    def _span_boundary_logits(self, hidden, span_starts, span_ends, span_positions):
        batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        left = (span_starts - 1).clamp(min=0)
        right = (span_ends + 1).clamp(max=hidden.shape[1] - 1)
        width = span_positions.shape[-1]
        offsets = torch.arange(1, width + 1, device=hidden.device).expand_as(span_positions)
        left_state = hidden[batch, left]
        right_state = hidden[batch, right]
        decoder_input = torch.cat((left_state[:, :, None, :].expand(-1, -1, width, -1),
                                   right_state[:, :, None, :].expand(-1, -1, width, -1),
                                   self.sbo_position(offsets)), dim=-1)
        hidden_state = self.sbo_norm(F.gelu(self.sbo_transform(decoder_input)))
        hidden_state = self.sbo_output_norm(F.gelu(self.sbo_output(hidden_state)))
        return F.linear(hidden_state, self.word_embeddings.weight, self.mlm_bias)

    def forward(self, tokens, mask=None, token_type_ids=None):
        return self.forward_pretraining(tokens, mask, token_type_ids)["mlm_logits"]

    @staticmethod
    def pretraining_loss(outputs, mlm_labels, span_labels=None, span_mask=None,
                         sbo_weight=1.0):
        mlm = F.cross_entropy(outputs["mlm_logits"].reshape(-1, outputs["mlm_logits"].shape[-1]),
                              mlm_labels.reshape(-1), ignore_index=-100)
        if span_labels is None:
            return mlm
        if "sbo_logits" not in outputs or span_labels.shape != outputs["sbo_logits"].shape[:-1]:
            raise ValueError("span labels must match supplied SBO outputs")
        valid = span_labels.ne(-100) if span_mask is None else span_mask.bool() & span_labels.ne(-100)
        sbo = F.cross_entropy(outputs["sbo_logits"][valid], span_labels[valid]) if bool(valid.any()) else outputs["sbo_logits"].sum() * 0
        return mlm + sbo_weight * sbo


class SpanMaskingCollator:
    def __init__(self, vocab_size, mask_token_id, pad_token_id=0,
                 special_token_ids=(0, 101, 102), masking_ratio=0.15,
                 geometric_p=0.2, span_lower=1, span_upper=10,
                 max_span_length=200, seed=0):
        if (isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 4
                or isinstance(masking_ratio, bool) or not isinstance(masking_ratio, (int, float))
                or not math.isfinite(masking_ratio) or not 0 < masking_ratio <= 1
                or isinstance(geometric_p, bool) or not isinstance(geometric_p, (int, float))
                or not math.isfinite(geometric_p) or not 0 < geometric_p <= 1):
            raise ValueError("invalid vocabulary, masking ratio, or geometric probability")
        for name, value in (("span_lower", span_lower), ("span_upper", span_upper),
                            ("max_span_length", max_span_length)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if span_upper < span_lower or span_upper > max_span_length:
            raise ValueError("span lengths must satisfy 1 <= span_lower <= span_upper <= max_span_length")
        if not 0 <= mask_token_id < vocab_size or not 0 <= pad_token_id < vocab_size:
            raise ValueError("special IDs must be within vocabulary")
        self.vocab_size, self.mask_token_id, self.pad_token_id = vocab_size, mask_token_id, pad_token_id
        if any(isinstance(token_id, bool) or not isinstance(token_id, int)
               or not 0 <= token_id < vocab_size for token_id in special_token_ids):
            raise ValueError("special_token_ids must be vocabulary IDs")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.special_ids = set(special_token_ids) | {pad_token_id, mask_token_id}
        self.ratio, self.geometric_p = float(masking_ratio), float(geometric_p)
        self.span_lower, self.span_upper, self.maximum = span_lower, span_upper, max_span_length
        lengths = torch.arange(span_lower, span_upper + 1, dtype=torch.float64)
        probabilities = self.geometric_p * (1 - self.geometric_p) ** (lengths - span_lower)
        self.length_probabilities = (probabilities / probabilities.sum()).float()
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, tokens, word_ids=None):
        if isinstance(tokens, (list, tuple)):
            tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.pad_token_id)
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.device.type != "cpu":
            raise ValueError("tokens must be a CPU LongTensor [batch, time]")
        b, t = tokens.shape
        if not b or not t or tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError("tokens must be non-empty valid vocabulary IDs")
        if word_ids is not None and (word_ids.shape != tokens.shape or word_ids.device.type != "cpu"
                                     or word_ids.dtype != torch.long):
            raise ValueError("word_ids must be a matching CPU LongTensor")
        spans_by_row, eligible = [], tokens.ne(self.pad_token_id)
        for special in self.special_ids:
            eligible &= tokens.ne(special)
        for row in range(b):
            available = set(eligible[row].nonzero().flatten().tolist())
            wanted = max(1, math.ceil(len(available) * self.ratio)) if available else 0
            spans, selected = [], set()
            while len(selected) < wanted:
                starts = sorted(available - selected)
                if not starts:
                    break
                start = starts[int(torch.randint(len(starts), (1,), generator=self.generator))]
                if word_ids is not None:
                    while start > 0 and word_ids[row, start - 1] == word_ids[row, start]:
                        start -= 1
                length = self.span_lower + int(torch.multinomial(
                    self.length_probabilities, 1, generator=self.generator))
                end = start
                words_masked = 0
                cursor = start
                while cursor in available and cursor not in selected and words_masked < length:
                    group_end = cursor
                    if word_ids is not None:
                        while (group_end + 1 in available and group_end + 1 not in selected
                               and word_ids[row, group_end + 1] == word_ids[row, cursor]):
                            group_end += 1
                    selected.update(range(cursor, group_end + 1))
                    end = group_end
                    words_masked += 1
                    cursor = group_end + 1
                if end < start:
                    continue
                if end - start + 1 > self.maximum:
                    raise ValueError("whole-word span exceeds max_span_length")
                spans.append((start, end))
            spans_by_row.append(spans)
        if not any(spans_by_row):
            raise ValueError("batch has no ordinary tokens eligible for span masking")
        count = max(1, max(map(len, spans_by_row)))
        width = min(self.maximum, max(1, max((e - s + 1 for spans in spans_by_row for s, e in spans), default=1)))
        original = tokens.clone()
        corrupt = tokens.clone()
        labels = torch.full_like(tokens, -100)
        starts = torch.zeros((b, count), dtype=torch.long)
        ends = torch.zeros_like(starts)
        positions = torch.zeros((b, count, width), dtype=torch.long)
        span_labels = torch.full_like(positions, -100)
        for row, spans in enumerate(spans_by_row):
            for index, (start, end) in enumerate(spans):
                starts[row, index], ends[row, index] = start, end
                draw = float(torch.rand((), generator=self.generator))
                for offset, pos in enumerate(range(start, end + 1)):
                    positions[row, index, offset] = pos
                    span_labels[row, index, offset] = original[row, pos]
                    labels[row, pos] = original[row, pos]
                    if draw < .8:
                        corrupt[row, pos] = self.mask_token_id
                    elif draw < .9:
                        corrupt[row, pos] = torch.randint(self.vocab_size, (), generator=self.generator)
        return {"tokens": corrupt, "labels": labels, "mask": tokens.ne(self.pad_token_id),
                "span_starts": starts, "span_ends": ends,
                "span_positions": positions, "span_labels": span_labels,
                "span_mask": span_labels.ne(-100)}
