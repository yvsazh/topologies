import math

import torch
from torch import nn
from torch.nn import functional as F


def _rms_norm(x, weight, eps):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(x.dtype)


def _apply_rope(x, positions, theta=10000.0):
    half = x.shape[-1] // 2
    inverse_frequency = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    phase = positions.float()[..., None] * inverse_frequency
    cosine = phase.cos().to(x.dtype)[:, None]
    sine = phase.sin().to(x.dtype)[:, None]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


def _swiglu(x, gate, up, down):
    return down(F.silu(gate(x)) * up(x))


def _argument_error(message):
    return ValueError(message)


class CogVLM(nn.Module):
    architecture_id = "cogvlm"
    PRESETS = {
        "tiny": {
            "vocab_size": 64,
            "vision_feature_size": 16,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "num_layers": 1,
            "max_positions": 128,
            "rms_norm_eps": 1e-6,
        },
        "small": {
            "vocab_size": 512,
            "vision_feature_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "num_layers": 2,
            "max_positions": 512,
            "rms_norm_eps": 1e-6,
        },
        "base": {
            "vocab_size": 4096,
            "vision_feature_size": 512,
            "hidden_size": 256,
            "intermediate_size": 768,
            "num_heads": 8,
            "num_layers": 6,
            "max_positions": 2048,
            "rms_norm_eps": 1e-6,
        },
    }

    def __init__(
        self,
        vocab_size=64,
        vision_feature_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        num_layers=1,
        max_positions=128,
        rms_norm_eps=1e-6,
        ignore_index=-100,
    ):
        super().__init__()
        dimensions = (vocab_size, vision_feature_size, hidden_size, intermediate_size,
                      num_heads, num_layers, max_positions)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in dimensions):
            raise ValueError("vocabulary, feature, model, head, layer, and position sizes must be positive integers")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (hidden_size // num_heads) % 2:
            raise ValueError("hidden_size divided by num_heads must be even for RoPE")
        if (isinstance(rms_norm_eps, bool) or not isinstance(rms_norm_eps, (int, float))
                or not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0):
            raise ValueError("rms_norm_eps must be a finite positive number")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise _argument_error("ignore_index must be an integer")

        self.vocab_size = vocab_size
        self.vision_feature_size = vision_feature_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_positions = max_positions
        self.rms_norm_eps = float(rms_norm_eps)
        self.ignore_index = ignore_index
        self.head_size = hidden_size // num_heads

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.vision_projection = nn.Linear(vision_feature_size, hidden_size, bias=False)
        self.vision_layer_norm = nn.LayerNorm(hidden_size)
        self.vision_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.vision_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.vision_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.layers = nn.ModuleList()
        self.attention_norm_weights = nn.ParameterList()
        self.ffn_norm_weights = nn.ParameterList()
        for _ in range(num_layers):
            block = nn.ModuleDict(
                {
                    "vision_qkv": nn.Linear(hidden_size, 3 * hidden_size, bias=False),
                    "text_qkv": nn.Linear(hidden_size, 3 * hidden_size, bias=False),
                    "vision_out": nn.Linear(hidden_size, hidden_size, bias=False),
                    "text_out": nn.Linear(hidden_size, hidden_size, bias=False),
                    "vision_gate": nn.Linear(hidden_size, intermediate_size, bias=False),
                    "vision_up": nn.Linear(hidden_size, intermediate_size, bias=False),
                    "vision_down": nn.Linear(intermediate_size, hidden_size, bias=False),
                    "text_gate": nn.Linear(hidden_size, intermediate_size, bias=False),
                    "text_up": nn.Linear(hidden_size, intermediate_size, bias=False),
                    "text_down": nn.Linear(intermediate_size, hidden_size, bias=False),
                }
            )
            with torch.no_grad():
                for suffix in ("qkv", "out", "gate", "up", "down"):
                    block[f"vision_{suffix}"].weight.copy_(block[f"text_{suffix}"].weight)
            self.layers.append(block)
            self.attention_norm_weights.append(nn.Parameter(torch.ones(hidden_size)))
            self.ffn_norm_weights.append(nn.Parameter(torch.ones(hidden_size)))
        self.final_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _adapt_vision(self, vision_features):
        hidden = F.gelu(self.vision_layer_norm(self.vision_projection(vision_features)))
        return _swiglu(hidden, self.vision_gate, self.vision_up, self.vision_down)

    def _attention(self, hidden, layer_index, vision_length, position_ids=None):
        batch_size, sequence_length, _ = hidden.shape
        block = self.layers[layer_index]
        vision_hidden = hidden[:, :vision_length]
        text_hidden = hidden[:, vision_length:]
        vision_qkv = block["vision_qkv"](vision_hidden)
        text_qkv = block["text_qkv"](text_hidden)
        qkv = torch.cat((vision_qkv, text_qkv), dim=1)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.reshape(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)
        key = key.reshape(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)
        value = value.reshape(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=hidden.device).expand(batch_size, -1)
            if vision_length:
                position_ids = torch.cat(
                    (torch.zeros((batch_size, vision_length), device=hidden.device, dtype=torch.long),
                     position_ids[:, vision_length:]),
                    dim=1,
                )
        query = _apply_rope(query, position_ids)
        key = _apply_rope(key, position_ids)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        causal_mask = torch.triu(
            torch.ones((sequence_length, sequence_length), device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        probabilities = F.softmax(scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min), dim=-1)
        attended = torch.matmul(probabilities, value).transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        vision_output = block["vision_out"](attended[:, :vision_length])
        text_output = block["text_out"](attended[:, vision_length:])
        return torch.cat((vision_output, text_output), dim=1)

    def _ffn(self, hidden, block, vision_length):
        vision_hidden = hidden[:, :vision_length]
        text_hidden = hidden[:, vision_length:]
        vision_output = _swiglu(
            vision_hidden, block["vision_gate"], block["vision_up"], block["vision_down"]
        )
        text_output = _swiglu(text_hidden, block["text_gate"], block["text_up"], block["text_down"])
        return torch.cat((vision_output, text_output), dim=1)

    @staticmethod
    def _causal_loss(logits, labels, ignore_index=-100):
        if (not torch.is_tensor(logits) or logits.ndim != 3 or not logits.is_floating_point()
                or not torch.is_tensor(labels) or labels.shape != logits.shape[:2]):
            raise ValueError("logits and labels must have shapes [batch, sequence, vocabulary] and [batch, sequence]")
        if labels.dtype != torch.long:
            raise ValueError("labels must be a LongTensor")
        if labels.device != logits.device:
            raise ValueError("labels and logits must be on the same device")
        valid_labels = labels[labels != ignore_index]
        if valid_labels.numel() and (
            int(valid_labels.min()) < 0 or int(valid_labels.max()) >= logits.shape[-1]
        ):
            raise ValueError("labels must contain valid token IDs or ignore_index")
        shifted_labels = labels[:, 1:]
        if shifted_labels.numel() == 0 or not bool((shifted_labels != ignore_index).any()):
            raise ValueError("labels must contain at least one next-token target")
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=ignore_index,
        )

    def forward(self, input_ids, vision_features=None, labels=None):
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a LongTensor with shape [batch, text_sequence]")
        batch_size, text_length = input_ids.shape
        if vision_features is None:
            vision_length = 0
            vision_hidden = None
        else:
            if (not torch.is_tensor(vision_features) or vision_features.ndim != 3
                    or vision_features.shape[0] != batch_size
                    or vision_features.shape[-1] != self.vision_feature_size
                    or vision_features.shape[1] == 0 or not vision_features.is_floating_point()):
                raise ValueError("vision_features must be a nonempty floating tensor [batch, patches, vision_feature_size]")
            if vision_features.device != input_ids.device:
                raise ValueError("vision_features and input_ids must be on the same device")
            vision_length = vision_features.shape[1]
            vision_hidden = self._adapt_vision(vision_features)
        if input_ids.numel() and (int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size):
            raise ValueError("input_ids must contain token IDs in [0, vocab_size)")
        sequence_length = vision_length + text_length
        if sequence_length == 0:
            raise ValueError("at least one text token or visual feature is required")
        if sequence_length > self.max_positions:
            raise ValueError("combined text and vision sequence exceeds max_positions")
        text_hidden = self.token_embedding(input_ids)
        hidden = text_hidden if vision_hidden is None else torch.cat((vision_hidden, text_hidden), dim=1)
        position_ids = torch.arange(sequence_length, device=input_ids.device).expand(batch_size, -1)
        if vision_length:
            position_ids = torch.cat(
                (torch.zeros((batch_size, vision_length), device=input_ids.device, dtype=torch.long),
                 position_ids[:, vision_length:]),
                dim=1,
            )
        for index, block in enumerate(self.layers):
            normalized = _rms_norm(hidden, self.attention_norm_weights[index], self.rms_norm_eps)
            hidden = hidden + self._attention(normalized, index, vision_length, position_ids)
            normalized = _rms_norm(hidden, self.ffn_norm_weights[index], self.rms_norm_eps)
            hidden = hidden + self._ffn(normalized, block, vision_length)
        logits = self.lm_head(_rms_norm(hidden, self.final_norm_weight, self.rms_norm_eps))
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self._causal_loss(logits, labels, self.ignore_index)
        return output
