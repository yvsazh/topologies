import math

import torch
from torch import nn
from torch.nn import functional as F


class VisualBERT(nn.Module):
    architecture_id = "visualbert"
    PRESETS = {
        "tiny": {
            "vocab_size": 30522,
            "visual_dim": 512,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 64,
            "max_positions": 32,
            "type_vocab_size": 2,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "layer_norm_eps": 1e-12,
        },
        "small": {
            "vocab_size": 30522,
            "visual_dim": 512,
            "hidden_size": 128,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 512,
            "max_positions": 128,
            "type_vocab_size": 2,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "layer_norm_eps": 1e-12,
        },
        "base": {
            "vocab_size": 30522,
            "visual_dim": 512,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
            "max_positions": 512,
            "type_vocab_size": 2,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "layer_norm_eps": 1e-12,
        },
    }

    @staticmethod
    def _value_error(message):
        return ValueError(message)

    def __init__(
        self,
        vocab_size=30522,
        visual_dim=512,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_positions=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        positive = {
            "vocab_size": vocab_size,
            "visual_dim": visual_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "max_positions": max_positions,
            "type_vocab_size": type_vocab_size,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if type_vocab_size < 2:
            raise ValueError("type_vocab_size must be at least 2")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        for name, value in (
            ("hidden_dropout_prob", hidden_dropout_prob),
            ("attention_probs_dropout_prob", attention_probs_dropout_prob),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise self._value_error(f"{name} must be a finite number in [0, 1)")
            if not math.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be a finite number in [0, 1)")
        if isinstance(layer_norm_eps, bool) or not isinstance(layer_norm_eps, (int, float)):
            raise self._value_error("layer_norm_eps must be a finite positive number")
        if not math.isfinite(layer_norm_eps) or layer_norm_eps <= 0.0:
            raise ValueError("layer_norm_eps must be a finite positive number")

        self.vocab_size = vocab_size
        self.visual_dim = visual_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_positions = max_positions
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = float(hidden_dropout_prob)
        self.attention_probs_dropout_prob = float(attention_probs_dropout_prob)
        self.layer_norm_eps = float(layer_norm_eps)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_positions, hidden_size)
        self.visual_projection = nn.Linear(visual_dim, hidden_size)
        self.visual_segment_embeddings = nn.Embedding(1, hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.embedding_dropout = nn.Dropout(hidden_dropout_prob)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "query": nn.Linear(hidden_size, hidden_size),
                "key": nn.Linear(hidden_size, hidden_size),
                "value": nn.Linear(hidden_size, hidden_size),
                "attention_output": nn.Linear(hidden_size, hidden_size),
                "attention_norm": nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                "attention_dropout": nn.Dropout(hidden_dropout_prob),
                "probability_dropout": nn.Dropout(attention_probs_dropout_prob),
                "intermediate": nn.Linear(hidden_size, intermediate_size),
                "output": nn.Linear(intermediate_size, hidden_size),
                "output_norm": nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                "output_dropout": nn.Dropout(hidden_dropout_prob),
            }))
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.mlm_transform = nn.Linear(hidden_size, hidden_size)
        self.mlm_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlm_decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.sentence_image_head = nn.Linear(hidden_size, 2)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _integer_tensor(tensor):
        return isinstance(tensor, torch.Tensor) and tensor.dtype in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        )

    def _validate_mask(self, mask, shape, name, device):
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if mask.device != device:
            raise ValueError(f"{name} must be on the same device as input_ids")
        if mask.dtype != torch.bool:
            if mask.is_floating_point() and not torch.isfinite(mask).all():
                raise ValueError(f"{name} must contain only zero and one")
            if torch.any((mask != 0) & (mask != 1)):
                raise ValueError(f"{name} must contain only zero and one")
        return mask.bool()

    def _embed_inputs(
        self,
        input_ids,
        visual_features,
        *,
        token_type_ids=None,
        image_text_alignment=None,
        text_attention_mask=None,
        visual_attention_mask=None,
    ):
        if not self._integer_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-two integer tensor")
        batch_size, text_length = input_ids.shape
        if batch_size <= 0 or text_length <= 0:
            raise ValueError("input_ids must contain a nonempty batch and text sequence")
        if not isinstance(visual_features, torch.Tensor) or visual_features.ndim != 3:
            raise ValueError("visual_features must be a rank-three tensor")
        if visual_features.shape[0] != batch_size:
            raise ValueError("input_ids and visual_features batch dimensions must match")
        if visual_features.shape[2] != self.visual_dim:
            raise ValueError(f"visual_features last dimension must equal visual_dim={self.visual_dim}")
        if not visual_features.is_floating_point():
            raise ValueError("visual_features must have a floating-point dtype")
        if visual_features.device != input_ids.device:
            raise ValueError("input_ids and visual_features must be on the same device")
        region_count = visual_features.shape[1]
        if text_length + region_count > self.max_positions:
            raise ValueError("combined text and visual sequence exceeds max_positions")
        input_ids = input_ids.long()
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            raise ValueError("input_ids contains token IDs outside the configured vocabulary")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        elif not self._integer_tensor(token_type_ids) or tuple(token_type_ids.shape) != (
            batch_size, text_length
        ):
            raise ValueError("token_type_ids must be an integer tensor with shape (B, T)")
        elif token_type_ids.device != input_ids.device:
            raise ValueError("token_type_ids must be on the same device as input_ids")
        token_type_ids = token_type_ids.long()
        if torch.any(token_type_ids < 0) or torch.any(token_type_ids >= self.type_vocab_size):
            raise ValueError("token_type_ids contains IDs outside the configured type vocabulary")
        if image_text_alignment is not None:
            if not self._integer_tensor(image_text_alignment) or image_text_alignment.ndim != 3:
                raise ValueError("image_text_alignment must be a rank-three integer tensor")
            if tuple(image_text_alignment.shape[:2]) != (batch_size, region_count):
                raise ValueError("image_text_alignment must have shape (B, R, A)")
            if image_text_alignment.device != input_ids.device:
                raise ValueError("image_text_alignment must be on the same device as input_ids")
            invalid = (image_text_alignment < -1) | (image_text_alignment >= text_length)
            if torch.any(invalid):
                raise ValueError("alignment positions must be -1 or valid text position indices")
        text_mask = self._validate_mask(
            text_attention_mask, (batch_size, text_length), "text_attention_mask", input_ids.device
        )
        visual_mask = self._validate_mask(
            visual_attention_mask,
            (batch_size, region_count),
            "visual_attention_mask",
            input_ids.device,
        )
        text_positions = torch.arange(text_length, device=input_ids.device)
        text = self.token_embeddings(input_ids)
        text = text + self.type_embeddings(token_type_ids)
        text = text + self.position_embeddings(text_positions).unsqueeze(0)
        visual = self.visual_projection(visual_features)
        visual = visual + self.visual_segment_embeddings.weight[0].view(1, 1, -1)
        if image_text_alignment is not None:
            alignment_mask = image_text_alignment >= 0
            alignment_positions = image_text_alignment.clamp_min(0).long()
            aligned_positions = self.position_embeddings(alignment_positions)
            visual = visual + (aligned_positions * alignment_mask.unsqueeze(-1)).sum(dim=2)
        embedded = torch.cat((text, visual), dim=1)
        return embedded, torch.cat((text_mask, visual_mask), dim=1)

    def _run_layer(self, hidden, key_mask, layer_index):
        layer = self.layers[layer_index]
        batch_size, sequence_length, _ = hidden.shape
        head_size = self.hidden_size // self.num_heads

        def split_heads(projection):
            return projection(hidden).view(
                batch_size, sequence_length, self.num_heads, head_size
            ).transpose(1, 2)

        query = split_heads(layer["query"])
        key = split_heads(layer["key"])
        value = split_heads(layer["value"])
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_size)
        scores = scores.masked_fill(~key_mask[:, None, None, :], -10000.0)
        probabilities = F.softmax(scores, dim=-1)
        attended = torch.matmul(layer["probability_dropout"](probabilities), value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.hidden_size
        )
        attended = layer["attention_output"](attended)
        hidden = layer["attention_norm"](hidden + layer["attention_dropout"](attended))
        feed_forward = layer["intermediate"](hidden)
        feed_forward = F.gelu(feed_forward)
        feed_forward = layer["output"](feed_forward)
        hidden = layer["output_norm"](hidden + layer["output_dropout"](feed_forward))
        return hidden, probabilities

    def forward(
        self,
        input_ids,
        visual_features,
        *,
        token_type_ids=None,
        image_text_alignment=None,
        text_attention_mask=None,
        visual_attention_mask=None,
        mlm_labels=None,
        sentence_image_labels=None,
    ):
        hidden, key_mask = self._embed_inputs(
            input_ids,
            visual_features,
            token_type_ids=token_type_ids,
            image_text_alignment=image_text_alignment,
            text_attention_mask=text_attention_mask,
            visual_attention_mask=visual_attention_mask,
        )
        hidden = self.embedding_dropout(self.embedding_layer_norm(hidden))
        attention_weights = []
        for index in range(self.num_layers):
            hidden, weights = self._run_layer(hidden, key_mask, index)
            attention_weights.append(weights)
        pooled = torch.tanh(self.pooler(hidden[:, 0]))
        mlm_hidden = F.gelu(self.mlm_transform(hidden))
        mlm_hidden = self.mlm_layer_norm(mlm_hidden)
        mlm_logits = F.linear(mlm_hidden, self.token_embeddings.weight, self.mlm_decoder_bias)
        sentence_image_logits = self.sentence_image_head(pooled)
        outputs = {
            "sequence_output": hidden,
            "pooled_output": pooled,
            "mlm_logits": mlm_logits,
            "sentence_image_logits": sentence_image_logits,
            "attention_weights": tuple(attention_weights),
        }
        losses = []
        if mlm_labels is not None:
            if not self._integer_tensor(mlm_labels) or tuple(mlm_labels.shape) != tuple(
                mlm_logits.shape[:2]
            ):
                raise ValueError("mlm_labels must be an integer tensor with shape (B, T+R)")
            labels = mlm_labels.to(device=mlm_logits.device, dtype=torch.long)
            invalid = (labels != -1) & ((labels < 0) | (labels >= self.vocab_size))
            if torch.any(invalid):
                raise ValueError("mlm_labels must be -1 or valid token IDs")
            valid_labels = labels != -1
            if torch.any(valid_labels):
                mlm_loss = F.cross_entropy(
                    mlm_logits.reshape(-1, self.vocab_size), labels.reshape(-1), ignore_index=-1
                )
            else:
                mlm_loss = mlm_logits.sum() * 0.0
            outputs["mlm_loss"] = mlm_loss
            losses.append(mlm_loss)
        if sentence_image_labels is not None:
            if not self._integer_tensor(sentence_image_labels) or tuple(
                sentence_image_labels.shape
            ) != (input_ids.shape[0],):
                raise ValueError("sentence_image_labels must be an integer tensor with shape (B,)")
            labels = sentence_image_labels.to(device=sentence_image_logits.device, dtype=torch.long)
            if torch.any(labels < 0) or torch.any(labels >= 2):
                raise ValueError("sentence_image_labels must contain class IDs 0 or 1")
            sentence_image_loss = F.cross_entropy(sentence_image_logits, labels)
            outputs["sentence_image_loss"] = sentence_image_loss
            losses.append(sentence_image_loss)
        if losses:
            outputs["loss"] = sum(losses[1:], losses[0])
        return outputs
