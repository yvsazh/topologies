import torch
from torch import nn


class ABMIL(nn.Module):
    architecture_id = "abmil"
    PRESETS = {
        "tiny": {"encoder_dims": (64,), "attention_dim": 64, "gated": True},
        "small": {"encoder_dims": (256, 128, 64), "attention_dim": 64, "gated": True},
    }

    def __init__(self, input_dim, encoder_dims=(256, 128, 64), attention_dim=64, gated=True):
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(encoder_dims, (tuple, list)) or not encoder_dims:
            raise ValueError("encoder_dims must be a non-empty tuple or list of positive integers")
        if any(not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in encoder_dims):
            raise ValueError("every encoder_dims value must be a positive integer")
        if not isinstance(attention_dim, int) or isinstance(attention_dim, bool) or attention_dim < 1:
            raise ValueError("attention_dim must be a positive integer")
        if gated is not True and gated is not False:
            raise ValueError("gated must be a Boolean")

        self.input_dim = input_dim
        self.encoder_dims = tuple(encoder_dims)
        self.attention_dim = attention_dim
        self.gated = gated

        encoder_layers = []
        previous_dim = input_dim
        for width in self.encoder_dims:
            encoder_layers.extend((nn.Linear(previous_dim, width), nn.ReLU()))
            previous_dim = width
        self.encoder = nn.Sequential(*encoder_layers)
        self.attention_v = nn.Linear(previous_dim, attention_dim, bias=False)
        self.attention_u = nn.Linear(previous_dim, attention_dim, bias=False) if gated else None
        self.attention_w = nn.Linear(attention_dim, 1, bias=False)
        self.classifier = nn.Linear(previous_dim, 1)

    def pool_embeddings(self, embeddings, mask=None):
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            raise ValueError("embeddings must be a rank-3 tensor with shape [batch, instances, features]")
        if not embeddings.is_floating_point():
            raise ValueError("embeddings must have a floating-point dtype")
        if embeddings.shape[-1] != self.encoder_dims[-1]:
            raise ValueError(f"last dimension must be {self.encoder_dims[-1]}")
        batch_size, instance_count = embeddings.shape[:2]
        if instance_count < 1:
            raise ValueError("each bag must contain at least one valid instance")
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or tuple(mask.shape) != (batch_size, instance_count):
                raise ValueError("mask shape must match [batch, instances]")
            if mask.dtype != torch.bool:
                raise ValueError("mask must have a Boolean dtype")
            if mask.device != embeddings.device:
                raise ValueError("mask and embeddings must be on the same device")
            if not torch.all(mask.any(dim=1)):
                raise ValueError("each bag must have at least one valid instance")

        values = torch.tanh(self.attention_v(embeddings))
        if self.gated:
            values = values * torch.sigmoid(self.attention_u(embeddings))
        scores = self.attention_w(values).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * embeddings, dim=1)
        return pooled, weights

    def forward(self, x, mask=None, return_attention=False):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a rank-3 tensor with shape [batch, instances, input_dim]")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"last dimension must be input_dim={self.input_dim}")
        if return_attention is not True and return_attention is not False:
            raise ValueError("return_attention must be a Boolean")
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or tuple(mask.shape) != tuple(x.shape[:2]):
                raise ValueError("mask shape must match [batch, instances]")
            if mask.dtype != torch.bool:
                raise ValueError("mask must have a Boolean dtype")
            if mask.device != x.device:
                raise ValueError("mask and x must be on the same device")
            if not torch.all(mask.any(dim=1)):
                raise ValueError("each bag must have at least one valid instance")
        if x.shape[1] < 1:
            raise ValueError("each bag must contain at least one valid instance")

        batch_size, instance_count, _ = x.shape
        embeddings = self.encoder(x.reshape(batch_size * instance_count, self.input_dim))
        embeddings = embeddings.reshape(batch_size, instance_count, self.encoder_dims[-1])
        pooled, weights = self.pool_embeddings(embeddings, mask)
        probabilities = torch.sigmoid(self.classifier(pooled))
        if return_attention:
            return probabilities, weights
        return probabilities
