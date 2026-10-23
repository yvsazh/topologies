import math

import torch
import torch.nn.functional as F
from torch import nn


class SeasonalContrast(nn.Module):
    component_id = "seco"

    def __init__(self, feature_dim, hidden_dim=None, embedding_dim=128, queue_size=16384,
                 temperature=0.07, encoder_momentum=0.999, per_example_output=False):
        super().__init__()
        dimensions = {
            "feature_dim": feature_dim,
            "hidden_dim": feature_dim if hidden_dim is None else hidden_dim,
            "embedding_dim": embedding_dim,
            "queue_size": queue_size,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
                or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        if isinstance(encoder_momentum, bool) or not isinstance(encoder_momentum, (int, float)) \
                or not math.isfinite(encoder_momentum) or not 0 <= encoder_momentum < 1:
            raise ValueError("encoder_momentum must be in [0, 1)")
        if per_example_output is not True and per_example_output is not False:
            raise ValueError("per_example_output must be a boolean")

        self.feature_dim = feature_dim
        self.hidden_dim = dimensions["hidden_dim"]
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.temperature = float(temperature)
        self.encoder_momentum = float(encoder_momentum)
        self.per_example_output = per_example_output
        self.query_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, embedding_dim, bias=False),
            )
            for _ in range(3)
        ])
        self.key_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, embedding_dim, bias=False),
            )
            for _ in range(3)
        ])
        self.key_heads.load_state_dict(self.query_heads.state_dict())
        for parameter in self.key_heads.parameters():
            parameter.requires_grad_(False)
        queue = F.normalize(torch.randn(3, embedding_dim, queue_size), dim=1)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(3, dtype=torch.long))
        self.logits = ()
        self.losses = torch.empty(0)
        self.query_projected = torch.empty(0)
        self.key_projected = torch.empty(0)

    def _update_key_heads(self):
        with torch.no_grad():
            for query_head, key_head in zip(self.query_heads, self.key_heads):
                for query_parameter, key_parameter in zip(query_head.parameters(), key_head.parameters()):
                    key_parameter.mul_(self.encoder_momentum).add_(
                        query_parameter, alpha=1 - self.encoder_momentum
                    )

    def forward(self, query_features, key_features):
        if not torch.is_tensor(query_features) or query_features.ndim != 2:
            raise ValueError("query_features must have shape [B, D]")
        if not torch.is_tensor(key_features) or key_features.ndim != 3 \
                or key_features.shape[1] != 3:
            raise ValueError("key_features must have shape [B, 3, D]")
        if query_features.shape[0] != key_features.shape[0]:
            raise ValueError("query and key batch sizes must match")
        if query_features.shape[1] != self.feature_dim or key_features.shape[2] != self.feature_dim:
            raise ValueError("query and key feature dimensions must match feature_dim")
        if query_features.shape[0] <= 0:
            raise ValueError("batch size must be positive")
        if not query_features.is_floating_point() or not key_features.is_floating_point():
            raise ValueError("query_features and key_features must be floating point tensors")
        if query_features.device != key_features.device or query_features.dtype != key_features.dtype:
            raise ValueError("query_features and key_features must share device and dtype")
        batch_size = query_features.shape[0]
        if self.queue_size % batch_size:
            raise ValueError("queue_size must be divisible by batch size")

        if self.training:
            self._update_key_heads()
        query_projected = torch.stack([
            F.normalize(head(query_features), dim=-1)
            for head in self.query_heads
        ], dim=1)
        with torch.no_grad():
            key_projected = torch.stack([
                F.normalize(head(key_features.reshape(-1, self.feature_dim)), dim=-1)
                .reshape(batch_size, 3, self.embedding_dim)
                for head in self.key_heads
            ], dim=1)

        queue_snapshot = self.queue.detach().clone()
        logits = []
        losses = []
        per_example_losses = []
        targets = torch.zeros(batch_size, dtype=torch.long, device=query_features.device)
        for space in range(3):
            query = query_projected[:, space]
            positive = (query * key_projected[:, space, space]).sum(dim=-1, keepdim=True)
            negative_queue = query @ queue_snapshot[space]
            logit_parts = [positive / self.temperature, negative_queue / self.temperature]
            if space:
                hard_views = [view for view in range(3) if view != space]
                hard_negatives = torch.stack([
                    (query * key_projected[:, space, view]).sum(dim=-1)
                    for view in hard_views
                ], dim=-1)
                logit_parts.append(hard_negatives / self.temperature)
            space_logits = torch.cat(logit_parts, dim=1)
            logits.append(space_logits)
            space_loss = F.cross_entropy(space_logits, targets, reduction="none")
            per_example_losses.append(space_loss)
            losses.append(space_loss.mean())

        per_example_loss = torch.stack(per_example_losses).sum(dim=0)
        loss = per_example_loss.mean()
        self.loss = loss.detach()
        self.logits = tuple(value.detach() for value in logits)
        self.losses = torch.stack(losses).detach()
        self.query_projected = query_projected.detach()
        self.key_projected = key_projected.detach()

        if self.training:
            with torch.no_grad():
                for space in range(3):
                    pointer = int(self.queue_ptr[space].item())
                    self.queue[space, :, pointer:pointer + batch_size] = \
                        key_projected[:, space, space].T
                    self.queue_ptr[space] = (pointer + batch_size) % self.queue_size

        return per_example_loss.unsqueeze(-1) if self.per_example_output else loss
