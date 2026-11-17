import math
from numbers import Real

import torch
from torch import nn


def _is_finite_real(value):
    try:
        return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def valid_curvature(value):
    try:
        return math.isfinite(value) and value > 0
    except (TypeError, ValueError, OverflowError):
        return False


class RotH(nn.Module):
    component_id = "roth"
    PRESETS = {
        "tiny": {"num_entities": 8, "num_relations": 3, "embedding_dim": 2},
        "small": {"num_entities": 64, "num_relations": 8, "embedding_dim": 16},
        "base": {"num_entities": 256, "num_relations": 32, "embedding_dim": 32},
    }

    def __init__(
        self,
        curvature=1.0,
        *,
        num_entities=8,
        num_relations=3,
        embedding_dim=4,
        init_size=0.001,
        curvature_raw_init=1.0,
    ):
        super().__init__()
        self.curvature = curvature
        for name, value in (
            ("num_entities", num_entities),
            ("num_relations", num_relations),
            ("embedding_dim", embedding_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if embedding_dim % 2:
            raise ValueError("embedding_dim must be even for pairwise Givens rotations")
        if not _is_finite_real(init_size) or init_size < 0:
            raise ValueError("init_size must be a finite non-negative number")
        if not _is_finite_real(curvature_raw_init):
            raise ValueError("curvature_raw_init must be a finite number")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Parameter(torch.empty(num_entities, embedding_dim))
        self.relation_embeddings = nn.Parameter(torch.empty(num_relations, embedding_dim))
        self.rotation = nn.Parameter(torch.empty(num_relations, embedding_dim))
        self.raw_curvature = nn.Parameter(torch.empty(num_relations, 1))
        self.head_bias = nn.Parameter(torch.empty(num_entities, 1))
        self.tail_bias = nn.Parameter(torch.empty(num_entities, 1))
        nn.init.normal_(self.entity_embeddings, mean=0.0, std=init_size)
        nn.init.normal_(self.relation_embeddings, mean=0.0, std=init_size)
        nn.init.uniform_(self.rotation, -1.0, 1.0)
        nn.init.constant_(self.raw_curvature, curvature_raw_init)
        nn.init.zeros_(self.head_bias)
        nn.init.zeros_(self.tail_bias)

    @property
    def curvature(self):
        return self._curvature

    @curvature.setter
    def curvature(self, value):
        if not valid_curvature(value):
            raise ValueError("curvature magnitude must be finite and greater than zero")
        self._curvature = value

    @staticmethod
    def valid_curvature(value):
        return valid_curvature(value)

    def set_curvature(self, value):
        self.curvature = value
        return self.curvature

    @staticmethod
    def _expmap0(vector, curvature):
        sqrt_curvature = torch.sqrt(curvature)
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        scaled_norm = sqrt_curvature * norm
        denominator = scaled_norm.clamp_min(torch.finfo(vector.dtype).tiny)
        scale = torch.where(scaled_norm > 0, torch.tanh(scaled_norm) / denominator, 1.0)
        return scale * vector

    @staticmethod
    def _mobius_add(left, right, curvature):
        left_norm_sq = left.square().sum(dim=-1, keepdim=True)
        right_norm_sq = right.square().sum(dim=-1, keepdim=True)
        dot = (left * right).sum(dim=-1, keepdim=True)
        alpha = 1.0 + 2.0 * curvature * dot + curvature * right_norm_sq
        beta = 1.0 - curvature * left_norm_sq
        denominator = 1.0 + 2.0 * curvature * dot + curvature.square() * left_norm_sq * right_norm_sq
        denominator = denominator.clamp_min(torch.finfo(left.dtype).tiny)
        return (alpha * left + beta * right) / denominator

    def forward(self, triples):
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if not isinstance(triples, torch.Tensor) or triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("triples must be an integer tensor with shape [batch, 3]")
        if triples.dtype not in integer_dtypes:
            raise ValueError("triples must use an integer dtype")
        if triples.shape[0] == 0:
            raise ValueError("triples batch must contain at least one triple")
        triples = triples.to(device=self.entity_embeddings.device, dtype=torch.long)
        heads, relations, tails = triples.unbind(dim=1)
        if (
            torch.any(heads < 0)
            or torch.any(heads >= self.num_entities)
            or torch.any(tails < 0)
            or torch.any(tails >= self.num_entities)
        ):
            raise ValueError("head and tail IDs must be within the configured entity range")
        if torch.any(relations < 0) or torch.any(relations >= self.num_relations):
            raise ValueError("relation IDs must be within the configured relation range")

        curvature = torch.nn.functional.softplus(self.raw_curvature[relations]).clamp_min(
            torch.finfo(self.entity_embeddings.dtype).tiny
        )
        head = self._expmap0(self.entity_embeddings[heads], curvature)
        tail = self._expmap0(self.entity_embeddings[tails], curvature)
        pair = self.rotation[relations].reshape(-1, self.embedding_dim // 2, 2)
        pair_norm = torch.linalg.vector_norm(pair, dim=-1, keepdim=True)
        pair = pair / pair_norm.clamp_min(torch.finfo(pair.dtype).tiny)
        cosine = pair[..., 0]
        sine = pair[..., 1]
        head_pairs = head.reshape(-1, self.embedding_dim // 2, 2)
        rotated = torch.stack(
            (
                cosine * head_pairs[..., 0] - sine * head_pairs[..., 1],
                sine * head_pairs[..., 0] + cosine * head_pairs[..., 1],
            ),
            dim=-1,
        ).reshape(-1, self.embedding_dim)
        relation = self._expmap0(self.relation_embeddings[relations], curvature)
        query = self._mobius_add(rotated, relation, curvature)
        difference = self._mobius_add(-query, tail, curvature)
        sqrt_curvature = torch.sqrt(curvature)
        distance_argument = sqrt_curvature * torch.linalg.vector_norm(
            difference, dim=-1, keepdim=True
        )
        distance_argument = distance_argument.clamp(max=1.0 - torch.finfo(difference.dtype).eps)
        distance = 2.0 / sqrt_curvature * torch.atanh(distance_argument)
        score = -distance.squeeze(-1).square()
        return (
            score
            + self.head_bias[heads].squeeze(-1)
            + self.tail_bias[tails].squeeze(-1)
        )
