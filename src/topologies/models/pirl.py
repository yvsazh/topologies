import math

import torch
from torch import nn
from torch.nn import functional as F


class PIRL(nn.Module):
    architecture_id = "pirl"
    PRESETS = {
        "tiny": {
            "input_dim": 32,
            "projection_dim": 128,
            "temperature": 0.07,
            "lambd": 0.5,
            "memory_momentum": 0.5,
            "num_negatives": 4,
        },
        "small": {
            "input_dim": 512,
            "projection_dim": 128,
            "temperature": 0.07,
            "lambd": 0.5,
            "memory_momentum": 0.5,
            "num_negatives": 4096,
        },
        "base": {
            "input_dim": 2048,
            "projection_dim": 128,
            "temperature": 0.07,
            "lambd": 0.5,
            "memory_momentum": 0.5,
            "num_negatives": 32000,
        },
    }

    def __init__(
        self,
        input_dim=2048,
        projection_dim=128,
        temperature=0.07,
        lambd=0.5,
        memory_momentum=0.5,
        num_negatives=32000,
    ):
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(projection_dim, int) or isinstance(projection_dim, bool) or projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer")
        if not isinstance(num_negatives, int) or isinstance(num_negatives, bool) or num_negatives <= 0:
            raise ValueError("num_negatives must be a positive integer")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        if not isinstance(lambd, (int, float)) or isinstance(lambd, bool) or not math.isfinite(lambd) or not 0.0 <= lambd <= 1.0:
            raise ValueError("lambd must be a finite number in [0, 1]")
        if not isinstance(memory_momentum, (int, float)) or isinstance(memory_momentum, bool) or not math.isfinite(memory_momentum) or not 0.0 <= memory_momentum <= 1.0:
            raise ValueError("memory_momentum must be a finite number in [0, 1]")
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.temperature = float(temperature)
        self.lambd = float(lambd)
        self.memory_momentum = float(memory_momentum)
        self.num_negatives = num_negatives
        self.image_projection = nn.Linear(input_dim, projection_dim)
        self.patch_projection = nn.Linear(input_dim, projection_dim)
        self.jigsaw_projection = nn.Linear(9 * projection_dim, projection_dim)

    def _validate_feature_tensor(self, features, name, dimensions):
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if features.ndim != dimensions:
            raise ValueError(f"{name} must have {dimensions} dimensions")
        if not features.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        weight = self.image_projection.weight
        if features.device != weight.device or features.dtype != weight.dtype:
            raise ValueError(f"{name} must match the model parameter device and dtype")

    def encode_image(self, image_features):
        self._validate_feature_tensor(image_features, "image_features", 2)
        if image_features.shape[0] == 0 or image_features.shape[1] != self.input_dim:
            raise ValueError("image_features must have shape [B, input_dim] with B positive")
        return self.image_projection(image_features)

    def encode_patches(self, patch_features):
        self._validate_feature_tensor(patch_features, "patch_features", 3)
        if patch_features.shape[0] == 0 or patch_features.shape[1] != 9 or patch_features.shape[2] != self.input_dim:
            raise ValueError("patch_features must contain nine patches and have shape [B, 9, input_dim]")
        projected = self.patch_projection(patch_features)
        return self.jigsaw_projection(projected.flatten(start_dim=1))

    def _validate_bank_and_ids(self, memory_bank, sample_ids, batch_size, negative_ids=None):
        if not isinstance(memory_bank, torch.Tensor) or memory_bank.ndim != 2:
            raise ValueError("memory_bank must have shape [dataset_size, projection_dim]")
        if memory_bank.shape[0] == 0 or memory_bank.shape[1] != self.projection_dim:
            raise ValueError("memory_bank must have shape [dataset_size, projection_dim]")
        if not memory_bank.is_floating_point():
            raise ValueError("memory_bank must have a floating-point dtype")
        if not isinstance(sample_ids, torch.Tensor) or sample_ids.dtype != torch.long or sample_ids.ndim != 1 or sample_ids.shape[0] != batch_size:
            raise ValueError("sample_ids must be a long tensor with shape [B]")
        if sample_ids.device != memory_bank.device:
            raise ValueError("sample_ids and memory_bank must be on the same device")
        if torch.any(sample_ids < 0) or torch.any(sample_ids >= memory_bank.shape[0]):
            raise ValueError("sample_ids must be within the memory bank range")
        if negative_ids is not None:
            if not isinstance(negative_ids, torch.Tensor) or negative_ids.dtype != torch.long or negative_ids.ndim != 2 or negative_ids.shape != (batch_size, self.num_negatives):
                raise ValueError("negative_ids must be a long tensor with shape [B, num_negatives]")
            if negative_ids.device != memory_bank.device:
                raise ValueError("negative_ids and memory_bank must be on the same device")
            if torch.any(negative_ids < 0) or torch.any(negative_ids >= memory_bank.shape[0]):
                raise ValueError("negative_ids must be within the memory bank range")
            if torch.any(negative_ids == sample_ids.unsqueeze(1)):
                raise ValueError("negative_ids must exclude each row's sample_id")

    def nce_loss(self, memory_anchors, queries, negative_embeddings):
        self._validate_feature_tensor(memory_anchors, "memory_anchors", 2)
        self._validate_feature_tensor(queries, "queries", 2)
        self._validate_feature_tensor(negative_embeddings, "negative_embeddings", 3)
        batch_size = queries.shape[0]
        if batch_size == 0 or memory_anchors.shape != queries.shape or queries.shape[1] != self.projection_dim:
            raise ValueError("memory_anchors and queries must have matching shape [B, projection_dim]")
        if negative_embeddings.shape != (batch_size, self.num_negatives, self.projection_dim):
            raise ValueError("negative_embeddings must have shape [B, num_negatives, projection_dim]")
        anchor_vectors = F.normalize(memory_anchors, dim=-1)
        query_vectors = F.normalize(queries, dim=-1)
        negative_vectors = F.normalize(negative_embeddings, dim=-1)
        positive_logits = (anchor_vectors * query_vectors).sum(dim=-1) / self.temperature
        negative_logits = torch.bmm(negative_vectors, query_vectors.unsqueeze(-1)).squeeze(-1) / self.temperature
        positive_losses = F.softplus(torch.logsumexp(negative_logits, dim=-1) - positive_logits)
        prefix = torch.logcumsumexp(negative_logits, dim=-1)
        suffix = torch.flip(torch.logcumsumexp(torch.flip(negative_logits, dims=(-1,)), dim=-1), dims=(-1,))
        negative_infinity = torch.full_like(positive_logits.unsqueeze(-1), -torch.inf)
        before = torch.cat((negative_infinity, prefix[:, :-1]), dim=-1)
        after = torch.cat((suffix[:, 1:], negative_infinity), dim=-1)
        other_negative_logits = torch.logaddexp(before, after)
        other_logits = torch.logaddexp(positive_logits.unsqueeze(-1), other_negative_logits)
        negative_losses = F.softplus(negative_logits - other_logits)
        return (positive_losses + negative_losses.sum(dim=-1)).mean()

    def loss_from_embeddings(self, image_embeddings, patch_embeddings, memory_bank, sample_ids, negative_ids):
        self._validate_feature_tensor(image_embeddings, "image_embeddings", 2)
        self._validate_feature_tensor(patch_embeddings, "patch_embeddings", 2)
        if image_embeddings.shape[0] == 0 or image_embeddings.shape != patch_embeddings.shape or image_embeddings.shape[1] != self.projection_dim:
            raise ValueError("image_embeddings and patch_embeddings must have matching shape [B, projection_dim]")
        self._validate_bank_and_ids(memory_bank, sample_ids, image_embeddings.shape[0], negative_ids)
        if memory_bank.device != image_embeddings.device or memory_bank.dtype != image_embeddings.dtype:
            raise ValueError("memory_bank must match the embedding device and dtype")
        anchors = memory_bank.index_select(0, sample_ids).detach()
        negatives = memory_bank[negative_ids].detach()
        patch_loss = self.nce_loss(anchors, patch_embeddings, negatives)
        image_loss = self.nce_loss(anchors, image_embeddings, negatives)
        return self.lambd * patch_loss + (1.0 - self.lambd) * image_loss

    def update_memory(self, memory_bank, sample_ids, image_embeddings):
        self._validate_feature_tensor(image_embeddings, "image_embeddings", 2)
        if image_embeddings.shape[1] != self.projection_dim or image_embeddings.shape[0] == 0:
            raise ValueError("image_embeddings must have shape [B, projection_dim] with B positive")
        self._validate_bank_and_ids(memory_bank, sample_ids, image_embeddings.shape[0])
        if memory_bank.device != image_embeddings.device or memory_bank.dtype != image_embeddings.dtype:
            raise ValueError("memory_bank must match the embedding device and dtype")
        if torch.unique(sample_ids).numel() != sample_ids.numel():
            raise ValueError("sample_ids must be unique when updating the memory bank")
        with torch.no_grad():
            previous = memory_bank.index_select(0, sample_ids)
            updated = self.memory_momentum * previous + (1.0 - self.memory_momentum) * image_embeddings.detach()
            memory_bank.index_copy_(0, sample_ids, updated)
        return memory_bank

    def forward(self, image_features, patch_features, memory_bank, sample_ids, negative_ids):
        if not isinstance(image_features, torch.Tensor) or not isinstance(patch_features, torch.Tensor):
            raise TypeError("image_features and patch_features must be tensors")
        if image_features.ndim != 2 or patch_features.ndim != 3:
            raise ValueError("image_features and patch_features must have shapes [B, input_dim] and [B, 9, input_dim]")
        if image_features.shape[0] != patch_features.shape[0]:
            raise ValueError("image_features and patch_features must have the same batch size")
        image_embeddings = self.encode_image(image_features)
        patch_embeddings = self.encode_patches(patch_features)
        return self.loss_from_embeddings(image_embeddings, patch_embeddings, memory_bank, sample_ids, negative_ids)
