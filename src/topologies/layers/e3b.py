import math

import torch
from torch import nn


class E3B(nn.Module):
    component_id = "e3b"

    def __init__(self, features=256, ridge=0.1):
        super().__init__()
        if not isinstance(features, int) or isinstance(features, bool) or features <= 0:
            raise ValueError("features must be a positive integer")
        if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or ridge <= 0:
            raise ValueError("ridge must be a positive finite number")
        try:
            ridge = float(ridge)
        except OverflowError:
            ridge = math.inf
        if not math.isfinite(ridge):
            raise ValueError("ridge must be a positive finite number")
        self.features = features
        self.ridge = ridge

    def forward(self, embedding, stored_embeddings):
        if not isinstance(embedding, torch.Tensor) or embedding.ndim not in (1, 2):
            raise ValueError("embedding must have shape [features] or [batch, features]")
        if embedding.shape[-1] != self.features:
            raise ValueError(f"embedding must have {self.features} features")
        valid_history_dimensions = (2,) if embedding.ndim == 1 else (2, 3)
        if not isinstance(stored_embeddings, torch.Tensor) \
                or stored_embeddings.ndim not in valid_history_dimensions:
            raise ValueError("stored_embeddings must have shape [N, features] or [batch, N, features]")
        if stored_embeddings.shape[-1] != self.features:
            raise ValueError(f"stored_embeddings must have {self.features} features")
        if embedding.ndim == 2 and stored_embeddings.ndim == 3 \
                and stored_embeddings.shape[0] != embedding.shape[0]:
            raise ValueError("batched stored_embeddings must match the embedding batch size")
        if not embedding.is_floating_point() or not stored_embeddings.is_floating_point():
            raise ValueError("embedding and stored_embeddings must have floating-point dtype")
        if embedding.device != stored_embeddings.device:
            raise ValueError("embedding and stored_embeddings must be on the same device")
        if embedding.dtype != stored_embeddings.dtype:
            raise ValueError("embedding and stored_embeddings must have the same dtype")

        output_dtype = embedding.dtype
        if output_dtype in (torch.float16, torch.bfloat16):
            embedding = embedding.float()
            stored_embeddings = stored_embeddings.float()
        identity = torch.eye(self.features, dtype=embedding.dtype, device=embedding.device)
        if embedding.ndim == 1:
            covariance = self.ridge * identity + stored_embeddings.transpose(0, 1) @ stored_embeddings
            solution = torch.linalg.solve(covariance, embedding)
            return torch.dot(embedding, solution).to(dtype=output_dtype)
        if stored_embeddings.ndim == 2:
            covariance = self.ridge * identity + stored_embeddings.transpose(0, 1) @ stored_embeddings
            solution = torch.linalg.solve(covariance, embedding.transpose(0, 1)).transpose(0, 1)
        else:
            covariance = self.ridge * identity.unsqueeze(0) \
                + stored_embeddings.transpose(1, 2) @ stored_embeddings
            solution = torch.linalg.solve(covariance, embedding.unsqueeze(-1)).squeeze(-1)
        return (embedding * solution).sum(dim=-1, keepdim=True).to(dtype=output_dtype)
