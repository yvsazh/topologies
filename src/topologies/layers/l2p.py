import math

import torch
from torch import nn
from torch.nn import functional as F


class _L2POutput(torch.Tensor):
    def __iter__(self):
        yield self.prompt_embeddings
        yield self.indices
        yield self.pull_loss


class L2P(nn.Module):
    component_id = "l2p"

    def __init__(self, pool_size, top_k, prompt_length, embed_dim, key_dim, pull_weight=0.5):
        super().__init__()
        dimensions = {
            "pool_size": pool_size,
            "top_k": top_k,
            "prompt_length": prompt_length,
            "embed_dim": embed_dim,
            "key_dim": key_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if top_k > pool_size:
            raise ValueError("top_k must not exceed pool_size")
        valid_pull_weight = isinstance(pull_weight, (int, float)) and not isinstance(
            pull_weight, bool
        )
        if valid_pull_weight:
            try:
                valid_pull_weight = math.isfinite(float(pull_weight)) and pull_weight >= 0
            except OverflowError:
                valid_pull_weight = False
        if not valid_pull_weight:
            raise ValueError("pull_weight must be a finite nonnegative number")
        self.pool_size = pool_size
        self.top_k = top_k
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.pull_weight = float(pull_weight)
        self.keys = nn.Parameter(torch.empty(pool_size, key_dim))
        self.prompts = nn.Parameter(torch.empty(pool_size, prompt_length, embed_dim))
        nn.init.normal_(self.keys)
        nn.init.normal_(self.prompts)

    def _validate_query(self, query):
        if not isinstance(query, torch.Tensor) or query.ndim != 2:
            raise ValueError("query must be a rank 2 tensor with shape (batch, key_dim)")
        if query.shape[0] <= 0:
            raise ValueError("query batch dimension must be positive")
        if query.shape[1] != self.key_dim:
            raise ValueError(f"query key_dim must match configured key_dim {self.key_dim}")
        if not query.is_floating_point():
            raise ValueError("query must have a floating point dtype")
        if query.device != self.keys.device:
            raise ValueError("query and keys must be on the same device")
        if query.dtype != self.keys.dtype:
            raise ValueError("query and keys must have the same dtype")

    def select_indices(self, query):
        self._validate_query(query)
        query_unit = F.normalize(query, p=2, dim=1, eps=1e-8)
        keys_unit = F.normalize(self.keys, p=2, dim=1, eps=1e-8)
        similarities = query_unit @ keys_unit.transpose(0, 1)
        return similarities.topk(self.top_k, dim=1, largest=True, sorted=True).indices

    def key_pull_loss(self, query, indices):
        self._validate_query(query)
        if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
            raise ValueError("indices must be a rank 2 tensor with shape (batch, top_k)")
        if indices.shape != (query.shape[0], self.top_k):
            raise ValueError("indices shape must match (batch, top_k)")
        if indices.dtype != torch.long:
            raise ValueError("indices must have dtype torch.long")
        if indices.device != self.keys.device:
            raise ValueError("indices and keys must be on the same device")
        if bool(((indices < 0) | (indices >= self.pool_size)).any()):
            raise ValueError("indices must refer to entries in the prompt pool")
        selected_keys = self.keys[indices]
        similarities = F.cosine_similarity(
            query.unsqueeze(1), selected_keys, dim=-1, eps=1e-8
        )
        return self.pull_weight * (1.0 - similarities).sum(dim=1)

    def forward(self, query, token_embeddings):
        self._validate_query(query)
        if not isinstance(token_embeddings, torch.Tensor) or token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings must be a rank 3 tensor with shape (batch, sequence, embed_dim)"
            )
        if token_embeddings.shape[0] != query.shape[0]:
            raise ValueError("query and token_embeddings batch dimensions must match")
        if token_embeddings.shape[1] <= 0:
            raise ValueError("token_embeddings sequence dimension must be positive")
        if token_embeddings.shape[2] != self.embed_dim:
            raise ValueError(
                f"token_embeddings embed_dim must match configured embed_dim {self.embed_dim}"
            )
        if not token_embeddings.is_floating_point():
            raise ValueError("token_embeddings must have a floating point dtype")
        if token_embeddings.device != self.prompts.device:
            raise ValueError("token_embeddings and prompts must be on the same device")
        if token_embeddings.dtype != self.prompts.dtype:
            raise ValueError("token_embeddings and prompts must have the same dtype")
        indices = self.select_indices(query)
        selected_prompts = self.prompts[indices].reshape(
            query.shape[0], self.top_k * self.prompt_length, self.embed_dim
        )
        prefixed_embeddings = torch.cat((selected_prompts, token_embeddings), dim=1)
        pull_loss = self.key_pull_loss(query, indices)
        tensor_output = prefixed_embeddings + (pull_loss - pull_loss.detach()).reshape(-1, 1, 1)
        output = tensor_output.as_subclass(_L2POutput)
        output.prompt_embeddings = prefixed_embeddings
        output.indices = indices
        output.pull_loss = pull_loss
        return output
