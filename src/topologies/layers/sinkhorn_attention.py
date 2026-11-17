import math

import torch
from torch import nn


class SinkhornAttention(nn.Module):
    component_id = "sinkhorn-attention"

    def __init__(
        self,
        model_dim,
        num_heads=1,
        block_size=8,
        sinkhorn_iterations=5,
        temperature=0.75,
        gumbel_noise=True,
    ):
        super().__init__()
        if isinstance(model_dim, bool) or not isinstance(model_dim, int) or model_dim <= 0:
            raise ValueError("model_dim must be a positive integer")
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if (
            isinstance(sinkhorn_iterations, bool)
            or not isinstance(sinkhorn_iterations, int)
            or sinkhorn_iterations < 0
        ):
            raise ValueError("sinkhorn_iterations must be a nonnegative integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive number")
        if gumbel_noise is not True and gumbel_noise is not False:
            raise ValueError("gumbel_noise must be a boolean")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temperature = float(temperature)
        self.gumbel_noise = gumbel_noise
        self.head_dim = model_dim // num_heads
        self.sort_networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.ReLU(),
                nn.Linear(self.head_dim, self.head_dim),
                nn.ReLU(),
            )
            for _ in range(num_heads)
        )
        self.output_projection = nn.Linear(model_dim, model_dim)

    def _split_heads(self, x):
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _pool_blocks(self, x):
        batch, heads, length, head_dim = x.shape
        block_count = length // self.block_size
        return x.reshape(
            batch, heads, block_count, self.block_size, head_dim
        ).sum(dim=-2)

    def _relaxed_sort(self, x, permutation):
        batch, heads, length, head_dim = x.shape
        block_count = length // self.block_size
        blocks = x.reshape(batch, heads, block_count, self.block_size, head_dim)
        sorted_blocks = torch.einsum("bhij,bhjcd->bhicd", permutation, blocks)
        return sorted_blocks.reshape(batch, heads, length, head_dim)

    def _sinkhorn_normalize(self, logits):
        log_matrix = logits
        for _ in range(self.sinkhorn_iterations):
            log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)
            log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-2, keepdim=True)
        return log_matrix.exp()

    def _block_attention(self, query, key, sorted_key, sorted_value):
        batch, heads, length, head_dim = query.shape
        block_count = length // self.block_size
        query_blocks = query.reshape(
            batch, heads, block_count, self.block_size, head_dim
        )
        key_blocks = key.reshape(batch, heads, block_count, self.block_size, head_dim)
        sorted_key_blocks = sorted_key.reshape(
            batch, heads, block_count, self.block_size, head_dim
        )
        sorted_value_blocks = sorted_value.reshape(
            batch, heads, block_count, self.block_size, head_dim
        )
        sorted_scores = torch.einsum("bhmcd,bhmed->bhmce", query_blocks, sorted_key_blocks)
        local_scores = torch.einsum("bhmcd,bhmed->bhmce", query_blocks, key_blocks)
        weights = (sorted_scores + local_scores).softmax(dim=-1)
        output = torch.einsum("bhmce,bhmed->bhmcd", weights, sorted_value_blocks)
        return output.reshape(batch, heads, length, head_dim)

    def forward(self, query, key, value, sort_input=None):
        if sort_input is None:
            sort_input = query
        streams = (query, key, value, sort_input)
        if any(not isinstance(stream, torch.Tensor) for stream in streams):
            raise ValueError("query, key, value, and sort_input must be tensors")
        if any(stream.ndim != 3 for stream in streams):
            raise ValueError("all input streams must have shape (batch, sequence, model_dim)")
        if any(stream.shape != query.shape for stream in streams):
            raise ValueError("query, key, value, and sort_input must have the same shape")
        if query.shape[0] <= 0 or query.shape[1] <= 0 or query.shape[2] != self.model_dim:
            raise ValueError("input shape must have a positive batch and sequence and match model_dim")
        if query.shape[1] % self.block_size != 0:
            raise ValueError("sequence length must be divisible by block_size")
        if any(not stream.is_floating_point() for stream in streams):
            raise ValueError("all input streams must have floating point dtypes")
        if any(stream.dtype != query.dtype or stream.device != query.device for stream in streams):
            raise ValueError("all input streams must share the same dtype and device")
        if query.dtype != self.output_projection.weight.dtype:
            raise ValueError("input dtype must match the module parameter dtype")
        if query.device != self.output_projection.weight.device:
            raise ValueError("input device must match the module parameter device")

        query_heads = self._split_heads(query)
        key_heads = self._split_heads(key)
        value_heads = self._split_heads(value)
        sort_heads = self._split_heads(sort_input)
        pooled = self._pool_blocks(sort_heads)
        sorting_logits = []
        for head, network in enumerate(self.sort_networks):
            source_features = network(pooled[:, head])
            destination_features = pooled[:, head].relu()
            sorting_logits.append(source_features @ destination_features.transpose(-2, -1))
        sorting_logits = torch.stack(sorting_logits, dim=1)
        if self.training and self.gumbel_noise:
            epsilon = torch.finfo(sorting_logits.dtype).eps
            uniform = torch.rand_like(sorting_logits).clamp(epsilon, 1.0 - epsilon)
            noise = -torch.log(-torch.log(uniform))
            sorting_logits = sorting_logits + noise
        permutation = self._sinkhorn_normalize(sorting_logits / self.temperature)
        sorted_key = self._relaxed_sort(key_heads, permutation)
        sorted_value = self._relaxed_sort(value_heads, permutation)
        output = self._block_attention(
            query_heads, key_heads, sorted_key, sorted_value
        )
        output = output.transpose(1, 2).contiguous().reshape(
            query.shape[0], query.shape[1], self.model_dim
        )
        return self.output_projection(output)
