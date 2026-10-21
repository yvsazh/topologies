import math

import torch
from torch import nn


class PagedAttention(nn.Module):
    component_id = "paged-attention"

    def __init__(self, block_size=16):
        super().__init__()
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size

    @staticmethod
    def _invalid_tensor(name):
        raise ValueError(f"{name} must be a torch.Tensor")

    def forward(self, query, key_cache, value_cache, block_table, sequence_lengths):
        tensors = {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "block_table": block_table,
            "sequence_lengths": sequence_lengths,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                self._invalid_tensor(name)

        if query.ndim != 3:
            raise ValueError("query must have shape [batch, heads, head_dim]")
        if key_cache.ndim != 4 or value_cache.ndim != 4:
            raise ValueError("key_cache and value_cache must have shape [blocks, heads, block_size, head_dim]")
        if key_cache.shape != value_cache.shape:
            raise ValueError("key_cache and value_cache must have identical shapes")
        if block_table.ndim != 2:
            raise ValueError("block_table must have shape [batch, logical_blocks]")
        if sequence_lengths.ndim == 2 and sequence_lengths.shape[1] == 1:
            sequence_lengths = sequence_lengths[:, 0]
        if sequence_lengths.ndim != 1:
            raise ValueError("sequence_lengths must have shape [batch]")

        batch, heads, head_dim = query.shape
        physical_blocks, cache_heads, cache_block_size, cache_head_dim = key_cache.shape
        if batch <= 0 or heads <= 0 or head_dim <= 0:
            raise ValueError("query dimensions must be positive")
        if physical_blocks <= 0:
            raise ValueError("key/value caches must contain at least one physical block")
        if cache_heads != heads or cache_head_dim != head_dim:
            raise ValueError("cache head count and head dimension must match query")
        if cache_block_size != self.block_size:
            raise ValueError("cache block size must match block_size")
        if block_table.shape[0] != batch or sequence_lengths.shape[0] != batch:
            raise ValueError("block_table and sequence_lengths batch dimensions must match query")
        if query.device != key_cache.device or query.device != value_cache.device:
            raise ValueError("query and key/value caches must be on the same device")
        if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
            raise ValueError("query and key/value caches must have the same dtype")
        if not query.is_floating_point():
            raise ValueError("query and key/value caches must use a floating-point dtype")
        if block_table.device != query.device or sequence_lengths.device != query.device:
            raise ValueError("block_table and sequence_lengths must be on the same device as query")

        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if block_table.dtype not in integer_dtypes:
            raise ValueError("block_table must use an integer dtype")
        if sequence_lengths.dtype not in integer_dtypes:
            raise ValueError("sequence_lengths must use an integer dtype")

        outputs = []
        max_capacity = physical_blocks * self.block_size
        for batch_index in range(batch):
            sequence_length = int(sequence_lengths[batch_index].item())
            if sequence_length <= 0:
                raise ValueError("sequence lengths must be positive")
            if sequence_length > max_capacity:
                raise ValueError("sequence length exceeds physical cache capacity")

            logical_blocks = (sequence_length + self.block_size - 1) // self.block_size
            if logical_blocks > block_table.shape[1]:
                raise ValueError("block_table has too few entries for a valid sequence prefix")
            physical_indices = block_table[batch_index, :logical_blocks].to(torch.long)
            if torch.any(physical_indices < 0) or torch.any(physical_indices >= physical_blocks):
                raise ValueError("block_table contains an out-of-range physical block index")

            positions = torch.arange(sequence_length, device=query.device)
            mapped_blocks = physical_indices[positions // self.block_size]
            slots = positions % self.block_size
            keys = key_cache[mapped_blocks, :, slots, :].permute(1, 0, 2)
            values = value_cache[mapped_blocks, :, slots, :].permute(1, 0, 2)
            scores = torch.einsum("hd,hld->hl", query[batch_index], keys) / math.sqrt(head_dim)
            probabilities = torch.softmax(scores, dim=-1)
            outputs.append(torch.einsum("hl,hld->hd", probabilities, values))

        return torch.stack(outputs, dim=0)
