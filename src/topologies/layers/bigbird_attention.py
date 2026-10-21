import torch
from torch import nn


class BigBirdAttention(nn.Module):
    component_id = "bigbird-attention"
    PRESETS = {"tiny": {"hidden_size": 8, "num_heads": 2, "block_size": 2, "window_blocks": 1, "num_random_blocks": 1}}

    def __init__(self, hidden_size, num_heads, block_size=64, window_blocks=3, num_random_blocks=3, dropout=0.0, seed=0):
        super().__init__()
        if hidden_size % num_heads or block_size < 1 or window_blocks < 1 or num_random_blocks < 0:
            raise ValueError("invalid BigBird attention dimensions")
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size, self.window_blocks = block_size, window_blocks
        self.num_random_blocks, self.seed = num_random_blocks, seed
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def selected_random_blocks(self, query_block, block_count, global_blocks=(), seed=None):
        excluded = set(global_blocks) | {query_block}
        excluded.update(range(max(0, query_block - self.window_blocks), min(block_count, query_block + self.window_blocks + 1)))
        candidates = [i for i in range(block_count) if i not in excluded]
        generator = torch.Generator(device="cpu").manual_seed(self.seed if seed is None else seed + query_block)
        if not candidates or self.num_random_blocks == 0:
            return []
        return [candidates[i] for i in torch.randperm(len(candidates), generator=generator).tolist()[:self.num_random_blocks]]

    def forward(self, hidden_states, global_attention_mask=None, seed=None):
        batch, length, _ = hidden_states.shape
        block_count = (length + self.block_size - 1) // self.block_size
        mask = torch.zeros((batch, length), dtype=torch.bool, device=hidden_states.device) if global_attention_mask is None else global_attention_mask.bool()
        q, k, v = [layer(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2) for layer in (self.query, self.key, self.value)]
        outputs = []
        for b in range(batch):
            global_blocks = torch.nonzero(mask[b], as_tuple=False).flatten().div(self.block_size, rounding_mode="floor").unique().tolist()
            allowed = torch.zeros((length, length), dtype=torch.bool, device=hidden_states.device)
            for qb in range(block_count):
                qs, qe = qb * self.block_size, min(length, (qb + 1) * self.block_size)
                blocks = set(global_blocks)
                blocks.update(range(max(0, qb - self.window_blocks), min(block_count, qb + self.window_blocks + 1)))
                blocks.update(self.selected_random_blocks(qb, block_count, global_blocks, seed))
                for kb in blocks:
                    allowed[qs:qe, kb * self.block_size:min(length, (kb + 1) * self.block_size)] = True
            scores = q[b] @ k[b].transpose(-1, -2) / self.head_dim ** 0.5
            scores = scores.masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
            outputs.append(self.dropout(torch.softmax(scores, dim=-1)) @ v[b])
        context = torch.stack(outputs).transpose(1, 2).contiguous().view(batch, length, self.hidden_size)
        return self.output(context)
