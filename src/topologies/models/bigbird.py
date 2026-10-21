import torch
from torch import nn
from torch.nn import functional as F


class BigBirdAttention(nn.Module):
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
        order = torch.randperm(len(candidates), generator=generator).tolist()
        return [candidates[i] for i in order[:self.num_random_blocks]]

    def forward(self, hidden_states, global_attention_mask=None, seed=None):
        batch, length, _ = hidden_states.shape
        count = (length + self.block_size - 1) // self.block_size
        mask = torch.zeros((batch, length), dtype=torch.bool, device=hidden_states.device) if global_attention_mask is None else global_attention_mask.bool()
        q, k, v = [layer(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2) for layer in (self.query, self.key, self.value)]
        outputs = []
        for b in range(batch):
            globals_ = torch.nonzero(mask[b], as_tuple=False).flatten().div(self.block_size, rounding_mode="floor").unique().tolist()
            allowed = torch.zeros((length, length), dtype=torch.bool, device=hidden_states.device)
            for qb in range(count):
                qs, qe = qb * self.block_size, min(length, (qb + 1) * self.block_size)
                blocks = set(globals_)
                blocks.update(range(max(0, qb - self.window_blocks), min(count, qb + self.window_blocks + 1)))
                blocks.update(self.selected_random_blocks(qb, count, globals_, seed))
                for kb in blocks:
                    allowed[qs:qe, kb * self.block_size:min(length, (kb + 1) * self.block_size)] = True
            scores = q[b] @ k[b].transpose(-1, -2) / self.head_dim ** 0.5
            scores = scores.masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
            outputs.append(self.dropout(torch.softmax(scores, dim=-1)) @ v[b])
        context = torch.stack(outputs).transpose(1, 2).contiguous().view(batch, length, self.hidden_size)
        return self.output(context)


class BigBirdLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, block_size, window_blocks, num_random_blocks, dropout):
        super().__init__()
        self.attention = BigBirdAttention(hidden_size, num_heads, block_size, window_blocks, num_random_blocks, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, global_attention_mask=None):
        x = self.attention_norm(x + self.attention(x, global_attention_mask))
        return self.output_norm(x + self.output(F.gelu(self.intermediate(x))))


class BigBird(nn.Module):
    architecture_id = "bigbird"
    PRESETS = {"tiny": {"vocab_size": 32, "context_length": 32, "hidden_size": 16, "num_layers": 2, "num_heads": 2, "intermediate_size": 32, "block_size": 4, "window_blocks": 1, "num_random_blocks": 1}, "base": {"vocab_size": 50358, "context_length": 4096, "hidden_size": 768, "num_layers": 12, "num_heads": 12, "intermediate_size": 3072, "block_size": 64, "window_blocks": 3, "num_random_blocks": 3}}

    def __init__(self, vocab_size=32, context_length=32, hidden_size=16, num_layers=2, num_heads=2, intermediate_size=32, block_size=4, window_blocks=1, num_random_blocks=1, pad_token_id=0, mask_token_id=4, dropout=0.0):
        super().__init__()
        self.vocab_size, self.context_length = vocab_size, context_length
        self.hidden_size, self.num_layers, self.num_heads = hidden_size, num_layers, num_heads
        self.block_size, self.window_blocks, self.num_random_blocks = block_size, window_blocks, num_random_blocks
        self.pad_token_id, self.mask_token_id = pad_token_id, mask_token_id
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(context_length, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(BigBirdLayer(hidden_size, num_heads, intermediate_size, block_size, window_blocks, num_random_blocks, dropout) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.weight

    def forward(self, tokens, attention_mask=None, global_attention_mask=None):
        if tokens.shape[1] > self.context_length:
            raise ValueError("input exceeds configured context_length")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        valid = tokens.ne(self.pad_token_id) if attention_mask is None else attention_mask.bool()
        x = self.dropout(self.embedding_norm(self.embeddings(tokens) + self.position_embeddings(positions)))
        for layer in self.layers:
            x = layer(x, global_attention_mask) * valid.unsqueeze(-1)
        return self.lm_head(self.final_norm(x))
