import torch
from torch import nn

class MatrixFactorization(nn.Module):
    architecture_id = 'matrix-factorization'

    def __init__(self, num_users: int, num_items: int, embedding_dim: int=16, biases: bool=True, score_mode: str='regression'):
        super().__init__()
        if min(num_users, num_items, embedding_dim) < 1:
            raise ValueError('user/item counts and embedding_dim must be positive')
        if score_mode not in {'regression', 'binary', 'ranking'}:
            raise ValueError('score_mode must be regression, binary, or ranking')
        self.num_users, self.num_items, self.embedding_dim = (num_users, num_items, embedding_dim)
        self.score_mode = score_mode
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1) if biases else None
        self.item_bias = nn.Embedding(num_items, 1) if biases else None
        self.global_bias = nn.Parameter(torch.zeros(())) if biases else None
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        if user_ids.shape != item_ids.shape:
            raise ValueError('user_ids and item_ids must have identical shapes')
        if user_ids.dtype not in (torch.int32, torch.int64) or item_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('IDs must be integer tensors')
        score = (self.user_embedding(user_ids) * self.item_embedding(item_ids)).sum(dim=-1)
        if self.user_bias is not None:
            score = score + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1) + self.global_bias
        return score
