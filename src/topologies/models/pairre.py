import torch
from torch import nn
from torch.nn import functional as F


class PairRE(nn.Module):
    architecture_id = "pairre"
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        if num_entities <= 0 or num_relations <= 0 or dim <= 0:
            raise ValueError("num_entities, num_relations, and dim must be positive")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.head_scale = nn.Embedding(num_relations, dim)
        self.tail_scale = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.entity_embeddings.weight, -1.0, 1.0)
        nn.init.uniform_(self.head_scale.weight, -1.0, 1.0)
        nn.init.uniform_(self.tail_scale.weight, -1.0, 1.0)

    def forward(self, heads, relations, tails):
        head = F.normalize(self.entity_embeddings(heads), p=2, dim=-1)
        tail = F.normalize(self.entity_embeddings(tails), p=2, dim=-1)
        head_scale = self.head_scale(relations)
        tail_scale = self.tail_scale(relations)
        return -torch.linalg.vector_norm(head * head_scale - tail * tail_scale, ord=1, dim=-1)
