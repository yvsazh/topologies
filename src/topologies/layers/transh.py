import torch
from torch import nn
from torch.nn import functional as F


class TransH(nn.Module):
    architecture_id = "transh"
    component_id = "transh"
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.normal_embeddings = nn.Embedding(num_relations, dim)
        self.translation_embeddings = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.normal_embeddings.weight)
        nn.init.xavier_uniform_(self.translation_embeddings.weight)

    def forward(self, heads, relations, tails):
        normal = F.normalize(self.normal_embeddings(relations), p=2, dim=-1)
        head = self.entity_embeddings(heads)
        tail = self.entity_embeddings(tails)
        projected_head = head - (head * normal).sum(dim=-1, keepdim=True) * normal
        projected_tail = tail - (tail * normal).sum(dim=-1, keepdim=True) * normal
        translation = self.translation_embeddings(relations)
        translation = translation - (translation * normal).sum(dim=-1, keepdim=True) * normal
        return torch.linalg.vector_norm(projected_head + translation - projected_tail, ord=2, dim=-1)
