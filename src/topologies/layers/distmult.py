import torch
from torch import nn


class DistMult(nn.Module):
    component_id = "distmult"
    def __init__(self, num_entities=8, num_relations=4, embedding_dim=4):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(self, head, relation, tail):
        if head.shape != relation.shape or head.shape != tail.shape:
            raise ValueError("head, relation, and tail must have matching shapes")
        return (head * relation * tail).sum(dim=-1)

    def forward(self, head_ids, relation_ids, tail_ids):
        head = self.entity_embeddings(head_ids)
        relation = self.relation_embeddings(relation_ids)
        tail = self.entity_embeddings(tail_ids)
        return self.score(head, relation, tail)
