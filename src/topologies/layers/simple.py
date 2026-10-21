import torch
from torch import nn


class SimplE(nn.Module):
    component_id = "simple"
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        if min(num_entities, num_relations, embedding_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.entity_head = nn.Embedding(num_entities, embedding_dim)
        self.entity_tail = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        self.relation_inverse = nn.Embedding(num_relations, embedding_dim)
        for embedding in (self.entity_head, self.entity_tail, self.relation, self.relation_inverse):
            nn.init.xavier_uniform_(embedding.weight)

    def score_embeddings(self, head_head, head_tail, relation, relation_inverse, tail_head, tail_tail):
        if any(tensor.shape != head_head.shape for tensor in (head_tail, relation, relation_inverse, tail_head, tail_tail)):
            raise ValueError("all SimplE embedding tensors must have the same shape")
        forward = (head_head * relation * tail_tail).sum(dim=-1)
        reverse = (head_tail * relation_inverse * tail_head).sum(dim=-1)
        return (forward + reverse) / 2

    def forward(self, head, relation, tail):
        return self.score_embeddings(
            self.entity_head(head),
            self.entity_tail(head),
            self.relation(relation),
            self.relation_inverse(relation),
            self.entity_head(tail),
            self.entity_tail(tail),
        )
