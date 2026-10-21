import torch
from torch import nn


class ComplEx(nn.Module):
    component_id = "complex"
    def __init__(self, num_entities=4, num_relations=2, embedding_dim=2, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {', '.join(kwargs)}")
        if num_entities <= 0 or num_relations <= 0 or embedding_dim <= 0:
            raise ValueError("entity, relation, and embedding dimensions must be positive")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_imag = nn.Embedding(num_entities, embedding_dim)
        self.relation_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_imag = nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for embedding in (self.entity_real, self.entity_imag, self.relation_real, self.relation_imag):
            nn.init.xavier_uniform_(embedding.weight)

    @staticmethod
    def score(head, relation, tail):
        if head.shape[-1] % 2 or relation.shape[-1] % 2 or tail.shape[-1] % 2:
            raise ValueError("ComplEx embeddings must have an even final dimension")
        if head.shape[-1] != relation.shape[-1] or head.shape[-1] != tail.shape[-1]:
            raise ValueError("ComplEx embeddings must have matching dimensions")
        head_real, head_imag = head.chunk(2, dim=-1)
        rel_real, rel_imag = relation.chunk(2, dim=-1)
        tail_real, tail_imag = tail.chunk(2, dim=-1)
        return (head_real * rel_real * tail_real
                + head_imag * rel_real * tail_imag
                + head_real * rel_imag * tail_imag
                - head_imag * rel_imag * tail_real).sum(dim=-1)

    def forward(self, head, relation, tail):
        head_vector = torch.cat((self.entity_real(head), self.entity_imag(head)), dim=-1)
        relation_vector = torch.cat((self.relation_real(relation), self.relation_imag(relation)), dim=-1)
        tail_vector = torch.cat((self.entity_real(tail), self.entity_imag(tail)), dim=-1)
        return self.score(head_vector, relation_vector, tail_vector)
