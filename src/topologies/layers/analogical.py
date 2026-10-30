import torch
from torch import nn


class Analogical(nn.Module):
    component_id = "analogical"

    def __init__(self, num_entities=8, num_relations=4, embedding_dim=4, num_real_blocks=2):
        super().__init__()
        for name, value in (
            ("num_entities", num_entities),
            ("num_relations", num_relations),
            ("embedding_dim", embedding_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(num_real_blocks) is not int:
            raise ValueError("num_real_blocks must be an integer")
        if num_real_blocks < 0 or num_real_blocks > embedding_dim:
            raise ValueError("num_real_blocks must be between 0 and embedding_dim")
        if (embedding_dim - num_real_blocks) % 2 != 0:
            raise ValueError("embedding_dim - num_real_blocks must be even")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_real_blocks = num_real_blocks
        self.num_complex_blocks = (embedding_dim - num_real_blocks) // 2
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(self, head, relation, tail):
        if head.shape != relation.shape or head.shape != tail.shape:
            raise ValueError("head, relation, and tail embeddings must have matching shapes")
        if head.shape[-1] != self.embedding_dim:
            raise ValueError("embedding vectors must match embedding_dim")

        num_real_blocks = self.num_real_blocks
        real_score = (
            head[..., :num_real_blocks]
            * relation[..., :num_real_blocks]
            * tail[..., :num_real_blocks]
        ).sum(dim=-1)
        head_pairs = head[..., num_real_blocks:].reshape(
            *head.shape[:-1], self.num_complex_blocks, 2
        )
        relation_x = relation[..., num_real_blocks::2]
        relation_y = relation[..., num_real_blocks + 1 :: 2]
        tail_pairs = tail[..., num_real_blocks:].reshape(
            *tail.shape[:-1], self.num_complex_blocks, 2
        )
        pair_score = (
            relation_x
            * (
                head_pairs[..., 0] * tail_pairs[..., 0]
                + head_pairs[..., 1] * tail_pairs[..., 1]
            )
            + relation_y
            * (
                head_pairs[..., 1] * tail_pairs[..., 0]
                - head_pairs[..., 0] * tail_pairs[..., 1]
            )
        ).sum(dim=-1)
        return real_score + pair_score

    def forward(self, head_ids, relation_ids, tail_ids):
        if not all(isinstance(ids, torch.Tensor) for ids in (head_ids, relation_ids, tail_ids)):
            raise ValueError("head_ids, relation_ids, and tail_ids must be tensors")
        if head_ids.shape != relation_ids.shape or head_ids.shape != tail_ids.shape:
            raise ValueError("head_ids, relation_ids, and tail_ids must have matching shapes")
        if any(ids.dtype not in (torch.int32, torch.int64) for ids in (head_ids, relation_ids, tail_ids)):
            raise ValueError("head_ids, relation_ids, and tail_ids must have integer dtypes")

        head = self.entity_embeddings(head_ids)
        relation = self.relation_embeddings(relation_ids)
        tail = self.entity_embeddings(tail_ids)
        return self.score(head, relation, tail)
