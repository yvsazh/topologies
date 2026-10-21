import torch
from torch import nn


class TuckER(nn.Module):
    component_id = "tucker"
    def __init__(self, num_entities=4, num_relations=3, entity_dim=3, relation_dim=2, core=None):
        super().__init__()
        if min(num_entities, num_relations, entity_dim, relation_dim) <= 0:
            raise ValueError("embedding counts and dimensions must be positive")
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        if core is None:
            self.core = nn.Parameter(torch.empty(entity_dim, relation_dim, entity_dim))
            nn.init.xavier_uniform_(self.core)
        else:
            tensor = torch.as_tensor(core, dtype=torch.get_default_dtype())
            if tensor.ndim != 3 or tuple(tensor.shape) != (entity_dim, relation_dim, entity_dim):
                raise ValueError("core shape must match entity_dim, relation_dim, entity_dim")
            self.core = nn.Parameter(tensor.clone())

    def forward(self, head_ids, relation_ids, tail_ids):
        head = self.entity_embeddings(head_ids)
        relation = self.relation_embeddings(relation_ids)
        tail = self.entity_embeddings(tail_ids)
        return torch.einsum("...i,ijk,...j,...k->...", head, self.core, relation, tail)


def score(core, head, relation, tail):
    core = torch.as_tensor(core)
    head = torch.as_tensor(head, dtype=core.dtype, device=core.device)
    relation = torch.as_tensor(relation, dtype=core.dtype, device=core.device)
    tail = torch.as_tensor(tail, dtype=core.dtype, device=core.device)
    if core.ndim != 3:
        raise ValueError("core must be three-dimensional")
    if any(vector.ndim != 1 for vector in (head, relation, tail)):
        raise ValueError("head, relation, and tail must be vectors")
    if tuple(core.shape) != (head.numel(), relation.numel(), tail.numel()):
        raise ValueError("core dimensions must match head, relation, and tail lengths")
    return torch.einsum("ijk,i,j,k->", core, head, relation, tail)
