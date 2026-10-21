import torch
from torch import nn
from torch.nn import functional as F


def hamilton(left, right):
    left = torch.as_tensor(left)
    right = torch.as_tensor(right, dtype=left.dtype, device=left.device)
    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("quaternions must have four components")
    a, b, c, d = left.unbind(dim=-1)
    e, f, g, h = right.unbind(dim=-1)
    return torch.stack((a * e - b * f - c * g - d * h,
                        a * f + b * e + c * h - d * g,
                        a * g - b * h + c * e + d * f,
                        a * h + b * g - c * f + d * e), dim=-1)


def rotate_head(head, relation):
    head = torch.as_tensor(head)
    if not head.is_floating_point():
        head = head.float()
    relation = torch.as_tensor(relation, dtype=head.dtype, device=head.device)
    return hamilton(head, F.normalize(relation, p=2, dim=-1))


class QuatE(nn.Module):
    component_id = "quate"
    def __init__(self, num_entities=4, num_relations=2, embedding_dim=4, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {', '.join(kwargs)}")
        if num_entities <= 0 or num_relations <= 0 or embedding_dim <= 0:
            raise ValueError("entity, relation, and embedding sizes must be positive")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 4)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim * 4)
        nn.init.uniform_(self.entity_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embeddings.weight, -0.1, 0.1)

    def forward(self, head, relation, tail):
        device = self.entity_embeddings.weight.device
        head = torch.as_tensor(head, dtype=torch.long, device=device)
        relation = torch.as_tensor(relation, dtype=torch.long, device=device)
        tail = torch.as_tensor(tail, dtype=torch.long, device=device)
        head_q = self.entity_embeddings(head).reshape(*head.shape, self.embedding_dim, 4)
        relation_q = self.relation_embeddings(relation).reshape(*relation.shape, self.embedding_dim, 4)
        tail_q = self.entity_embeddings(tail).reshape(*tail.shape, self.embedding_dim, 4)
        rotated = rotate_head(head_q, relation_q)
        return (rotated * tail_q).sum(dim=(-1, -2))
