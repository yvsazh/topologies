import torch
from torch import nn


def score(head, relation, tail):
    head = torch.as_tensor(head)
    tail = torch.as_tensor(tail, dtype=head.dtype, device=head.device)
    relation = torch.as_tensor(relation, dtype=head.dtype, device=head.device)
    if head.ndim == 0 or head.shape[-1] % 2:
        raise ValueError("embedding dimension must be even")
    if tail.ndim == 0 or tail.shape[-1] % 2:
        raise ValueError("embedding dimension must be even")
    if head.shape != tail.shape:
        raise ValueError("head and tail must have identical shapes")
    pairs = head.shape[-1] // 2
    if relation.shape[-1:] != (pairs,):
        raise ValueError("relation must contain one angle per complex pair")

    h = head.reshape(*head.shape[:-1], pairs, 2)
    t = tail.reshape(*tail.shape[:-1], pairs, 2)
    angle = relation
    cosine, sine = torch.cos(angle), torch.sin(angle)
    modulus = torch.sqrt(cosine.square() + sine.square())
    cosine, sine = cosine / modulus, sine / modulus
    real = h[..., 0] * cosine - h[..., 1] * sine
    imag = h[..., 0] * sine + h[..., 1] * cosine
    rotated = torch.stack((real, imag), dim=-1)
    return -torch.linalg.vector_norm((rotated - t).reshape(*head.shape[:-1], -1), dim=-1)


class RotatE(nn.Module):
    component_id = "rotate"

    def __init__(self, embedding_dim=None, **kwargs):
        super().__init__()
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument: {unexpected}")
        if embedding_dim is not None and (not isinstance(embedding_dim, int) or embedding_dim <= 0 or embedding_dim % 2):
            raise ValueError("embedding_dim must be a positive even integer")
        self.embedding_dim = embedding_dim

    def forward(self, head, relation, tail):
        return score(head, relation, tail)

    score = staticmethod(score)
