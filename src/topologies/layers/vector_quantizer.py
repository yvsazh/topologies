import torch
from torch import nn

class VectorQuantizer(nn.Module):
    component_id = 'vector-quantizer'

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        if min(num_embeddings, embedding_dim) <= 0 or commitment_cost < 0:
            raise ValueError('invalid quantizer configuration')
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        d = x.shape[-1]
        if d != self.embedding.embedding_dim:
            raise ValueError('last dimension mismatch')
        flat = x.reshape(-1, d)
        w = self.embedding.weight
        distances = flat.square().sum(-1, keepdim=True) + w.square().sum(-1) - 2 * flat @ w.t()
        indices = distances.argmin(-1)
        quant = self.embedding(indices).view_as(x)
        loss = torch.nn.functional.mse_loss(quant, x.detach()) + self.commitment_cost * torch.nn.functional.mse_loss(x, quant.detach())
        return (x + (quant - x).detach(), indices.reshape(x.shape[:-1]), loss)
