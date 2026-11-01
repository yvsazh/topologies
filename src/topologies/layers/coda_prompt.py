import torch
from torch import nn
from torch.nn import functional as F


class CODAPrompt(nn.Module):
    component_id = "coda-prompt"

    def __init__(self, embed_dim=768, prompt_len=8, num_components=100):
        super().__init__()
        dimensions = {
            "embed_dim": embed_dim,
            "prompt_len": prompt_len,
            "num_components": num_components,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if prompt_len % 2 != 0:
            raise ValueError("prompt_len must be even to split the prefix into keys and values")
        self.embed_dim = embed_dim
        self.prompt_len = prompt_len
        self.num_components = num_components
        self.keys = nn.Parameter(torch.empty(embed_dim, num_components))
        self.attention = nn.Parameter(torch.empty(embed_dim, num_components))
        self.prompt_components = nn.Parameter(torch.empty(prompt_len, embed_dim, num_components))
        nn.init.orthogonal_(self.keys)
        nn.init.orthogonal_(self.attention)
        nn.init.orthogonal_(self.prompt_components)

    def attention_weights(self, query):
        if not isinstance(query, torch.Tensor) or query.ndim != 2:
            raise ValueError("query must be a rank 2 tensor with shape (batch, embed_dim)")
        if query.shape[1] != self.embed_dim:
            raise ValueError(f"query embed_dim must match configured embed_dim {self.embed_dim}")
        attended_query = query.unsqueeze(-1) * self.attention.unsqueeze(0)
        return F.cosine_similarity(attended_query, self.keys.unsqueeze(0), dim=1)

    def forward(self, query):
        coefficients = self.attention_weights(query)
        return torch.einsum("bm,ldm->bld", coefficients, self.prompt_components)
