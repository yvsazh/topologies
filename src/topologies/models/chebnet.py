import torch
from torch import nn


class ChebNet(nn.Module):
    architecture_id = "chebnet"
    PRESETS = {
        "tiny": {"in_features": 2, "out_features": 2, "K": 2},
        "small": {"in_features": 16, "out_features": 16, "K": 5},
        "base": {"in_features": 64, "out_features": 64, "K": 25},
    }

    def __init__(self, in_features, out_features, K):
        super().__init__()
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("K", K),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, scaled_laplacian):
        if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3):
            raise ValueError("x must be a rank 2 or 3 tensor")
        if x.shape[-2] == 0:
            raise ValueError("x must contain at least one node")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x must have feature width {self.in_features}")
        if not x.is_floating_point():
            raise ValueError("x must have a floating point dtype")
        if x.device != self.weight.device:
            raise ValueError("x and the model parameters must be on the same device")
        if x.dtype != self.weight.dtype:
            raise ValueError("x and the model parameters must have the same dtype")
        if not isinstance(scaled_laplacian, torch.Tensor) or scaled_laplacian.ndim != 2:
            raise ValueError("scaled_laplacian must be a rank 2 tensor")
        if scaled_laplacian.shape != (x.shape[-2], x.shape[-2]):
            raise ValueError("scaled_laplacian must be square and match the node count")
        if scaled_laplacian.layout not in (
            torch.strided,
            torch.sparse_coo,
            torch.sparse_csr,
        ):
            raise ValueError("scaled_laplacian must use a dense, COO, or CSR layout")
        if scaled_laplacian.device != x.device:
            raise ValueError("scaled_laplacian and x must be on the same device")
        if scaled_laplacian.dtype != x.dtype:
            raise ValueError("scaled_laplacian and x must have the same dtype")

        unbatched = x.ndim == 2
        values = x.unsqueeze(0) if unbatched else x
        terms = [values]
        if self.K > 1:
            if scaled_laplacian.layout == torch.strided:
                terms.append(torch.matmul(scaled_laplacian, values))
            else:
                flattened = values.permute(1, 0, 2).reshape(values.shape[1], -1)
                first = torch.sparse.mm(scaled_laplacian, flattened)
                terms.append(first.reshape(values.shape[1], values.shape[0], -1).permute(1, 0, 2))
            for _ in range(2, self.K):
                if scaled_laplacian.layout == torch.strided:
                    next_term = 2 * torch.matmul(scaled_laplacian, terms[-1]) - terms[-2]
                else:
                    flattened = terms[-1].permute(1, 0, 2).reshape(values.shape[1], -1)
                    propagated = torch.sparse.mm(scaled_laplacian, flattened)
                    propagated = propagated.reshape(
                        values.shape[1], values.shape[0], -1
                    ).permute(1, 0, 2)
                    next_term = 2 * propagated - terms[-2]
                terms.append(next_term)
        polynomial_terms = torch.stack(terms, dim=1)
        output = torch.einsum("bknf,kfo->bno", polynomial_terms, self.weight)
        return output.squeeze(0) if unbatched else output
