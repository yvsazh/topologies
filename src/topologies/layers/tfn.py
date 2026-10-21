import torch
from torch import nn


class TFNSelfInteraction(nn.Module):
    component_id = "tfn"

    def __init__(self, in_channels=1, out_channels=None, order=1, bias=False):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        if min(in_channels, out_channels) < 1 or order < 0:
            raise ValueError("channel counts must be positive and order must be nonnegative")
        if order > 0 and bias:
            raise ValueError("bias is only equivariant for scalar (order-zero) features")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = order
        self.representation_dim = 2 * order + 1
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, features):
        if (features.ndim < 2
                or features.shape[-2:] != (self.in_channels, self.representation_dim)):
            raise ValueError(
                "features must end in (in_channels, 2 * order + 1)"
            )
        output = torch.einsum("oc,...cm->...om", self.weight, features)
        if self.bias is not None:
            output = output + self.bias[:, None]
        return output


class TFN(nn.Module):
    def __init__(self, scalar_channels=1, vector_channels=1, output_scalars=None, output_vectors=None):
        super().__init__()
        if min(scalar_channels, vector_channels) < 1:
            raise ValueError("channel counts must be positive")
        output_scalars = scalar_channels if output_scalars is None else output_scalars
        output_vectors = vector_channels if output_vectors is None else output_vectors
        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.scalar_mix = nn.Linear(scalar_channels + vector_channels, output_scalars)
        self.vector_mix = nn.Parameter(torch.empty(output_vectors, vector_channels))
        self.vector_gate = nn.Linear(scalar_channels, output_vectors)
        nn.init.xavier_uniform_(self.vector_mix)

    def forward(self, scalar, vector):
        scalar = torch.as_tensor(scalar)
        vector = torch.as_tensor(vector, dtype=scalar.dtype, device=scalar.device)
        if scalar.ndim < 2 or scalar.shape[-1] != self.scalar_channels:
            raise ValueError("scalar must end in scalar_channels")
        if vector.shape[:-2] != scalar.shape[:-1] or vector.shape[-2:] != (self.vector_channels, 3):
            raise ValueError("vector must have shape scalar.shape[:-1] + (vector_channels, 3)")
        magnitudes = torch.linalg.vector_norm(vector, dim=-1)
        scalar_out = self.scalar_mix(torch.cat((scalar, magnitudes), dim=-1))
        vector_out = torch.einsum("oc,...c d->...o d", self.vector_mix, vector)
        vector_out = vector_out * (1.0 + self.vector_gate(scalar).unsqueeze(-1))
        return scalar_out, vector_out
