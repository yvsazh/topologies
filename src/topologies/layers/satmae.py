import torch
from torch import nn


class SatMAE(nn.Module):
    component_id = "satmae"

    def __init__(self, mode="temporal", d_model=1024, metadata_dim=None):
        super().__init__()
        if mode not in ("temporal", "spectral"):
            raise ValueError("mode must be 'temporal' or 'spectral'")
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if metadata_dim is None:
            metadata_dim = 384 if mode == "temporal" else 256
        if (
            not isinstance(metadata_dim, int)
            or isinstance(metadata_dim, bool)
            or metadata_dim <= 0
        ):
            raise ValueError("metadata_dim must be a positive integer")
        if metadata_dim >= d_model:
            raise ValueError("metadata_dim must be smaller than d_model")
        spatial_dim = d_model - metadata_dim
        if spatial_dim % 4 != 0:
            raise ValueError("spatial encoding width must be divisible by 4")
        if mode == "temporal" and metadata_dim % 6 != 0:
            raise ValueError(
                "temporal metadata_dim must be divisible by 6"
            )
        if mode == "spectral" and metadata_dim % 2 != 0:
            raise ValueError("spectral metadata_dim must be even")
        self.mode = mode
        self.d_model = d_model
        self.metadata_dim = metadata_dim

    def _encode_1d(self, positions, width, output_dtype):
        frequency_count = width // 2
        calculation_dtype = (
            torch.float64 if output_dtype == torch.float64 else torch.float32
        )
        frequencies = torch.pow(
            torch.tensor(10000.0, device=positions.device, dtype=calculation_dtype),
            -torch.arange(
                frequency_count, device=positions.device, dtype=calculation_dtype
            )
            / frequency_count,
        )
        angles = positions.to(calculation_dtype).unsqueeze(-1) * frequencies
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).to(
            output_dtype
        )

    def forward(self, patches, timestamps=None):
        if not isinstance(patches, torch.Tensor):
            raise TypeError("patches must be a torch.Tensor")
        if patches.ndim != 5:
            raise ValueError("patches must have shape [B,Q,H,W,D]")
        if any(size <= 0 for size in patches.shape[:-1]):
            raise ValueError("patches batch, sequence, and grid dimensions must be positive")
        if patches.shape[-1] != self.d_model:
            raise ValueError("patches final dimension must equal d_model")
        if not patches.is_floating_point():
            raise ValueError("patches must have a floating-point dtype")
        batch_size, sequence_size, height, width, _ = patches.shape
        y_positions, x_positions = torch.meshgrid(
            torch.arange(height, device=patches.device),
            torch.arange(width, device=patches.device),
            indexing="ij",
        )
        spatial_width = (self.d_model - self.metadata_dim) // 2
        spatial_encoding = torch.cat(
            (
                self._encode_1d(x_positions, spatial_width, patches.dtype),
                self._encode_1d(y_positions, spatial_width, patches.dtype),
            ),
            dim=-1,
        )[None, None, :, :, :]
        spatial_encoding = spatial_encoding.expand(
            batch_size, sequence_size, height, width, -1
        )
        if self.mode == "temporal":
            if not isinstance(timestamps, torch.Tensor):
                raise ValueError("timestamps must be a torch.Tensor in temporal mode")
            if timestamps.shape != (batch_size, sequence_size, 3):
                raise ValueError("timestamps must have shape [B,Q,3]")
            timestamp_values = timestamps.to(device=patches.device)
            temporal_width = self.metadata_dim // 3
            metadata_encoding = torch.cat(
                tuple(
                    self._encode_1d(
                        timestamp_values[..., index], temporal_width, patches.dtype
                    )
                    for index in range(3)
                ),
                dim=-1,
            )[:, :, None, None, :]
        else:
            group_positions = torch.arange(sequence_size, device=patches.device)
            metadata_encoding = self._encode_1d(
                group_positions, self.metadata_dim, patches.dtype
            )[None, :, None, None, :]
        metadata_encoding = metadata_encoding.expand(
            batch_size, sequence_size, height, width, -1
        )
        positional_encoding = torch.cat(
            (spatial_encoding, metadata_encoding), dim=-1
        )
        return patches + positional_encoding
