import torch
from torch import nn


class GaussianSplatting(nn.Module):
    component_id = "gaussian-splatting"
    PRESETS = {"tiny": {"height": 4, "width": 5}, "small": {"height": 16, "width": 16}}

    def __init__(self, height: int = 16, width: int = 16):
        super().__init__()
        if (not isinstance(height, int) or isinstance(height, bool) or height <= 0
                or not isinstance(width, int) or isinstance(width, bool) or width <= 0):
            raise ValueError("height and width must be positive integers")
        self.height = height
        self.width = width

    def forward(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        values = (means, covariances, opacities, colors, depths)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise ValueError("means, covariances, opacities, colors, and depths must be tensors")
        if any(not value.is_floating_point() for value in values[:4]):
            raise ValueError("means, covariances, opacities, and colors must be floating point tensors")
        if depths.is_complex() or depths.dtype == torch.bool:
            raise ValueError("depths must be real numeric values")
        if means.ndim == 3:
            if means.shape[0] == 0 or means.shape[2] != 2:
                raise ValueError("batched means must have shape [B, N, 2] with B greater than zero")
            batch, count = means.shape[:2]
            if covariances.shape != (batch, count, 2, 2):
                raise ValueError("batched covariances must have shape [B, N, 2, 2]")
            if opacities.shape != (batch, count):
                raise ValueError("batched opacities must have shape [B, N]")
            if colors.ndim != 3 or colors.shape[:2] != (batch, count) or colors.shape[2] == 0:
                raise ValueError("batched colors must have shape [B, N, C] with C greater than zero")
            if depths.shape != (batch, count):
                raise ValueError("batched depths must have shape [B, N]")
            return torch.stack(tuple(
                self.forward(means[index], covariances[index], opacities[index], colors[index], depths[index])
                for index in range(batch)
            ))
        if means.ndim != 2 or means.shape[1] != 2:
            raise ValueError("means must have shape [N, 2]")
        count = means.shape[0]
        if covariances.shape != (count, 2, 2):
            raise ValueError("covariances must have shape [N, 2, 2]")
        if opacities.shape != (count,):
            raise ValueError("opacities must have shape [N]")
        if colors.ndim != 2 or colors.shape[0] != count or colors.shape[1] == 0:
            raise ValueError("colors must have shape [N, C] with C greater than zero")
        if depths.shape != (count,):
            raise ValueError("depths must have shape [N]")
        if any(value.device != means.device for value in values):
            raise ValueError("all inputs must be on the same device")
        if any(value.dtype != means.dtype for value in values[:4]):
            raise ValueError("means, covariances, opacities, and colors must have the same floating point dtype")
        if any(not torch.isfinite(value).all().item() for value in values):
            raise ValueError("all inputs must contain only finite values")
        if ((opacities < 0) | (opacities >= 1)).any().item():
            raise ValueError("opacities must be in the range [0, 1)")
        if not torch.allclose(covariances, covariances.transpose(-1, -2), atol=1e-6, rtol=1e-5):
            raise ValueError("covariances must be symmetric positive definite matrices")
        covariance_xx = covariances[:, 0, 0]
        covariance_xy = 0.5 * (covariances[:, 0, 1] + covariances[:, 1, 0])
        covariance_yy = covariances[:, 1, 1]
        determinant = covariance_xx * covariance_yy - covariance_xy.square()
        if ((covariance_xx <= 0) | (determinant <= 0)).any().item():
            raise ValueError("covariances must be symmetric positive definite matrices")
        if count == 0:
            return means.new_zeros((self.height, self.width, colors.shape[1]))
        inverse_xx = covariance_yy / determinant
        inverse_xy = -covariance_xy / determinant
        inverse_yy = covariance_xx / determinant
        y, x = torch.meshgrid(
            torch.arange(self.height, device=means.device, dtype=means.dtype),
            torch.arange(self.width, device=means.device, dtype=means.dtype),
            indexing="ij",
        )
        delta_x = x.unsqueeze(0) - means[:, 0, None, None]
        delta_y = y.unsqueeze(0) - means[:, 1, None, None]
        quadratic = (
            inverse_xx[:, None, None] * delta_x.square()
            + 2 * inverse_xy[:, None, None] * delta_x * delta_y
            + inverse_yy[:, None, None] * delta_y.square()
        )
        alpha = opacities[:, None, None] * torch.exp(-0.5 * quadratic)
        order = torch.argsort(depths, stable=True)
        alpha = alpha[order]
        ordered_colors = colors[order]
        transmittance = torch.cat(
            (torch.ones_like(alpha[:1]), torch.cumprod(1 - alpha, dim=0)[:-1]), dim=0
        )
        weights = transmittance * alpha
        return (weights[..., None] * ordered_colors[:, None, None, :]).sum(dim=0)
