import torch
from torch import nn


class SWAG(nn.Module):
    component_id = "swag"

    def __init__(self, rank=20, moment_update_frequency=1):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 2:
            raise ValueError("rank must be an integer of at least 2")
        if (
            not isinstance(moment_update_frequency, int)
            or isinstance(moment_update_frequency, bool)
            or moment_update_frequency < 1
        ):
            raise ValueError("moment_update_frequency must be a positive integer")
        self.rank = rank
        self.moment_update_frequency = moment_update_frequency

    def forward(self, trajectory, z_diag=None, z_low_rank=None):
        if not isinstance(trajectory, torch.Tensor):
            raise TypeError("trajectory must be a tensor")
        if trajectory.ndim not in (2, 3):
            raise ValueError("trajectory must be a two-dimensional or three-dimensional tensor")
        if any(size == 0 for size in trajectory.shape):
            raise ValueError("trajectory must be nonempty")
        if not trajectory.is_floating_point():
            raise ValueError("trajectory must have a floating-point dtype")

        batched_trajectories = trajectory.ndim == 3
        selected = trajectory[..., self.moment_update_frequency - 1 :: self.moment_update_frequency, :]
        selected_count, parameter_count = selected.shape[-2:]
        if selected_count < 2:
            raise ValueError("trajectory must contain at least two selected snapshots")

        mean = selected.mean(dim=-2)
        diagonal_variance = (selected.square().mean(dim=-2) - mean.square()).clamp_min(0)
        deviation_count = min(self.rank, selected_count)
        first_deviation = selected_count - deviation_count
        deviation_suffix = selected[..., first_deviation:, :]
        prefix_sum = selected[..., :first_deviation, :].sum(dim=-2)
        running_counts = torch.arange(
            first_deviation + 1,
            selected_count + 1,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        if batched_trajectories:
            running_counts = running_counts.reshape(1, deviation_count, 1)
        else:
            running_counts = running_counts.reshape(deviation_count, 1)
        running_means = (prefix_sum.unsqueeze(-2) + deviation_suffix.cumsum(dim=-2)) / running_counts
        deviations = deviation_suffix - running_means

        batch_size = None
        for noise, width, name in (
            (z_diag, parameter_count, "z_diag"),
            (z_low_rank, deviation_count, "z_low_rank"),
        ):
            if noise is None:
                continue
            if not isinstance(noise, torch.Tensor) or noise.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional tensor")
            if noise.shape[0] == 0 or noise.shape[1] != width:
                raise ValueError(f"{name} must have shape (batch, {width}) with a nonempty batch")
            if not noise.is_floating_point():
                raise ValueError(f"{name} must have a floating-point dtype")
            if noise.dtype != trajectory.dtype or noise.device != trajectory.device:
                raise ValueError(f"{name} must match the trajectory dtype and device")
            if batched_trajectories and noise.shape[0] != trajectory.shape[0]:
                raise ValueError(f"{name} must have one noise row per trajectory")
            if batch_size is not None and noise.shape[0] != batch_size:
                raise ValueError("z_diag and z_low_rank must have the same batch size")
            batch_size = noise.shape[0]

        if batch_size is None:
            batch_size = trajectory.shape[0] if batched_trajectories else 1
        if z_diag is None:
            z_diag = torch.randn(
                batch_size,
                parameter_count,
                dtype=trajectory.dtype,
                device=trajectory.device,
            )
        if z_low_rank is None:
            z_low_rank = torch.randn(
                batch_size,
                deviation_count,
                dtype=trajectory.dtype,
                device=trajectory.device,
            )

        positive_variance = diagonal_variance > 0
        safe_variance = torch.where(positive_variance, diagonal_variance, torch.ones_like(diagonal_variance))
        diagonal_scale = torch.where(
            positive_variance,
            torch.sqrt(safe_variance / 2),
            torch.zeros_like(diagonal_variance),
        )
        diagonal_sample = z_diag * diagonal_scale
        if batched_trajectories:
            low_rank_sample = torch.bmm(z_low_rank.unsqueeze(1), deviations).squeeze(1)
        else:
            low_rank_sample = z_low_rank @ deviations
        low_rank_sample = low_rank_sample / (2 * (deviation_count - 1)) ** 0.5
        return mean + diagonal_sample + low_rank_sample
