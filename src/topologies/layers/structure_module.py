import torch
from torch import nn


def frame_aligned_point_error(
    predicted_points,
    target_points,
    predicted_rotations,
    predicted_translations,
    target_rotations,
    target_translations,
    clamp_distance=None,
    length_scale=1.0,
):
    if predicted_points.ndim != 3 or target_points.shape != predicted_points.shape:
        raise ValueError("predicted_points and target_points must have equal (batch, points, 3) shapes")
    if predicted_points.shape[-1] != 3:
        raise ValueError("point coordinates must have width 3")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    if clamp_distance is not None and clamp_distance < 0:
        raise ValueError("clamp_distance must be nonnegative")

    batch = predicted_points.shape[0]
    if predicted_rotations.ndim != 4 or predicted_rotations.shape[0] != batch:
        raise ValueError("predicted_rotations must have shape (batch, frames, 3, 3)")
    expected_rotations = (batch, predicted_rotations.shape[1], 3, 3)
    expected_translations = (batch, predicted_rotations.shape[1], 3)
    if predicted_rotations.shape != expected_rotations or predicted_translations.shape != expected_translations:
        raise ValueError("predicted frames must have shapes (batch, frames, 3, 3) and (batch, frames, 3)")
    if target_rotations.ndim != 4 or target_translations.ndim != 3:
        raise ValueError("target frames must have shapes (batch, frames, 3, 3) and (batch, frames, 3)")
    if target_rotations.shape != predicted_rotations.shape or target_translations.shape != predicted_translations.shape:
        raise ValueError("predicted and target frame shapes must match")

    predicted_delta = predicted_points[:, None] - predicted_translations[:, :, None]
    target_delta = target_points[:, None] - target_translations[:, :, None]
    predicted_local = torch.matmul(
        predicted_rotations.transpose(-1, -2)[:, :, None], predicted_delta.unsqueeze(-1)
    ).squeeze(-1)
    target_local = torch.matmul(
        target_rotations.transpose(-1, -2)[:, :, None], target_delta.unsqueeze(-1)
    ).squeeze(-1)
    error = torch.linalg.vector_norm(predicted_local - target_local, dim=-1)
    if clamp_distance is not None:
        error = error.clamp(max=clamp_distance)
    return error / length_scale


class FrameAlignedPointError(nn.Module):
    component_id = "structure-module"

    def __init__(self, clamp_distance=None, length_scale=1.0):
        super().__init__()
        if length_scale <= 0:
            raise ValueError("length_scale must be positive")
        if clamp_distance is not None and clamp_distance < 0:
            raise ValueError("clamp_distance must be nonnegative")
        self.clamp_distance = clamp_distance
        self.length_scale = length_scale

    def forward(self, predicted_points, target_points, predicted_rotations,
                predicted_translations, target_rotations, target_translations):
        return frame_aligned_point_error(
            predicted_points,
            target_points,
            predicted_rotations,
            predicted_translations,
            target_rotations,
            target_translations,
            clamp_distance=self.clamp_distance,
            length_scale=self.length_scale,
        )


class StructureModule(nn.Module):
    def __init__(self, single_dim=16, heads=4, head_dim=8, update_scale=0.1):
        super().__init__()
        if single_dim < 1 or heads < 1 or head_dim < 1:
            raise ValueError("dimensions must be positive")
        self.single_dim = single_dim
        self.heads = heads
        self.head_dim = head_dim
        self.update_scale = update_scale
        self.norm = nn.LayerNorm(single_dim)
        self.query = nn.Linear(single_dim, heads * head_dim, bias=False)
        self.key = nn.Linear(single_dim, heads * head_dim, bias=False)
        self.value = nn.Linear(single_dim, heads * head_dim, bias=False)
        self.translation = nn.Linear(heads * head_dim, 1)
        self.output = nn.Linear(heads * head_dim, single_dim)

    def forward(self, single, positions):
        single = torch.as_tensor(single)
        positions = torch.as_tensor(positions, dtype=single.dtype, device=single.device)
        if single.ndim != 3 or positions.shape != (*single.shape[:2], 3):
            raise ValueError("expected single (batch, residues, channels) and positions (batch, residues, 3)")
        if single.shape[-1] != self.single_dim:
            raise ValueError("single feature width does not match single_dim")
        batch, residues, _ = single.shape
        normalized = self.norm(single)
        query = self.query(normalized).view(batch, residues, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(normalized).view(batch, residues, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(normalized).view(batch, residues, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / self.head_dim ** 0.5
        attention = torch.softmax(logits, dim=-1)
        point_context = torch.matmul(attention, positions[:, None]).mean(dim=1)
        feature_context = torch.matmul(attention, value).transpose(1, 2).reshape(batch, residues, -1)
        displacement = point_context - positions
        scale = torch.tanh(self.translation(feature_context)) * self.update_scale
        updated_positions = positions + scale * displacement
        return updated_positions, single + self.output(feature_context)
