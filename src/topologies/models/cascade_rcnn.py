import torch
from torch import nn
from torch.nn import functional as F


def _box_tensor(boxes):
    try:
        boxes = torch.as_tensor(boxes)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("boxes must be numeric tensors") from exc
    if boxes.dtype == torch.bool or boxes.is_complex():
        raise TypeError("boxes must contain real coordinates")
    if not boxes.is_floating_point():
        boxes = boxes.to(dtype=torch.get_default_dtype())
    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError("boxes must end in four (x1, y1, x2, y2) coordinates")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("box coordinates must be finite")
    return boxes


def _box_geometry(boxes):
    boxes = _box_tensor(boxes)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    width, height = x2 - x1, y2 - y1
    if bool(((width <= 0) | (height <= 0)).any()):
        raise ValueError("boxes must have positive width and height")
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, width, height


def encode_box_deltas(reference_boxes, target_boxes):
    reference_boxes = _box_tensor(reference_boxes)
    target_boxes = _box_tensor(target_boxes).to(
        device=reference_boxes.device, dtype=reference_boxes.dtype
    )
    if reference_boxes.shape != target_boxes.shape:
        raise ValueError("reference_boxes and target_boxes must have the same shape")
    ref_x, ref_y, ref_width, ref_height = _box_geometry(reference_boxes)
    target_x, target_y, target_width, target_height = _box_geometry(target_boxes)
    return torch.stack((
        (target_x - ref_x) / ref_width,
        (target_y - ref_y) / ref_height,
        torch.log(target_width / ref_width),
        torch.log(target_height / ref_height),
    ), dim=-1)


def decode_box_deltas(reference_boxes, deltas):
    reference_boxes = _box_tensor(reference_boxes)
    try:
        deltas = torch.as_tensor(deltas)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("deltas must be numeric tensors") from exc
    if deltas.dtype == torch.bool or deltas.is_complex():
        raise TypeError("deltas must contain real values")
    deltas = deltas.to(device=reference_boxes.device, dtype=reference_boxes.dtype)
    if deltas.shape != reference_boxes.shape or deltas.shape[-1] != 4:
        raise ValueError("deltas must have the same shape as reference_boxes and end in four values")

    center_x, center_y, width, height = _box_geometry(reference_boxes)
    dx, dy, dw, dh = deltas.unbind(dim=-1)
    predicted_center_x = center_x + dx * width
    predicted_center_y = center_y + dy * height
    predicted_width = width * torch.exp(dw)
    predicted_height = height * torch.exp(dh)
    return torch.stack((
        predicted_center_x - predicted_width * 0.5,
        predicted_center_y - predicted_height * 0.5,
        predicted_center_x + predicted_width * 0.5,
        predicted_center_y + predicted_height * 0.5,
    ), dim=-1)


class CascadeRCNN(nn.Module):
    architecture_id = "cascade-rcnn"

    def __init__(self, in_channels=4, num_stages=2, hidden_channels=None):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (in_channels, num_stages)):
            raise ValueError("in_channels and num_stages must be positive")
        if hidden_channels is None:
            hidden_channels = in_channels
        if type(hidden_channels) is not int or hidden_channels < 1:
            raise ValueError("hidden_channels must be a positive integer or None")
        self.in_channels = in_channels
        self.stages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 4),
            )
            for _ in range(num_stages)
        )

    def _sample(self, features, boxes):
        batch, channels, height, width = features.shape
        boxes = _box_tensor(boxes)
        if boxes.ndim != 3 or boxes.shape[0] != batch:
            raise ValueError("boxes must have shape (batch, count, 4) or (count, 4)")
        center_x, center_y, _, _ = _box_geometry(boxes)
        grid_x = center_x * (2.0 / max(width - 1, 1)) - 1.0
        grid_y = center_y * (2.0 / max(height - 1, 1)) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
        sampled = F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2).reshape(batch, -1, channels)

    def forward(self, features, boxes):
        if features.ndim != 4:
            raise ValueError("features must have shape (batch, channels, height, width)")
        if features.shape[1] != self.in_channels:
            raise ValueError("feature channel count does not match in_channels")
        current = _box_tensor(boxes).to(device=features.device, dtype=features.dtype)
        if current.ndim == 2:
            current = current.unsqueeze(0).expand(features.shape[0], -1, -1)
        if current.ndim != 3 or current.shape[0] != features.shape[0]:
            raise ValueError("boxes must have shape (batch, count, 4) or (count, 4)")
        _box_geometry(current)
        deltas = []
        for stage in self.stages:
            delta = stage(self._sample(features, current))
            deltas.append(delta)
            current = decode_box_deltas(current, delta)
        return {"boxes": current, "deltas": deltas}
