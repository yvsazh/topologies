import math
from numbers import Real

import torch
from torch import nn


class DIoUNMS(nn.Module):
    component_id = "diou-nms"

    def __init__(self, nms_threshold=0.45):
        super().__init__()
        invalid_threshold_type = isinstance(nms_threshold, bool) or not isinstance(
            nms_threshold, Real
        )
        if invalid_threshold_type:
            raise ValueError("nms_threshold must be a finite number in [0, 1]")
        if not math.isfinite(float(nms_threshold)) or not 0 <= nms_threshold <= 1:
            raise ValueError("nms_threshold must be a finite number in [0, 1]")
        self.nms_threshold = float(nms_threshold)

    def forward(self, boxes, scores):
        invalid_tensor_inputs = not isinstance(boxes, torch.Tensor) or not isinstance(
            scores, torch.Tensor
        )
        if invalid_tensor_inputs:
            raise ValueError("boxes and scores must be torch.Tensor values")
        single_set = boxes.ndim == 2 and scores.ndim == 1
        batched_sets = boxes.ndim == 3 and scores.ndim == 2
        if not single_set and not batched_sets:
            raise ValueError("boxes and scores must have shapes [N, 4]/[N] or [B, N, 4]/[B, N]")
        if boxes.shape[-1] != 4:
            raise ValueError("boxes must use xyxy format with four coordinates")
        if boxes.shape[-2] != scores.shape[-1]:
            raise ValueError("scores must have shape [N] matching boxes")
        if batched_sets and boxes.shape[0] != scores.shape[0]:
            raise ValueError("batched boxes and scores must have the same batch size")
        if not boxes.is_floating_point() or not scores.is_floating_point():
            raise ValueError("boxes and scores must use floating-point dtypes")
        if boxes.device != scores.device:
            raise ValueError("boxes and scores must be on the same device")
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise ValueError("boxes and scores must contain only finite values")
        if torch.any(boxes[..., 2:] <= boxes[..., :2]):
            raise ValueError("boxes must have positive width and height")

        if single_set:
            kept = self._keep_indices(boxes, scores)
            return torch.tensor(kept, dtype=torch.long, device=boxes.device)

        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for batch_index, (box_set, score_set) in enumerate(zip(boxes, scores)):
            kept = self._keep_indices(box_set, score_set)
            keep_mask[batch_index, kept] = True
        return scores * keep_mask.to(dtype=scores.dtype)

    def _keep_indices(self, boxes, scores):
        order = sorted(
            range(boxes.shape[0]), key=lambda index: (-float(scores[index].detach()), index)
        )
        kept = []
        with torch.no_grad():
            while order:
                current = order.pop(0)
                kept.append(current)
                if not order:
                    continue
                candidate_indices = torch.tensor(order, dtype=torch.long, device=boxes.device)
                current_box = boxes[current]
                candidates = boxes[candidate_indices]
                intersection_min = torch.maximum(current_box[:2], candidates[:, :2])
                intersection_max = torch.minimum(current_box[2:], candidates[:, 2:])
                intersection = (intersection_max - intersection_min).clamp_min(0).prod(dim=1)
                current_size = current_box[2:] - current_box[:2]
                candidate_sizes = candidates[:, 2:] - candidates[:, :2]
                union = current_size.prod() + candidate_sizes.prod(dim=1) - intersection
                iou = intersection / union
                current_center = (current_box[:2] + current_box[2:]) / 2
                candidate_centers = (candidates[:, :2] + candidates[:, 2:]) / 2
                center_distance = (current_center - candidate_centers).pow(2).sum(dim=1)
                enclosing_min = torch.minimum(current_box[:2], candidates[:, :2])
                enclosing_max = torch.maximum(current_box[2:], candidates[:, 2:])
                enclosing_diagonal = (enclosing_max - enclosing_min).pow(2).sum(dim=1)
                criterion = iou - center_distance / enclosing_diagonal
                order = [
                    index
                    for index, value in zip(order, criterion)
                    if float(value) < self.nms_threshold
                ]
        return kept
