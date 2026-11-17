import math
from numbers import Real

import torch
from torch import nn


class SoftNMS(nn.Module):
    component_id = "soft-nms"

    def __init__(
        self,
        method="gaussian",
        sigma=0.5,
        iou_threshold=0.3,
        score_threshold=None,
        return_scores_only=False,
    ):
        super().__init__()
        if not isinstance(method, str) or method not in ("gaussian", "linear"):
            raise ValueError("method must be 'gaussian' or 'linear'")
        if isinstance(sigma, bool) or not isinstance(sigma, Real) or not math.isfinite(float(sigma)) or sigma <= 0:
            raise ValueError("sigma must be a finite positive number")
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, Real)
            or not math.isfinite(float(iou_threshold))
            or not 0 <= iou_threshold <= 1
        ):
            raise ValueError("iou_threshold must be a finite number in [0, 1]")
        if score_threshold is not None and (
            isinstance(score_threshold, bool)
            or not isinstance(score_threshold, Real)
            or not math.isfinite(float(score_threshold))
        ):
            raise ValueError("score_threshold must be None or a finite number")
        if return_scores_only is not True and return_scores_only is not False:
            raise ValueError("return_scores_only must be a bool")
        self.method = method
        self.sigma = float(sigma)
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = None if score_threshold is None else float(score_threshold)
        self.return_scores_only = return_scores_only

    def forward(self, boxes, scores):
        if not torch.is_tensor(boxes) or not torch.is_tensor(scores):
            raise ValueError("boxes and scores must be torch.Tensor instances")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4)")
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        elif scores.ndim != 1:
            raise ValueError("scores must have shape (N,) or (N, 1)")
        if boxes.shape[0] != scores.shape[0]:
            raise ValueError("boxes and scores must have the same number of detections")
        if not boxes.is_floating_point() or not scores.is_floating_point():
            raise ValueError("boxes and scores must have floating-point dtypes")
        if boxes.device != scores.device:
            raise ValueError("boxes and scores must be on the same device")
        if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
            raise ValueError("boxes and scores must contain only finite values")
        if boxes.shape[0] and not bool(
            ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all()
        ):
            raise ValueError("boxes must have positive width and height")

        updated = [scores[index] for index in range(scores.shape[0])]
        active = list(range(scores.shape[0]))
        if self.score_threshold is not None:
            active = [index for index in active if float(scores[index].detach().item()) >= self.score_threshold]

        kept = []
        while active:
            active_scores = torch.stack([updated[index] for index in active])
            selected_position = int(torch.argmax(active_scores).item())
            selected_index = active.pop(selected_position)
            kept.append(selected_index)
            if not active:
                continue

            candidate_indices = torch.tensor(active, dtype=torch.long, device=boxes.device)
            candidates = boxes.index_select(0, candidate_indices)
            selected = boxes[selected_index]
            left_top = torch.where(selected[:2] >= candidates[:, :2], selected[:2], candidates[:, :2])
            right_bottom = torch.where(selected[2:] <= candidates[:, 2:], selected[2:], candidates[:, 2:])
            intersection_size = (right_bottom - left_top).clamp_min(0)
            intersection = intersection_size[:, 0] * intersection_size[:, 1]
            selected_area = (selected[2] - selected[0]) * (selected[3] - selected[1])
            candidate_areas = (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1])
            overlaps = intersection / (selected_area + candidate_areas - intersection)

            if self.method == "gaussian":
                decay = torch.exp(-(overlaps * overlaps) / self.sigma)
            else:
                decay = torch.where(overlaps >= self.iou_threshold, 1.0 - overlaps, torch.ones_like(overlaps))
            decayed = torch.stack([updated[index] for index in active]) * decay
            for position, index in enumerate(active):
                updated[index] = decayed[position]

            if self.score_threshold is not None:
                active = [
                    index
                    for index in active
                    if float(updated[index].detach().item()) >= self.score_threshold
                ]

        kept_indices = torch.tensor(kept, dtype=torch.long, device=scores.device)
        updated_scores = torch.stack(updated) if updated else scores.clone()
        if self.return_scores_only:
            return updated_scores
        return kept_indices, updated_scores
