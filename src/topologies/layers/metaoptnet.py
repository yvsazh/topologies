import math

import torch
from torch import nn


def _project_capped_simplex(values, labels, cap):
    classes = values.shape[1]
    valid = ~torch.nn.functional.one_hot(labels, num_classes=classes).to(torch.bool)
    positive = values.clamp_min(0).masked_fill(~valid, 0)
    over = positive.sum(dim=1, keepdim=True) > cap
    sentinel = torch.finfo(values.dtype).min / 4
    ordered = values.masked_fill(~valid, sentinel).sort(dim=1, descending=True).values
    cumulative = ordered.cumsum(dim=1) - cap
    ranks = torch.arange(1, classes + 1, device=values.device, dtype=values.dtype).unsqueeze(0)
    eligible = (ordered - cumulative / ranks > 0) & (ordered > sentinel / 2)
    rho = eligible.sum(dim=1).clamp_min(1)
    theta = cumulative.gather(1, (rho - 1).unsqueeze(1)) / rho.to(values.dtype).unsqueeze(1)
    projected = (values - theta).clamp_min(0).masked_fill(~valid, 0)
    return torch.where(over, projected, positive)


def _projected_dual_solution(hessian, labels, cap):
    examples, classes = labels.shape[0], hessian.shape[0] // labels.shape[0]
    linear = torch.ones((examples, classes), device=hessian.device, dtype=hessian.dtype)
    linear.scatter_(1, labels.unsqueeze(1), 0)
    largest = torch.linalg.eigvalsh(hessian).amax().clamp_min(1e-8)
    estimate = torch.zeros_like(linear)
    extrapolated = estimate.clone()
    momentum = 1.0
    tolerance = max(torch.finfo(hessian.dtype).eps * 20, 1e-10)
    for _ in range(1024):
        gradient = (hessian @ extrapolated.reshape(-1)).reshape_as(extrapolated) - linear
        updated = _project_capped_simplex(extrapolated - gradient / largest, labels, cap)
        next_momentum = (1 + math.sqrt(1 + 4 * momentum * momentum)) / 2
        extrapolated = updated + ((momentum - 1) / next_momentum) * (updated - estimate)
        difference = (updated - estimate).abs().amax()
        estimate = updated
        momentum = next_momentum
        if difference.item() <= tolerance:
            break
    return estimate


def _implicit_dual_solution(hessian, labels, initial, cap):
    classes = initial.shape[1]
    flat_initial = initial.detach().reshape(-1)
    free_tolerance = max(cap * 1e-6, torch.finfo(initial.dtype).eps * 20)
    cap_tolerance = max(cap * 1e-4, torch.finfo(initial.dtype).eps * 100)
    free_indices = torch.nonzero(flat_initial > free_tolerance, as_tuple=False).flatten()
    if free_indices.numel() == 0:
        return torch.zeros_like(initial)
    row_sums = initial.detach().sum(dim=1)
    capped_rows = torch.nonzero(row_sums >= cap - cap_tolerance, as_tuple=False).flatten()
    free_rows = torch.div(free_indices, classes, rounding_mode="floor")
    if capped_rows.numel():
        constraints = (capped_rows.unsqueeze(1) == free_rows.unsqueeze(0)).to(initial.dtype)
        constraints = constraints * (free_indices.unsqueeze(0) % classes != labels[capped_rows].unsqueeze(1)).to(initial.dtype)
        active_constraints = constraints.sum(dim=1) > 0
        constraints = constraints[active_constraints]
    else:
        constraints = initial.new_zeros((0, free_indices.numel()))
    reduced_hessian = hessian.index_select(0, free_indices).index_select(1, free_indices)
    linear = torch.ones(free_indices.numel(), device=initial.device, dtype=initial.dtype)
    if constraints.shape[0]:
        top = torch.cat((reduced_hessian, constraints.transpose(0, 1)), dim=1)
        bottom = torch.cat((constraints, initial.new_zeros((constraints.shape[0], constraints.shape[0]))), dim=1)
        system = torch.cat((top, bottom), dim=0)
        target = torch.cat((linear, initial.new_full((constraints.shape[0],), cap)))
    else:
        system = reduced_hessian
        target = linear
    try:
        solved = torch.linalg.solve(system, target)[:free_indices.numel()]
    except RuntimeError:
        solved = (torch.linalg.pinv(system) @ target)[:free_indices.numel()]
    return torch.zeros_like(flat_initial).scatter(0, free_indices, solved).reshape_as(initial)


class MetaOptNet(nn.Module):
    component_id = "metaoptnet"

    def __init__(self, num_classes, scale_init, C=0.1):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2")
        if isinstance(scale_init, bool) or not isinstance(scale_init, (int, float)) or not math.isfinite(scale_init):
            raise ValueError("scale_init must be a finite number")
        if isinstance(C, bool) or not isinstance(C, (int, float)) or not math.isfinite(C) or C <= 0:
            raise ValueError("C must be a finite positive number")
        self.num_classes = num_classes
        self.C = float(C)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, support_features, support_labels, query_features):
        if not all(torch.is_tensor(value) for value in (support_features, support_labels, query_features)):
            raise ValueError("support features, support labels, and query features must be tensors")
        if support_features.ndim != 3 or query_features.ndim != 3:
            raise ValueError("support and query features must have shape [B, N or Q, D]")
        if support_features.shape[0] != query_features.shape[0] or support_features.shape[2] != query_features.shape[2]:
            raise ValueError("support and query feature dimensions must match")
        if query_features.shape[1] < 1:
            raise ValueError("query features must be nonempty")
        if support_features.shape[2] < 1:
            raise ValueError("feature dimension must be positive")
        if not support_features.is_floating_point() or not query_features.is_floating_point():
            raise ValueError("support and query features must use floating point dtypes")
        if support_features.dtype not in (torch.float32, torch.float64) or query_features.dtype not in (
            torch.float32, torch.float64
        ):
            raise ValueError("support and query features must use float32 or float64 dtypes")
        if support_features.dtype != query_features.dtype or support_features.device != query_features.device:
            raise ValueError("support and query features must share dtype and device")
        if support_labels.shape != support_features.shape[:2] or support_labels.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError("support labels must be an integer tensor with shape [B, N]")
        if support_labels.device != support_features.device:
            raise ValueError("support labels and features must share a device")
        if not torch.isfinite(support_features).all() or not torch.isfinite(query_features).all():
            raise ValueError("support and query features must be finite")
        if torch.any(support_labels < 0) or torch.any(support_labels >= self.num_classes):
            raise ValueError(f"labels must be in [0, {self.num_classes - 1}]")
        support_labels = support_labels.to(dtype=torch.long)
        expected_classes = torch.arange(self.num_classes, device=support_labels.device)
        for episode_labels in support_labels:
            if not torch.equal(torch.unique(episode_labels), expected_classes):
                raise ValueError("every class must appear in each support episode")
        batch, count, dimension = support_features.shape
        class_eye = torch.eye(self.num_classes, dtype=support_features.dtype, device=support_features.device)
        label_vectors = torch.nn.functional.one_hot(support_labels, num_classes=self.num_classes).to(support_features.dtype)
        class_effect = label_vectors.unsqueeze(2) - class_eye.unsqueeze(0).unsqueeze(0)
        coefficients = class_effect.unsqueeze(-1) * support_features.unsqueeze(2).unsqueeze(2)
        coefficients = coefficients.reshape(batch, count * self.num_classes, self.num_classes * dimension)
        hessians = coefficients @ coefficients.transpose(1, 2)
        solved = []
        for index in range(batch):
            with torch.no_grad():
                initial = _projected_dual_solution(hessians[index].detach(), support_labels[index], self.C)
            solved.append(_implicit_dual_solution(hessians[index], support_labels[index], initial, self.C))
        beta = torch.stack(solved)
        alpha = -beta + label_vectors * beta.sum(dim=2, keepdim=True)
        weights = torch.einsum("bnk,bnd->bdk", alpha, support_features)
        return self.scale.to(dtype=query_features.dtype) * torch.einsum("bqd,bdk->bqk", query_features, weights)
