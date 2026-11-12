import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


class PGDAttack(nn.Module):
    component_id = "pgd-attack"

    def __init__(
        self,
        epsilon=0.3,
        step_size=0.01,
        max_steps=40,
        random_start=True,
        clip_min=0.0,
        clip_max=1.0,
    ):
        super().__init__()
        if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real):
            error = ValueError("epsilon must be a finite positive number")
            raise error
        try:
            epsilon = float(epsilon)
        except OverflowError as error:
            raise ValueError("epsilon must be a finite positive number") from error
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        if isinstance(step_size, bool) or not isinstance(step_size, numbers.Real):
            error = ValueError("step_size must be a finite positive number")
            raise error
        try:
            step_size = float(step_size)
        except OverflowError as error:
            raise ValueError("step_size must be a finite positive number") from error
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be a finite positive number")
        if isinstance(max_steps, bool) or not isinstance(max_steps, numbers.Integral) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(random_start, bool):
            error = ValueError("random_start must be a boolean")
            raise error
        for name, value in (("clip_min", clip_min), ("clip_max", clip_max)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                error = ValueError(f"{name} must be a finite number")
                raise error
            try:
                finite_value = float(value)
            except OverflowError as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not math.isfinite(finite_value):
                raise ValueError(f"{name} must be a finite number")
            if name == "clip_min":
                clip_min = finite_value
            else:
                clip_max = finite_value
        if clip_min >= clip_max:
            raise ValueError("clip_min must be less than clip_max")
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_steps = int(max_steps)
        self.random_start = random_start
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def _logits(self, model, x):
        if isinstance(model, torch.Tensor):
            features = x.reshape(x.shape[0], -1)
            if model.ndim == 2:
                if model.shape[0] == 1 and model.shape[1] == features.shape[1]:
                    if not model.is_floating_point() or model.device != x.device or model.dtype != x.dtype:
                        raise ValueError("model weights must match the input device and floating-point dtype")
                    score = F.linear(features, model)
                    logits = torch.cat((score, -score), dim=1)
                elif model.shape[0] < 2 or model.shape[1] != features.shape[1]:
                    raise ValueError("model weights must have shape [num_classes, flattened_features]")
                else:
                    if not model.is_floating_point() or model.device != x.device or model.dtype != x.dtype:
                        raise ValueError("model weights must match the input device and floating-point dtype")
                    logits = F.linear(features, model)
            elif (model.ndim == 3 and model.shape[0] == x.shape[0]
                  and model.shape[1] >= 2 and model.shape[2] == features.shape[1]):
                if not model.is_floating_point() or model.device != x.device or model.dtype != x.dtype:
                    raise ValueError("model weights must match the input device and floating-point dtype")
                logits = torch.bmm(model, features.unsqueeze(-1)).squeeze(-1)
            else:
                raise ValueError("model weights must have shape [num_classes, flattened_features]")
        elif callable(model):
            logits = model(x)
        else:
            error = ValueError("model must be callable or a linear classifier weight tensor")
            raise error
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("model must return logits with shape [batch_size, num_classes]")
        if logits.shape[0] != x.shape[0] or logits.shape[1] < 2:
            raise ValueError("model logits must match the batch and contain at least two classes")
        if not logits.is_floating_point() or logits.device != x.device:
            raise ValueError("model logits must be floating point and on the input device")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("model logits must contain only finite values")
        return logits

    def forward(self, model, x, y):
        if not isinstance(x, torch.Tensor) or x.ndim < 2 or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor with shape [batch_size, ...]")
        if x.shape[0] == 0 or any(size == 0 for size in x.shape[1:]):
            raise ValueError("x must contain a nonempty batch and nonempty examples")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x must contain only finite values")
        if bool(((x < self.clip_min) | (x > self.clip_max)).any()):
            raise ValueError("x values must lie within the configured input range")
        if (not isinstance(y, torch.Tensor) or y.ndim not in (1, 2)
                or y.shape[0] != x.shape[0] or (y.ndim == 2 and y.shape[1] != 1)):
            raise ValueError("y must be an integer label tensor with shape [batch_size] or [batch_size, 1]")
        if y.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("y must contain integer class labels")
        if y.device != x.device:
            raise ValueError("x and y must be on the same device")
        labels = y.reshape(-1).to(dtype=torch.long)
        clean = x.detach()
        adversarial = clean.clone()
        if self.random_start:
            noise = torch.empty_like(adversarial).uniform_(-self.epsilon, self.epsilon)
            adversarial = (clean + noise).clamp(self.clip_min, self.clip_max)
        for _ in range(self.max_steps):
            with torch.enable_grad():
                candidate = adversarial.detach().requires_grad_(True)
                logits = self._logits(model, candidate)
                if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
                    raise ValueError("y contains a class label outside the model logits")
                loss = F.cross_entropy(logits, labels, reduction="sum")
                gradient = torch.autograd.grad(loss, candidate)[0]
            adversarial = candidate.detach() + self.step_size * gradient.sign()
            adversarial = torch.maximum(
                torch.minimum(adversarial, clean + self.epsilon), clean - self.epsilon
            )
            adversarial = adversarial.clamp(self.clip_min, self.clip_max)
        return adversarial.detach() + (x - x.detach())
