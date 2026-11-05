import math
import numbers

import torch
import torch.nn.functional as F
from torch import nn


class FriendlyAdversarialPGD(nn.Module):
    component_id = "fat"

    def __init__(
        self,
        epsilon=0.031,
        step_size=0.007,
        max_steps=10,
        tau=0,
        random_start=False,
        clip_min=0.0,
        clip_max=1.0,
    ):
        super().__init__()
        self.epsilon = self._finite_number(epsilon, "epsilon", minimum=0.0)
        self.step_size = self._finite_number(step_size, "step_size", minimum=0.0, strict=True)
        self.max_steps = self._nonnegative_integer(max_steps, "max_steps")
        self.tau = self._nonnegative_integer(tau, "tau")
        if not isinstance(random_start, bool):
            raise self._invalid_value("random_start must be a boolean")
        self.random_start = random_start
        if (clip_min is None) != (clip_max is None):
            raise ValueError("clip_min and clip_max must both be numbers or both be None")
        if clip_min is None:
            self.clip_min = None
            self.clip_max = None
        else:
            self.clip_min = self._finite_number(clip_min, "clip_min")
            self.clip_max = self._finite_number(clip_max, "clip_max")
            if self.clip_min >= self.clip_max:
                raise ValueError("clip_min must be less than clip_max")

    @staticmethod
    def _invalid_value(message):
        return ValueError(message)

    @staticmethod
    def _finite_number(value, name, minimum=None, strict=False):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise FriendlyAdversarialPGD._invalid_value(f"{name} must be a finite number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if minimum is not None and (value <= minimum if strict else value < minimum):
            relation = "greater than" if strict else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
        return value

    @staticmethod
    def _nonnegative_integer(value, name):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(value)

    @staticmethod
    def _logits(model, x, batch_size):
        logits = model(x)
        if not isinstance(logits, torch.Tensor):
            raise FriendlyAdversarialPGD._invalid_value("model must return a logits tensor")
        if logits.ndim != 2 or logits.shape[0] != batch_size or logits.shape[1] < 2:
            raise ValueError("model must return logits with shape [batch_size, num_classes]")
        return logits

    def forward(self, model, x, y):
        if isinstance(model, torch.Tensor):
            if model.ndim != 2 or model.shape[0] < 1:
                raise ValueError("tensor model weights must have shape [num_classes, input_features]")
            weights = model

            def model(current):
                flattened = current.reshape(current.shape[0], -1)
                if weights.shape == flattened.shape:
                    scores = (flattened * weights).sum(dim=1)
                    return torch.stack((scores, torch.zeros_like(scores)), dim=1)
                return F.linear(flattened, weights)

        if not callable(model):
            raise self._invalid_value("model must be callable")
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor with a batch dimension")
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one example")
        if not isinstance(y, torch.Tensor):
            raise self._invalid_value("y must be a label tensor with shape [batch_size] or [batch_size, 1]")
        if (y.ndim not in (1, 2) or y.shape[0] != x.shape[0]
                or (y.ndim == 2 and y.shape[1] != 1)):
            raise ValueError("y must be a label tensor with shape [batch_size] or [batch_size, 1]")
        if y.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("y must contain integer class labels")
        labels = y.reshape(-1).to(device=x.device, dtype=torch.long)
        original = x.detach()
        adversarial = original.clone()
        if self.random_start:
            noise = torch.empty_like(adversarial).uniform_(-self.epsilon, self.epsilon)
            adversarial = original + noise
            adversarial = torch.maximum(torch.minimum(adversarial, original + self.epsilon), original - self.epsilon)
            if self.clip_min is not None:
                adversarial = adversarial.clamp(self.clip_min, self.clip_max)
        tau_remaining = torch.full((x.shape[0],), self.tau, device=x.device, dtype=torch.long)
        active = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        for step in range(self.max_steps):
            with torch.no_grad():
                logits = self._logits(model, adversarial, x.shape[0])
                if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
                    raise ValueError("y contains a class label outside the model logits")
                predictions = logits.argmax(dim=1)
                misclassified = predictions.ne(labels)
                stopped = active & misclassified & tau_remaining.eq(0)
                active = active & ~stopped
                tau_remaining = torch.where(active & misclassified, tau_remaining - 1, tau_remaining)
            if not bool(active.any()):
                break
            with torch.enable_grad():
                if step == 0 and not self.random_start and x.requires_grad:
                    candidate = x
                else:
                    candidate = adversarial.detach().requires_grad_(True)
                logits = self._logits(model, candidate, x.shape[0])
                losses = F.cross_entropy(logits, labels, reduction="none")
                gradient = torch.autograd.grad((losses * active.to(losses.dtype)).sum(), candidate)[0]
            update = self.step_size * gradient.sign()
            update = update * active.reshape((x.shape[0],) + (1,) * (x.ndim - 1)).to(update.dtype)
            adversarial = candidate.detach() + update
            adversarial = torch.maximum(torch.minimum(adversarial, original + self.epsilon), original - self.epsilon)
            if self.clip_min is not None:
                adversarial = adversarial.clamp(self.clip_min, self.clip_max)
        return adversarial.detach()
