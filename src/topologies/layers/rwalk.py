import math
import numbers

import torch
from torch import nn


class RWalk(nn.Module):
    component_id = "rwalk"

    def __init__(
        self,
        parameters=None,
        strength=1000.0,
        fisher_decay=0.9,
        importance_interval=10,
        path_denominator_epsilon=1e-3,
        normalization_epsilon=1e-32,
        feature_passthrough=False,
    ):
        super().__init__()
        if not isinstance(feature_passthrough, bool):
            error = ValueError("feature_passthrough must be a boolean")
            raise error
        self.feature_passthrough = feature_passthrough
        self.strength = self._finite_real(strength, "strength")
        if self.strength < 0:
            raise ValueError("strength must be nonnegative")
        self.fisher_decay = self._finite_real(fisher_decay, "fisher_decay")
        if not 0 <= self.fisher_decay <= 1:
            raise ValueError("fisher_decay must be in [0, 1]")
        if isinstance(importance_interval, bool) or not isinstance(importance_interval, int):
            error = ValueError("importance_interval must be a positive integer")
            raise error
        if importance_interval <= 0:
            raise ValueError("importance_interval must be a positive integer")
        self.importance_interval = importance_interval
        self.path_denominator_epsilon = self._finite_real(
            path_denominator_epsilon, "path_denominator_epsilon"
        )
        if self.path_denominator_epsilon <= 0:
            raise ValueError("path_denominator_epsilon must be positive")
        self.normalization_epsilon = self._finite_real(
            normalization_epsilon, "normalization_epsilon"
        )
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        object.__setattr__(self, "_tracked_parameters", None)
        object.__setattr__(self, "_parameter_shapes", None)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_pending_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("_tasks_seen", torch.zeros((), dtype=torch.long))
        if parameters is not None:
            values = self._parameter_sequence(parameters, "parameters")
            self._initialize(values)

    @staticmethod
    def _finite_real(value, name):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            error = ValueError(f"{name} must be a finite real number")
            raise error
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite real number")
        return result

    @staticmethod
    def _parameter_sequence(values, name):
        if isinstance(values, torch.Tensor):
            result = (values,)
        else:
            try:
                result = tuple(values)
            except TypeError as error:
                raise ValueError(f"{name} must be a non-empty sequence of tensors") from error
        if not result:
            raise ValueError(f"{name} must not be empty")
        for value in result:
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ValueError(f"{name} must contain only floating-point tensors")
            if value.numel() == 0:
                raise ValueError(f"{name} tensors must not be empty")
        devices = {value.device for value in result}
        if len(devices) != 1:
            raise ValueError(f"{name} tensors must be on the same device")
        return result

    def _initialize(self, parameters):
        values = self._parameter_sequence(parameters, "parameters")
        if self._parameter_shapes is not None:
            self._check_shapes(values, "parameters")
            return
        object.__setattr__(self, "_tracked_parameters", values)
        object.__setattr__(self, "_parameter_shapes", tuple(value.shape for value in values))
        for index, parameter in enumerate(values):
            current = parameter.detach().clone()
            self.register_buffer(f"_anchor_{index}", current.clone())
            self.register_buffer(f"_fisher_{index}", torch.zeros_like(current))
            self.register_buffer(f"_consolidated_fisher_{index}", torch.zeros_like(current))
            self.register_buffer(f"_importance_{index}", torch.zeros_like(current))
            self.register_buffer(f"_task_importance_{index}", torch.zeros_like(current))
            self.register_buffer(f"_previous_{index}", current.clone())
            self.register_buffer(f"_interval_start_{index}", current.clone())
            self.register_buffer(f"_interval_loss_{index}", torch.zeros_like(current))

    def _check_shapes(self, values, name):
        values = self._parameter_sequence(values, name)
        if len(values) != len(self._parameter_shapes):
            raise ValueError(f"{name} count does not match initialized parameter count")
        for index, (value, shape) in enumerate(zip(values, self._parameter_shapes)):
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"{name}[{index}] shape does not match initialized parameter shape")
        return values

    def _states(self, prefix):
        if self._parameter_shapes is None:
            return ()
        return tuple(getattr(self, f"_{prefix}_{index}") for index in range(len(self._parameter_shapes)))

    @property
    def anchor(self):
        return self._states("anchor")

    @property
    def fisher(self):
        return self._states("fisher")

    @property
    def consolidated_fisher(self):
        return self._states("consolidated_fisher")

    @property
    def importance(self):
        return self._states("importance")

    @property
    def task_importance(self):
        return self._states("task_importance")

    def normalized_weights(self):
        if self._parameter_shapes is None:
            raise ValueError("parameters must be initialized before normalizing importance")
        fisher = self.consolidated_fisher
        importance = self.importance
        normalized = []
        for values in (fisher, importance):
            flattened = torch.cat([value.detach().reshape(-1) for value in values])
            minimum = flattened.min()
            maximum = flattened.max()
            denominator = maximum - minimum + self.normalization_epsilon
            normalized.append(tuple((value - minimum) / denominator for value in values))
        return normalized[0], normalized[1]

    def forward(self, parameters=None):
        if (
            self.feature_passthrough
            and self._parameter_shapes is None
            and isinstance(parameters, torch.Tensor)
        ):
            return parameters
        values = self._tracked_parameters if parameters is None else self._parameter_sequence(
            parameters, "parameters"
        )
        if values is None:
            raise ValueError("parameters must be provided on the first forward call")
        if self._parameter_shapes is None:
            self._initialize(values)
        else:
            values = self._check_shapes(values, "parameters")
        if int(self._tasks_seen.item()) == 0:
            return values[0].sum() * 0.0
        normalized_fisher, normalized_importance = self.normalized_weights()
        terms = [
            ((fisher + importance) * (parameter - anchor).square()).sum()
            for parameter, anchor, fisher, importance in zip(
                values, self.anchor, normalized_fisher, normalized_importance
            )
        ]
        return self.strength * torch.stack(terms).sum()

    def update(self, task_loss_gradients, post_optimizer_parameters):
        post_values = self._parameter_sequence(post_optimizer_parameters, "post_optimizer_parameters")
        if self._parameter_shapes is None:
            self._initialize(post_values)
        else:
            post_values = self._check_shapes(post_values, "post_optimizer_parameters")
        gradients = self._parameter_sequence(task_loss_gradients, "task_loss_gradients")
        gradients = self._check_shapes(gradients, "task_loss_gradients")
        step = int(self._step.item()) + 1
        with torch.no_grad():
            for index, (gradient, parameter) in enumerate(zip(gradients, post_values)):
                gradient = gradient.to(device=self.fisher[index].device, dtype=self.fisher[index].dtype)
                parameter = parameter.to(device=self.fisher[index].device, dtype=self.fisher[index].dtype)
                if not bool(torch.isfinite(gradient).all()):
                    raise ValueError(f"task_loss_gradients[{index}] must be finite")
                if not bool(torch.isfinite(parameter).all()):
                    raise ValueError(f"post_optimizer_parameters[{index}] must be finite")
                if step % self.importance_interval == 0:
                    fisher = self.fisher[index]
                    fisher.mul_(1.0 - self.fisher_decay).addcmul_(
                        gradient, gradient, value=self.fisher_decay
                    )
                displacement = parameter - self._previous(index)
                self._interval_loss(index).addcmul_(gradient, displacement, value=-1.0)
                self._previous(index).copy_(parameter)
            self._step.fill_(step)
            pending = int(self._pending_steps.item()) + 1
            self._pending_steps.fill_(pending)
            if step % self.importance_interval == 0:
                self._flush_path_importance(post_values)

    def _anchor(self, index):
        return getattr(self, f"_anchor_{index}")

    def _previous(self, index):
        return getattr(self, f"_previous_{index}")

    def _interval_start(self, index):
        return getattr(self, f"_interval_start_{index}")

    def _interval_loss(self, index):
        return getattr(self, f"_interval_loss_{index}")

    def _flush_path_importance(self, post_values):
        with torch.no_grad():
            for index, parameter in enumerate(post_values):
                current = parameter.to(
                    device=self.fisher[index].device, dtype=self.fisher[index].dtype
                )
                displacement = current - self._interval_start(index)
                denominator = (
                    0.5 * self.fisher[index] * displacement.square()
                    + self.path_denominator_epsilon
                )
                increment = (self._interval_loss(index) / denominator).clamp_min(0.0)
                self.task_importance[index].add_(increment)
                self._interval_start(index).copy_(current)
                self._interval_loss(index).zero_()
            self._pending_steps.zero_()

    def consolidate(self, parameters=None):
        if self._parameter_shapes is None:
            if parameters is None:
                raise ValueError("parameters must be initialized before consolidation")
            self._initialize(parameters)
        values = self._tracked_parameters if parameters is None else self._check_shapes(
            parameters, "parameters"
        )
        with torch.no_grad():
            if int(self._pending_steps.item()) > 0:
                self._flush_path_importance(values)
            for index, parameter in enumerate(values):
                current = parameter.detach().to(
                    device=self.fisher[index].device, dtype=self.fisher[index].dtype
                )
                self.importance[index].copy_(
                    0.5 * (self.importance[index] + self.task_importance[index])
                )
                self.consolidated_fisher[index].copy_(self.fisher[index])
                self.task_importance[index].zero_()
                self._anchor(index).copy_(current)
                self._previous(index).copy_(current)
                self._interval_start(index).copy_(current)
            self._tasks_seen.add_(1)
            self._step.zero_()
