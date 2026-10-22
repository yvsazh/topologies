import copy
import math
import numbers

import torch
from torch import nn


class _ParameterSnapshot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, snapshot):
        return snapshot.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class Tent(nn.Module):
    component_id = "tent"

    def __init__(
        self,
        model=None,
        optimizer=None,
        steps=1,
        episodic=False,
        learning_rate=0.001,
        input_channels=1,
        width=4,
        num_classes=3,
    ):
        super().__init__()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if type(episodic) is not bool or episodic not in (False, True):
            raise ValueError("episodic must be a boolean")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, numbers.Real)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a finite positive number")
        for name, value in (
            ("input_channels", input_channels),
            ("width", width),
            ("num_classes", num_classes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if model is not None and not isinstance(model, nn.Module):
            raise ValueError("model must be a torch.nn.Module or None")
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer must be a torch.optim.Optimizer or None")
        if model is None:
            model = nn.Sequential(
                nn.Conv2d(input_channels, width, kernel_size=3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(width, num_classes),
            )
        self.model = self.configure_model(model)
        params = self.collect_params(self.model)
        if not params:
            raise ValueError("model must contain affine BatchNorm2d parameters")
        if optimizer is None:
            optimizer = torch.optim.Adam(params, lr=float(learning_rate))
        else:
            optimized_params = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            if len(optimized_params) != len(params) or {
                id(parameter) for parameter in optimized_params
            } != {id(parameter) for parameter in params}:
                raise ValueError("optimizer must contain only the model's BatchNorm2d affine parameters")
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.model_state = copy.deepcopy(self.model.state_dict())
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    @staticmethod
    def configure_model(model):
        model.train()
        model.requires_grad_(False)
        batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        if not batch_norms:
            raise ValueError("model must contain at least one BatchNorm2d layer")
        for module in batch_norms:
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("model must contain affine BatchNorm2d parameters")
        return model

    @staticmethod
    def collect_params(model):
        return [
            parameter
            for module in model.modules()
            if isinstance(module, nn.BatchNorm2d)
            for parameter in (module.weight, module.bias)
            if parameter is not None and parameter.requires_grad
        ]

    @staticmethod
    def entropy(logits):
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("logits must have shape (batch, classes) with at least two classes")
        if not logits.is_floating_point():
            raise ValueError("logits must use a floating-point dtype")
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)

    def _functional_forward(self, x):
        parameters = {
            name: _ParameterSnapshot.apply(parameter, parameter.detach().clone())
            for name, parameter in self.model.named_parameters()
        }
        buffers = {
            name: buffer.detach().clone()
            for name, buffer in self.model.named_buffers()
        }
        return torch.func.functional_call(self.model, (parameters, buffers), (x,))

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("input must be an image tensor with shape (batch, channels, height, width)")
        if not x.is_floating_point():
            raise ValueError("input must use a floating-point dtype")
        if x.shape[0] < 1 or x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError("input batch, height, and width dimensions must be positive")
        if x.shape[0] * x.shape[2] * x.shape[3] < 2:
            raise ValueError("BatchNorm2d requires at least two values per channel")
        if self.episodic:
            self.reset()
        return_differentiable = torch.is_grad_enabled()
        return_outputs = None
        with torch.enable_grad():
            for index in range(self.steps):
                cpu_rng_state = torch.random.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
                outputs = self.model(x)
                if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
                    raise ValueError("model must return classification logits with shape (batch, classes)")
                if outputs.shape[0] != x.shape[0] or outputs.shape[1] < 2:
                    raise ValueError("model logits must have shape (batch, classes) with at least two classes")
                if return_differentiable and index == self.steps - 1:
                    updated_cpu_rng_state = torch.random.get_rng_state()
                    updated_cuda_rng_state = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
                    try:
                        torch.random.set_rng_state(cpu_rng_state)
                        if cuda_rng_state is not None:
                            torch.cuda.set_rng_state(cuda_rng_state, x.device)
                        return_outputs = self._functional_forward(x)
                    finally:
                        torch.random.set_rng_state(updated_cpu_rng_state)
                        if updated_cuda_rng_state is not None:
                            torch.cuda.set_rng_state(updated_cuda_rng_state, x.device)
                loss = self.entropy(outputs).mean()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        return return_outputs if return_differentiable else outputs.detach()

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
