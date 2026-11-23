import math

import torch


class _GraphMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr,
        momentum=0.95,
        nesterov=True,
        tau=0.5,
        ns_steps=5,
        weight_decay=5e-4,
        epsilon_metric=1e-5,
        metric_sample_size=2048,
        max_full_dim=512,
        eigen_refresh_steps=10,
        eigen_change_tol=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
        auxiliary_weight_decay=0.0,
        ns_dtype=None,
    ):
        self._pending = {}
        self._groups = {}
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "tau": tau,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "epsilon_metric": epsilon_metric,
            "metric_sample_size": metric_sample_size,
            "max_full_dim": max_full_dim,
            "eigen_refresh_steps": eigen_refresh_steps,
            "eigen_change_tol": eigen_change_tol,
            "betas": betas,
            "eps": eps,
            "auxiliary_weight_decay": auxiliary_weight_decay,
            "ns_dtype": ns_dtype,
        }
        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        options = {**self.defaults, **param_group}
        for key in ("lr", "weight_decay", "auxiliary_weight_decay", "eigen_change_tol"):
            if not math.isfinite(options[key]) or options[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("epsilon_metric", "eps"):
            if not math.isfinite(options[key]) or options[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(options["momentum"]) or not 0 <= options["momentum"] < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not math.isfinite(options["tau"]) or not 0 <= options["tau"] <= 1:
            raise ValueError("tau must be in [0, 1]")
        if not isinstance(options["nesterov"], bool):
            raise TypeError("nesterov must be a bool")
        if not isinstance(options["betas"], (tuple, list)) or len(options["betas"]) != 2:
            raise TypeError("betas must contain two numbers")
        if not all(math.isfinite(beta) and 0 <= beta < 1 for beta in options["betas"]):
            raise ValueError("betas must contain two finite numbers in [0, 1)")
        for key, minimum in (("ns_steps", 1), ("metric_sample_size", 0), ("max_full_dim", 0)):
            if not isinstance(options[key], int) or isinstance(options[key], bool) or options[key] < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if not isinstance(options["eigen_refresh_steps"], int) or isinstance(
            options["eigen_refresh_steps"], bool
        ):
            raise TypeError("eigen_refresh_steps must be an integer")
        if options["ns_dtype"] not in (None, torch.float32, torch.float64, torch.bfloat16):
            raise ValueError("ns_dtype must be None, float32, float64 or bfloat16")
        super().add_param_group(param_group)
        for parameter in self.param_groups[-1]["params"]:
            if not parameter.is_floating_point():
                raise ValueError("GraphMuon requires real floating-point parameters")
            self._groups[parameter] = self.param_groups[-1]

    def register_graph(self, parameter, *, input_axis=1):
        if parameter not in self._groups or parameter.ndim != 2:
            raise ValueError("parameter must be a 2D tensor owned by this optimizer")
        if input_axis not in (0, 1):
            raise ValueError("input_axis must be 0 or 1")
        state = self.state[parameter]
        if "input_axis" in state and state["input_axis"] != input_axis:
            raise ValueError("cannot change input_axis after registration")
        state["input_axis"] = input_axis

    def needs_observation(self, parameter):
        return parameter in self.state and "input_axis" in self.state[parameter]

    @torch.no_grad()
    def observe(self, parameter, x_g, *, input_axis=1):
        self.register_graph(parameter, input_axis=input_axis)
        if not isinstance(x_g, torch.Tensor) or x_g.ndim < 2:
            raise ValueError("inputs must have shape [..., input dimension]")
        if x_g.shape[-1] != parameter.shape[input_axis]:
            raise ValueError("input dimension must match the selected parameter axis")
        if x_g.device != parameter.device:
            raise ValueError("inputs and parameter must be on the same device")
        x = x_g.detach().reshape(-1, x_g.shape[-1])
        rows = x.shape[0]
        if rows == 0:
            raise ValueError("cannot estimate covariance from zero rows")
        group = self._groups[parameter]
        if 0 < group["metric_sample_size"] < rows:
            indices = torch.randperm(rows, device=x.device)[: group["metric_sample_size"]]
            x = x[indices]
        dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        x = x.to(dtype)
        if x.shape[1] <= group["max_full_dim"]:
            moment = x.T @ x / x.shape[0]
        else:
            moment = x.square().mean(dim=0)
        weighted = moment * rows
        if parameter in self._pending:
            previous, count = self._pending[parameter]
            self._pending[parameter] = (previous + weighted, count + rows)
        else:
            self._pending[parameter] = (weighted, rows)

    @staticmethod
    def _zeropower(matrix, steps, dtype):
        work = matrix.to(dtype=dtype)
        transposed = work.shape[0] > work.shape[1]
        if transposed:
            work = work.T
        work = work / work.norm().clamp_min(1e-7)
        for _ in range(steps):
            gram = work @ work.T
            polynomial = torch.addmm(gram, gram, gram, beta=-4.7750, alpha=2.0315)
            work = torch.addmm(work, polynomial, work, beta=3.4445)
        return (work.T if transposed else work).to(dtype=matrix.dtype)

    @staticmethod
    def _normalized_covariance(covariance, epsilon):
        if covariance.ndim == 1:
            mean_energy = covariance.mean()
            if mean_energy <= 0:
                return torch.ones_like(covariance) + epsilon
            return covariance / mean_energy + epsilon
        mean_energy = covariance.diagonal().mean()
        if mean_energy <= 0:
            return torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device) * (
                1 + epsilon
            )
        normalized = covariance / mean_energy
        return (normalized + normalized.T) * 0.5 + torch.eye(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        ) * epsilon

    @staticmethod
    def _metric_power(normalized, state, group, step):
        if normalized.ndim == 1:
            return normalized.clamp_min(group["epsilon_metric"]).pow(group["tau"] / 2)
        epsilon = group["epsilon_metric"]
        refresh = group["eigen_refresh_steps"]
        cached = "eigenvalues" in state and "eigenvectors" in state
        age = step - state.get("eigen_step", 0)
        changed = True
        if cached:
            reference = state["eigen_reference"]
            relative_change = torch.linalg.vector_norm(normalized - reference) / (
                torch.linalg.vector_norm(reference) + epsilon
            )
            changed = relative_change.item() > group["eigen_change_tol"]
        if not cached or refresh <= 0 or age >= refresh or changed:
            values, vectors = torch.linalg.eigh(normalized)
            state["eigenvalues"] = values
            state["eigenvectors"] = vectors
            state["eigen_reference"] = normalized.clone()
            state["eigen_step"] = step
        else:
            values, vectors = state["eigenvalues"], state["eigenvectors"]
        return values.clamp_min(epsilon).pow(group["tau"] / 2), vectors

    def _graph_direction(self, matrix, covariance, state, group, step):
        direction_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
        metric = self._normalized_covariance(covariance, group["epsilon_metric"])
        state["normalized_covariance"] = metric
        if metric.ndim == 1:
            power = self._metric_power(metric, state, group, step)
            mapped = matrix * power[:, None]
            return self._zeropower(mapped, group["ns_steps"], self._ns_dtype(group, direction_dtype)) / power[
                :, None
            ]
        power, vectors = self._metric_power(metric, state, group, step)
        mapped = vectors @ (power[:, None] * (vectors.T @ matrix))
        orthogonal = self._zeropower(mapped, group["ns_steps"], self._ns_dtype(group, direction_dtype))
        return vectors @ ((vectors.T @ orthogonal) / power[:, None])

    @staticmethod
    def _ns_dtype(group, parameter_dtype):
        return group["ns_dtype"] or parameter_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or gradient.is_complex():
                    raise RuntimeError("GraphMuon requires dense, real gradients")
                if self.needs_observation(parameter) and parameter not in self._pending:
                    raise RuntimeError("Missing propagated inputs: call observe() before step()")
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    self._update(parameter, group)
        self._pending.clear()
        return loss

    def _update(self, parameter, group):
        state = self.state[parameter]
        state["step"] = state.get("step", 0) + 1
        dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        gradient = parameter.grad.to(dtype)
        if parameter.ndim != 2:
            beta1, beta2 = group["betas"]
            first = state.setdefault("exp_avg", torch.zeros_like(parameter, dtype=dtype))
            second = state.setdefault("exp_avg_sq", torch.zeros_like(parameter, dtype=dtype))
            first.mul_(beta1).add_(gradient, alpha=1 - beta1)
            second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            first_hat = first / (1 - beta1**state["step"])
            second_hat = second / (1 - beta2**state["step"])
            parameter.mul_(1 - group["lr"] * group["auxiliary_weight_decay"])
            parameter.add_(first_hat / (second_hat.sqrt() + group["eps"]), alpha=-group["lr"])
            return
        buffer = state.setdefault("momentum_buffer", torch.zeros_like(parameter, dtype=dtype))
        buffer.mul_(group["momentum"]).add_(gradient)
        momentum = gradient + group["momentum"] * buffer if group["nesterov"] else buffer
        input_axis = state.get("input_axis", 1)
        matrix = momentum.T if input_axis == 1 else momentum
        if parameter in self._pending:
            total, count = self._pending[parameter]
            covariance = total / count
            state["covariance"] = covariance
        else:
            covariance = state.get("covariance")
        if covariance is None:
            direction = self._zeropower(
                matrix, group["ns_steps"], self._ns_dtype(group, dtype)
            ).to(dtype)
            state.pop("normalized_covariance", None)
        else:
            covariance = covariance.to(dtype)
            direction = self._graph_direction(matrix, covariance, state, group, state["step"])
            direction = direction.to(dtype)
            product = covariance[:, None] * direction if covariance.ndim == 1 else covariance @ direction
            norm = direction.square().sum().clamp_min(torch.finfo(dtype).tiny)
            scale = (((direction * product).sum() / norm).clamp_min(0) + group["epsilon_metric"]).sqrt()
            direction = direction / scale
        rate = group["lr"] * math.sqrt(max(1, matrix.shape[0] / matrix.shape[1]))
        parameter.mul_(1 - rate * group["weight_decay"])
        update = direction.T if input_axis == 1 else direction
        parameter.add_(update.to(parameter.dtype), alpha=-rate)


def graph_muon(params, *, lr, **hyperparameters):
    allowed = {
        "momentum",
        "nesterov",
        "tau",
        "ns_steps",
        "weight_decay",
        "epsilon_metric",
        "metric_sample_size",
        "max_full_dim",
        "eigen_refresh_steps",
        "eigen_change_tol",
        "betas",
        "eps",
        "auxiliary_weight_decay",
        "ns_dtype",
    }
    unknown = set(hyperparameters) - allowed
    if unknown:
        raise ValueError(f"unsupported GraphMuon parameters: {', '.join(sorted(unknown))}")
    return _GraphMuon(params, lr=lr, **hyperparameters)
