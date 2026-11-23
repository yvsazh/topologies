import math

import torch


def _raise_value_error(message):
    error = ValueError(message)
    raise error


class _StableDriftControl(torch.optim.Optimizer):
    _AUTO_FALLBACK_BUDGET_SQ = 0.06
    _CALIBRATION_MAX_STEPS = 32
    _CALIBRATION_MIN_STEPS = 12
    _CALIBRATION_WINDOW = 5
    _HORIZON_CAP = 256
    _STATE_KEY = "_stable_drift_control"

    def __init__(
        self,
        params,
        *,
        lr,
        betas,
        eps,
        weight_decay,
        drift_budget_sq,
        dual_lr,
        lambda_init,
        deadzone,
        violation_memory,
        lambda_max,
    ):
        budget_is_auto = drift_budget_sq == "auto"
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "drift_budget_sq": drift_budget_sq,
            "dual_lr": dual_lr,
            "lambda_init": lambda_init,
            "deadzone": deadzone,
            "violation_memory": violation_memory,
            "lambda_max": lambda_max,
        }
        super().__init__(params, defaults)
        self._validate_groups(budget_is_auto)
        self._auto_budget = budget_is_auto
        self.defaults["drift_budget_sq"] = (
            self._AUTO_FALLBACK_BUDGET_SQ
            if self._auto_budget
            else self.param_groups[0]["drift_budget_sq"]
        )
        self._task_index = -1
        self._dual_lambda = 0.0
        self._drift_sq = 0.0
        self._calibrating = False
        self._calibration_steps = 0
        self._calibration_buffer = []
        self._violation_sum = 0.0
        self._violation_steps = 0

    @staticmethod
    def _number(value, name):
        if isinstance(value, bool):
            _raise_value_error(f"{name} must be a finite number")
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    def _validate_groups(self, budget_is_auto):
        if not self.param_groups:
            raise ValueError("at least one parameter group is required")
        controller_keys = (
            "drift_budget_sq",
            "dual_lr",
            "lambda_init",
            "deadzone",
            "violation_memory",
            "lambda_max",
        )
        controller_values = []
        betas_value = None
        for group in self.param_groups:
            group["lr"] = self._number(group["lr"], "lr")
            if group["lr"] <= 0.0:
                raise ValueError("lr must be positive")
            group["eps"] = self._number(group["eps"], "eps")
            if group["eps"] < 0.0:
                raise ValueError("eps must be non-negative")
            group["weight_decay"] = self._number(group["weight_decay"], "weight_decay")
            if group["weight_decay"] < 0.0:
                raise ValueError("weight_decay must be non-negative")
            try:
                pair = tuple(group["betas"])
            except TypeError as exc:
                raise ValueError("betas must contain exactly two values in [0, 1)") from exc
            if len(pair) != 2:
                raise ValueError("betas must contain exactly two values in [0, 1)")
            pair = tuple(self._number(value, "betas") for value in pair)
            if any(value < 0.0 or value >= 1.0 for value in pair):
                raise ValueError("betas must contain exactly two values in [0, 1)")
            group["betas"] = pair
            if betas_value is None:
                betas_value = pair
            elif pair != betas_value:
                raise ValueError("all parameter groups must use the same betas")

            budget = group["drift_budget_sq"]
            group_auto = isinstance(budget, str) and budget == "auto"
            if isinstance(budget, str) and not group_auto:
                raise ValueError("drift_budget_sq must be a positive number or 'auto'")
            if group_auto:
                budget = self._AUTO_FALLBACK_BUDGET_SQ
            else:
                budget = self._number(budget, "drift_budget_sq")
                if budget <= 0.0:
                    raise ValueError("drift_budget_sq must be positive")
            group["drift_budget_sq"] = budget
            if group_auto != budget_is_auto:
                raise ValueError("all parameter groups must use the same drift_budget_sq mode")

            group["dual_lr"] = self._number(group["dual_lr"], "dual_lr")
            if group["dual_lr"] <= 0.0:
                raise ValueError("dual_lr must be positive")
            group["lambda_init"] = self._number(group["lambda_init"], "lambda_init")
            if group["lambda_init"] < 0.0:
                raise ValueError("lambda_init must be non-negative")
            group["deadzone"] = self._number(group["deadzone"], "deadzone")
            if group["deadzone"] < 0.0 or group["deadzone"] >= 1.0:
                raise ValueError("deadzone must be in [0, 1)")
            if not isinstance(group["violation_memory"], bool):
                _raise_value_error("violation_memory must be a bool")
            group["lambda_max"] = self._number(group["lambda_max"], "lambda_max")
            if group["lambda_max"] <= 0.0:
                raise ValueError("lambda_max must be positive")
            controller_values.append(tuple(group[key] for key in controller_keys))

        if any(values != controller_values[0] for values in controller_values[1:]):
            raise ValueError("all parameter groups must use the same drift controller settings")

    @property
    def task_index(self):
        return self._task_index

    @property
    def dual_lambda(self):
        return self._dual_lambda

    @property
    def drift_sq(self):
        return self._drift_sq

    @property
    def drift_budget_sq(self):
        return self.param_groups[0]["drift_budget_sq"]

    @torch.no_grad()
    def start_task(self):
        self._task_index += 1
        if self._task_index == 1 and self._auto_budget:
            self._calibrate_budget()
            self._auto_budget = False
        self._calibrating = self._task_index == 0 and self._auto_budget
        self._violation_sum = 0.0
        self._violation_steps = 0
        if self._task_index == 0:
            self._dual_lambda = 0.0
        elif self._task_index == 1:
            self._dual_lambda = self.param_groups[0]["lambda_init"]
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                if parameter.requires_grad:
                    state["anchor"] = parameter.detach().clone(
                        memory_format=torch.preserve_format
                    )
                else:
                    state.pop("anchor", None)

    def _calibrate_budget(self):
        if self._calibration_steps < self._CALIBRATION_MIN_STEPS:
            budget = self._AUTO_FALLBACK_BUDGET_SQ
        else:
            tail_size = min(
                self._CALIBRATION_MAX_STEPS,
                max(math.ceil(0.1 * self._calibration_steps), self._CALIBRATION_MIN_STEPS),
            )
            tail = self._calibration_buffer[-tail_size:]
            sums = [
                max(sum(tail[index : index + self._CALIBRATION_WINDOW]), 1e-12)
                for index in range(len(tail) - self._CALIBRATION_WINDOW + 1)
            ]
            values = torch.tensor(sums, dtype=torch.float64)
            quantile = torch.quantile(values, 0.75).item()
            beta1, beta2 = self.param_groups[0]["betas"]
            horizon = min(
                math.ceil(math.sqrt(1.0 / ((1.0 - beta1) * (1.0 - beta2)))),
                self._HORIZON_CAP,
            )
            budget = horizon / self._CALIBRATION_WINDOW * quantile
            budget = float(budget)
        for group in self.param_groups:
            group["drift_budget_sq"] = budget

    @torch.no_grad()
    def step(self, closure=None):
        if self._task_index < 0:
            raise RuntimeError("Call start_task() before the first optimizer step")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        updates = []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("StableDriftControl does not support sparse gradients")
                state = self.state[parameter]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                if self._task_index == 0 and self._auto_budget and state.get("anchor") is None:
                    state["anchor"] = parameter.detach().clone(
                        memory_format=torch.preserve_format
                    )
                state["step"] += 1
                if self._task_index > 0:
                    anchor = state.get("anchor")
                    if anchor is None:
                        anchor = parameter.detach().clone(memory_format=torch.preserve_format)
                        state["anchor"] = anchor
                    gradient = gradient.add(parameter - anchor, alpha=2.0 * self._dual_lambda)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                step = state["step"]
                moment = exp_avg / (1.0 - beta1**step)
                variance = exp_avg_sq / (1.0 - beta2**step)
                direction = moment / (variance.sqrt() + group["eps"])
                updates.append((group, parameter, direction))

        if not updates:
            return loss

        calibration_update_sq = None
        if self._calibrating:
            for group, parameter, direction in updates:
                lr = group["lr"]
                decay = 1.0 - lr * group["weight_decay"]
                displacement = parameter * (decay - 1.0) - lr * decay * direction
                update_sq = (displacement * displacement).sum().to(torch.float64)
                if calibration_update_sq is None:
                    calibration_update_sq = update_sq
                else:
                    calibration_update_sq = calibration_update_sq + update_sq

        for group, parameter, direction in updates:
            parameter.add_(direction, alpha=-group["lr"])
            if group["weight_decay"] != 0.0:
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])

        if self._calibrating and calibration_update_sq is not None:
            self._calibration_steps += 1
            self._calibration_buffer.append(float(calibration_update_sq.item()))
            if len(self._calibration_buffer) > self._CALIBRATION_MAX_STEPS:
                self._calibration_buffer = self._calibration_buffer[-self._CALIBRATION_MAX_STEPS :]

        if self._task_index > 0:
            drift = None
            for group in self.param_groups:
                for parameter in group["params"]:
                    state = self.state.get(parameter)
                    anchor = state.get("anchor") if state is not None else None
                    if anchor is None:
                        continue
                    distance = parameter - anchor
                    contribution = (distance * distance).sum().to(torch.float64)
                    drift = contribution if drift is None else drift + contribution
            self._drift_sq = float(drift.item()) if drift is not None else 0.0
            self._update_dual()
        return loss

    def _update_dual(self):
        group = self.param_groups[0]
        budget = group["drift_budget_sq"]
        deadzone = group["deadzone"]
        lower_sq = budget * (1.0 - deadzone) ** 2
        upper_sq = budget * (1.0 + deadzone) ** 2
        prior_average = 0.0
        if group["violation_memory"] and self._violation_steps:
            prior_average = self._violation_sum / self._violation_steps
        if self._drift_sq > upper_sq:
            self._dual_lambda += group["dual_lr"] * (
                self._drift_sq - upper_sq + prior_average
            )
        elif self._drift_sq < lower_sq:
            self._dual_lambda *= (self._drift_sq / lower_sq) ** group["dual_lr"]
        self._dual_lambda = min(max(self._dual_lambda, 0.0), group["lambda_max"])
        if group["violation_memory"]:
            self._violation_sum += max(self._drift_sq - budget, 0.0)
            self._violation_steps += 1

    def state_dict(self):
        result = super().state_dict()
        result[self._STATE_KEY] = {
            "task_index": self._task_index,
            "dual_lambda": self._dual_lambda,
            "drift_sq": self._drift_sq,
            "auto_budget": self._auto_budget,
            "calibrating": self._calibrating,
            "calibration_steps": self._calibration_steps,
            "calibration_buffer": list(self._calibration_buffer),
            "violation_sum": self._violation_sum,
            "violation_steps": self._violation_steps,
        }
        return result

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict) or self._STATE_KEY not in state_dict:
            raise ValueError("state_dict is missing StableDriftControl task state")
        task_state = state_dict[self._STATE_KEY]
        optimizer_state = {key: value for key, value in state_dict.items() if key != self._STATE_KEY}
        super().load_state_dict(optimizer_state)
        self._task_index = int(task_state["task_index"])
        self._dual_lambda = float(task_state["dual_lambda"])
        self._drift_sq = float(task_state["drift_sq"])
        self._auto_budget = bool(task_state["auto_budget"])
        self._calibrating = bool(task_state["calibrating"])
        self._calibration_steps = int(task_state["calibration_steps"])
        self._calibration_buffer = list(task_state["calibration_buffer"])
        self._violation_sum = float(task_state["violation_sum"])
        self._violation_steps = int(task_state["violation_steps"])


def stable_drift_control(params, *, lr, **hyperparameters) -> torch.optim.Optimizer:
    defaults = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "drift_budget_sq": "auto",
        "dual_lr": 0.03,
        "lambda_init": 1.0,
        "deadzone": 0.05,
        "violation_memory": True,
        "lambda_max": 100.0,
    }
    unsupported = set(hyperparameters) - set(defaults)
    if unsupported:
        raise ValueError(f"unsupported StableDriftControl parameters: {', '.join(sorted(unsupported))}")
    defaults.update(hyperparameters)
    defaults["lr"] = _StableDriftControl._number(lr, "lr")
    if defaults["lr"] <= 0.0:
        raise ValueError("lr must be positive")
    return _StableDriftControl(params, **defaults)
