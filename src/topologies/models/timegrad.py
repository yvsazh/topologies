import torch
from torch import nn


class TimeGrad(nn.Module):
    architecture_id = "timegrad"
    PRESETS = {
        "tiny": {
            "input_size": 1,
            "covariate_size": 0,
            "hidden_size": 8,
            "recurrent_layers": 1,
            "rnn_type": "lstm",
            "diffusion_steps": 4,
            "beta_start": 0.0001,
            "beta_end": 0.1,
            "time_embedding_dim": 16,
            "max_noise_steps": 500,
            "residual_layers": 2,
            "residual_channels": 8,
            "kernel_size": 3,
        },
        "small": {
            "input_size": 1,
            "covariate_size": 0,
            "hidden_size": 16,
            "recurrent_layers": 1,
            "rnn_type": "lstm",
            "diffusion_steps": 32,
            "beta_start": 0.0001,
            "beta_end": 0.1,
            "time_embedding_dim": 32,
            "max_noise_steps": 500,
            "residual_layers": 4,
            "residual_channels": 8,
            "kernel_size": 3,
        },
        "base": {
            "input_size": 1,
            "covariate_size": 0,
            "hidden_size": 40,
            "recurrent_layers": 2,
            "rnn_type": "lstm",
            "diffusion_steps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.1,
            "time_embedding_dim": 32,
            "max_noise_steps": 500,
            "residual_layers": 8,
            "residual_channels": 8,
            "kernel_size": 3,
        },
    }

    def __init__(
        self,
        input_size=1,
        covariate_size=0,
        hidden_size=40,
        recurrent_layers=2,
        rnn_type="lstm",
        diffusion_steps=100,
        beta_start=0.0001,
        beta_end=0.1,
        time_embedding_dim=32,
        max_noise_steps=500,
        residual_layers=8,
        residual_channels=8,
        kernel_size=3,
    ):
        super().__init__()

        integer_arguments = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "recurrent_layers": recurrent_layers,
            "diffusion_steps": diffusion_steps,
            "time_embedding_dim": time_embedding_dim,
            "max_noise_steps": max_noise_steps,
            "residual_layers": residual_layers,
            "residual_channels": residual_channels,
            "kernel_size": kernel_size,
        }
        for name, value in integer_arguments.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(covariate_size, int) or isinstance(covariate_size, bool) or covariate_size < 0:
            raise ValueError("covariate_size must be a non-negative integer")
        if not isinstance(rnn_type, str) or rnn_type.lower() not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        if time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even")
        if max_noise_steps < diffusion_steps:
            raise ValueError("max_noise_steps must be at least diffusion_steps")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for circular padding")
        if (
            not isinstance(beta_start, (int, float))
            or isinstance(beta_start, bool)
            or not 0.0 < beta_start < 1.0
        ):
            raise ValueError("beta_start must be between zero and one")
        if (
            not isinstance(beta_end, (int, float))
            or isinstance(beta_end, bool)
            or not 0.0 < beta_end < 1.0
            or beta_end < beta_start
        ):
            raise ValueError("beta_end must be between beta_start and one")

        self.input_size = input_size
        self.covariate_size = covariate_size
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.rnn_type = rnn_type.lower()
        self.diffusion_steps = diffusion_steps
        self.time_embedding_dim = time_embedding_dim
        self.max_noise_steps = max_noise_steps
        self.residual_layers = residual_layers
        self.residual_channels = residual_channels
        self.kernel_size = kernel_size

        recurrent = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = recurrent(
            input_size + covariate_size,
            hidden_size,
            num_layers=recurrent_layers,
            batch_first=True,
        )
        self.denoiser = nn.ModuleDict(
            {
                "input_projection": nn.Conv1d(1, residual_channels, kernel_size=1),
                "time_projection": nn.Linear(time_embedding_dim, residual_channels),
                "context_projection": nn.Linear(hidden_size, residual_channels),
                "gated_convolutions": nn.ModuleList(
                    nn.Conv1d(
                        residual_channels,
                        residual_channels * 2,
                        kernel_size=kernel_size,
                        dilation=2 ** (index % 2),
                    )
                    for index in range(residual_layers)
                ),
                "condition_projections": nn.ModuleList(
                    nn.Conv1d(residual_channels, residual_channels * 2, kernel_size=1)
                    for _ in range(residual_layers)
                ),
                "residual_projections": nn.ModuleList(
                    nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
                    for _ in range(residual_layers)
                ),
                "skip_projections": nn.ModuleList(
                    nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
                    for _ in range(residual_layers)
                ),
                "output_projection": nn.Sequential(
                    nn.ReLU(),
                    nn.Conv1d(residual_channels, residual_channels, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv1d(residual_channels, 1, kernel_size=1),
                ),
            }
        )

        betas = torch.linspace(float(beta_start), float(beta_end), diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance)

    def _validate_history(self, history, history_covariates):
        if not isinstance(history, torch.Tensor) or history.ndim != 3:
            raise ValueError("history must have shape [batch, length, input_size]")
        if history.shape[1] == 0 or history.shape[2] != self.input_size:
            raise ValueError("history length and input_size must be valid")
        if not history.is_floating_point():
            raise ValueError("history must have a floating point dtype")
        if history_covariates is None:
            if self.covariate_size:
                raise ValueError("history_covariates are required when covariate_size is positive")
        elif (
            not isinstance(history_covariates, torch.Tensor)
            or history_covariates.ndim != 3
            or history_covariates.shape[:2] != history.shape[:2]
            or history_covariates.shape[2] != self.covariate_size
            or not history_covariates.is_floating_point()
        ):
            raise ValueError("history_covariates must match history batch, length, and covariate size")
        return history

    def _validate_target(self, target, batch_size, name):
        if not isinstance(target, torch.Tensor) or target.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, input_size]")
        if target.shape[0] != batch_size:
            raise ValueError(f"{name} batch must match history batch")
        if target.shape[1] != self.input_size:
            raise ValueError(f"{name} input_size must match the model")
        if not target.is_floating_point():
            raise ValueError(f"{name} must have a floating point dtype")

    def _canonical_steps(self, diffusion_step, batch_size, device):
        if isinstance(diffusion_step, torch.Tensor):
            if diffusion_step.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("diffusion_step must contain integer indices")
            if diffusion_step.ndim == 0:
                steps = diffusion_step.to(device=device, dtype=torch.long).expand(batch_size)
            elif diffusion_step.ndim == 1 and diffusion_step.shape[0] == batch_size:
                steps = diffusion_step.to(device=device, dtype=torch.long)
            else:
                raise ValueError("diffusion_step must be a scalar or have one value per batch item")
        elif isinstance(diffusion_step, int) and not isinstance(diffusion_step, bool):
            steps = torch.full((batch_size,), diffusion_step, device=device, dtype=torch.long)
        else:
            self._invalid_step()
        if torch.any(steps < 1).item() or torch.any(steps > self.diffusion_steps).item():
            raise ValueError("diffusion_step must be in the range [1, diffusion_steps]")
        return steps

    def _invalid_step(self):
        raise ValueError("diffusion_step must be an integer, scalar tensor, or batch tensor")

    def normalize_context(self, history):
        if not isinstance(history, torch.Tensor) or history.ndim != 3:
            raise ValueError("history must have shape [batch, length, input_size]")
        if history.shape[1] == 0 or history.shape[2] != self.input_size:
            raise ValueError("history length and input_size must be valid")
        if not history.is_floating_point():
            raise ValueError("history must have a floating point dtype")
        scales = history.mean(dim=1)
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        return history / scales.unsqueeze(1), scales

    def _encode(self, history, history_covariates=None):
        self._validate_history(history, history_covariates)
        normalized, scales = self.normalize_context(history)
        if history_covariates is not None:
            normalized = torch.cat((normalized, history_covariates), dim=-1)
        _, state = self.rnn(normalized)
        return state, scales

    def _state_features(self, state):
        hidden = state[0] if self.rnn_type == "lstm" else state
        return hidden[-1]

    def _time_embedding(self, steps):
        half_dim = self.time_embedding_dim // 2
        positions = steps.to(dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            -torch.log(torch.tensor(float(self.max_noise_steps), device=steps.device))
            * torch.arange(half_dim, device=steps.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        angles = positions * frequencies.unsqueeze(0)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def _predict_noise(self, noisy_target, state, steps):
        features = self._state_features(state)
        time_features = self._time_embedding(steps).to(dtype=noisy_target.dtype)
        time_features = self.denoiser["time_projection"](time_features)
        context_features = self.denoiser["context_projection"](features)
        x = self.denoiser["input_projection"](noisy_target.unsqueeze(1))
        skips = []
        for index, (convolution, condition, residual, skip) in enumerate(
            zip(
                self.denoiser["gated_convolutions"],
                self.denoiser["condition_projections"],
                self.denoiser["residual_projections"],
                self.denoiser["skip_projections"],
            )
        ):
            dilation = 2 ** (index % 2)
            padding = dilation * (self.kernel_size // 2)
            circular_indices = torch.arange(-padding, x.shape[-1] + padding, device=x.device) % x.shape[-1]
            convolved = convolution(x.index_select(-1, circular_indices))
            conditioned = time_features + context_features
            convolved = convolved + condition(conditioned.unsqueeze(-1))
            gate, value = convolved.chunk(2, dim=1)
            activated = torch.sigmoid(gate) * torch.tanh(value)
            x = (x + residual(activated)) / (2.0**0.5)
            skips.append(skip(activated))
        output = torch.stack(skips, dim=0).sum(dim=0)
        return self.denoiser["output_projection"](output).squeeze(1)

    def forward(self, history, noisy_target, diffusion_step, history_covariates=None):
        self._validate_history(history, history_covariates)
        self._validate_target(noisy_target, history.shape[0], "noisy_target")
        steps = self._canonical_steps(diffusion_step, history.shape[0], history.device)
        state, _ = self._encode(history, history_covariates)
        return self._predict_noise(noisy_target, state, steps)

    def q_sample(self, target, noise, diffusion_step):
        if not isinstance(target, torch.Tensor) or target.ndim != 2:
            raise ValueError("target must have shape [batch, input_size]")
        self._validate_target(target, target.shape[0], "target")
        self._validate_target(noise, target.shape[0], "noise")
        if noise.shape != target.shape:
            raise ValueError("noise shape must match target shape")
        steps = self._canonical_steps(diffusion_step, target.shape[0], target.device)
        alpha_bar = self.alpha_bars.index_select(0, steps - 1).to(dtype=target.dtype).unsqueeze(1)
        return alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise

    def training_loss(
        self,
        history,
        target,
        noise=None,
        diffusion_step=None,
        history_covariates=None,
    ):
        self._validate_history(history, history_covariates)
        self._validate_target(target, history.shape[0], "target")
        state, scales = self._encode(history, history_covariates)
        normalized_target = target / scales
        if noise is None:
            noise = torch.randn_like(normalized_target)
        self._validate_target(noise, history.shape[0], "noise")
        if noise.shape != normalized_target.shape:
            raise ValueError("noise shape must match target shape")
        if diffusion_step is None:
            diffusion_step = torch.randint(
                1,
                self.diffusion_steps + 1,
                (history.shape[0],),
                device=history.device,
            )
        steps = self._canonical_steps(diffusion_step, history.shape[0], history.device)
        noisy_target = self.q_sample(normalized_target, noise, steps)
        prediction = self._predict_noise(noisy_target, state, steps)
        return torch.mean((noise - prediction) ** 2)

    def _reverse_normalized(self, state, initial_noise):
        sample = initial_noise
        for step in range(self.diffusion_steps, 0, -1):
            steps = torch.full((sample.shape[0],), step, device=sample.device, dtype=torch.long)
            predicted_noise = self._predict_noise(sample, state, steps)
            beta = self.betas[step - 1].to(dtype=sample.dtype)
            alpha = self.alphas[step - 1].to(dtype=sample.dtype)
            alpha_bar = self.alpha_bars[step - 1].to(dtype=sample.dtype)
            variance = self.posterior_variance[step - 1].to(dtype=sample.dtype)
            sample = (sample - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
            if step > 1:
                sample = sample + variance.sqrt() * torch.randn_like(sample)
        return sample

    def reverse_sample(self, history, initial_noise=None, history_covariates=None):
        self._validate_history(history, history_covariates)
        state, scales = self._encode(history, history_covariates)
        if initial_noise is None:
            initial_noise = torch.randn(
                history.shape[0], self.input_size, device=history.device, dtype=history.dtype
            )
        self._validate_target(initial_noise, history.shape[0], "initial_noise")
        sample = self._reverse_normalized(state, initial_noise)
        return sample * scales

    def sample(self, history, horizon, history_covariates=None, future_covariates=None):
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self._validate_history(history, history_covariates)
        if future_covariates is None:
            if self.covariate_size:
                raise ValueError("future_covariates are required when covariate_size is positive")
        elif (
            not isinstance(future_covariates, torch.Tensor)
            or future_covariates.shape != (history.shape[0], horizon, self.covariate_size)
            or not future_covariates.is_floating_point()
        ):
            raise ValueError("future_covariates must have shape [batch, horizon, covariate_size]")
        state, scales = self._encode(history, history_covariates)
        forecasts = []
        for index in range(horizon):
            initial_noise = torch.randn(
                history.shape[0], self.input_size, device=history.device, dtype=history.dtype
            )
            normalized_sample = self._reverse_normalized(state, initial_noise)
            forecasts.append(normalized_sample * scales)
            recurrent_input = normalized_sample
            if future_covariates is not None:
                recurrent_input = torch.cat((recurrent_input, future_covariates[:, index]), dim=-1)
            _, state = self.rnn(recurrent_input.unsqueeze(1), state)
        return torch.stack(forecasts, dim=1)
