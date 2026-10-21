

import math

import torch
from torch import nn


class NoisePredictor(nn.Module):
    def __init__(self, observation_dim, action_dim, horizon, hidden_dim, num_steps):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim * horizon + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim * horizon),
        )
        self.time_dim = hidden_dim
        self.num_steps = num_steps

    def forward(self, observation, noisy_action, timestep):
        flat_action = noisy_action.flatten(start_dim=1)
        phase = timestep.to(dtype=noisy_action.dtype).reshape(-1, 1)
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=noisy_action.device, dtype=noisy_action.dtype)
            / max(half - 1, 1)
        )
        angles = phase * frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.time_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.time_dim - embedding.shape[-1]))
        features = torch.cat((observation, flat_action, embedding), dim=-1)
        return self.network(features).reshape_as(noisy_action)


class DiffusionPolicy(nn.Module):


    component_id = "diffusion-policy"
    PRESETS = {
        "tiny": {"observation_dim": 3, "action_dim": 2, "horizon": 4, "hidden_dim": 32, "num_steps": 8},
        "base": {"observation_dim": 6, "action_dim": 3, "horizon": 8, "hidden_dim": 128, "num_steps": 32},
    }

    def __init__(self, observation_dim=3, action_dim=2, horizon=4, hidden_dim=32, num_steps=8,
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()
        if min(observation_dim, action_dim, horizon, hidden_dim, num_steps) <= 0:
            raise ValueError("dimensions and num_steps must be positive")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("require 0 < beta_start < beta_end < 1")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_steps = num_steps
        self.noise_predictor = NoisePredictor(observation_dim, action_dim, horizon, hidden_dim, num_steps)
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat((torch.ones(1), alpha_bars[:-1]))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars))
        self.register_buffer("posterior_coef_x0", betas * alpha_bars_prev.sqrt() / (1.0 - alpha_bars))
        self.register_buffer("posterior_coef_xt", (1.0 - alpha_bars_prev) * alphas.sqrt() / (1.0 - alpha_bars))

    def predict_noise(self, observation, noisy_action, timestep):
        observation = torch.as_tensor(observation, dtype=noisy_action.dtype, device=noisy_action.device)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.shape != (noisy_action.shape[0], self.observation_dim):
            raise ValueError("observation must have shape (batch, observation_dim)")
        return self.noise_predictor(observation, noisy_action, timestep)

    def posterior_mean(self, noisy_action, predicted_noise, timestep):

        t = torch.as_tensor(timestep, dtype=torch.long, device=noisy_action.device)
        if t.ndim == 0:
            t = t.expand(noisy_action.shape[0])
        if t.shape != (noisy_action.shape[0],) or torch.any(t < 0) or torch.any(t >= self.num_steps):
            raise ValueError("timestep must be valid for each batch item")
        a_bar = self.alpha_bars[t].view(-1, 1, 1)
        predicted_x0 = (noisy_action - (1.0 - a_bar).sqrt() * predicted_noise) / a_bar.sqrt()
        return (self.posterior_coef_x0[t].view(-1, 1, 1) * predicted_x0
                + self.posterior_coef_xt[t].view(-1, 1, 1) * noisy_action)

    def denoise_step(self, observation, noisy_action, timestep, noise=None):
        t = torch.as_tensor(timestep, dtype=torch.long, device=noisy_action.device)
        if t.ndim == 0:
            t = t.expand(noisy_action.shape[0])
        predicted_noise = self.predict_noise(observation, noisy_action, t)
        mean = self.posterior_mean(noisy_action, predicted_noise, t)
        if noise is None:
            noise = torch.randn_like(noisy_action)
        return mean + (self.posterior_variance[t].view(-1, 1, 1).sqrt() * noise)

    @torch.no_grad()
    def sample(self, observation, initial_noise=None, generator=None):
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.betas.device)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError("observation must have shape (batch, observation_dim)")
        shape = (observation.shape[0], self.horizon, self.action_dim)
        action = torch.randn(shape, device=observation.device, dtype=observation.dtype, generator=generator) if initial_noise is None else torch.as_tensor(initial_noise, dtype=observation.dtype, device=observation.device)
        if action.shape != shape:
            raise ValueError("initial_noise must match (batch, horizon, action_dim)")
        for index in reversed(range(self.num_steps)):
            t = torch.full((action.shape[0],), index, dtype=torch.long, device=action.device)
            predicted_noise = self.predict_noise(observation, action, t)
            action = self.posterior_mean(action, predicted_noise, t)
            if index > 0:
                action = action + self.posterior_variance[t].view(-1, 1, 1).sqrt() * torch.randn(action.shape, device=action.device, dtype=action.dtype, generator=generator)
        return action

    def forward(self, observation, noisy_action, timestep):
        return self.predict_noise(observation, noisy_action, timestep)
