import copy
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


class TiCo(nn.Module):
    architecture_id = "tico"
    PRESETS = {
        "tiny": {
            "in_channels": 3,
            "encoder_width": 8,
            "projector_hidden_dim": 32,
            "embedding_dim": 16,
            "alpha_start": 0.99,
            "alpha_end": 1.0,
            "beta": 0.9,
            "rho": 8.0,
        },
        "small": {
            "in_channels": 3,
            "encoder_width": 32,
            "projector_hidden_dim": 256,
            "embedding_dim": 128,
            "alpha_start": 0.99,
            "alpha_end": 1.0,
            "beta": 0.9,
            "rho": 8.0,
        },
    }

    def __init__(
        self,
        in_channels=3,
        encoder_width=8,
        projector_hidden_dim=32,
        embedding_dim=16,
        alpha_start=0.99,
        alpha_end=1.0,
        beta=0.9,
        rho=8.0,
    ):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("encoder_width", encoder_width),
            ("projector_hidden_dim", projector_hidden_dim),
            ("embedding_dim", embedding_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("alpha_start", alpha_start),
            ("alpha_end", alpha_end),
            ("beta", beta),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if alpha_start > alpha_end:
            raise ValueError("alpha_start must not exceed alpha_end")
        if (
            isinstance(rho, bool)
            or not isinstance(rho, Real)
            or not math.isfinite(rho)
            or rho <= 0.0
        ):
            raise ValueError("rho must be a finite positive number")

        self.in_channels = in_channels
        self.encoder_width = encoder_width
        self.projector_hidden_dim = projector_hidden_dim
        self.embedding_dim = embedding_dim
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.beta = float(beta)
        self.rho = float(rho)
        self.online_encoder = nn.Sequential(
            nn.Conv2d(in_channels, encoder_width, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, encoder_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_width, encoder_width, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, encoder_width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        self.online_projector = nn.Sequential(
            nn.Linear(encoder_width, projector_hidden_dim),
            nn.LayerNorm(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, embedding_dim),
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.target_encoder.requires_grad_(False)
        self.target_projector.requires_grad_(False)
        self.register_buffer("covariance", torch.zeros(embedding_dim, embedding_dim))

    @staticmethod
    def normalize_embeddings(embeddings):
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [batch, features]")
        if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            raise ValueError("embeddings must have nonempty batch and feature dimensions")
        if not embeddings.is_floating_point():
            raise ValueError("embeddings must have a floating point dtype")
        return F.normalize(embeddings, dim=1)

    def momentum_at(self, step, total_steps):
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if step > total_steps:
            raise ValueError("step must not exceed total_steps")
        progress = (1.0 - math.cos(math.pi * step / total_steps)) / 2.0
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress

    @torch.no_grad()
    def update_target(self, alpha=None):
        if alpha is None:
            alpha = self.alpha_start
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not math.isfinite(alpha)
            or not 0.0 <= alpha <= 1.0
        ):
            raise ValueError("alpha must be finite and in [0, 1]")
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target.mul_(alpha).add_(online, alpha=1.0 - alpha)
        for target, online in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            target.mul_(alpha).add_(online, alpha=1.0 - alpha)

    def update_covariance(self, online_embeddings):
        if (
            not isinstance(online_embeddings, torch.Tensor)
            or online_embeddings.ndim != 2
            or online_embeddings.shape[0] == 0
            or online_embeddings.shape[1] != self.embedding_dim
        ):
            raise ValueError(f"online_embeddings must have nonempty shape [batch, {self.embedding_dim}]")
        if not online_embeddings.is_floating_point():
            raise ValueError("online_embeddings must have a floating point dtype")
        if online_embeddings.device != self.covariance.device:
            raise ValueError("online_embeddings and the model must be on the same device")
        if online_embeddings.dtype != self.covariance.dtype:
            raise ValueError("online_embeddings and the model must have the same dtype")
        batch_covariance = online_embeddings.transpose(0, 1) @ online_embeddings
        batch_covariance = batch_covariance / online_embeddings.shape[0]
        previous = self.covariance.detach().clone()
        updated = self.beta * previous + (1.0 - self.beta) * batch_covariance
        with torch.no_grad():
            self.covariance.copy_(updated.detach())
        return updated

    def loss_from_embeddings(self, online_embeddings, target_embeddings, covariance=None):
        if (
            not isinstance(online_embeddings, torch.Tensor)
            or not isinstance(target_embeddings, torch.Tensor)
            or online_embeddings.ndim != 2
            or target_embeddings.shape != online_embeddings.shape
            or online_embeddings.shape[0] == 0
            or online_embeddings.shape[1] != self.embedding_dim
        ):
            raise ValueError(
                f"online_embeddings and target_embeddings must share nonempty shape [batch, {self.embedding_dim}]"
            )
        if not online_embeddings.is_floating_point() or not target_embeddings.is_floating_point():
            raise ValueError("embeddings must have a floating point dtype")
        if online_embeddings.device != target_embeddings.device:
            raise ValueError("embeddings must be on the same device")
        if online_embeddings.dtype != target_embeddings.dtype:
            raise ValueError("embeddings must have the same dtype")
        if covariance is None:
            covariance = self.covariance
        if (
            not isinstance(covariance, torch.Tensor)
            or covariance.shape != (self.embedding_dim, self.embedding_dim)
        ):
            raise ValueError(f"covariance must have shape [{self.embedding_dim}, {self.embedding_dim}]")
        if covariance.device != online_embeddings.device or covariance.dtype != online_embeddings.dtype:
            raise ValueError("covariance must share the embeddings' device and dtype")
        invariance = 1.0 - (online_embeddings * target_embeddings).sum(dim=1).mean()
        contrast = ((online_embeddings @ covariance) * online_embeddings).sum(dim=1).mean()
        return invariance + self.rho * contrast

    def _validate_views(self, view_one, view_two):
        if not isinstance(view_one, torch.Tensor) or not isinstance(view_two, torch.Tensor):
            raise TypeError("both views must be tensors")
        if view_one.ndim != 4 or view_two.ndim != 4:
            raise ValueError("both views must have shape [batch, channels, height, width]")
        if view_one.shape != view_two.shape:
            raise ValueError("both views must have the same shape")
        if view_one.shape[0] == 0 or view_one.shape[2] == 0 or view_one.shape[3] == 0:
            raise ValueError("views must have nonempty batch, height, and width dimensions")
        if view_one.shape[1] != self.in_channels:
            raise ValueError(f"views must have {self.in_channels} channels")
        if view_one.device != view_two.device or view_one.dtype != view_two.dtype:
            raise ValueError("both views must have the same device and dtype")
        if not view_one.is_floating_point() or not view_two.is_floating_point():
            raise ValueError("both views must have a floating point dtype")
        parameter = next(self.online_encoder.parameters())
        if view_one.device != parameter.device or view_one.dtype != parameter.dtype:
            raise ValueError("views must share the model parameters' device and dtype")

    def forward(self, view_one, view_two, return_details=False):
        self._validate_views(view_one, view_two)
        if not isinstance(return_details, bool):
            raise TypeError("return_details must be a boolean")
        online = self.online_projector(self.online_encoder(view_one))
        online = self.normalize_embeddings(online)
        with torch.no_grad():
            target = self.target_projector(self.target_encoder(view_two))
            target = self.normalize_embeddings(target)
        covariance = self.update_covariance(online) if self.training else self.covariance
        loss = self.loss_from_embeddings(online, target, covariance)
        if not return_details:
            return loss
        return {
            "online_embeddings": online,
            "target_embeddings": target,
            "covariance": covariance,
            "loss": loss,
        }
