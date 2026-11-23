import math

import torch
from torch import nn
from torch.nn import functional as F


class RecOESubspaceOOD(nn.Module):
    architecture_id = "recoe-subspace-ood"
    PRESETS = {
        "tiny": {
            "input_dim": 4,
            "feature_dim": 8,
            "num_classes": 3,
            "rank": 2,
            "margin": 0.5,
            "beta": 0.001,
            "gamma": 0.5,
            "ema_momentum": 0.99,
            "eps": 1e-6,
        },
        "small": {
            "input_dim": 16,
            "feature_dim": 32,
            "num_classes": 4,
            "rank": 8,
            "margin": 0.5,
            "beta": 0.001,
            "gamma": 0.5,
            "ema_momentum": 0.99,
            "eps": 1e-6,
        },
    }

    def __init__(
        self,
        input_dim,
        feature_dim,
        num_classes,
        rank,
        margin=0.5,
        beta=0.001,
        gamma=0.5,
        ema_momentum=0.99,
        eps=1e-6,
    ):
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "rank": rank,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if rank > feature_dim:
            raise ValueError("rank cannot exceed feature_dim")
        for name, value in (("margin", margin), ("beta", beta), ("gamma", gamma)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            isinstance(ema_momentum, bool)
            or not isinstance(ema_momentum, (int, float))
            or not math.isfinite(ema_momentum)
            or not 0 <= ema_momentum < 1
        ):
            raise ValueError("ema_momentum must be in [0, 1)")
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise ValueError("eps must be finite and positive")

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.rank = rank
        self.register_buffer("_active_rank", torch.tensor(max(1, math.ceil(rank / 2)), dtype=torch.long))
        self.margin = float(margin)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.ema_momentum = float(ema_momentum)
        self.eps = float(eps)
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.basis_in = nn.Parameter(torch.zeros(feature_dim, rank))
        self.basis_aux = nn.Parameter(torch.zeros(feature_dim, rank))
        with torch.no_grad():
            self.basis_in[:, : self.active_rank].copy_(
                self._random_orthonormal(feature_dim, self.active_rank)
            )
            self.basis_aux[:, : self.active_rank].copy_(
                self._random_orthonormal(feature_dim, self.active_rank)
            )
        self.register_buffer("mu_in", torch.zeros(1, feature_dim))
        self.register_buffer("mu_aux", torch.zeros(1, feature_dim))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _random_orthonormal(rows, columns, device=None, dtype=None):
        matrix = torch.randn(rows, columns, device=device, dtype=dtype)
        return torch.linalg.qr(matrix, mode="reduced").Q.contiguous()

    @staticmethod
    def _validate_feature_matrix(features, name):
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty [batch, feature_dim] tensor")

    def _validate_feature_width(self, features, name):
        self._validate_feature_matrix(features, name)
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"{name} must have feature width {self.feature_dim}")

    @property
    def active_rank(self):
        return int(self._active_rank.item())

    @property
    def U_in(self):
        return torch.linalg.qr(self.basis_in[:, : self.active_rank], mode="reduced").Q

    @property
    def U_aux(self):
        return torch.linalg.qr(self.basis_aux[:, : self.active_rank], mode="reduced").Q

    @staticmethod
    def normalize_features(features, eps=1e-6):
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be a [batch, feature_dim] tensor")
        return features / (features.norm(p=2, dim=1, keepdim=True) + eps)

    def update_ema_centers(self, features_in, features_aux):
        self._validate_feature_width(features_in, "features_in")
        self._validate_feature_width(features_aux, "features_aux")
        if not self.training:
            return
        mean_in = features_in.mean(dim=0, keepdim=True).detach()
        mean_aux = features_aux.mean(dim=0, keepdim=True).detach()
        with torch.no_grad():
            if not bool(self.ema_initialized.item()):
                self.mu_in.copy_(mean_in)
                self.mu_aux.copy_(mean_aux)
                self.ema_initialized.fill_(True)
            else:
                self.mu_in.mul_(self.ema_momentum).add_(mean_in, alpha=1 - self.ema_momentum)
                self.mu_aux.mul_(self.ema_momentum).add_(mean_aux, alpha=1 - self.ema_momentum)

    @staticmethod
    def reconstruction_error(centered_features, basis):
        if not isinstance(centered_features, torch.Tensor) or centered_features.ndim != 2:
            raise ValueError("centered_features must be a [batch, feature_dim] tensor")
        if not isinstance(basis, torch.Tensor) or basis.ndim != 2:
            raise ValueError("basis must be a [feature_dim, rank] tensor")
        if centered_features.shape[1] != basis.shape[0]:
            raise ValueError("centered_features and basis must have matching feature dimensions")
        projected = centered_features @ basis @ basis.T
        return (centered_features - projected).square().sum(dim=1)

    def orthogonality_loss(self):
        overlap = self.U_in.T @ self.U_aux
        return overlap.square().sum()

    def subspace_losses(self, features_in, features_aux):
        self._validate_feature_width(features_in, "features_in")
        self._validate_feature_width(features_aux, "features_aux")
        features_in = self.normalize_features(features_in, self.eps)
        features_aux = self.normalize_features(features_aux, self.eps)
        return self._subspace_losses_normalized(features_in, features_aux)

    def _subspace_losses_normalized(self, features_in, features_aux):
        mean_in = features_in.mean(dim=0, keepdim=True)
        mean_aux = features_aux.mean(dim=0, keepdim=True)
        self.update_ema_centers(features_in, features_aux)

        centered_in = features_in - mean_in
        centered_aux = features_aux - mean_aux
        in_on_in = self.reconstruction_error(centered_in, self.U_in)
        aux_on_aux = self.reconstruction_error(centered_aux, self.U_aux)
        in_on_aux = self.reconstruction_error(features_in - mean_aux, self.U_aux)
        aux_on_in = self.reconstruction_error(features_aux - mean_in, self.U_in)
        loss_fit = in_on_in.mean() + aux_on_aux.mean()
        loss_sep = F.relu(self.margin - in_on_aux).mean() + F.relu(
            self.margin - aux_on_in
        ).mean()
        loss_ortho = self.orthogonality_loss()
        loss_sub = loss_fit + loss_sep + self.beta * loss_ortho
        return {
            "loss_fit": loss_fit,
            "loss_sep": loss_sep,
            "loss_ortho": loss_ortho,
            "loss_sub": loss_sub,
            "error_in_fit": in_on_in,
            "error_aux_fit": aux_on_aux,
            "error_in_on_aux": in_on_aux,
            "error_aux_on_in": aux_on_in,
        }

    def _score_normalized(self, features):
        error_in = self.reconstruction_error(features - self.mu_in, self.U_in)
        error_aux = self.reconstruction_error(features - self.mu_aux, self.U_aux)
        return error_aux - error_in

    def score_features(self, features):
        self._validate_feature_width(features, "features")
        return self._score_normalized(self.normalize_features(features, self.eps))

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2 or inputs.shape[0] < 1:
            raise ValueError("inputs must be a nonempty [batch, input_dim] tensor")
        if inputs.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have width {self.input_dim}")
        raw_features = self.encoder(inputs)
        features = self.normalize_features(raw_features, self.eps)
        return {
            "logits": self.classifier(raw_features),
            "features": features,
            "score": self._score_normalized(features),
        }

    def training_objective(self, inputs_in, inputs_aux, labels):
        if not isinstance(inputs_in, torch.Tensor) or inputs_in.ndim != 2:
            raise ValueError("inputs_in must be a [batch, input_dim] tensor")
        if not isinstance(inputs_aux, torch.Tensor) or inputs_aux.ndim != 2:
            raise ValueError("inputs_aux must be a [batch, input_dim] tensor")
        if inputs_in.shape[1] != self.input_dim or inputs_aux.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have width {self.input_dim}")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.shape[0] != inputs_in.shape[0]:
            raise ValueError("labels must have shape [batch] matching inputs_in")
        if labels.numel() == 0 or labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError("labels must be valid class indices")
        raw_features_in = self.encoder(inputs_in)
        raw_features_aux = self.encoder(inputs_aux)
        features_in = self.normalize_features(raw_features_in, self.eps)
        features_aux = self.normalize_features(raw_features_aux, self.eps)
        losses = self._subspace_losses_normalized(features_in, features_aux)
        logits = self.classifier(raw_features_in)
        loss_ce = F.cross_entropy(logits, labels.long())
        loss_total = loss_ce + self.gamma * losses["loss_sub"]
        return {
            **losses,
            "logits": logits,
            "features_in": features_in,
            "features_aux": features_aux,
            "loss_ce": loss_ce,
            "loss_total": loss_total,
        }

    def next_rank(self):
        if self.active_rank >= self.rank:
            return self.rank
        return min(self.rank, self.active_rank + max(1, math.ceil((self.rank - self.active_rank) / 2)))

    @staticmethod
    def _orthogonal_append(basis, count):
        if count <= 0:
            return basis
        rows = basis.shape[0]
        for _ in range(3):
            candidates = torch.randn(rows, count, device=basis.device, dtype=basis.dtype)
            candidates = candidates - basis @ (basis.T @ candidates)
            new_directions, triangular = torch.linalg.qr(candidates, mode="reduced")
            if torch.all(torch.diagonal(triangular).abs() > 1e-8):
                return torch.cat((basis, new_directions[:, :count]), dim=1).contiguous()
        raise ValueError("could not initialize orthogonal directions for rank expansion")

    def expand_rank(self, new_rank=None):
        if new_rank is None:
            new_rank = self.next_rank()
        if isinstance(new_rank, bool) or not isinstance(new_rank, int):
            raise TypeError("new_rank must be an integer")
        if new_rank < self.active_rank or new_rank > self.rank:
            raise ValueError("new_rank must be between active_rank and target rank")
        if new_rank == self.active_rank:
            return False
        old_rank = self.active_rank
        with torch.no_grad():
            for raw_basis, current_basis in (
                (self.basis_in, self.U_in.detach()),
                (self.basis_aux, self.U_aux.detach()),
            ):
                expanded = self._orthogonal_append(current_basis, new_rank - old_rank)
                raw_basis[:, :new_rank].copy_(expanded)
        self._active_rank.fill_(new_rank)
        return True
