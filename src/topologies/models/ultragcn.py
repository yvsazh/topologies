import math

import torch
from torch import nn
from torch.nn import functional as F


class UltraGCN(nn.Module):
    architecture_id = "ultragcn"
    PRESETS = {
        "tiny": {
            "n_users": 4,
            "n_items": 6,
            "embedding_dim": 4,
            "n_neighbors": 2,
            "lambda_constraint": 1.0,
            "gamma_item": 2.5,
        },
        "small": {
            "n_users": 1000,
            "n_items": 2000,
            "embedding_dim": 64,
            "n_neighbors": 10,
            "lambda_constraint": 1.0,
            "gamma_item": 2.5,
        },
    }

    def __init__(
        self,
        n_users,
        n_items,
        interactions,
        embedding_dim=64,
        negative_samples_per_positive=300,
        n_neighbors=10,
        lambda_constraint=1.0,
        gamma_item=2.5,
        embedding_init_std=1e-4,
        l2_reg=1e-4,
    ):
        super().__init__()
        dimensions = {
            "n_users": n_users,
            "n_items": n_items,
            "embedding_dim": embedding_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(negative_samples_per_positive, bool)
            or not isinstance(negative_samples_per_positive, int)
            or negative_samples_per_positive <= 0
        ):
            raise ValueError("negative_samples_per_positive must be a positive integer")
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
            self._invalid_argument("n_neighbors must be an integer between 1 and n_items")
        if n_neighbors < 1 or n_neighbors > n_items:
            raise ValueError("n_neighbors must be between 1 and n_items")
        for name, value in (
            ("lambda_constraint", lambda_constraint),
            ("gamma_item", gamma_item),
            ("l2_reg", l2_reg),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self._invalid_argument(f"{name} must be a finite nonnegative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if (
            isinstance(embedding_init_std, bool)
            or not isinstance(embedding_init_std, (int, float))
            or not math.isfinite(embedding_init_std)
            or embedding_init_std <= 0
        ):
            raise ValueError("embedding_init_std must be a finite positive number")
        try:
            edge_pairs = torch.as_tensor(interactions)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("interactions must be an integer tensor with shape [2, E]") from exc
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if edge_pairs.ndim != 2 or edge_pairs.shape[0] != 2 or edge_pairs.shape[1] == 0:
            raise ValueError("interactions must have nonempty shape [2, E]")
        if edge_pairs.dtype not in integer_types:
            raise ValueError("interactions must contain integer user and item IDs")
        edge_pairs = edge_pairs.to(dtype=torch.long, device="cpu")
        if bool((edge_pairs[0] < 0).any()) or bool((edge_pairs[0] >= n_users).any()):
            raise ValueError("interaction user IDs must be in [0, n_users)")
        if bool((edge_pairs[1] < 0).any()) or bool((edge_pairs[1] >= n_items).any()):
            raise ValueError("interaction item IDs must be in [0, n_items)")
        if torch.unique(edge_pairs.t(), dim=0).shape[0] != edge_pairs.shape[1]:
            raise ValueError("interactions must contain unique user-item pairs")
        user_degrees = torch.bincount(edge_pairs[0], minlength=n_users)
        if bool((user_degrees == 0).any()):
            raise ValueError("every user must have at least one interaction")
        item_degrees = torch.bincount(edge_pairs[1], minlength=n_items)
        neighbor_ids, neighbor_weights = self._build_item_neighbors(
            edge_pairs, n_users, n_items, n_neighbors
        )
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.negative_samples_per_positive = negative_samples_per_positive
        self.n_neighbors = n_neighbors
        self.lambda_constraint = float(lambda_constraint)
        self.gamma_item = float(gamma_item)
        self.embedding_init_std = float(embedding_init_std)
        self.l2_reg = float(l2_reg)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=self.embedding_init_std)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.embedding_init_std)
        self.register_buffer("interactions", edge_pairs.contiguous())
        self.register_buffer("user_degrees", user_degrees)
        self.register_buffer("item_degrees", item_degrees)
        self.register_buffer("item_neighbor_ids", neighbor_ids)
        self.register_buffer("item_neighbor_weights", neighbor_weights)

    @staticmethod
    def _invalid_argument(message):
        raise ValueError(message)

    @staticmethod
    def _build_item_neighbors(interactions, n_users, n_items, n_neighbors):
        order = torch.argsort(interactions[0], stable=True)
        sorted_users = interactions[0, order]
        sorted_items = interactions[1, order]
        counts = torch.bincount(sorted_users, minlength=n_users).tolist()
        cooccurrence = torch.zeros((n_items, n_items), dtype=torch.float32)
        offset = 0
        for count in counts:
            user_items = sorted_items[offset:offset + count]
            if count:
                cooccurrence[user_items[:, None], user_items[None, :]] += 1.0
            offset += count
        item_degrees = cooccurrence.sum(dim=1)
        off_diagonal_degrees = item_degrees - cooccurrence.diagonal()
        denominator = off_diagonal_degrees.clamp_min(1.0)
        column_degrees = item_degrees.clamp_min(1.0)
        weights = cooccurrence / denominator[:, None]
        weights = weights * torch.sqrt(item_degrees[:, None] / column_degrees[None, :])
        weights = torch.where(off_diagonal_degrees[:, None] > 0, weights, 0.0)
        neighbor_ids = torch.argsort(weights, dim=1, descending=True, stable=True)[:, :n_neighbors]
        neighbor_weights = weights.gather(1, neighbor_ids)
        return neighbor_ids, neighbor_weights

    def _id_vector(self, values, name, upper_bound):
        device = self.user_embedding.weight.device
        try:
            values = torch.as_tensor(values, device=device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must contain integer IDs") from exc
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if values.dtype not in integer_types:
            raise ValueError(f"{name} must contain integer IDs")
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = values.to(dtype=torch.long)
        if values.numel() and (bool((values < 0).any()) or bool((values >= upper_bound).any())):
            raise ValueError(f"{name} IDs must be in [0, {upper_bound})")
        return values

    def _training_ids(self, user_ids, positive_item_ids, negative_item_ids):
        user_ids = self._id_vector(user_ids, "user_ids", self.n_users)
        positive_item_ids = self._id_vector(
            positive_item_ids, "positive_item_ids", self.n_items
        )
        if user_ids.shape != positive_item_ids.shape:
            raise ValueError("user_ids and positive_item_ids must have the same shape")
        device = self.user_embedding.weight.device
        try:
            negative_item_ids = torch.as_tensor(negative_item_ids, device=device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("negative_item_ids must contain integer IDs") from exc
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if negative_item_ids.dtype not in integer_types:
            raise ValueError("negative_item_ids must contain integer IDs")
        if negative_item_ids.ndim == 1:
            negative_item_ids = negative_item_ids.unsqueeze(-1)
        if negative_item_ids.ndim != 2 or negative_item_ids.shape[0] != user_ids.shape[0]:
            raise ValueError("negative_item_ids must have shape [B, R]")
        if negative_item_ids.shape[1] == 0:
            raise ValueError("negative_item_ids must contain at least one negative per user")
        negative_item_ids = negative_item_ids.to(dtype=torch.long)
        if bool((negative_item_ids < 0).any()) or bool((negative_item_ids >= self.n_items).any()):
            raise ValueError(f"negative_item_ids IDs must be in [0, {self.n_items})")
        return user_ids, positive_item_ids, negative_item_ids

    def constraint_coefficients(self, user_ids, item_ids):
        user_ids = self._id_vector(user_ids, "user_ids", self.n_users)
        item_ids = self._id_vector(item_ids, "item_ids", self.n_items)
        if user_ids.shape != item_ids.shape:
            raise ValueError("user_ids and item_ids must have the same shape")
        user_degrees = self.user_degrees[user_ids].to(dtype=self.user_embedding.weight.dtype)
        item_degrees = self.item_degrees[item_ids].to(dtype=self.user_embedding.weight.dtype)
        return user_degrees.reciprocal() * torch.sqrt(
            (user_degrees + 1.0) / (item_degrees + 1.0)
        )

    def forward(self, user_ids, item_ids):
        user_ids = self._id_vector(user_ids, "user_ids", self.n_users)
        item_ids = self._id_vector(item_ids, "item_ids", self.n_items)
        if user_ids.shape != item_ids.shape:
            raise ValueError("user_ids and item_ids must have the same shape")
        return (self.user_embedding(user_ids) * self.item_embedding(item_ids)).sum(dim=-1)

    def objective_loss(self, user_ids, positive_item_ids, negative_item_ids):
        user_ids, positive_item_ids, negative_item_ids = self._training_ids(
            user_ids, positive_item_ids, negative_item_ids
        )
        positive_scores = self(user_ids, positive_item_ids)
        negative_scores = (
            self.user_embedding(user_ids).unsqueeze(1)
            * self.item_embedding(negative_item_ids)
        ).sum(dim=-1)
        return F.softplus(-positive_scores).sum() + F.softplus(negative_scores).sum()

    def constraint_loss(self, user_ids, positive_item_ids, negative_item_ids):
        user_ids, positive_item_ids, negative_item_ids = self._training_ids(
            user_ids, positive_item_ids, negative_item_ids
        )
        positive_scores = self(user_ids, positive_item_ids)
        negative_scores = (
            self.user_embedding(user_ids).unsqueeze(1)
            * self.item_embedding(negative_item_ids)
        ).sum(dim=-1)
        positive_weights = self.constraint_coefficients(user_ids, positive_item_ids)
        negative_weights = self.constraint_coefficients(
            user_ids.unsqueeze(1).expand_as(negative_item_ids).reshape(-1),
            negative_item_ids.reshape(-1),
        ).reshape_as(negative_item_ids)
        positive_loss = (positive_weights * F.softplus(-positive_scores)).sum()
        negative_loss = (negative_weights * F.softplus(negative_scores)).sum()
        return positive_loss + negative_loss

    def item_item_loss(self, user_ids, positive_item_ids):
        user_ids = self._id_vector(user_ids, "user_ids", self.n_users)
        positive_item_ids = self._id_vector(
            positive_item_ids, "positive_item_ids", self.n_items
        )
        if user_ids.shape != positive_item_ids.shape:
            raise ValueError("user_ids and positive_item_ids must have the same shape")
        neighbor_ids = self.item_neighbor_ids[positive_item_ids]
        neighbor_weights = self.item_neighbor_weights[positive_item_ids]
        scores = (
            self.user_embedding(user_ids).unsqueeze(1) * self.item_embedding(neighbor_ids)
        ).sum(dim=-1)
        return (neighbor_weights * F.softplus(-scores)).sum()

    def training_loss(self, user_ids, positive_item_ids, negative_item_ids):
        loss = self.objective_loss(user_ids, positive_item_ids, negative_item_ids)
        loss = loss + self.lambda_constraint * self.constraint_loss(
            user_ids, positive_item_ids, negative_item_ids
        )
        if self.gamma_item:
            loss = loss + self.gamma_item * self.item_item_loss(user_ids, positive_item_ids)
        if self.l2_reg:
            regularization = self.user_embedding.weight.square().sum()
            regularization = regularization + self.item_embedding.weight.square().sum()
            loss = loss + 0.5 * self.l2_reg * regularization
        return loss
