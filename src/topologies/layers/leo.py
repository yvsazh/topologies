import math

import torch
from torch import nn
from torch.nn import functional as F


class Leo(nn.Module):
    component_id = "leo"
    PRESETS = {
        "tiny": {
            "input_dim": 4,
            "latent_dim": 2,
            "relation_hidden_dim": 4,
            "relation_layers": 3,
            "latent_steps": 1,
            "parameter_steps": 0,
            "p_keep": 1.0,
        },
        "small": {
            "input_dim": 16,
            "latent_dim": 4,
            "relation_hidden_dim": 8,
            "relation_layers": 3,
            "latent_steps": 2,
            "parameter_steps": 1,
            "p_keep": 1.0,
        },
        "base": {
            "input_dim": 640,
            "latent_dim": 64,
            "relation_hidden_dim": 128,
            "relation_layers": 3,
            "latent_steps": 5,
            "parameter_steps": 5,
            "alpha_z_init": 1.0,
            "alpha_w_init": 0.001,
            "beta": 0.124171967,
            "gamma": 1.33365371e-9,
            "lambda_1": 0.000108982953,
            "lambda_2": 303.216647,
            "p_keep": 0.711524088,
        },
    }

    def __init__(
        self,
        input_dim=640,
        latent_dim=64,
        relation_hidden_dim=128,
        relation_layers=3,
        latent_steps=5,
        parameter_steps=5,
        alpha_z_init=1.0,
        alpha_w_init=0.001,
        beta=0.124171967,
        gamma=1.33365371e-9,
        lambda_1=0.000108982953,
        lambda_2=303.216647,
        p_keep=0.711524088,
    ):
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "relation_hidden_dim": relation_hidden_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("relation_layers", relation_layers), ("latent_steps", latent_steps),
                            ("parameter_steps", parameter_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if relation_layers < 2:
            raise ValueError("relation_layers must be at least 2")
        scalars = {
            "alpha_z_init": alpha_z_init,
            "alpha_w_init": alpha_w_init,
            "beta": beta,
            "gamma": gamma,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "p_keep": p_keep,
        }
        for name, value in scalars.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("alpha_z_init", "alpha_w_init", "beta", "gamma", "lambda_1", "lambda_2"):
            if scalars[name] < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < p_keep <= 1.0:
            raise ValueError("p_keep must be in the interval (0, 1]")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.relation_layers = relation_layers
        self.latent_steps = latent_steps
        self.parameter_steps = parameter_steps
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.p_keep = float(p_keep)
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        relation = [nn.Linear(2 * latent_dim, relation_hidden_dim, bias=False), nn.ReLU()]
        for _ in range(relation_layers - 2):
            relation.extend((nn.Linear(relation_hidden_dim, relation_hidden_dim, bias=False), nn.ReLU()))
        relation.append(nn.Linear(relation_hidden_dim, 2 * latent_dim, bias=False))
        self.relation = nn.Sequential(*relation)
        self.decoder = nn.Linear(latent_dim, 2 * input_dim, bias=False)
        self.alpha_z = nn.Parameter(torch.full((latent_dim,), float(alpha_z_init)))
        self.alpha_w = nn.Parameter(torch.full((input_dim,), float(alpha_w_init)))

    @staticmethod
    def _integer_types():
        return (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

    def _prepare_episode(self, support_x, support_y, query_x, query_y=None):
        support_x = torch.as_tensor(support_x)
        support_y = torch.as_tensor(support_y)
        query_x = torch.as_tensor(query_x)
        if support_x.ndim != 4:
            raise ValueError("support_x must have shape [B, N, K, D]")
        if support_y.ndim != 3 or tuple(support_y.shape) != tuple(support_x.shape[:3]):
            raise ValueError("support_y must have shape [B, N, K] matching support_x")
        if query_x.ndim != 3:
            raise ValueError("query_x must have shape [B, Q, D]")
        if any(size == 0 for size in support_x.shape) or any(size == 0 for size in query_x.shape):
            raise ValueError("episode batch, classes, support, query, and feature dimensions must be nonempty")
        if support_x.shape[0] != query_x.shape[0]:
            raise ValueError("support and query batch dimensions must match")
        if support_x.shape[-1] != query_x.shape[-1] or support_x.shape[-1] != self.input_dim:
            raise ValueError("support and query feature dimensions must match input_dim")
        if not torch.is_floating_point(support_x) or not torch.is_floating_point(query_x):
            raise ValueError("support_x and query_x must be floating point tensors")
        if support_x.device != query_x.device or support_x.dtype != query_x.dtype:
            raise ValueError("support_x and query_x must have the same device and dtype")
        if support_y.dtype not in self._integer_types():
            raise ValueError("support_y must contain integer class indices")
        support_y = support_y.to(device=support_x.device, dtype=torch.long)
        expected = torch.arange(support_x.shape[1], device=support_x.device).view(1, -1, 1)
        expected = expected.expand_as(support_y)
        if not torch.equal(support_y, expected):
            raise ValueError("support_y must match the class-grouped support axis")
        if query_y is not None:
            query_y = torch.as_tensor(query_y)
            if query_y.ndim != 2 or tuple(query_y.shape) != tuple(query_x.shape[:2]):
                raise ValueError("query_y must have shape [B, Q] matching query_x")
            if query_y.dtype not in self._integer_types():
                raise ValueError("query_y must contain integer class indices")
            query_y = query_y.to(device=support_x.device, dtype=torch.long)
            if (query_y < 0).any().item() or (query_y >= support_x.shape[1]).any().item():
                raise ValueError("query_y must refer to a support class")
        return support_x, support_y, query_x, query_y

    def encode_posterior(self, support_x):
        support_x = torch.as_tensor(support_x)
        if support_x.ndim != 4 or any(size == 0 for size in support_x.shape):
            raise ValueError("support_x must be a nonempty tensor with shape [B, N, K, D]")
        if support_x.shape[-1] != self.input_dim:
            raise ValueError("support_x feature dimension must match input_dim")
        if not torch.is_floating_point(support_x):
            raise ValueError("support_x must be a floating point tensor")
        encoded = self.encoder(support_x)
        left = encoded[:, :, :, None, None, :].expand(-1, -1, -1, encoded.shape[1], encoded.shape[2], -1)
        right = encoded[:, None, None, :, :, :].expand(-1, encoded.shape[1], encoded.shape[2], -1, -1, -1)
        pair_codes = torch.cat((left, right), dim=-1)
        pair_outputs = self.relation(pair_codes)
        grouped_outputs = pair_outputs.mean(dim=(2, 3, 4))
        return grouped_outputs.split(self.latent_dim, dim=-1)

    def decoder_parameters(self, codes):
        codes = torch.as_tensor(codes)
        if codes.ndim != 3 or codes.shape[-1] != self.latent_dim:
            raise ValueError("codes must have shape [B, N, latent_dim]")
        if any(size == 0 for size in codes.shape) or not torch.is_floating_point(codes):
            raise ValueError("codes must be nonempty floating point tensors")
        mean, scale = self.decoder(codes).chunk(2, dim=-1)
        return mean, scale

    def decode(self, codes, *, sample=None, epsilon=None):
        if sample not in (None, True, False):
            raise ValueError("sample must be True, False, or None")
        mean, scale = self.decoder_parameters(codes)
        if epsilon is not None:
            epsilon = torch.as_tensor(epsilon, dtype=mean.dtype, device=mean.device)
            if tuple(epsilon.shape) != tuple(mean.shape):
                raise ValueError("epsilon must have the same shape as decoded parameters")
            return mean + scale * epsilon
        stochastic = self.training if sample is None else sample
        if stochastic:
            return mean + scale * torch.randn_like(scale)
        return mean

    def classification_loss(self, logits, labels):
        logits = torch.as_tensor(logits)
        labels = torch.as_tensor(labels)
        if logits.ndim != 3 or labels.ndim != 2 or tuple(labels.shape) != tuple(logits.shape[:2]):
            raise ValueError("logits and labels must have shapes [B, S, N] and [B, S]")
        if logits.shape[-1] == 0 or logits.shape[1] == 0:
            raise ValueError("logit class and sample dimensions must be nonempty")
        if labels.dtype not in self._integer_types():
            raise ValueError("labels must contain integer class indices")
        labels = labels.to(device=logits.device, dtype=torch.long)
        if (labels < 0).any().item() or (labels >= logits.shape[-1]).any().item():
            raise ValueError("labels must refer to an available class")
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        )
        return losses.reshape(logits.shape[:2]).sum(dim=1)

    def support_loss(self, weights, support_x, support_y):
        weights = torch.as_tensor(weights)
        support_x = torch.as_tensor(support_x)
        support_y = torch.as_tensor(support_y)
        if weights.ndim != 3 or weights.shape[-1] != self.input_dim:
            raise ValueError("weights must have shape [B, N, input_dim]")
        if support_x.ndim != 4 or tuple(support_x.shape[:2]) != tuple(weights.shape[:2]):
            raise ValueError("support_x must have shape [B, N, K, D] matching weights")
        if tuple(support_y.shape) != tuple(support_x.shape[:3]):
            raise ValueError("support_y must match the batch, class, and support dimensions")
        flattened_x = support_x.reshape(support_x.shape[0], -1, support_x.shape[-1])
        flattened_y = support_y.reshape(support_y.shape[0], -1)
        logits = torch.einsum("bsd,bnd->bsn", flattened_x, weights)
        return self.classification_loss(logits, flattened_y)

    @staticmethod
    def _gradient_step(values, loss_fn, learning_rate):
        with torch.enable_grad():
            if not values.requires_grad:
                values = values.detach().requires_grad_(True)
            loss = loss_fn(values)
            gradient = torch.autograd.grad(loss.sum(), values, create_graph=True)[0]
            return values - learning_rate * gradient

    @staticmethod
    def _step_count(steps, name):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return steps

    def _drop_features(self, features, stochastic):
        return F.dropout(features, p=1.0 - self.p_keep, training=stochastic)

    def adapt_latents(self, codes, support_x, support_y, *, steps=None, sample=None):
        if sample not in (None, True, False):
            raise ValueError("sample must be True, False, or None")
        count = self.latent_steps if steps is None else self._step_count(steps, "steps")
        stochastic = self.training if sample is None else sample
        labels = torch.as_tensor(support_y, device=support_x.device, dtype=torch.long)
        adapted = codes
        for _ in range(count):
            features = self._drop_features(support_x, stochastic)
            adapted = self._gradient_step(
                adapted,
                lambda current, current_features=features, current_labels=labels: self.support_loss(
                    self.decode(current, sample=stochastic), current_features, current_labels
                ),
                self.alpha_z,
            )
        return adapted

    def adapt_parameters(self, weights, support_x, support_y, *, steps=None, sample=None):
        if sample not in (None, True, False):
            raise ValueError("sample must be True, False, or None")
        count = self.parameter_steps if steps is None else self._step_count(steps, "steps")
        stochastic = self.training if sample is None else sample
        labels = torch.as_tensor(support_y, device=support_x.device, dtype=torch.long)
        if support_x.ndim == 4:
            support_x = support_x.reshape(support_x.shape[0], -1, support_x.shape[-1])
            labels = labels.reshape(labels.shape[0], -1)
        adapted = weights
        for _ in range(count):
            features = self._drop_features(support_x, stochastic)
            adapted = self._gradient_step(
                adapted,
                lambda current, current_features=features, current_labels=labels: self.classification_loss(
                    torch.einsum("bsd,bnd->bsn", current_features, current), current_labels
                ),
                self.alpha_w,
            )
        return adapted

    def regularization(self):
        weight_decay = sum(parameter.square().sum() for module in (self.encoder, self.relation, self.decoder)
                           for parameter in module.parameters())
        decoder_rows = self.decoder.weight.transpose(0, 1)
        normalized = decoder_rows / torch.linalg.vector_norm(decoder_rows, dim=-1, keepdim=True).clamp_min(1e-12)
        correlation = normalized @ normalized.transpose(-1, -2)
        identity = torch.eye(self.latent_dim, dtype=correlation.dtype, device=correlation.device)
        orthogonality = torch.linalg.matrix_norm(correlation - identity, ord=2)
        return self.lambda_1 * weight_decay + self.lambda_2 * orthogonality

    def meta_objective(
        self,
        query_logits,
        query_y,
        posterior_mean,
        posterior_scale,
        initial_codes,
        adapted_codes,
        *,
        regularization=None,
    ):
        query_loss = self.classification_loss(query_logits, query_y)
        variance = posterior_scale.square().clamp_min(1e-12)
        kl = 0.5 * (
            posterior_mean.square() + variance - 1.0 - variance.log()
        ).sum(dim=(-1, -2))
        reconstruction = (adapted_codes.detach() - initial_codes).square().sum(dim=(-1, -2))
        penalty = self.regularization() if regularization is None else torch.as_tensor(
            regularization, dtype=query_logits.dtype, device=query_logits.device
        )
        return query_loss + self.beta * kl + self.gamma * reconstruction + penalty

    def forward(self, support_x, support_y, query_x, query_y=None, *, sample=None, return_aux=False):
        if sample not in (None, True, False):
            raise ValueError("sample must be True, False, or None")
        support_x, support_y, query_x, query_y = self._prepare_episode(
            support_x, support_y, query_x, query_y
        )
        stochastic = self.training if sample is None else sample
        encoded_support = self._drop_features(support_x, stochastic)
        posterior_mean, posterior_scale = self.encode_posterior(encoded_support)
        if stochastic:
            initial_codes = posterior_mean + posterior_scale * torch.randn_like(posterior_scale)
        else:
            initial_codes = posterior_mean
        adapted_codes = self.adapt_latents(
            initial_codes, support_x, support_y, sample=stochastic
        )
        weights = self.decode(adapted_codes, sample=stochastic)
        flattened_support = support_x.reshape(support_x.shape[0], -1, support_x.shape[-1])
        flattened_labels = support_y.reshape(support_y.shape[0], -1)
        weights = self.adapt_parameters(
            weights, flattened_support, flattened_labels, steps=self.parameter_steps, sample=stochastic
        )
        query_features = self._drop_features(query_x, stochastic)
        logits = torch.einsum("bqd,bnd->bqn", query_features, weights)
        if not return_aux:
            return logits
        objective = None
        if query_y is not None:
            objective = self.meta_objective(
                logits,
                query_y,
                posterior_mean,
                posterior_scale,
                initial_codes,
                adapted_codes,
            )
        return {
            "logits": logits,
            "query_logits": logits,
            "weights": weights,
            "posterior_mean": posterior_mean,
            "posterior_scale": posterior_scale,
            "initial_codes": initial_codes,
            "adapted_codes": adapted_codes,
            "meta_objective": objective,
        }
