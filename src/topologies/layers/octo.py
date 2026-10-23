import math

import torch
from torch import nn
from torch.nn import functional as F


class OctoPolicyReadout(nn.Module):
    component_id = "octo"

    class _SelfAttention(nn.Module):
        def __init__(self, d_model, num_heads, attention_dropout):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.attention_dropout = attention_dropout
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x, allowed):
            batch, sequence, width = x.shape
            qkv = self.qkv(x).view(batch, sequence, 3, self.num_heads, self.head_dim)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            allowed = allowed.clone()
            empty_rows = ~allowed.any(dim=-1)
            diagonal = torch.eye(sequence, dtype=torch.bool, device=x.device).unsqueeze(0)
            allowed |= empty_rows.unsqueeze(-1) & diagonal
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed[:, None, :, :],
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
            output = output.transpose(1, 2).reshape(batch, sequence, width)
            return self.out_proj(output)

    class _TransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, mlp_dim, dropout, attention_dropout):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = OctoPolicyReadout._SelfAttention(d_model, num_heads, attention_dropout)
            self.attention_output_dropout = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x, allowed):
            x = x + self.attention_output_dropout(self.attn(self.norm1(x), allowed))
            return x + self.mlp(self.norm2(x))

    def __init__(
        self,
        d_model=384,
        num_layers=12,
        num_heads=6,
        mlp_dim=1536,
        max_horizon=10,
        num_readout_tokens=1,
        dropout=0.0,
        attention_dropout=0.0,
        repeat_task_tokens=False,
    ):
        super().__init__()
        dimensions = {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "mlp_dim": mlp_dim,
            "max_horizon": max_horizon,
            "num_readout_tokens": num_readout_tokens,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        for name, value in (
            ("dropout", dropout),
            ("attention_dropout", attention_dropout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or value >= 1
            ):
                raise ValueError(f"{name} must be in [0, 1)")
        if repeat_task_tokens is not True and repeat_task_tokens is not False:
            raise ValueError("repeat_task_tokens must be a boolean")

        self.d_model = d_model
        self.max_horizon = max_horizon
        self.num_readout_tokens = num_readout_tokens
        self.repeat_task_tokens = repeat_task_tokens
        self.readout_positional_embeddings = nn.Parameter(
            torch.empty(1, max_horizon, num_readout_tokens, d_model)
        )
        nn.init.normal_(self.readout_positional_embeddings, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            self._TransformerBlock(d_model, num_heads, mlp_dim, dropout, attention_dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def build_attention_mask(self, task_valid, observation_valid, timestep_valid=None):
        if not isinstance(task_valid, torch.Tensor) or task_valid.ndim != 2:
            raise ValueError("task_valid must have shape (batch, task_tokens)")
        if task_valid.dtype != torch.bool:
            raise ValueError("task_valid must be a boolean tensor")
        if not isinstance(observation_valid, torch.Tensor) or observation_valid.ndim != 3:
            raise ValueError("observation_valid must have shape (batch, horizon, observation_tokens)")
        if observation_valid.dtype != torch.bool:
            raise ValueError("observation_valid must be a boolean tensor")
        batch, horizon, observation_count = observation_valid.shape
        if task_valid.shape[0] != batch:
            raise ValueError("task_valid and observation_valid batch sizes must match")
        if timestep_valid is None:
            timestep_valid = torch.ones(batch, horizon, dtype=torch.bool, device=observation_valid.device)
        elif (
            not isinstance(timestep_valid, torch.Tensor)
            or timestep_valid.shape != (batch, horizon)
            or timestep_valid.dtype != torch.bool
        ):
            raise ValueError("timestep_valid must be a boolean tensor with shape (batch, horizon)")
        if task_valid.device != observation_valid.device or timestep_valid.device != observation_valid.device:
            raise ValueError("all validity masks must be on the same device")

        task_count = task_valid.shape[1]
        if self.repeat_task_tokens:
            task_keys = task_valid[:, None, :].expand(batch, horizon, task_count).reshape(batch, -1)
            task_times = torch.arange(horizon, device=observation_valid.device).repeat_interleave(task_count)
            task_keys = task_keys & timestep_valid.repeat_interleave(task_count, dim=1)
        else:
            task_keys = task_valid
            task_times = torch.full((task_count,), -1, device=observation_valid.device)

        observation_keys = observation_valid & timestep_valid.unsqueeze(-1)
        observation_times = torch.arange(horizon, device=observation_valid.device).repeat_interleave(
            observation_count
        )
        readout_keys = timestep_valid.unsqueeze(-1).expand(batch, horizon, self.num_readout_tokens).reshape(
            batch, -1
        )
        readout_times = torch.arange(horizon, device=observation_valid.device).repeat_interleave(
            self.num_readout_tokens
        )

        valid_keys = torch.cat((task_keys, observation_keys.reshape(batch, -1), readout_keys), dim=1)
        task_times = task_times.to(dtype=torch.long)
        times = torch.cat((task_times, observation_times, readout_times))
        task_end = task_keys.shape[1]
        observation_end = task_end + observation_keys.shape[1] * observation_keys.shape[2]
        sequence = valid_keys.shape[1]
        query_types = torch.cat(
            (
                torch.zeros(task_end, dtype=torch.long, device=valid_keys.device),
                torch.ones(observation_end - task_end, dtype=torch.long, device=valid_keys.device),
                torch.full((sequence - observation_end,), 2, dtype=torch.long, device=valid_keys.device),
            )
        )
        key_types = query_types
        q_types = query_types.view(1, sequence, 1)
        k_types = key_types.view(1, 1, sequence)
        q_times = times.view(1, sequence, 1)
        k_times = times.view(1, 1, sequence)
        task_rule = (q_types == 0) & (k_types == 0) & (k_times <= q_times)
        observation_rule = (q_types == 1) & (
            ((k_types == 0) & (k_times <= q_times))
            | ((k_types == 1) & (k_times <= q_times))
        )
        readout_rule = (q_types == 2) & (
            ((k_types == 0) & (k_times <= q_times))
            | ((k_types == 1) & (k_times <= q_times))
            | ((k_types == 2) & (k_times <= q_times))
        )
        return (task_rule | observation_rule | readout_rule) & valid_keys[:, None, :]

    def forward(
        self,
        task_tokens,
        observation_tokens,
        task_mask=None,
        observation_mask=None,
        timestep_mask=None,
    ):
        if not isinstance(task_tokens, torch.Tensor) or task_tokens.ndim != 3:
            raise ValueError("task_tokens must have shape (batch, task_tokens, d_model)")
        if not isinstance(observation_tokens, torch.Tensor) or observation_tokens.ndim != 4:
            raise ValueError("observation_tokens must have shape (batch, horizon, observation_tokens, d_model)")
        batch, task_count, width = task_tokens.shape
        observation_batch, horizon, observation_count, observation_width = observation_tokens.shape
        if batch < 1 or task_count < 1 or horizon < 1 or observation_count < 1:
            raise ValueError("batch, task, horizon, and observation dimensions must be positive")
        if width != self.d_model or observation_width != self.d_model:
            raise ValueError(f"task_tokens and observation_tokens must have final dimension {self.d_model}")
        if batch != observation_batch:
            raise ValueError("task_tokens and observation_tokens batch sizes must match")
        if task_tokens.device != observation_tokens.device or task_tokens.dtype != observation_tokens.dtype:
            raise ValueError("task_tokens and observation_tokens must share a device and dtype")
        if not task_tokens.is_floating_point() or not observation_tokens.is_floating_point():
            raise ValueError("task_tokens and observation_tokens must be floating point")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon must not exceed max_horizon ({self.max_horizon})")

        if task_mask is None:
            task_mask = torch.ones(batch, task_count, dtype=torch.bool, device=task_tokens.device)
        elif not isinstance(task_mask, torch.Tensor) or task_mask.shape != (batch, task_count):
            raise ValueError("task_mask must have shape (batch, task_tokens)")
        elif task_mask.dtype != torch.bool:
            raise ValueError("task_mask must be a boolean tensor")
        if observation_mask is None:
            observation_mask = torch.ones(
                batch, horizon, observation_count, dtype=torch.bool, device=observation_tokens.device
            )
        elif (
            not isinstance(observation_mask, torch.Tensor)
            or observation_mask.shape != (batch, horizon, observation_count)
        ):
            raise ValueError("observation_mask must have shape (batch, horizon, observation_tokens)")
        elif observation_mask.dtype != torch.bool:
            raise ValueError("observation_mask must be a boolean tensor")
        if timestep_mask is not None and (
            not isinstance(timestep_mask, torch.Tensor)
            or timestep_mask.shape != (batch, horizon)
            or timestep_mask.dtype != torch.bool
        ):
            raise ValueError("timestep_mask must be a boolean tensor with shape (batch, horizon)")
        if task_mask.device != task_tokens.device or observation_mask.device != observation_tokens.device:
            raise ValueError("all masks must be on the same device as their token tensors")
        if timestep_mask is not None and timestep_mask.device != observation_tokens.device:
            raise ValueError("timestep_mask must be on the same device as the token tensors")

        if timestep_mask is None:
            timestep_valid = torch.ones(batch, horizon, dtype=torch.bool, device=observation_tokens.device)
        else:
            timestep_valid = timestep_mask
        allowed = self.build_attention_mask(task_mask, observation_mask, timestep_valid)
        task_input = task_tokens
        if self.repeat_task_tokens:
            task_input = task_tokens[:, None, :, :].expand(batch, horizon, task_count, width).reshape(
                batch, horizon * task_count, width
            )
        readout = self.readout_positional_embeddings[:, :horizon].expand(batch, -1, -1, -1)
        tokens = torch.cat(
            (
                task_input,
                observation_tokens.reshape(batch, horizon * observation_count, width),
                readout.reshape(batch, horizon * self.num_readout_tokens, width),
            ),
            dim=1,
        )
        for layer in self.layers:
            tokens = layer(tokens, allowed)
        tokens = self.final_norm(tokens)
        readout_start = task_input.shape[1] + horizon * observation_count
        return tokens[:, readout_start:].reshape(batch, horizon, self.num_readout_tokens, width)
