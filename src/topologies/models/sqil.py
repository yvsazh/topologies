import math

import torch
from torch import nn


class SQILSoftBellmanLoss(nn.Module):
    architecture_id = "sqil"
    PRESETS = {"tiny": {"lambda_samp": 1.0}, "small": {"lambda_samp": 1.0}}

    def __init__(self, lambda_samp=1.0):
        super().__init__()
        if type(lambda_samp) not in (int, float):
            raise ValueError("lambda_samp must be a finite nonnegative number")
        if not math.isfinite(lambda_samp) or lambda_samp < 0.0:
            raise ValueError("lambda_samp must be a finite nonnegative number")
        self.lambda_samp = float(lambda_samp)

    def forward(
        self,
        q_demo,
        actions_demo,
        next_q_demo,
        q_sample,
        actions_sample,
        next_q_sample,
        discount,
    ):
        if type(discount) not in (int, float):
            raise ValueError("discount must be a finite number in [0, 1]")
        if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be a finite number in [0, 1]")

        q_values = (q_demo, next_q_demo, q_sample, next_q_sample)
        if any(not torch.is_tensor(values) for values in q_values):
            raise ValueError("all Q-values must be tensors")
        for name, values in zip(
            ("q_demo", "next_q_demo", "q_sample", "next_q_sample"), q_values
        ):
            if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
                raise ValueError(f"{name} must have shape [batch, action_count] with positive sizes")
            if not torch.is_floating_point(values):
                raise ValueError(f"{name} must have a floating-point dtype")
        if q_demo.shape != next_q_demo.shape:
            raise ValueError("q_demo and next_q_demo must have matching shapes")
        if q_sample.shape != next_q_sample.shape:
            raise ValueError("q_sample and next_q_sample must have matching shapes")
        if q_demo.shape[1] != q_sample.shape[1]:
            raise ValueError("demonstration and sampled Q-values must have matching action counts")
        if any(values.dtype != q_demo.dtype for values in q_values):
            raise ValueError("all Q-values must have matching dtypes")
        if any(values.device != q_demo.device for values in q_values):
            raise ValueError("all Q-values must be on the same device")

        action_values = (actions_demo, actions_sample)
        q_batches = (q_demo, q_sample)
        names = ("actions_demo", "actions_sample")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        for actions, q_batch, name in zip(action_values, q_batches, names):
            if not torch.is_tensor(actions) or actions.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional integer tensor")
            if actions.dtype not in integer_dtypes:
                raise ValueError(f"{name} must have an integer dtype")
            if actions.shape[0] != q_batch.shape[0]:
                raise ValueError(f"{name} must match the corresponding Q-value batch size")
            if actions.device != q_batch.device:
                raise ValueError(f"{name} and its Q-values must be on the same device")
            if torch.any(actions < 0) or torch.any(actions >= q_batch.shape[1]):
                raise ValueError(f"{name} contains an action outside the available action range")

        demo_values = q_demo.gather(1, actions_demo.long().unsqueeze(1)).squeeze(1)
        sample_values = q_sample.gather(1, actions_sample.long().unsqueeze(1)).squeeze(1)
        demo_targets = 1.0 + discount * torch.logsumexp(next_q_demo, dim=1)
        sample_targets = discount * torch.logsumexp(next_q_sample, dim=1)
        demo_loss = (demo_values - demo_targets).square().mean()
        sample_loss = (sample_values - sample_targets).square().mean()
        return demo_loss + self.lambda_samp * sample_loss
