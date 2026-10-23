import math
from numbers import Integral, Real

import torch
from torch import nn


def _value_error(message):
    error_type = ValueError
    return error_type(message)


class RLIQM(nn.Module):
    component_id = "rl-iqm"

    def __init__(self):
        super().__init__()

    @staticmethod
    def _validate_scores(scores, allow_batch=False):
        if not isinstance(scores, torch.Tensor):
            raise _value_error("scores must be a floating-point tensor with shape (runs, tasks)")
        if scores.ndim != 2 and not (allow_batch and scores.ndim == 3):
            raise ValueError("scores must have shape (runs, tasks)")
        if any(size < 1 for size in scores.shape):
            raise ValueError("scores must contain at least one run and one task")
        if scores.layout != torch.strided:
            raise ValueError("scores must be a dense floating-point tensor")
        if not scores.dtype.is_floating_point:
            raise ValueError("scores must use a floating-point dtype")

    @staticmethod
    def _trimmed_mean(values):
        ordered = torch.sort(values, dim=-1).values
        count = ordered.shape[-1]
        trim = count // 4
        end = count - trim if trim else count
        return ordered[..., trim:end].mean(dim=-1)

    def forward(self, scores):
        self._validate_scores(scores, allow_batch=True)
        if scores.ndim == 2:
            return self._trimmed_mean(scores.reshape(-1))
        return self._trimmed_mean(scores.reshape(scores.shape[0], -1))

    def confidence_interval(
        self,
        scores,
        num_resamples=50000,
        confidence_level=0.95,
        seed=None,
    ):
        self._validate_scores(scores)
        if not isinstance(num_resamples, Integral) or isinstance(num_resamples, bool):
            raise _value_error("num_resamples must be a positive integer")
        if num_resamples < 1:
            raise ValueError("num_resamples must be a positive integer")
        if not isinstance(confidence_level, Real) or isinstance(confidence_level, bool):
            raise _value_error("confidence_level must be a finite number strictly between 0 and 1")
        if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be a finite number strictly between 0 and 1")
        if seed is not None:
            if not isinstance(seed, Integral) or isinstance(seed, bool):
                raise ValueError("seed must be an integer or None")
            if not -(2**63) <= seed <= 2**64 - 1:
                raise ValueError("seed must be in the range [-2**63, 2**64 - 1]")

        runs, tasks = scores.shape
        total = runs * tasks
        chunk_size = max(1, min(1024, 1_000_000 // total))
        generator = None
        if seed is not None:
            generator = torch.Generator(device=scores.device)
            generator.manual_seed(int(seed))
        task_indices = torch.arange(tasks, device=scores.device).view(1, 1, tasks)
        estimates = []
        with torch.no_grad():
            for start in range(0, int(num_resamples), chunk_size):
                size = min(chunk_size, int(num_resamples) - start)
                indices = torch.randint(
                    runs,
                    (size, runs, tasks),
                    device=scores.device,
                    generator=generator,
                )
                resampled = scores.detach()[indices, task_indices]
                estimates.append(self._trimmed_mean(resampled.reshape(size, total)))
        bootstrap_iqms = torch.cat(estimates)
        tail = (1.0 - float(confidence_level)) / 2.0
        quantiles = torch.tensor(
            [tail, 1.0 - tail], device=scores.device, dtype=torch.float64
        )
        interval = torch.quantile(bootstrap_iqms.to(torch.float64), quantiles)
        return interval.to(dtype=scores.dtype)
