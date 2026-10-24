import math

import torch
from torch import nn


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(value, name):
    numeric_value = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not numeric_value:
        raise ValueError(f"{name} must be a finite number in (0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be a finite number in (0, 1)")
    return result


def _beta_continued_fraction(a, b, x):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for index in range(1, 201):
        twice = 2 * index
        coefficient = index * (b - index) * x / ((qam + twice) * (a + twice))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + index) * (qab + index) * x / ((a + twice) * (qap + twice))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return result


def _regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_quantile(probability, a, b):
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    if b == 1.0:
        return probability ** (1.0 / a)
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if _regularized_beta(middle, a, b) < probability:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def _binomial_two_sided_pvalue(successes, trials):
    if trials == 0:
        return 1.0
    lower = min(successes, trials - successes)
    tail = _regularized_beta(0.5, trials - lower, lower + 1.0)
    return min(1.0, 2.0 * tail)


def _normal_inverse_cdf(probability):
    return float(torch.special.ndtri(torch.tensor(probability, dtype=torch.float64)))


class RandomizedSmoothing(nn.Module):
    component_id = "randomized-smoothing"

    def __init__(self, base_classifier=None, sigma=None, n0=100, n=100000, alpha=0.001):
        super().__init__()
        if sigma is None or isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
            raise ValueError("sigma must be a finite number greater than zero")
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be a finite number greater than zero")
        if base_classifier is not None and not isinstance(base_classifier, nn.Module):
            raise ValueError("base_classifier must be a torch.nn.Module or None")
        self.sigma = sigma
        self.n0 = _positive_integer(n0, "n0")
        self.n = _positive_integer(n, "n")
        self.alpha = _probability(alpha, "alpha")
        self.base_classifier = nn.Identity() if base_classifier is None else base_classifier

    def forward(self, x, noise=None, generator=None):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor")
        if noise is None:
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        elif not isinstance(noise, torch.Tensor) or noise.shape != x.shape:
            raise ValueError("noise must be a tensor with the same shape as x")
        return self.base_classifier(x + self.sigma * noise)

    def _validate_batch(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim < 2:
            raise ValueError("x must be a floating-point tensor with a batch and feature dimensions")
        if x.shape[0] == 0 or any(size == 0 for size in x.shape[1:]):
            raise ValueError("x must have a non-empty batch and feature dimensions")

    def _sample_counts(self, x, num_samples, *, generator=None, noise_samples=None):
        self._validate_batch(x)
        _positive_integer(num_samples, "num_samples")
        if noise_samples is not None:
            expected = (x.shape[0], num_samples, *x.shape[1:])
            if not isinstance(noise_samples, torch.Tensor) or tuple(noise_samples.shape) != expected:
                raise ValueError(f"noise_samples must have shape {expected}")
            if not noise_samples.is_floating_point():
                raise ValueError("noise_samples must be a floating-point tensor")
        batch_size = x.shape[0]
        counts = None
        chunk_size = 1024
        for start in range(0, num_samples, chunk_size):
            stop = min(start + chunk_size, num_samples)
            chunk_count = stop - start
            if noise_samples is None:
                noise = torch.randn(
                    (batch_size, chunk_count, *x.shape[1:]),
                    dtype=x.dtype,
                    device=x.device,
                    generator=generator,
                )
            else:
                noise = noise_samples[:, start:stop].to(device=x.device, dtype=x.dtype)
            perturbed = x.unsqueeze(1) + self.sigma * noise
            flat = perturbed.reshape(batch_size * chunk_count, *x.shape[1:])
            logits = self.base_classifier(flat)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise ValueError("base_classifier must return a [batch, classes] logits tensor")
            if logits.shape[0] != batch_size * chunk_count or logits.shape[1] < 2:
                raise ValueError("base_classifier must return logits for at least two classes")
            if counts is None:
                counts = torch.zeros(batch_size, logits.shape[1], dtype=torch.long, device=x.device)
            elif logits.shape[1] != counts.shape[1]:
                raise ValueError("base_classifier must return a consistent number of classes")
            predictions = logits.argmax(dim=1).reshape(batch_size, chunk_count)
            counts.scatter_add_(1, predictions, torch.ones_like(predictions, dtype=counts.dtype))
        return counts

    def predict(self, x, n=None, alpha=None, *, generator=None, noise_samples=None):
        num_samples = self.n if n is None else _positive_integer(n, "n")
        significance = self.alpha if alpha is None else _probability(alpha, "alpha")
        counts = self._sample_counts(
            x, num_samples, generator=generator, noise_samples=noise_samples
        )
        top_values = counts.topk(2, dim=1).values
        labels = counts.argmax(dim=1)
        should_abstain = torch.tensor(
            [
                _binomial_two_sided_pvalue(int(row[0]), int(row[0] + row[1])) > significance
                for row in top_values.tolist()
            ],
            dtype=torch.bool,
            device=x.device,
        )
        return torch.where(should_abstain, -1, labels), counts, should_abstain

    def certify(
        self,
        x,
        n0=None,
        n=None,
        alpha=None,
        *,
        generator=None,
        selection_noise=None,
        estimation_noise=None,
    ):
        selection_samples = self.n0 if n0 is None else _positive_integer(n0, "n0")
        estimation_samples = self.n if n is None else _positive_integer(n, "n")
        significance = self.alpha if alpha is None else _probability(alpha, "alpha")
        selected_counts = self._sample_counts(
            x, selection_samples, generator=generator, noise_samples=selection_noise
        )
        selected_class = selected_counts.argmax(dim=1)
        estimated_counts = self._sample_counts(
            x, estimation_samples, generator=generator, noise_samples=estimation_noise
        )
        successes = estimated_counts.gather(1, selected_class.unsqueeze(1)).squeeze(1)
        lower_bounds = [
            0.0
            if count == 0
            else _beta_quantile(significance, count, estimation_samples - count + 1)
            for count in successes.tolist()
        ]
        p_lower = torch.tensor(lower_bounds, dtype=x.dtype, device=x.device)
        should_abstain = p_lower <= 0.5
        radii = torch.tensor(
            [
                0.0 if lower <= 0.5 else self.sigma * _normal_inverse_cdf(lower)
                for lower in lower_bounds
            ],
            dtype=x.dtype,
            device=x.device,
        )
        labels = torch.where(should_abstain, -1, selected_class)
        return labels, radii, should_abstain, p_lower
