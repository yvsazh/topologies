import torch
from torch import nn


def jensen_shannon(*logits):
    if len(logits) < 2:
        raise ValueError("Jensen-Shannon divergence needs at least two distributions")
    if any(values.shape != logits[0].shape for values in logits[1:]):
        raise ValueError("all logits must have the same shape")
    log_probabilities = [torch.log_softmax(values, dim=-1) for values in logits]
    probabilities = [values.exp() for values in log_probabilities]
    midpoint = torch.stack(probabilities, dim=0).mean(dim=0)
    midpoint_log = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    divergence = sum(
        (probability * (log_probability - midpoint_log)).sum(dim=-1)
        for probability, log_probability in zip(probabilities, log_probabilities)
    ) / len(logits)
    return divergence.mean()


class AugMix(nn.Module):
    component_id = "augmix"
    def __init__(self, mixture_width=3, alpha=1.0):
        super().__init__()
        if isinstance(mixture_width, bool) or not isinstance(mixture_width, int) or mixture_width < 1:
            raise ValueError("mixture width must be an integer of at least one")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.mixture_width = mixture_width
        self.alpha = float(alpha)

    def _augment(self, image, generator):
        result = image
        operations = int(torch.randint(1, 4, (), device=image.device, generator=generator).item())
        for _ in range(operations):
            operation = int(torch.randint(0, 3, (), device=image.device, generator=generator).item())
            if operation == 0 and result.ndim >= 3:
                dimension = result.ndim - 1 if bool(torch.randint(0, 2, (), device=image.device, generator=generator).item()) else result.ndim - 2
                result = torch.flip(result, dims=(dimension,))
            elif operation == 1 and result.ndim >= 3:
                dims = (result.ndim - 2, result.ndim - 1)
                shifts = (int(torch.randint(-2, 3, (), device=image.device, generator=generator).item()), int(torch.randint(-2, 3, (), device=image.device, generator=generator).item()))
                result = torch.roll(result, shifts=shifts, dims=dims)
            else:
                scale = 0.5 + torch.rand((), device=image.device, dtype=image.dtype, generator=generator)
                result = result * scale
        return result

    def forward(self, x, generator=None):
        if not isinstance(x, torch.Tensor) or x.ndim < 2 or not x.is_floating_point():
            raise ValueError("x must be a floating point tensor with batch and feature dimensions")
        concentration = torch.full((self.mixture_width,), self.alpha, dtype=torch.float32, device=x.device)
        chain_weights = torch._standard_gamma(concentration, generator=generator)
        chain_weights = chain_weights / chain_weights.sum()
        mix = torch.zeros_like(x)
        for index in range(self.mixture_width):
            mix = mix + chain_weights[index].to(dtype=x.dtype) * self._augment(x, generator)
        beta_pair = torch.full((2,), self.alpha, dtype=torch.float32, device=x.device)
        beta_samples = torch._standard_gamma(beta_pair, generator=generator)
        beta = beta_samples[0] / beta_samples.sum()
        beta = beta.to(dtype=x.dtype)
        return beta * x + (1 - beta) * mix
