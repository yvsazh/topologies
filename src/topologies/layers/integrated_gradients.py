import torch


class IntegratedGradients(torch.nn.Module):
    component_id = "integrated-gradients"
    def __init__(self, device="cpu"):
        super().__init__()
        self.to(device)

    def forward(self, grad_fn, x, baseline, steps):
        if torch.is_tensor(steps):
            if steps.numel() != 1 or (steps.is_floating_point() and steps.item() % 1):
                raise ValueError("steps must be at least 1 and an integer")
            steps = int(steps.item())
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be at least 1 and an integer")
        difference = x - baseline
        average_gradient = torch.zeros_like(difference)
        for index in range(1, steps + 1):
            alpha = index / steps
            point = baseline + alpha * difference
            gradient = grad_fn(point) if callable(grad_fn) else torch.as_tensor(
                grad_fn, dtype=point.dtype, device=point.device
            )
            average_gradient = average_gradient + gradient
        return (average_gradient / steps) * difference
