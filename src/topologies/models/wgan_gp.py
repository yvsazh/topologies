import torch
from torch import nn


def gradient_penalty(critic, real, fake):
    if real.shape != fake.shape or real.ndim < 2:
        raise ValueError("real and fake must have matching batch-shaped tensors")
    batch = real.shape[0]
    alpha_shape = (batch,) + (1,) * (real.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real.device, dtype=real.dtype)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        materialize_grads=True,
    )[0]
    norms = gradients.reshape(batch, -1).norm(2, dim=1)
    penalty = ((norms - 1) ** 2).mean()
    if not penalty.requires_grad:
        penalty = penalty + interpolated.sum() * 0
    return penalty


class WGANGP(nn.Module):
    architecture_id = "wgan-gp"
    def __init__(self, features=64, penalty_weight=10, input_dim=3):
        super().__init__()
        if features <= 0 or penalty_weight < 0 or input_dim <= 0:
            raise ValueError("features and input_dim must be positive and penalty_weight nonnegative")
        self.features = features
        self.penalty_weight = penalty_weight
        self.input_dim = input_dim
        self.critic = nn.Sequential(
            nn.Linear(input_dim, features),
            nn.LeakyReLU(0.2),
            nn.Linear(features, features),
            nn.LeakyReLU(0.2),
            nn.Linear(features, 1),
        )

    def forward(self, samples):
        if samples.ndim < 2:
            raise ValueError("samples must include a batch dimension and features")
        flattened = samples.reshape(samples.shape[0], -1)
        if flattened.shape[1] != self.input_dim:
            raise ValueError(f"flattened samples must have {self.input_dim} features")
        return self.critic(flattened).reshape(-1)

    def gradient_penalty(self, real, fake):
        return gradient_penalty(self, real, fake)

    def critic_loss(self, real, fake):
        return self(fake).mean() - self(real).mean() + self.penalty_weight * self.gradient_penalty(real, fake)
