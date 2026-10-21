import torch
from torch import nn


class JointEnergy(nn.Module):
    architecture_id = "joint-energy"
    component_id = "joint-energy"
    def __init__(self, features=4, num_classes=3, perturb_steps=1, step_size=0.01, noise_scale=0.01):
        super().__init__()
        if features <= 0 or num_classes <= 0:
            raise ValueError("features and num_classes must be positive")
        if perturb_steps < 0 or step_size < 0 or noise_scale < 0:
            raise ValueError("perturbation parameters must be nonnegative")
        self.features = features
        self.num_classes = num_classes
        self.perturb_steps = perturb_steps
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.classifier = nn.Linear(features, num_classes)

    def logits(self, values):
        if values.shape[-1] != self.features:
            raise ValueError("last input dimension must equal features")
        return self.classifier(values)

    def energy(self, values):
        return -torch.logsumexp(self.logits(values), dim=-1)

    def forward(self, values):
        return self.energy(values)

    def perturbed_sample(self, values):
        sample = values.detach()
        for _ in range(self.perturb_steps):
            sample = sample.requires_grad_(True)
            energy = self.energy(sample).sum()
            gradient = torch.autograd.grad(energy, sample)[0]
            sample = (sample - 0.5 * self.step_size * gradient
                      + self.noise_scale * torch.randn_like(sample)).detach()
        return sample

    def contrastive_difference(self, values):
        negative = self.perturbed_sample(values)
        return self.energy(values).mean() - self.energy(negative).mean()
