import torch
from torch import nn


def cycle_consistency_loss(translator_a_to_b, translator_b_to_a, real_a, real_b):
    reconstructed_a = translator_b_to_a(translator_a_to_b(real_a))
    reconstructed_b = translator_a_to_b(translator_b_to_a(real_b))
    if reconstructed_a.shape != real_a.shape or reconstructed_b.shape != real_b.shape:
        raise ValueError("each translated cycle must reconstruct its input shape")
    return (reconstructed_a - real_a).abs().mean() + (reconstructed_b - real_b).abs().mean()


class CycleGAN(nn.Module):
    architecture_id = "cyclegan"

    def __init__(self, translator_a_to_b=None, translator_b_to_a=None):
        super().__init__()
        if (translator_a_to_b is None) != (translator_b_to_a is None):
            raise ValueError("supply both translators or neither")
        if translator_a_to_b is None:
            translator_a_to_b = nn.Identity()
            translator_b_to_a = nn.Identity()
        if not isinstance(translator_a_to_b, nn.Module) or not isinstance(translator_b_to_a, nn.Module):
            raise TypeError("translators must be torch.nn.Module instances")
        self.translator_a_to_b = translator_a_to_b
        self.translator_b_to_a = translator_b_to_a

    def cycle_reconstruct(self, value, direction="a_to_b"):
        if direction == "a_to_b":
            return self.translator_b_to_a(self.translator_a_to_b(value))
        if direction == "b_to_a":
            return self.translator_a_to_b(self.translator_b_to_a(value))
        raise ValueError("direction must be 'a_to_b' or 'b_to_a'")

    def cycle_loss(self, real_a, real_b):
        return cycle_consistency_loss(
            self.translator_a_to_b, self.translator_b_to_a, real_a, real_b
        )

    def forward(self, real_a, real_b):
        return self.cycle_loss(real_a, real_b)
