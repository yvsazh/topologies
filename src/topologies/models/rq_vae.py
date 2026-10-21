import torch
from torch import nn


class RQVAE(nn.Module):
    architecture_id = "rq-vae"
    def __init__(self, channels=4, num_codebooks=2, codebook_size=6, commitment_cost=0.25):
        super().__init__()
        if channels <= 0 or num_codebooks <= 0 or codebook_size <= 0:
            raise ValueError("channels, num_codebooks, and codebook_size must be positive")
        self.channels = channels
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.codebooks = nn.ModuleList(
            nn.Embedding(codebook_size, channels) for _ in range(num_codebooks)
        )
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def quantize(self, latent):
        if latent.shape[-1] != self.channels:
            raise ValueError(f"expected last dimension {self.channels}, got {latent.shape[-1]}")
        residual = latent
        quantized_sum = torch.zeros_like(latent)
        indices = []
        stage_outputs = []
        quantization_loss = latent.new_zeros(())
        for codebook in self.codebooks:
            flat = residual.reshape(-1, self.channels)
            distances = flat.square().sum(dim=1, keepdim=True) + codebook.weight.square().sum(dim=1) - 2 * flat @ codebook.weight.t()
            code_indices = distances.argmin(dim=1)
            vectors = codebook(code_indices).reshape_as(latent)
            quantized_sum = quantized_sum + vectors
            stage_outputs.append(quantized_sum)
            codebook_loss = (residual.detach() - vectors).square().mean()
            commitment_loss = (residual - vectors.detach()).square().mean()
            quantization_loss = quantization_loss + codebook_loss + self.commitment_cost * commitment_loss
            residual = residual - vectors.detach()
            indices.append(code_indices.reshape(latent.shape[:-1]))
        straight_through = latent + (quantized_sum - latent).detach()
        return straight_through, torch.stack(indices, dim=-1), quantization_loss, tuple(stage_outputs)

    def forward(self, inputs):
        quantized, indices, loss, _ = self.quantize(inputs)
        return {"reconstruction": quantized, "indices": indices, "loss": loss}
