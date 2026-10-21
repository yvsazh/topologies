import torch
from torch import nn
from torch.nn import functional as F


class DALLE(nn.Module):
    architecture_id = "dall-e"

    def __init__(self, vocab_size=8192, width=64, layers=2, heads=4, max_length=1024):
        super().__init__()
        if vocab_size < 2 or width < 1 or layers < 1 or heads < 1 or width % heads or max_length < 1:
            raise ValueError("invalid model dimensions")
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.code_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_length, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True, norm_first=True)
        self.prior = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, codes):
        codes = torch.as_tensor(codes, device=self.code_embedding.weight.device)
        if codes.is_floating_point() or codes.is_complex():
            raise TypeError("codes must contain integer codebook indices")
        codes = codes.to(dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] == 0 or codes.shape[1] > self.max_length:
            raise ValueError("codes must have shape (batch, nonempty sequence)")
        if torch.any(codes < 0) or torch.any(codes >= self.vocab_size):
            raise ValueError("code index outside the quantized codebook")
        position = torch.arange(codes.shape[1], device=codes.device)
        hidden = self.code_embedding(codes) + self.position_embedding(position)[None, :, :]
        causal_mask = torch.triu(torch.ones(codes.shape[1], codes.shape[1], dtype=torch.bool, device=codes.device), diagonal=1)
        return self.output(self.prior(hidden, mask=causal_mask))

    def next_code_loss(self, codes):
        logits = self(codes)
        if logits.shape[1] < 2:
            raise ValueError("next-code loss requires at least two codes")
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            torch.as_tensor(codes, device=logits.device, dtype=torch.long)[:, 1:].reshape(-1),
        )

    @torch.no_grad()
    def generate(self, prefix, steps=1, temperature=1.0):
        prefix = torch.as_tensor(prefix, dtype=torch.long, device=self.code_embedding.weight.device)
        if prefix.ndim != 2 or prefix.shape[1] == 0:
            raise ValueError("prefix must have shape (batch, nonempty sequence)")
        if steps < 0 or temperature <= 0 or prefix.shape[1] + steps > self.max_length:
            raise ValueError("invalid generation parameters")
        result = prefix
        for _ in range(steps):
            logits = self(result)[:, -1] / temperature
            next_code = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            result = torch.cat((result, next_code), dim=1)
        return result
