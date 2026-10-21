import torch
from torch import nn
from torch.nn import functional as F


class SepFormer(nn.Module):
    architecture_id = "sepformer"

    def __init__(self, channels=4, chunk_size=2, num_heads=2, num_layers=1, dropout=0.0):
        super().__init__()
        if channels <= 0 or chunk_size <= 0 or num_heads <= 0 or num_layers <= 0:
            raise ValueError("channels, chunk_size, num_heads, and num_layers must be positive")
        if chunk_size < 2 or chunk_size % 2:
            raise ValueError("chunk_size must be even and at least two for 50% overlap")
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.chunk_size = chunk_size
        self.intra_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                channels, num_heads, channels * 4, dropout, batch_first=True, norm_first=True
            )
            for _ in range(num_layers)
        )
        self.inter_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                channels, num_heads, channels * 4, dropout, batch_first=True, norm_first=True
            )
            for _ in range(num_layers)
        )

    @staticmethod
    def _positions(length, channels, *, device, dtype):
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, channels, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / channels)
        )
        encoding = torch.zeros(length, channels, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return encoding

    def _split_chunks(self, x):
        batch, length, channels = x.shape
        step = self.chunk_size // 2
        padded_length = max(self.chunk_size, ((length + step - 1) // step) * step)
        padded = F.pad(x, (0, 0, 0, padded_length - length))
        chunks = torch.stack(
            [padded[:, start : start + self.chunk_size] for start in range(0, padded_length - self.chunk_size + 1, step)],
            dim=1,
        )
        return chunks, length

    def _merge_chunks(self, chunks, length):
        batch, count, width, channels = chunks.shape
        step = self.chunk_size // 2
        total_length = (count - 1) * step + width
        values = [chunks.new_zeros((batch, channels)) for _ in range(total_length)]
        counts = [0] * total_length
        for chunk_index in range(count):
            start = chunk_index * step
            for offset in range(width):
                values[start + offset] = values[start + offset] + chunks[:, chunk_index, offset]
                counts[start + offset] += 1
        merged = torch.stack(
            [value / count for value, count in zip(values, counts)], dim=1
        )
        return merged[:, :length]

    def intra_attention(self, chunks):
        batch, count, width, channels = chunks.shape
        if width != self.chunk_size or channels != self.channels:
            raise ValueError("chunks must have shape (batch, chunks, chunk_size, channels)")
        values = chunks.reshape(batch * count, width, channels)
        values = values + self._positions(width, channels, device=values.device, dtype=values.dtype)
        for layer in self.intra_layers:
            values = layer(values)
        return values.reshape(batch, count, width, channels)

    def inter_attention(self, chunks):
        batch, count, width, channels = chunks.shape
        values = chunks.permute(0, 2, 1, 3).reshape(batch * width, count, channels)
        values = values + self._positions(count, channels, device=values.device, dtype=values.dtype)
        for layer in self.inter_layers:
            values = layer(values)
        return values.reshape(batch, width, count, channels).permute(0, 2, 1, 3)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError("input must have shape (batch, sequence_length, channels)")
        if x.shape[1] == 0:
            raise ValueError("sequence_length must be positive")
        chunks, length = self._split_chunks(x)
        chunks = self.intra_attention(chunks)
        chunks = self.inter_attention(chunks)
        return self._merge_chunks(chunks, length)
