import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def greedy_ctc_collapse(logits, lengths, blank=0):
    if logits.ndim != 3 or lengths.shape != (logits.shape[0],):
        raise ValueError('expected [B,T,V] logits and [B] lengths')
    if not 0 <= blank < logits.shape[-1] or torch.any(lengths < 0) or torch.any(lengths > logits.shape[1]):
        raise ValueError('invalid blank id or lengths')
    result = []
    for row, length in zip(logits.argmax(-1), lengths):
        ids = row[:int(length)].tolist()
        out = []
        prev = None
        for token in ids:
            if token != prev and token != blank:
                out.append(token)
            prev = token
        result.append(out)
    return result

class CTCAcousticModel(nn.Module):
    architecture_id = 'ctc-acoustic-model'

    @classmethod
    def preset(cls, name, **geometry):
        widths = {'tiny': (16, 16, 1), 'small': (32, 32, 2), 'base': (64, 64, 3)}
        if name not in widths:
            raise ValueError('preset must be tiny, small, or base')
        conv, hidden, layers = widths[name]
        return cls(conv_channels=conv, hidden_size=hidden, rnn_layers=layers, **geometry)

    def __init__(self, vocab_size, in_channels=1, conv_channels=32, hidden_size=64, rnn_layers=2, blank_id=0):
        super().__init__()
        if min(vocab_size, in_channels, conv_channels, hidden_size, rnn_layers) <= 0 or not 0 <= blank_id < vocab_size:
            raise ValueError('invalid dimensions or blank_id')
        self.vocab_size, self.in_channels, self.blank_id = (vocab_size, in_channels, blank_id)
        self.conv = nn.Sequential(nn.Conv1d(in_channels, conv_channels, 5, stride=2, padding=2), nn.ReLU(), nn.Conv1d(conv_channels, conv_channels, 5, stride=2, padding=2), nn.ReLU())
        self.rnn = nn.GRU(conv_channels, hidden_size, batch_first=True, bidirectional=True, num_layers=rnn_layers)
        self.output = nn.Linear(hidden_size * 2, vocab_size)

    def output_lengths(self, lengths):
        return ((lengths + 1) // 2 + 1) // 2

    def forward(self, x, lengths):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError('input must be [B,C,T]')
        lengths = torch.as_tensor(lengths, device=x.device, dtype=torch.long)
        if lengths.shape != (x.shape[0],) or torch.any(lengths <= 0) or torch.any(lengths > x.shape[-1]):
            raise ValueError('invalid lengths')
        out_lengths = self.output_lengths(lengths)
        encoded = []
        for i in range(x.shape[0]):
            encoded.append(self.conv(x[i:i + 1, :, :int(lengths[i])]).transpose(1, 2))
        max_t = max((v.shape[1] for v in encoded))
        y = torch.cat([F.pad(v, (0, 0, 0, max_t - v.shape[1])) for v in encoded], 0)
        packed = pack_padded_sequence(y, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        y, _ = pad_packed_sequence(packed, batch_first=True, total_length=y.shape[1])
        return (self.output(y), out_lengths)
