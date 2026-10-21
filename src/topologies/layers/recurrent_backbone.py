import torch
from torch import nn

class RecurrentBackbone(nn.Module):
    component_id = 'recurrent-backbone'

    def __init__(self, input_size, hidden_size, num_layers=1, cell='gru', bidirectional=False, dropout=0.0):
        super().__init__()
        if min(input_size, hidden_size, num_layers) <= 0 or cell not in ('rnn', 'gru', 'lstm') or (not 0 <= dropout < 1):
            raise ValueError('invalid recurrent configuration')
        cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell]
        self.rnn = cls(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.cell = cell

    def forward(self, x, lengths=None, state=None):
        if x.ndim != 3 or x.shape[-1] != self.rnn.input_size:
            raise ValueError('expected [B,T,input_size]')
        if lengths is None:
            return self.rnn(x, state)
        if lengths.ndim != 1 or lengths.numel() != x.shape[0] or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError('lengths must be an integer [B] tensor')
        if bool(((lengths <= 0) | (lengths > x.shape[1])).any()):
            raise ValueError('lengths must be in [1,T]')
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, state = self.rnn(packed, state)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return (out, state)
