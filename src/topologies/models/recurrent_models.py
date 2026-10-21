import torch
from torch import nn


def _check_dropout(value):
    if not 0 <= value < 1:
        raise ValueError("dropout must be in [0, 1)")


def _rnn(kind, input_size, hidden_size, num_layers, dropout, batch_first=True):
    cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
    return cls(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first,
               dropout=dropout if num_layers > 1 else 0.0)


def _run(module, x, lengths, mask, state):
    b, t, _ = x.shape
    if lengths is not None and mask is not None:
        raise ValueError("provide lengths or mask, not both")
    if mask is not None:
        if mask.shape != (b, t) or (mask.dtype != torch.bool and mask.dtype.is_floating_point):
            raise ValueError("mask must be boolean or integer [B,T], True/1 means valid")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must contain only 0/1 values")
        mask = mask.bool()
        lengths = mask.sum(1)
        if (lengths == 0).any() or not torch.equal(mask, torch.arange(t, device=x.device)[None, :] < lengths[:, None]):
            raise ValueError("mask must contain nonempty right-padded valid prefixes")
    if lengths is None:
        return module(x, state)
    if lengths.shape != (b,) or lengths.dtype not in (torch.int32, torch.int64) or (lengths < 1).any() or (lengths > t).any():
        raise ValueError("lengths must be integer [B] values in [1,T]")
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    out, final = module(packed, state)
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=t)
    return out, final


class RecurrentLM(nn.Module):
    architecture_id = "recurrent-lm"

    def __init__(self, vocab_size=128, context_length=128, hidden_size=64, num_layers=1,
                 cell="gru", dropout=0.0, tie_weights=False):
        super().__init__()
        if min(vocab_size, context_length, hidden_size, num_layers) < 1 or cell not in ("rnn", "gru", "lstm"):
            raise ValueError("invalid recurrent LM dimensions or cell")
        _check_dropout(dropout)
        if tie_weights and hidden_size != self._embedding_size(hidden_size):
            raise ValueError("tied weights require matching embedding and hidden dimensions")
        self.vocab_size, self.context_length = vocab_size, context_length
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.recurrent = _rnn(cell, hidden_size, hidden_size, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, vocab_size, bias=not tie_weights)
        if tie_weights:
            self.head.weight = self.embedding.weight

    @staticmethod
    def _embedding_size(hidden_size):
        return hidden_size

    def forward(self, tokens, lengths=None, mask=None, state=None, return_state=False):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be int64 [B,T]")
        if not 1 <= tokens.shape[1] <= self.context_length:
            raise ValueError("sequence length exceeds context_length or is empty")
        x = self.embedding(tokens)
        out, final = _run(self.recurrent, x, lengths, mask, state)
        logits = self.head(self.dropout(out))
        return (logits, final) if return_state else logits


class RecurrentPredictor(nn.Module):
    architecture_id = "recurrent-predictor"

    def __init__(self, in_features=1, hidden_size=32, out_features=1, num_layers=1,
                 cell="gru", head="last", horizon=1, dropout=0.0):
        super().__init__()
        if min(in_features, hidden_size, out_features, num_layers, horizon) < 1 or cell not in ("rnn", "gru", "lstm") or head not in ("last", "per_step", "horizon"):
            raise ValueError("invalid predictor dimensions, cell, or head")
        _check_dropout(dropout)
        self.head_kind, self.horizon = head, horizon
        self.recurrent = _rnn(cell, in_features, hidden_size, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, out_features * horizon if head == "horizon" else out_features)

    def forward(self, x, lengths=None, mask=None, state=None, return_state=False):
        if x.ndim != 3 or not x.is_floating_point() or x.shape[-1] != self.recurrent.input_size:
            raise ValueError("x must be floating [B,T,in_features]")
        if x.shape[1] < 1:
            raise ValueError("sequence cannot be empty")
        out, final = _run(self.recurrent, x, lengths, mask, state)
        if self.head_kind in ("last", "horizon"):
            if lengths is None and mask is not None:
                lengths = mask.sum(1)
            y = out[:, -1] if lengths is None else out[torch.arange(x.shape[0], device=x.device), lengths.to(x.device) - 1]
            y = self.projection(self.dropout(y))
            if self.head_kind == "horizon":
                y = y.reshape(x.shape[0], self.horizon, -1)
            return (y, final) if return_state else y
        y = self.projection(self.dropout(out))
        return (y, final) if return_state else y
