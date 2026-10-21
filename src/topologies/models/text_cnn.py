import torch
from torch import nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    architecture_id = 'text-cnn'

    def __init__(self, vocab_size, num_classes, embedding_dim=64, kernel_sizes=(3, 4, 5), filters=64, padding_id=0, dropout=0.0, context_length=512, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in ('tiny', 'small', 'base'):
                raise ValueError('preset must be tiny, small, or base')
            embedding_dim, kernel_sizes, filters = {'tiny': (32, (2, 3), 32), 'small': (64, (2, 3, 4), 64), 'base': (128, (2, 3, 4, 5), 128)}[preset]
        for name, v in (('vocab_size', vocab_size), ('num_classes', num_classes), ('embedding_dim', embedding_dim), ('filters', filters), ('context_length', context_length)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f'{name} must be positive')
        if not kernel_sizes or any((isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in kernel_sizes)):
            raise ValueError('kernel_sizes must be positive integers')
        if not isinstance(padding_id, int) or not 0 <= padding_id < vocab_size:
            raise ValueError('padding_id must be in vocabulary range')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.vocab_size, self.context_length, self.padding_id = (vocab_size, context_length, padding_id)
        self.kernel_sizes = tuple(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_id)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, filters, k) for k in self.kernel_sizes])
        self.dropout, self.head = (nn.Dropout(dropout), nn.Linear(filters * len(self.kernel_sizes), num_classes))

    def forward(self, tokens, mask=None):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError('tokens must be LongTensor[B,T]')
        if not 0 < tokens.shape[1] <= self.context_length:
            raise ValueError('sequence length outside [1, context_length]')
        if tokens.device.type != 'meta' and tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError('token IDs outside vocabulary')
        if mask is None:
            valid = tokens != self.padding_id
        else:
            if mask.shape != tokens.shape or mask.device != tokens.device:
                raise ValueError('mask must match tokens shape and device')
            if mask.dtype != torch.bool and (not mask.is_floating_point()) and (mask.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
                raise ValueError('mask must contain boolean or numeric 0/1 values')
            if mask.device.type != 'meta' and (not torch.all((mask == 0) | (mask == 1))):
                raise ValueError('mask must contain 0/1; 1 means valid')
            valid = mask.bool()
        emb = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for k, conv in zip(self.kernel_sizes, self.convs):
            if tokens.shape[1] < k:
                pooled.append(emb.new_zeros((tokens.shape[0], conv.out_channels)))
                continue
            values = F.gelu(conv(emb))
            windows = valid.unfold(1, k, 1).all(-1)
            values = values.masked_fill(~windows[:, None], torch.finfo(values.dtype).min)
            maximum = values.max(-1).values
            pooled.append(torch.where(windows.any(-1, keepdim=True), maximum, torch.zeros_like(maximum)))
        return self.head(self.dropout(torch.cat(pooled, -1)))
