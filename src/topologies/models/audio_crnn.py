import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AudioCRNN(nn.Module):
    architecture_id = 'audio-crnn'

    @classmethod
    def preset(cls, name: str, **geometry):
        widths = {'tiny': (16, 16, 1), 'small': (32, 32, 1), 'base': (64, 64, 2)}
        if name not in widths:
            raise ValueError('preset must be tiny, small, or base')
        conv, hidden, layers = widths[name]
        return cls(conv_channels=conv, hidden_size=hidden, rnn_layers=layers, **geometry)

    def __init__(self, num_classes=10, input_mode='waveform', sample_rate=16000, in_channels=1, n_fft=256, hop_length=128, hidden_size=64, conv_channels=32, rnn_layers=1, dropout=0.0, mel_bins=None):
        super().__init__()
        if min(num_classes, sample_rate, in_channels, n_fft, hop_length, hidden_size, conv_channels, rnn_layers) <= 0:
            raise ValueError('dimensions and sample_rate must be positive')
        if input_mode not in ('waveform', 'spectrogram'):
            raise ValueError('input_mode must be waveform or spectrogram')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.input_mode, self.sample_rate, self.n_fft, self.hop_length = (input_mode, sample_rate, n_fft, hop_length)
        self.in_channels = in_channels
        self.mel_bins = mel_bins
        self.register_buffer('window', torch.hann_window(n_fft), persistent=True)
        self.conv = nn.Sequential(nn.Conv2d(1, conv_channels, 3, padding=1), nn.ReLU(), nn.Conv2d(conv_channels, conv_channels, 3, stride=2, padding=1), nn.ReLU())
        self.rnn = nn.GRU(conv_channels, hidden_size, batch_first=True, bidirectional=True, num_layers=rnn_layers, dropout=dropout if rnn_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def _features(self, x):
        if self.input_mode == 'waveform':
            if x.ndim == 2:
                x = x.unsqueeze(1)
            if x.ndim != 3 or x.shape[1] != self.in_channels:
                raise ValueError('waveform must be [B,T] or [B,C,T]')
            if x.shape[-1] < self.n_fft:
                raise ValueError('waveform shorter than n_fft')
            return torch.log1p(torch.stft(x.reshape(-1, x.shape[-1]), self.n_fft, self.hop_length, window=self.window.to(x), return_complex=True).abs().mean(0).reshape(x.shape[0], -1, x.shape[-1] // self.hop_length + 1)).unsqueeze(1)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('spectrogram must be [B,F,T] or [B,1,F,T]')
        return x

    def forward(self, x, lengths=None, sample_rate=None):
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raise ValueError('sample rate mismatch')
        if self.input_mode == 'waveform' and x.ndim == 2:
            x = x.unsqueeze(1)
        b = x.shape[0]
        raw_t = x.shape[-1]
        if lengths is None:
            lengths = torch.full((b,), raw_t, dtype=torch.long, device=x.device)
        if lengths.shape != (b,) or torch.any(lengths <= 0) or torch.any(lengths > raw_t):
            raise ValueError('invalid input lengths')
        if self.input_mode == 'waveform':
            feats = []
            for i in range(b):
                valid = x[i:i + 1, :, :int(lengths[i])]
                feats.append(self._features(valid))
            t = max((v.shape[-1] for v in feats))
            feat = torch.cat([torch.nn.functional.pad(v, (0, t - v.shape[-1])) for v in feats], 0)
            lengths = lengths // self.hop_length + 1
        else:
            feat = self._features(x)
            t = feat.shape[-1]
            if lengths is None:
                lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
            if lengths.shape != (b,) or torch.any(lengths <= 0) or torch.any(lengths > t):
                raise ValueError('spectrogram lengths must be valid frame counts')
        _, _, _, t = feat.shape
        y = self.conv(feat).mean(2).transpose(1, 2)
        out_lengths = (lengths + 1) // 2
        packed = pack_padded_sequence(y, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        y, _ = pad_packed_sequence(packed, batch_first=True, total_length=y.shape[1])
        return self.classifier(y).sum(1) / out_lengths.to(y).unsqueeze(1)
