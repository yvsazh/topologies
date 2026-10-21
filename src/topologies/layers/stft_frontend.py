import torch
from torch import nn

class STFTFrontend(nn.Module):
    component_id = 'stft-frontend'

    def __init__(self, n_fft, hop_length, win_length=None, sample_rate=16000, center=True, pad_mode='reflect'):
        super().__init__()
        if n_fft <= 0 or hop_length <= 0 or sample_rate <= 0:
            raise ValueError('FFT, hop and sample rate must be positive')
        win_length = n_fft if win_length is None else win_length
        if not 0 < win_length <= n_fft:
            raise ValueError('win_length must be in (0,n_fft]')
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.center = center
        self.pad_mode = pad_mode
        self.register_buffer('window', torch.hann_window(win_length), persistent=True)

    def forward(self, waveform, sample_rate=None):
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError('sample rate mismatch')
        if waveform.ndim not in (1, 2):
            raise ValueError('waveform must be [T] or [B,T]')
        return torch.stft(waveform, self.n_fft, self.hop_length, self.win_length, self.window.to(waveform), center=self.center, pad_mode=self.pad_mode, return_complex=True)
