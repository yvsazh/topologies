"""Signal processing in pure torch (CPU, deterministic): spectra, filters, features.

    S = synth.dsp.stft(x, n_fft=512, hop=128)             # complex [..., F, frames]
    M = synth.dsp.mel(x, sr=16_000, n_mels=40)             # power mel spectrogram
    C = synth.dsp.mfcc(x, sr=16_000, n_mfcc=13)
    f0 = synth.dsp.f0_autocorr(x, sr=16_000)               # Hz per frame (0 = unvoiced)
    y = synth.dsp.biquad(x, "lowpass", cutoff=800, q=0.707, sr=16_000)
    y = synth.dsp.fft_convolve(x, ir)                      # linear convolution
    feats = synth.dsp.audio_features(x, sr=16_000)         # fixed-size summary vector

Functions accept ``[T]`` or batched ``[..., T]`` signals and never reduce across
the leading (record) dimensions, so batched and per-record results agree.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# ---------------------------------------------------------------- spectra


def frame(x: torch.Tensor, size: int, hop: int) -> torch.Tensor:
    """``[..., T]`` -> ``[..., frames, size]`` (zero-padded to whole frames)."""
    t = x.shape[-1]
    n = max(1, 1 + math.ceil(max(0, t - size) / hop))
    need = (n - 1) * hop + size
    if need > t:
        x = torch.nn.functional.pad(x, (0, need - t))
    return x.unfold(-1, size, hop)


def stft(x: torch.Tensor, n_fft: int = 512, hop: int | None = None,
         window: str = "hann") -> torch.Tensor:
    """Complex STFT ``[..., n_fft // 2 + 1, frames]`` (no centering, Hann window)."""
    hop = hop or n_fft // 4
    w = torch.hann_window(n_fft, periodic=True, dtype=x.dtype) if window == "hann" else \
        torch.ones(n_fft, dtype=x.dtype)
    frames = frame(x, n_fft, hop) * w
    return torch.fft.rfft(frames, dim=-1).transpose(-1, -2)


def power_spectrogram(x: torch.Tensor, n_fft: int = 512, hop: int | None = None
                      ) -> torch.Tensor:
    s = stft(x, n_fft, hop)
    return s.real ** 2 + s.imag ** 2


def hz_to_mel(f: Any) -> Any:
    return 2595.0 * torch.log10(1.0 + torch.as_tensor(f) / 700.0)


def mel_to_hz(m: Any) -> Any:
    return 700.0 * (10 ** (torch.as_tensor(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 0.0,
                   fmax: float | None = None) -> torch.Tensor:
    """Triangular mel filters ``[n_mels, n_fft // 2 + 1]``."""
    fmax = fmax or sr / 2
    mels = torch.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(fmax)), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(float(c - lo), 1e-9)
        down = (hi - bins) / max(float(hi - c), 1e-9)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0)
    return fb


def mel(x: torch.Tensor, sr: int, n_fft: int = 512, hop: int | None = None,
        n_mels: int = 40) -> torch.Tensor:
    """Power mel spectrogram ``[..., n_mels, frames]``."""
    p = power_spectrogram(x.float(), n_fft, hop)
    return mel_filterbank(sr, n_fft, n_mels).to(p.dtype) @ p


def log_mel(x: torch.Tensor, sr: int, **kw: Any) -> torch.Tensor:
    return torch.log(mel(x, sr, **kw) + 1e-8)


def dct_matrix(n_out: int, n_in: int) -> torch.Tensor:
    k = torch.arange(n_out)[:, None]
    n = torch.arange(n_in)[None, :]
    m = torch.cos(math.pi / n_in * (n + 0.5) * k) * math.sqrt(2.0 / n_in)
    m[0] /= math.sqrt(2.0)
    return m


def mfcc(x: torch.Tensor, sr: int, n_mfcc: int = 13, n_mels: int = 40, **kw: Any
         ) -> torch.Tensor:
    """MFCCs ``[..., n_mfcc, frames]`` (DCT-II of the log mel spectrogram)."""
    lm = log_mel(x, sr, n_mels=n_mels, **kw)
    return dct_matrix(n_mfcc, n_mels) @ lm


# ---------------------------------------------------------------- pitch / loudness


def f0_autocorr(x: torch.Tensor, sr: int, *, fmin: float = 50.0, fmax: float = 2000.0,
                frame_size: int = 1024, hop: int = 256, threshold: float = 0.3
                ) -> torch.Tensor:
    """Fundamental frequency per frame (Hz) by normalized autocorrelation with parabolic
    peak interpolation; ``0`` where the frame is unvoiced."""
    fr = frame(x.float(), frame_size, hop)
    fr = fr - fr.mean(-1, keepdim=True)
    n = 2 * frame_size
    spec = torch.fft.rfft(fr, n=n, dim=-1)
    ac = torch.fft.irfft(spec.real ** 2 + spec.imag ** 2, n=n, dim=-1)[..., :frame_size]
    ac = ac / ac[..., :1].clamp_min(1e-12)
    lo = max(2, int(sr / fmax) - 1)
    hi = min(frame_size - 2, int(sr / fmin))
    # Search only after the autocorrelation first dips below zero: the zero-lag lobe is
    # not a period.
    lags = torch.arange(frame_size)
    neg = ac < 0
    first = torch.where(neg.any(-1), neg.float().argmax(-1), torch.full(neg.shape[:-1], lo))
    start = torch.clamp(first, min=lo)
    masked = torch.where((lags >= start[..., None]) & (lags < hi), ac,
                         torch.full_like(ac, -2.0))
    # The first local maximum within 85 % of the best one (avoids octave errors when a
    # later multiple of the period happens to sit closer to an integer lag).
    best = masked.max(-1, keepdim=True).values
    left = torch.nn.functional.pad(masked[..., :-1], (1, 0), value=-2.0)
    right = torch.nn.functional.pad(masked[..., 1:], (0, 1), value=-2.0)
    good = (masked >= left) & (masked >= right) & (masked >= 0.85 * best) & (masked > -1.5)
    peak = torch.where(good.any(-1), good.float().argmax(-1), masked.argmax(-1))
    peak = peak.clamp(1, frame_size - 2)
    val = torch.gather(ac, -1, peak[..., None])[..., 0]
    a = torch.gather(ac, -1, (peak - 1)[..., None])[..., 0]
    b = torch.gather(ac, -1, (peak + 1)[..., None])[..., 0]
    denom = a - 2 * val + b
    shift = torch.where(denom.abs() > 1e-12, 0.5 * (a - b) / denom, torch.zeros_like(denom))
    lag = peak.float() + shift.clamp(-1, 1)
    f0 = sr / lag
    return torch.where(val > threshold, f0, torch.zeros_like(f0))


def rms(x: torch.Tensor, frame_size: int = 1024, hop: int = 256) -> torch.Tensor:
    return frame(x.float(), frame_size, hop).pow(2).mean(-1).sqrt()


def loudness(x: torch.Tensor, frame_size: int = 1024, hop: int = 256) -> torch.Tensor:
    """Frame loudness in dB (``20 log10 rms``)."""
    return 20 * torch.log10(rms(x, frame_size, hop) + 1e-8)


def zero_crossing_rate(x: torch.Tensor, frame_size: int = 1024, hop: int = 256
                       ) -> torch.Tensor:
    fr = frame(x.float(), frame_size, hop)
    return (torch.sign(fr[..., 1:]) != torch.sign(fr[..., :-1])).float().mean(-1)


def spectral_shape(x: torch.Tensor, sr: int, n_fft: int = 512) -> dict[str, torch.Tensor]:
    """Per-frame spectral centroid (Hz), rolloff (85 %, Hz) and flatness (0..1)."""
    p = power_spectrogram(x.float(), n_fft)
    freqs = torch.linspace(0, sr / 2, p.shape[-2])[:, None]
    total = p.sum(-2).clamp_min(1e-12)
    centroid = (p * freqs).sum(-2) / total
    cum = p.cumsum(-2) / total[..., None, :]
    rolloff = (cum < 0.85).sum(-2).float() * (sr / 2) / (p.shape[-2] - 1)
    flat = torch.exp(torch.log(p + 1e-12).mean(-2)) / (p.mean(-2) + 1e-12)
    return {"centroid": centroid, "rolloff": rolloff, "flatness": flat}


def audio_features(x: torch.Tensor, sr: int, n_mels: int = 32) -> torch.Tensor:
    """A fixed-size vector per signal ``[..., D]``: log-mel mean/std, centroid, rolloff,
    flatness, ZCR and RMS mean/std, median voiced f0 (log), voiced share."""
    x = x.float()
    n_fft = 512 if x.shape[-1] >= 512 else max(16, 1 << (x.shape[-1].bit_length() - 1))
    lm = log_mel(x, sr, n_fft=n_fft, hop=n_fft // 2, n_mels=n_mels)
    sh = spectral_shape(x, sr, n_fft)
    z = zero_crossing_rate(x, n_fft, n_fft // 2)
    r = rms(x, n_fft, n_fft // 2)
    f0 = f0_autocorr(x, sr, frame_size=max(64, n_fft * 2), hop=n_fft // 2)
    voiced = (f0 > 0).float()
    f0_med = torch.where(voiced.sum(-1) > 0,
                         torch.log(torch.where(f0 > 0, f0, torch.full_like(f0, float("nan")))
                                   .nanmedian(-1).values.clamp_min(1.0)),
                         torch.zeros(f0.shape[:-1]))
    parts = [lm.mean(-1), lm.std(-1, unbiased=False),
             torch.stack([sh["centroid"].mean(-1) / sr, sh["centroid"].std(-1, unbiased=False) / sr,
                          sh["rolloff"].mean(-1) / sr, sh["flatness"].mean(-1),
                          z.mean(-1), z.std(-1, unbiased=False), r.mean(-1),
                          r.std(-1, unbiased=False), f0_med, voiced.mean(-1)], -1)]
    return torch.nan_to_num(torch.cat(parts, -1))


# ---------------------------------------------------------------- filters / convolution


def biquad_coefficients(kind: str, cutoff: float, sr: int, q: float = 0.7071,
                        gain_db: float = 0.0) -> tuple[list[float], list[float]]:
    """RBJ cookbook biquad ``(b, a)`` for lowpass / highpass / bandpass / notch / peak /
    lowshelf / highshelf / allpass."""
    w0 = 2 * math.pi * min(max(cutoff, 1e-3), sr / 2 * 0.999) / sr
    cos, sin = math.cos(w0), math.sin(w0)
    alpha = sin / (2 * q)
    amp = 10 ** (gain_db / 40)
    if kind == "lowpass":
        b = [(1 - cos) / 2, 1 - cos, (1 - cos) / 2]
        a = [1 + alpha, -2 * cos, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]
        a = [1 + alpha, -2 * cos, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cos, 1 - alpha]
    elif kind == "notch":
        b = [1.0, -2 * cos, 1.0]
        a = [1 + alpha, -2 * cos, 1 - alpha]
    elif kind == "allpass":
        b = [1 - alpha, -2 * cos, 1 + alpha]
        a = [1 + alpha, -2 * cos, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * amp, -2 * cos, 1 - alpha * amp]
        a = [1 + alpha / amp, -2 * cos, 1 - alpha / amp]
    elif kind in ("lowshelf", "highshelf"):
        sq = 2 * math.sqrt(amp) * alpha
        s = 1 if kind == "lowshelf" else -1
        b = [amp * ((amp + 1) - s * (amp - 1) * cos + sq),
             s * 2 * amp * ((amp - 1) - s * (amp + 1) * cos),
             amp * ((amp + 1) - s * (amp - 1) * cos - sq)]
        a = [(amp + 1) + s * (amp - 1) * cos + sq,
             -s * 2 * ((amp - 1) + s * (amp + 1) * cos),
             (amp + 1) + s * (amp - 1) * cos - sq]
    else:
        from topologies.synth.errors import unknown

        raise unknown("filter kind", kind, ["lowpass", "highpass", "bandpass", "notch",
                                            "peak", "lowshelf", "highshelf", "allpass"],
                      stage="dsp")
    return [x / a[0] for x in b], [x / a[0] for x in a]


def frequency_response(b: list[float], a: list[float], freqs: torch.Tensor, sr: int
                       ) -> torch.Tensor:
    """``|H(f)|`` of a biquad at ``freqs`` Hz."""
    z = torch.exp(-1j * 2 * math.pi * freqs.double() / sr)
    num = b[0] + b[1] * z + b[2] * z * z
    den = a[0] + a[1] * z + a[2] * z * z
    return (num / den).abs().float()


def iir_filter(x: torch.Tensor, b: list[float], a: list[float]) -> torch.Tensor:
    """Direct-form biquad over the last dimension (exact recursion, any batch shape)."""
    y = x.double().clone()
    xs = x.double()
    b0, b1, b2 = b
    _, a1, a2 = a
    out = torch.zeros_like(y)
    x1 = torch.zeros(y.shape[:-1], dtype=y.dtype)
    x2 = torch.zeros_like(x1)
    y1 = torch.zeros_like(x1)
    y2 = torch.zeros_like(x1)
    for t in range(y.shape[-1]):
        xt = xs[..., t]
        yt = b0 * xt + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[..., t] = yt
        x2, x1 = x1, xt
        y2, y1 = y1, yt
    return out.to(x.dtype)


def fir_from_biquad(b: list[float], a: list[float], n: int = 512) -> torch.Tensor:
    """Truncated impulse response of a (stable) biquad: lets filtering run as an FFT
    convolution (fast and batched)."""
    imp = torch.zeros(n, dtype=torch.float64)
    imp[0] = 1.0
    return iir_filter(imp, b, a)


def biquad(x: torch.Tensor, kind: str, cutoff: float, sr: int, q: float = 0.7071,
           gain_db: float = 0.0, *, taps: int = 1024) -> torch.Tensor:
    """Filter ``x`` with an RBJ biquad (as a ``taps``-long FIR via FFT convolution)."""
    b, a = biquad_coefficients(kind, cutoff, sr, q, gain_db)
    h = fir_from_biquad(b, a, taps).to(x.dtype)
    return fft_convolve(x, h)[..., : x.shape[-1]]


def fft_convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Linear convolution over the last dimension (``len(x) + len(h) - 1`` samples)."""
    n = x.shape[-1] + h.shape[-1] - 1
    size = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, size) * torch.fft.rfft(h, size), size)
    return y[..., :n]


def resample(x: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    """Band-limited resampling by FFT (exact length ``round(T * sr_out / sr_in)``)."""
    if sr_in == sr_out:
        return x
    t = x.shape[-1]
    n = max(1, round(t * sr_out / sr_in))
    spec = torch.fft.rfft(x, dim=-1)
    k = n // 2 + 1
    if k <= spec.shape[-1]:
        spec = spec[..., :k]
    else:
        spec = torch.nn.functional.pad(spec, (0, k - spec.shape[-1]))
    return torch.fft.irfft(spec, n=n, dim=-1) * (n / t)


def upsample_curve(c: torch.Tensor, n: int) -> torch.Tensor:
    """Control-rate curve ``[..., K]`` -> ``[..., n]`` by linear interpolation."""
    c = c.float()
    if c.shape[-1] == n:
        return c
    if c.shape[-1] == 1:
        return c.expand(*c.shape[:-1], n).clone()
    shape = c.shape
    flat = c.reshape(-1, 1, shape[-1])
    out = torch.nn.functional.interpolate(flat, size=n, mode="linear", align_corners=True)
    return out.reshape(*shape[:-1], n)


# ---------------------------------------------------------------- generic numeric features


def numeric_features(x: torch.Tensor, *, lags: int = 8, bands: int = 8) -> torch.Tensor:
    """Moments, quantiles, autocorrelation and band energies of each row of ``[N, T]``."""
    x = torch.nan_to_num(x.float())
    if x.dim() == 1:
        x = x[None]
    n, t = x.shape
    mean = x.mean(1)
    std = x.std(1, unbiased=False)
    q = torch.quantile(x, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]), dim=1).t() if t else \
        torch.zeros(n, 5)
    z = (x - mean[:, None]) / std.clamp_min(1e-8)[:, None]
    skew = z.pow(3).mean(1)
    kurt = z.pow(4).mean(1)
    acf = []
    for k in range(1, lags + 1):
        acf.append((z[:, :-k] * z[:, k:]).mean(1) if t > k else torch.zeros(n))
    spec = torch.fft.rfft(z, dim=1).abs().pow(2) if t > 1 else torch.zeros(n, 1)
    edges = torch.linspace(0, spec.shape[1], bands + 1).long()
    total = spec.sum(1).clamp_min(1e-12)
    band = [spec[:, edges[i]:max(edges[i] + 1, edges[i + 1])].sum(1) / total
            for i in range(bands)]
    return torch.nan_to_num(torch.stack([mean, std, skew, kurt, *q.t(), *acf, *band], 1))


__all__ = ["audio_features", "biquad", "biquad_coefficients", "dct_matrix", "f0_autocorr",
           "fft_convolve", "fir_from_biquad", "frame", "frequency_response", "hz_to_mel",
           "iir_filter", "log_mel", "loudness", "mel", "mel_filterbank", "mel_to_hz", "mfcc",
           "numeric_features", "power_spectrogram", "resample", "rms", "spectral_shape", "stft",
           "upsample_curve", "zero_crossing_rate"]
