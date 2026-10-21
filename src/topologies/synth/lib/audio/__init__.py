"""Deterministic torch audio synthesis and processing.

    tone = sine(f0=440, seconds=1, sr=16_000)
    filtered = tone | lowpass(cutoff=1200, sr=16_000)

All factories return nodes. Transform factories can be partially bound and composed.
"""

from __future__ import annotations

import importlib
import math

import torch

from topologies import synth

_EXAMPLE_AUDIO = {"call": "synth.Audio", "args": [[0.0] * 32], "sr": 16_000}


def _audio(x: torch.Tensor, sr: int) -> synth.Audio:
    return synth.Audio(x.float(), sr=sr)


def _duration(seconds: float, sr: int) -> int:
    if not math.isfinite(seconds) or seconds < 0 or sr <= 0:
        raise ValueError("seconds must be non-negative and sample rate positive")
    return round(seconds * sr)


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"f0": 440, "seconds": 1, "sr": 16000})
def sine(f0: float, seconds: float = 1.0, sr: int = 16_000, phase: float = 0.0) -> synth.Audio:
    """Sine oscillator; example: ``sine(f0=440, seconds=.2)``."""
    n = _duration(seconds, sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    return _audio(torch.sin(2 * math.pi * float(f0) * t + phase), sr)


def _wave(f0: float, seconds: float, sr: int, kind: str, width: float = .5) -> synth.Audio:
    n = _duration(seconds, sr)
    if not 0 < width < 1:
        raise ValueError("pulse width must be between zero and one")
    phase = (torch.arange(n, dtype=torch.float32) * (float(f0) / sr)).remainder(1)
    if kind == "saw":
        x = 2 * phase - 1
    elif kind == "square":
        x = torch.where(phase < .5, 1., -1.)
    elif kind == "pulse":
        x = torch.where(phase < width, 1., -1.)
    else:
        x = 1 - 4 * (phase - .5).abs()
    return _audio(x, sr)


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"f0": 220, "seconds": .2})
def saw(f0: float, seconds: float = 1., sr: int = 16_000):
    """Sawtooth oscillator."""
    return _wave(f0, seconds, sr, "saw")


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"f0": 220, "seconds": .2})
def square(f0: float, seconds: float = 1., sr: int = 16_000):
    """Square oscillator."""
    return _wave(f0, seconds, sr, "square")


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"f0": 220, "seconds": .2})
def triangle(f0: float, seconds: float = 1., sr: int = 16_000):
    """Triangle oscillator."""
    return _wave(f0, seconds, sr, "triangle")


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"f0": 220, "width": .25})
def pulse(f0: float, seconds: float = 1., sr: int = 16_000, width: float = .25) -> synth.Audio:
    """Pulse oscillator with duty width in (0, 1)."""
    return _wave(f0, seconds, sr, "pulse", width)


def _batch_rows(value, batch_size):
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
        return [value[i] for i in range(batch_size)]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return list(value)
    return [value] * batch_size


def _control_batch(value, count, n):
    curves = torch.stack([torch.as_tensor(v).float().flatten() for v in value])
    return synth.dsp.upsample_curve(curves, n) if curves.shape[-1] != n else curves


def _harmonic_batch(f0, harmonics, amplitudes, seconds, sr, ctx):
    """Vectorized additive renderer, grouped by compatible clip and control shapes."""
    batch_size = len(ctx)
    rows = [_batch_rows(v, batch_size) for v in (f0, harmonics, amplitudes, seconds, sr)]
    records = []
    for i in range(batch_size):
        f0_i, harms_i, amps_i, sec_i, sr_i = (column[i] for column in rows)
        rate = int(sr_i); n = _duration(float(sec_i), rate)
        weighted = ((isinstance(harms_i, torch.Tensor) and harms_i.ndim > 0)
                    or isinstance(harms_i, (list, tuple)))
        amp_i = torch.as_tensor(harms_i).float() if weighted else amps_i
        count = amp_i.shape[0] if weighted and amp_i.ndim > 1 else (
            amp_i.numel() if weighted else int(harms_i))
        ctrl = torch.as_tensor(f0_i).flatten()
        amps = torch.ones(count, 1) if amp_i is None else torch.as_tensor(amp_i).float()
        if amps.ndim == 1: amps = amps[:, None]
        records.append((i, rate, n, count, ctrl.numel(), amps.shape[-1], ctrl, amps))
    groups = {}
    for row in records:
        groups.setdefault(row[1:6], []).append(row)
    output = [None] * batch_size
    for (rate, n, count, _, _), group in groups.items():
        controls = torch.stack([row[6] for row in group])
        frequencies = _control_batch(controls, len(group), n)
        amp_curves = []
        for row in group:
            amp = row[7]
            amp_curves.append(synth.dsp.upsample_curve(amp, n) if amp.shape[-1] != n else amp)
        amps = torch.stack(amp_curves)
        phase = 2 * math.pi * torch.cumsum(frequencies, -1) / rate
        norm = torch.zeros((len(group), n))
        signal = torch.zeros_like(norm)
        for h in range(count):
            valid = (h + 1) * frequencies < rate / 2
            weight = amps[:, h]
            signal = signal + torch.sin(phase * (h + 1)) * weight * valid
            norm = norm + weight.abs()
        signal = signal / norm.clamp_min(1)
        for j, row in enumerate(group):
            output[row[0]] = _audio(signal[j], rate)
    return output


@synth.node(type=synth.Audio, tags=["audio", "ddsp"], example={"f0": 220, "seconds": .2},
            batch=_harmonic_batch, check_batch=True)
def harmonic(f0: float | torch.Tensor, harmonics: int | torch.Tensor = 16,
             amplitudes: torch.Tensor | None = None, seconds: float = 1., sr: int = 16_000,
             ctx=None) -> synth.Audio:
    """Additive harmonics; harmonics can be a count or per-partial amplitude weights."""
    n = _duration(seconds, sr)
    if isinstance(harmonics, (torch.Tensor, list, tuple)):
        amplitudes = torch.as_tensor(harmonics) if amplitudes is None else amplitudes
        count = amplitudes.shape[0] if amplitudes.ndim > 1 else amplitudes.numel()
    else:
        count = harmonics
    if count < 1:
        raise ValueError("harmonics must be positive")
    f = torch.as_tensor(f0, dtype=torch.float32).flatten()
    f = synth.dsp.upsample_curve(f, n) if n else f[:0]
    amps = torch.ones(count, n) if amplitudes is None else torch.as_tensor(amplitudes).float()
    if amps.ndim == 1:
        amps = amps[:, None]
    if amps.shape[-1] != n:
        amps = synth.dsp.upsample_curve(amps, n)
    phase = 2 * math.pi * torch.cumsum(f, 0) / sr
    y = torch.zeros(n); norm = torch.zeros(n)
    for h in range(count):
        weight = amps[h]
        valid = (h + 1) * f < sr / 2
        y = y + torch.sin(phase * (h + 1)) * weight * valid
        norm = norm + weight.abs()
    if ctx is not None:
        ctx.latent(f0=torch.as_tensor(f0).detach().clone())
    return _audio(y / norm.clamp_min(1), sr)


def _ddsp_batch(r, f0, loudness, harmonics, noise, seconds, sr, ctx):
    batch_size = len(ctx)
    columns = [_batch_rows(v, batch_size) for v in
               (r, f0, loudness, harmonics, noise, seconds, sr)]
    noise_rows = columns[4]
    rates = [int(v) for v in columns[6]]
    lengths = [_duration(float(s), rate) for s, rate in zip(columns[5], rates)]
    groups = {}
    for i, key in enumerate(zip(rates, lengths)):
        groups.setdefault(key, []).append(i)
    output = [None] * batch_size
    for (rate, n), indices in groups.items():
        group_ctx = [ctx[i] for i in indices]
        f0s = [columns[1][i] for i in indices]
        harmonics_i = [columns[3][i] for i in indices]
        harmonic_audio = _harmonic_batch(f0s, harmonics_i, [None] * len(indices),
                                         [n / rate] * len(indices), [rate] * len(indices),
                                         group_ctx)
        level_rows = [torch.as_tensor(columns[2][i]).float().flatten() for i in indices]
        noise_levels = [torch.as_tensor(noise_rows[i]).float().flatten() for i in indices]
        level = _control_batch(torch.stack(level_rows), len(indices), n)
        noise_level = _control_batch(torch.stack(noise_levels), len(indices), n)
        white = torch.stack([torch.randn(n, generator=columns[0][i].child("noise").torch)
                             for i in indices])
        signals = torch.stack([_tensor(value) for value in harmonic_audio]) * level + white * noise_level
        for j, i in enumerate(indices):
            output[i] = _audio(signals[j], rate)
    return output


@synth.node(type=synth.Audio, tags=["audio", "ddsp"], example={"f0": 220, "harmonics": 8},
            batch=_ddsp_batch, check_batch=True)
def ddsp(r, f0: float | torch.Tensor, loudness: float | torch.Tensor = 1., harmonics: int | torch.Tensor = 16,
         noise: float | torch.Tensor = 0., seconds: float = 1., sr: int = 16_000, ctx=None) -> synth.Audio:
    """Harmonic-plus-noise renderer with control-rate f0, amplitude and noise curves."""
    n = _duration(seconds, sr)
    harmonic_audio = _harmonic_batch([f0], [harmonics], [None], [seconds], [sr], [ctx])[0]
    level = synth.dsp.upsample_curve(torch.as_tensor(loudness).float().flatten(), n)
    noise_level = synth.dsp.upsample_curve(torch.as_tensor(noise).float().flatten(), n)
    noise_signal = torch.randn(n, generator=r.child("noise").torch)
    x = _tensor(harmonic_audio) * level + noise_signal * noise_level
    if ctx is not None: ctx.latent(f0=torch.as_tensor(f0).detach().clone())
    return _audio(x, sr)


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"seconds": .2})
def noise(r, seconds: float = 1., sr: int = 16_000, color: str = "white") -> synth.Audio:
    """White, pink (one-pole approximation), or brown noise."""
    n = _duration(seconds, sr)
    if color not in ("white", "pink", "brown"):
        raise ValueError("color must be white, pink, or brown")
    x = torch.randn(n, generator=r.torch)
    if color == "pink":
        x = torch.nn.functional.conv1d(torch.nn.functional.pad(x[None, None], (1, 0)),
                                       torch.tensor([[[.7, .3]]]))[0, 0]
    if color == "brown":
        x = x.cumsum(0) / max(n, 1) ** .5
    return _audio(x / x.std(unbiased=False).clamp_min(1e-8), sr)


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"seconds": .1})
def impulse(seconds: float = .1, sr: int = 16_000) -> synth.Audio:
    """Unit impulse at time zero."""
    x = torch.zeros(_duration(seconds, sr));
    if x.numel(): x[0] = 1
    return _audio(x, sr)


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"seconds": .1})
def click(r, seconds: float = .1, sr: int = 16_000, amplitude: float = 1.) -> synth.Audio:
    """Short decaying noise transient."""
    n = _duration(seconds, sr); k = min(n, max(1, round(.005 * sr)))
    x = torch.zeros(n)
    if k: x[:k] = torch.randn(k, generator=r.torch) * torch.exp(-torch.arange(k) / max(k / 5, 1)) * amplitude
    return _audio(x, sr)


@synth.node(type=synth.Audio, tags=["audio", "source"],
            example={"f0": 220, "table": [0., 1., 0., -1.], "seconds": .2})
def wavetable(f0: float | torch.Tensor, table: torch.Tensor, seconds: float = 1.,
              sr: int = 16_000) -> synth.Audio:
    """Periodic oscillator using linear interpolation in the supplied single-cycle table."""
    n = _duration(seconds, sr); tab = torch.as_tensor(table).float().flatten()
    if tab.numel() < 2: raise ValueError("wavetable must contain at least two samples")
    freq = synth.dsp.upsample_curve(torch.as_tensor(f0).float().flatten(), n)
    phase = (torch.cumsum(freq, 0) / sr).remainder(1) * tab.numel()
    i = phase.floor().long(); frac = phase - i
    return _audio(tab[i] * (1-frac) + tab[(i+1) % tab.numel()] * frac, sr)


@synth.node(type=synth.Audio, tags=["audio", "source"], example={"f0": 220, "mod_ratio": 2})
def fm(f0: float | torch.Tensor, mod_index: float = 2., mod_ratio: float = 2.,
       seconds: float = 1., sr: int = 16_000) -> synth.Audio:
    """Two-operator frequency modulation oscillator."""
    n = _duration(seconds, sr); freq = synth.dsp.upsample_curve(torch.as_tensor(f0).float().flatten(), n)
    phase = 2 * math.pi * torch.cumsum(freq, 0) / sr
    modphase = phase * mod_ratio
    return _audio(torch.sin(phase + mod_index * torch.sin(modphase)), sr)


@synth.node(type=synth.Audio, tags=["audio", "ddsp"],
            example={"magnitudes": [1.] * 257, "seconds": .2, "sr": 16000})
def filtered_noise(r, magnitudes: torch.Tensor, seconds: float = 1., sr: int = 16_000,
                   frame_size: int = 512, hop: int = 256) -> synth.Audio:
    """Time-varying spectral shaping of seeded noise frames with overlap-add."""
    n = _duration(seconds, sr); mag = torch.as_tensor(magnitudes).float()
    if mag.ndim == 1: mag = mag[None, :]
    if mag.shape[1] != frame_size // 2 + 1 or hop < 1:
        raise ValueError("magnitudes bins must equal frame_size//2+1 and hop must be positive")
    frames = synth.dsp.upsample_curve(mag.T, max(1, 1 + (n-frame_size + hop-1)//hop)).T
    out = torch.zeros(max(n, frame_size) + frame_size); win = torch.hann_window(frame_size)
    for k, spectrum_mag in enumerate(frames):
        white = torch.randn(frame_size, generator=r.child("frame", k).torch)
        spec = torch.fft.rfft(white) * spectrum_mag
        chunk = torch.fft.irfft(spec, n=frame_size) * win
        start = k * hop; out[start:start+frame_size] += chunk
    return _audio(out[:n], sr)


@synth.node(type=synth.Audio, tags=["audio", "filter"],
            example={"x": _EXAMPLE_AUDIO, "cutoff": 1000, "sr": 16000})
def lowpass(x: torch.Tensor, cutoff: float, sr: int | None = None, q: float = .707) -> synth.Audio:
    """Biquad low-pass; cutoff must be below Nyquist."""
    return _filtered(x, cutoff, sr, "lowpass", q)


def _tensor(x):
    return torch.as_tensor(synth.Audio.unwrap(x) if isinstance(x, synth.Audio) else x).float()


def _filtered(x, cutoff, sr, kind, q=.707, gain_db=0.):
    input_rate = x.sr if isinstance(x, synth.Audio) and x.sr else 16_000
    rate = input_rate if sr is None else sr
    if rate <= 0: raise ValueError("sample rate must be positive")
    if q <= 0: raise ValueError("filter Q must be positive")
    if isinstance(x, synth.Audio) and x.sr is not None and x.sr != rate:
        raise ValueError("filter sample rate must match audio sample rate")
    data = _tensor(x)
    if isinstance(cutoff, (torch.Tensor, list, tuple)):
        if data.shape[-1] == 0: return _audio(data, sr)
        cutoffs = synth.dsp.upsample_curve(torch.as_tensor(cutoff).float().flatten(), data.shape[-1])
        flat = data.reshape(-1, data.shape[-1]).double()
        output = torch.zeros_like(flat)
        x1 = torch.zeros(flat.shape[0]); x2 = torch.zeros_like(x1)
        y1 = torch.zeros_like(x1); y2 = torch.zeros_like(x1)
        for i, value in enumerate(cutoffs):
            hz = float(value)
            if not 0 < hz < rate / 2: raise ValueError("cutoff curve must lie below Nyquist")
            b, a = synth.dsp.biquad_coefficients(kind, hz, rate, q, gain_db)
            current = flat[:, i]
            result = b[0]*current + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2
            output[:, i] = result
            x2, x1, y2, y1 = x1, current, y1, result
        return _audio(output.reshape(data.shape).to(data.dtype), rate)
    if not 0 < cutoff < rate / 2: raise ValueError("cutoff must lie below Nyquist")
    return _audio(synth.dsp.biquad(data, kind, cutoff, rate, q, gain_db), rate)


@synth.node(type=synth.Audio, tags=["audio", "filter"],
            example={"x": _EXAMPLE_AUDIO, "cutoff": 1000, "sr": 16000})
def highpass(x, cutoff, sr=None, q=.707):
    """Biquad high-pass filter."""
    return _filtered(x, cutoff, sr, "highpass", q)


@synth.node(type=synth.Audio, tags=["audio", "filter"],
            example={"x": _EXAMPLE_AUDIO, "cutoff": 1000, "sr": 16000})
def bandpass(x, cutoff, sr=None, q=.707):
    """Biquad band-pass filter."""
    return _filtered(x, cutoff, sr, "bandpass", q)


@synth.node(type=synth.Audio, tags=["audio", "filter"],
            example={"x": _EXAMPLE_AUDIO, "cutoff": 1000, "sr": 16000})
def notch(x, cutoff, sr=None, q=.707):
    """Biquad notch filter."""
    return _filtered(x, cutoff, sr, "notch", q)


@synth.node(type=synth.Audio, tags=["audio", "filter"],
            example={"x": _EXAMPLE_AUDIO, "cutoff": 1000, "sr": 16000})
def peak(x, cutoff, sr=None, q=.707, gain_db=6.):
    """Biquad peaking filter."""
    return _filtered(x, cutoff, sr, "peak", q, gain_db)


@synth.node(type=synth.Audio, tags=["audio", "filter"],
            example={"x": _EXAMPLE_AUDIO, "cutoff": 1000, "sr": 16000})
def shelf(x, cutoff, sr=None, q=.707, gain_db=6.):
    """Biquad low-shelf filter with configurable gain."""
    return _filtered(x, cutoff, sr, "lowshelf", q, gain_db)


@synth.node(type=synth.Audio, tags=["audio", "filter"],
            example={"x": _EXAMPLE_AUDIO, "cutoff": 1000, "sr": 16000})
def allpass(x, cutoff, sr=None, q=.707):
    """Biquad all-pass filter."""
    return _filtered(x, cutoff, sr, "allpass", q)


@synth.node(type=synth.Audio, tags=["audio", "filter"],
            example={"x": _EXAMPLE_AUDIO, "delay_seconds": .01, "sr": 16000})
def comb(x, delay_seconds: float = .01, feedback: float = .5, sr: int | None = None):
    """Feedback comb filter with fixed sample delay."""
    if not 0 <= feedback < 1: raise ValueError("feedback must be in [0,1)")
    rate = sr or (x.sr if isinstance(x, synth.Audio) and x.sr else 16_000)
    if delay_seconds <= 0 or rate <= 0: raise ValueError("delay and sample rate must be positive")
    if isinstance(x, synth.Audio) and x.sr is not None and x.sr != rate:
        raise ValueError("filter sample rate must match audio sample rate")
    data = _tensor(x); delay_samples = max(1, round(delay_seconds*rate))
    y = torch.zeros_like(data)
    for i in range(data.shape[-1]):
        y[..., i] = data[..., i] + (feedback*y[..., i-delay_samples] if i >= delay_samples else 0.)
    return _audio(y, rate)


@synth.node(type=synth.Audio, tags=["audio", "effect"], example={"x": _EXAMPLE_AUDIO, "seconds": .2})
def delay(x, seconds: float = .25, feedback: float = .35, repeats: int = 4, wet: float = .5):
    """Feedback delay with a finite number of repeats and extended output tail."""
    if seconds <= 0 or repeats < 0 or not 0 <= feedback < 1 or not 0 <= wet <= 1:
        raise ValueError("delay time must be positive, repeats non-negative and gains in range")
    data = _tensor(x); rate = x.sr if isinstance(x, synth.Audio) else 16_000
    d = max(1, round(seconds*rate)); out = torch.nn.functional.pad(data, (0, d*repeats))
    for k in range(1, repeats+1):
        start = d*k
        out[..., start:start+data.shape[-1]] += data[..., :max(0, out.shape[-1]-start)] * (wet*feedback**(k-1))
    return _audio(out, rate)


@synth.node(type=synth.Audio, tags=["audio", "effect"], example={"x": _EXAMPLE_AUDIO, "depth": .003})
def chorus(x, depth: float = .003, rate: float = 0.8, mix: float = .5):
    """Modulated delay chorus with linear fractional-sample interpolation."""
    if depth < 0 or rate <= 0 or not 0 <= mix <= 1:
        raise ValueError("depth non-negative, rate positive, and mix in [0,1] required")
    data = _tensor(x); sample_rate = x.sr if isinstance(x, synth.Audio) else 16_000
    n = data.shape[-1]; t = torch.arange(n)/sample_rate
    delay_samps = (depth*sample_rate*(1+torch.sin(2*math.pi*rate*t))).clamp_min(0)
    position = torch.arange(n)-delay_samps
    lo = position.floor().long().clamp(0, max(0,n-1)); hi = (lo+1).clamp(max=max(0,n-1))
    frac = position-lo
    wet = data[..., lo]*(1-frac) + data[..., hi]*frac if n else data
    return _audio(data*(1-mix)+wet*mix, sample_rate)


@synth.node(type=synth.Audio, tags=["audio", "effect"], example={"x": _EXAMPLE_AUDIO, "threshold": .5})
def compressor(x, threshold: float = .5, ratio: float = 4., attack: float = .01,
               release: float = .1):
    """Feed-forward peak compressor with smoothed gain reduction."""
    if threshold <= 0 or ratio < 1 or attack <= 0 or release <= 0:
        raise ValueError("threshold/attack/release must be positive and ratio at least one")
    data = _tensor(x); rate = x.sr if isinstance(x, synth.Audio) else 16_000
    env = torch.zeros_like(data); a = math.exp(-1/(attack*rate)); r = math.exp(-1/(release*rate))
    for i in range(data.shape[-1]):
        coeff = a if bool((data[..., i].abs() > env[..., i-1].abs()).all()) and i else r
        env[..., i] = coeff*(env[..., i-1] if i else 0.) + (1-coeff)*data[..., i].abs()
    gain = torch.where(env > threshold, (threshold/env.clamp_min(1e-12)) ** (1-1/ratio), 1.)
    return _audio(data*gain, rate)


@synth.node(type=synth.Audio, tags=["audio", "effect"], example={"x": _EXAMPLE_AUDIO, "bands": [{"cutoff": 1000, "gain_db": 3}], "sr": 16000})
def eq(x, bands, sr: int | None = None):
    """Apply ordered peaking biquad equalizer bands."""
    result = x
    for band in bands:
        result = _filtered(result, band["cutoff"], sr, "peak", band.get("q", .707),
                           band.get("gain_db", 0.))
    return result


@synth.node(type=synth.Audio, tags=["audio", "effect"], example={"x": _EXAMPLE_AUDIO, "position": -.25})
def pan(x, position: float = 0.):
    """Constant-power stereo panning from mono or balance panning for stereo input."""
    if not -1 <= position <= 1: raise ValueError("pan position must be in [-1,1]")
    data = _tensor(x); rate = x.sr if isinstance(x, synth.Audio) else 16_000
    left, right = math.cos((position+1)*math.pi/4), math.sin((position+1)*math.pi/4)
    if data.ndim == 1: out = torch.stack((data*left, data*right))
    elif data.ndim == 2 and data.shape[0] == 2:
        out = torch.stack((data[0]*left*math.sqrt(2), data[1]*right*math.sqrt(2)))
    else: raise ValueError("pan expects mono or stereo audio")
    return _audio(out, rate)


@synth.node(type=synth.Audio, tags=["audio", "effect"], example={"x": _EXAMPLE_AUDIO, "sr_out": 8000})
def resample(x, sr_out: int):
    """FFT resampling; updates the wrapped output sample rate."""
    if sr_out <= 0: raise ValueError("output sample rate must be positive")
    rate = x.sr if isinstance(x, synth.Audio) else 16_000
    if rate <= 0: raise ValueError("input sample rate must be positive")
    return _audio(synth.dsp.resample(_tensor(x), rate, sr_out), sr_out)


@synth.node(type=synth.Audio, tags=["audio", "envelope"], example={"x": _EXAMPLE_AUDIO, "threshold": .01})
def gate(x, threshold: float = .01, floor: float = 0.):
    """Noise gate that attenuates samples below a non-negative threshold."""
    if threshold < 0 or not 0 <= floor <= 1: raise ValueError("threshold and floor must be non-negative/in range")
    rate = x.sr if isinstance(x, synth.Audio) else 16_000
    data = _tensor(x); mask = data.abs() >= threshold
    return _audio(data*torch.where(mask, 1., floor), rate)


@synth.node(type=synth.Audio, tags=["audio", "effect"], example={"x": _EXAMPLE_AUDIO, "amount": .2})
def gain(x: torch.Tensor, amount: float = 1.) -> synth.Audio:
    """Linear gain transform."""
    return _audio(_tensor(x) * amount, x.sr if isinstance(x, synth.Audio) else 16_000)


@synth.node(type=synth.Audio, tags=["audio", "effect"],
            example={"x": _EXAMPLE_AUDIO, "decay": .2, "sr": 16000})
def reverb(r, x: torch.Tensor, decay: float = .5, sr: int | None = None) -> synth.Audio:
    """FFT convolution with a deterministic exponentially decaying synthetic IR."""
    rate = sr or (x.sr if isinstance(x, synth.Audio) and x.sr else 16_000)
    if decay < 0 or rate <= 0: raise ValueError("decay and sample rate must be non-negative/positive")
    if isinstance(x, synth.Audio) and x.sr and sr is not None and x.sr != sr:
        raise ValueError("reverb sample rate must match audio sample rate")
    n = max(1, round(decay * rate)); t = torch.arange(n) / rate
    ir = torch.randn(n, generator=r.torch) * torch.exp(-6. * t / max(decay, 1e-6)); ir[0] += 1
    return _audio(synth.dsp.fft_convolve(_tensor(x), ir), rate)


def _partial(effect):
    options = {"x": _EXAMPLE_AUDIO}
    if effect.__name__ == "distortion": options["drive"] = 2.
    if effect.__name__ == "bitcrush": options["bits"] = 8
    return synth.node(type=synth.Audio, tags=["audio", "effect"], example=options)(effect)


@_partial
def distortion(x, drive: float = 2.):
    """Tanh waveshaping distortion."""
    return _audio(torch.tanh(_tensor(x) * drive), x.sr if isinstance(x, synth.Audio) else 16_000)


@_partial
def bitcrush(x, bits: int = 8):
    """Uniform quantization bit-depth reduction."""
    if bits < 1: raise ValueError("bits must be positive")
    t = _tensor(x); levels = 2 ** (bits - 1)
    return _audio((t * levels).round() / levels, x.sr if isinstance(x, synth.Audio) else 16_000)


def stft(x, n_fft=512, hop=None): return synth.dsp.stft(_tensor(x), n_fft, hop)
def mel(x, sr, n_fft=512, hop=None, n_mels=40): return synth.dsp.mel(_tensor(x), sr, n_fft, hop, n_mels)
def mfcc(x, sr, n_mfcc=13, **kwargs): return synth.dsp.mfcc(_tensor(x), sr, n_mfcc, **kwargs)
def f0_autocorr(x, sr, **kwargs): return synth.dsp.f0_autocorr(_tensor(x), sr, **kwargs)
def loudness(x, **kwargs): return synth.dsp.loudness(_tensor(x), **kwargs)


def _control_size(steps):
    if steps < 1: raise ValueError("steps must be positive")
    return steps


@synth.node(type=synth.Tensor, tags=["audio", "curve"], example={"start": 100, "end": 400})
def curve_linear(start: float, end: float, steps: int = 128):
    """Linear control-rate curve node."""
    return torch.linspace(start, end, _control_size(steps))


@synth.node(type=synth.Tensor, tags=["audio", "curve"], example={"points": [[0, 0], [1, 1]]})
def curve_from_points(points, steps: int = 128):
    """Interpolate ``(normalized_time, value)`` controls into a curve node."""
    n = _control_size(steps); p = torch.as_tensor(points).float()
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 2:
        raise ValueError("points must be an [N,2] array of time/value pairs")
    if bool((p[1:, 0] <= p[:-1, 0]).any()): raise ValueError("point times must increase")
    t = torch.linspace(float(p[0, 0]), float(p[-1, 0]), n)
    hi = torch.searchsorted(p[:, 0].contiguous(), t).clamp(1, p.shape[0]-1)
    lo = hi - 1
    ratio = (t-p[lo, 0]) / (p[hi, 0]-p[lo, 0]).clamp_min(1e-12)
    return p[lo, 1] + ratio * (p[hi, 1]-p[lo, 1])


@synth.node(type=synth.Tensor, tags=["audio", "curve"], example={"attack": .1})
def curve_adsr(attack: float = .1, decay: float = .1, sustain: float = .7,
               release: float = .1, steps: int = 128):
    """ADSR control curve over normalized time."""
    n = _control_size(steps)
    if min(attack, decay, release) < 0 or not 0 <= sustain <= 1:
        raise ValueError("ADSR durations must be non-negative and sustain in [0,1]")
    na, nd, nr = round(attack*n), round(decay*n), round(release*n)
    remain = max(0, n-na-nd-nr)
    seq = [torch.linspace(0, 1, max(1, na)), torch.linspace(1, sustain, max(1, nd)),
           torch.full((remain,), sustain), torch.linspace(sustain, 0, max(1, nr))]
    return torch.cat(seq)[:n]


def _adsr_curve(attack, decay, sustain, release, n):
    na, nd, nr = round(attack*n), round(decay*n), round(release*n)
    remain = max(0, n-na-nd-nr)
    return torch.cat((torch.linspace(0, 1, max(1, na)),
                      torch.linspace(1, sustain, max(1, nd)), torch.full((remain,), sustain),
                      torch.linspace(sustain, 0, max(1, nr))))[:n]


@synth.node(type=synth.Tensor, tags=["audio", "curve"], example={"base": 220, "depth": 4, "rate": 5})
def curve_vibrato(base: float, depth: float, rate: float, steps: int = 128):
    """Sinusoidal pitch modulation curve."""
    n = _control_size(steps); t = torch.arange(n) / n
    return base + depth * torch.sin(2 * math.pi * rate * t)


@synth.node(type=synth.Tensor, tags=["audio", "curve"], example={"start": 100, "end": 440})
def curve_glide(start: float, end: float, steps: int = 128):
    """Linear frequency glide curve."""
    return torch.linspace(start, end, _control_size(steps))


class _Curves:
    """Control-rate curve constructors."""
    linear = staticmethod(curve_linear)
    adsr = staticmethod(curve_adsr)
    vibrato = staticmethod(curve_vibrato)
    glide = staticmethod(curve_glide)
    from_points = staticmethod(curve_from_points)


curve = _Curves()


@synth.node(type=synth.Audio, tags=["audio", "envelope"], example={"x": _EXAMPLE_AUDIO, "attack": .01})
def adsr(x, attack=.01, decay=.1, sustain=.7, release=.1, sr=16000):
    """Apply an ADSR envelope to audio."""
    t = _tensor(x); env = _adsr_curve(attack, decay, sustain, release, max(1, t.shape[-1]))
    return _audio(t * env[:t.shape[-1]], x.sr if isinstance(x, synth.Audio) else sr)


@synth.node(type=synth.Audio, tags=["audio", "envelope"], example={"x": _EXAMPLE_AUDIO, "decay": .2})
def exp_decay(x, decay=.2, sr=None):
    """Apply exponential decay envelope."""
    t = _tensor(x)
    rate = sr or (x.sr if isinstance(x, synth.Audio) and x.sr else 16_000)
    if decay <= 0 or rate <= 0: raise ValueError("decay and sample rate must be positive")
    return _audio(t * torch.exp(-torch.arange(t.shape[-1]) / (decay*rate)), rate)


@synth.node(type=synth.Audio, tags=["audio", "composition"], example={"x": _EXAMPLE_AUDIO, "length": 1000})
def pad(x, length):
    """Pad audio to a sample count."""
    if length < 0: raise ValueError("length must be non-negative")
    t = _tensor(x); return _audio(torch.nn.functional.pad(t, (0, max(0, length-t.shape[-1]))), x.sr)


@synth.node(type=synth.Audio, tags=["audio", "composition"], example={"x": _EXAMPLE_AUDIO, "start": 0, "length": 1000})
def crop(x, start, length):
    """Crop samples from audio."""
    if start < 0 or length < 0: raise ValueError("start and length must be non-negative")
    return _audio(_tensor(x)[..., start:start+length], x.sr)


@synth.node(type=synth.Audio, tags=["audio", "composition"], example={"items": [_EXAMPLE_AUDIO]})
def concat(items):
    """Concatenate same-rate audio values."""
    if not items: raise ValueError("items cannot be empty")
    sr = items[0].sr
    if any(item.sr != sr or _tensor(item).shape[:-1] != _tensor(items[0]).shape[:-1]
           for item in items): raise ValueError("sample rates and channel shapes must match")
    return _audio(torch.cat([_tensor(item) for item in items]), sr)


@synth.node(type=synth.Audio, tags=["audio", "composition"], example={"items": [[_EXAMPLE_AUDIO, 1.]]})
def mix(items):
    """Mix ``(audio, gain)`` pairs, padding shorter sources with silence."""
    if not items: raise ValueError("items cannot be empty")
    sr = items[0][0].sr; shape = _tensor(items[0][0]).shape[:-1]
    n = max(_tensor(x).shape[-1] for x, _ in items)
    out = torch.zeros((*shape, n))
    for x, g in items:
        if x.sr != sr or _tensor(x).shape[:-1] != shape: raise ValueError("sample rates and channel shapes must match")
        v = _tensor(x); out[..., :v.shape[-1]] += v * g
    return _audio(out, sr)


@synth.node(type=synth.Audio, tags=["audio", "composition"], example={"x": _EXAMPLE_AUDIO, "length": 1000})
def random_crop(r, x, length: int):
    """Seeded crop of requested sample length; short signals are zero-padded."""
    data = _tensor(x)
    if length < 0: raise ValueError("length must be non-negative")
    if data.shape[-1] <= length:
        return _audio(torch.nn.functional.pad(data, (0, length-data.shape[-1])), x.sr)
    start = r.randint(0, data.shape[-1]-length)
    return _audio(data[..., start:start+length], x.sr)


def _resolve_instrument(value):
    if value in ("sine", "saw", "square", "triangle"):
        return globals()[value]
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("instrument must be a preset name or importable module:factory path")
    module_name, qualname = value.split(":", 1)
    target = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    if not isinstance(target, synth.NodeFactory):
        raise TypeError(f"Instrument path {value!r} does not resolve to a public node factory")
    return target


def sequence(events, instrument="sine", instrument_kwargs=None, seconds=1., sr=16_000, ctx=None):
    """Render event dictionaries with exact labels using per-event child nodes."""
    n = _duration(seconds, sr)
    if isinstance(events, dict):
        if "values" not in events:
            raise ValueError("event mapping must contain a 'values' list")
        events = events["values"]
    if not isinstance(events, (list, tuple)):
        raise TypeError("events must be a list or a process result with a values list")
    factory = _resolve_instrument(instrument)
    shared_kwargs = dict(instrument_kwargs or {})
    rendered = []; onsets = []; note_numbers = []; f0_values = []; params_latent = []
    durations = []; velocities = []; instruments = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise TypeError(f"event {index} must be a mapping with onset and duration")
        if "onset" not in event or "duration" not in event:
            raise ValueError(f"event {index} requires onset and duration")
        onset = float(event["onset"]); duration = float(event["duration"])
        if not math.isfinite(onset) or onset < 0:
            raise ValueError(f"event {index} onset must be finite and non-negative")
        count = _duration(duration, sr); start = round(onset * sr)
        note = event.get("note")
        frequency = float(event["f0"]) if event.get("f0") is not None else (
            440. * 2 ** ((int(note)-69)/12) if note is not None else None)
        if frequency is None or not math.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"event {index} requires a positive f0 or MIDI note")
        kwargs = {**shared_kwargs, **dict(event.get("instrument_kwargs") or {})}
        velocity = float(event.get("velocity", 1.))
        if not math.isfinite(velocity) or velocity < 0:
            raise ValueError(f"event {index} velocity must be finite and non-negative")
        child = factory(f0=frequency, seconds=duration, sr=sr, **kwargs)
        if ctx is None:
            raise ValueError("sequence rendering requires a node sampling context")
        child = child.named(f"{ctx.path}.events[{index}].instrument")
        value = ctx.sample(child)
        if not isinstance(value, synth.Audio):
            raise TypeError(f"instrument for event {index} must return synth.Audio")
        if value.sr != sr:
            raise ValueError(f"instrument for event {index} returned {value.sr} Hz; expected {sr} Hz")
        signal = _tensor(value)
        if signal.ndim not in (1, 2):
            raise ValueError(f"instrument for event {index} returned invalid audio shape {signal.shape}")
        signal = signal[..., :count] * velocity
        stop = min(n, start + signal.shape[-1])
        rendered.append((start, signal[..., :max(0, stop-start)]))
        onsets.append(start)
        note_numbers.append(None if note is None else int(note))
        f0_values.append(frequency)
        params_latent.append(dict(kwargs))
        durations.append(duration); velocities.append(velocity); instruments.append(instrument)
    channels = {1 if signal.ndim == 1 else signal.shape[0] for _, signal in rendered}
    if len(channels) > 1:
        raise ValueError("all event instruments must return the same channel count")
    channel_count = next(iter(channels), 1)
    out = torch.zeros(n) if channel_count == 1 else torch.zeros(channel_count, n)
    for start, signal in rendered:
        if signal.shape[-1]:
            target = out[start:start+signal.shape[-1]] if out.ndim == 1 else \
                out[..., start:start+signal.shape[-1]]
            target += signal
    if ctx is not None:
        ctx.latent(onsets=onsets, note_numbers=note_numbers, f0=f0_values,
                   durations=durations, velocities=velocities, instruments=instruments,
                   instrument_parameters=params_latent)
    return _audio(out, sr)


class _SequenceFactory(synth.NodeFactory):
    """Factory wrapper that records importable instruments by reference path."""

    def __call__(self, *args, **kwargs):
        args = list(args)
        position = 1
        factory = kwargs.get("instrument") if "instrument" in kwargs else (
            args[position] if len(args) > position else None)
        if isinstance(factory, synth.NodeFactory):
            target = factory.fn
            if "<locals>" in target.__qualname__ or "<lambda>" in target.__qualname__:
                raise ValueError("custom instruments for saved sequences must be module-level factories")
            ref = f"{target.__module__}:{target.__qualname__}"
            if "instrument" in kwargs:
                kwargs["instrument"] = ref
            elif len(args) > position:
                args[position] = ref
            else:
                kwargs["instrument"] = ref
        return super().__call__(*args, **kwargs)


sequence = _SequenceFactory(
    sequence, type=synth.Audio, tags=["audio", "composition", "sequence"],
    example={"events": [{"onset": 0., "note": 60, "duration": .2}], "sr": 16_000})


__all__ = [
    "adsr", "allpass", "bandpass", "bitcrush", "chorus", "click", "comb", "compressor",
    "concat", "crop", "curve", "curve_adsr", "curve_from_points", "curve_glide", "curve_linear",
    "curve_vibrato", "ddsp", "delay", "distortion", "eq", "exp_decay", "f0_autocorr",
    "filtered_noise", "fm", "gain", "gate", "harmonic", "highpass", "impulse", "loudness",
    "lowpass", "mel", "mfcc", "mix", "noise", "notch", "pad", "pan", "peak", "pulse",
    "random_crop", "resample", "reverb", "saw", "sequence", "shelf", "sine", "square", "stft",
    "triangle", "wavetable",
]
