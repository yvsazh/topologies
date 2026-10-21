"""Blob codecs (stdlib + torch only): PNG images, WAV audio, raw tensors, JSON.

    data = png_encode(img)            # img: float [C, H, W] in [0, 1] (C = 1, 3 or 4) or uint8
    img2 = png_decode(data)           # float [C, H, W], exact for 8-bit levels
    data = wav_encode(x, sr=16_000)   # float32 IEEE WAV (lossless); pcm16=True for 16-bit PCM
    x2, sr = wav_decode(data)
    data = tensor_encode(t); t2 = tensor_decode(data)      # header + little-endian bytes

Every codec is deterministic (same value -> same bytes), so blobs are content
addressed and datasets are byte-identical across machines.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

import torch

from topologies.synth.errors import SynthError

# ---------------------------------------------------------------- PNG

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_COLOR = {1: 0, 2: 4, 3: 2, 4: 6}  # channels -> PNG color type (gray, gray+alpha, RGB, RGBA)
_CHANNELS = {v: k for k, v in _COLOR.items()}


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(
        ">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """``[C, H, W]`` (or ``[H, W]``) float in ``[0, 1]`` / uint8 -> uint8 ``[C, H, W]``."""
    x = img.detach().cpu()
    if x.dim() == 2:
        x = x[None]
    if x.dim() != 3:
        raise SynthError(f"An image must be [C, H, W]; got shape {tuple(x.shape)}.",
                         stage="types")
    if x.dtype == torch.uint8:
        return x
    if x.dtype == torch.bool:
        return x.to(torch.uint8) * 255
    return (x.float().clamp(0, 1) * 255).round().to(torch.uint8)


def png_encode(img: torch.Tensor, *, level: int = 6) -> bytes:
    """8-bit PNG of an image tensor (filter 0, deterministic zlib stream)."""
    x = to_uint8(img)
    c, h, w = x.shape
    if c not in _COLOR:
        raise SynthError(f"PNG needs 1, 2, 3 or 4 channels; got {c}.", stage="types",
                         hint="Store other channel counts as synth.Tensor.")
    rows = x.permute(1, 2, 0).reshape(h, w * c)
    raw = torch.cat([torch.zeros(h, 1, dtype=torch.uint8), rows], dim=1).contiguous()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, _COLOR[c], 0, 0, 0)
    return (_PNG_SIG + _chunk(b"IHDR", ihdr) +
            _chunk(b"IDAT", zlib.compress(raw.numpy().tobytes(), level)) + _chunk(b"IEND", b""))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def png_decode(data: bytes) -> torch.Tensor:
    """Float ``[C, H, W]`` in ``[0, 1]`` from 8-bit PNG bytes (any filter, no interlace)."""
    if data[:8] != _PNG_SIG:
        raise SynthError("Not a PNG file.", stage="types")
    pos, idat, ihdr = 8, [], None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + n
    if ihdr is None:
        raise SynthError("PNG without IHDR.", stage="types")
    w, h, depth, color, _, _, interlace = ihdr
    if depth != 8 or interlace or color not in _CHANNELS:
        raise SynthError("Only 8-bit, non-interlaced gray/RGB(A) PNGs are supported.",
                         stage="types", hint="Re-save the image as 8-bit PNG.")
    c = _CHANNELS[color]
    raw = zlib.decompress(b"".join(idat))
    stride = w * c
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(h, stride + 1)
    filters = t[:, 0]
    pix = t[:, 1:].clone()
    if bool((filters != 0).any()):
        out = bytearray(h * stride)
        src = bytes(raw)
        for y in range(h):
            f = src[y * (stride + 1)]
            line = bytearray(src[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
            prev = out[(y - 1) * stride:y * stride] if y else bytearray(stride)
            for i in range(stride):
                a = line[i - c] if i >= c else 0
                b = prev[i]
                cc = prev[i - c] if i >= c else 0
                if f == 1:
                    line[i] = (line[i] + a) & 255
                elif f == 2:
                    line[i] = (line[i] + b) & 255
                elif f == 3:
                    line[i] = (line[i] + (a + b) // 2) & 255
                elif f == 4:
                    line[i] = (line[i] + _paeth(a, b, cc)) & 255
            out[y * stride:(y + 1) * stride] = line
        pix = torch.frombuffer(out, dtype=torch.uint8).view(h, stride).clone()
    return pix.view(h, w, c).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------- WAV


def wav_encode(x: torch.Tensor, sr: int, *, pcm16: bool = False) -> bytes:
    """WAV bytes of ``[T]`` / ``[C, T]`` audio: float32 IEEE (lossless) or 16-bit PCM."""
    a = x.detach().cpu().float()
    if a.dim() == 1:
        a = a[None]
    if a.dim() != 2:
        raise SynthError(f"Audio must be [T] or [C, T]; got {tuple(a.shape)}.", stage="types")
    ch = a.shape[0]
    inter = a.t().contiguous()
    if pcm16:
        body = (inter.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().astype("<i2")\
            .tobytes()
        fmt, bits = 1, 16
    else:
        body = inter.numpy().astype("<f4").tobytes()
        fmt, bits = 3, 32
    block = ch * bits // 8
    hdr = struct.pack("<4sI4s4sIHHIIHH", b"RIFF", 36 + len(body), b"WAVE", b"fmt ", 16, fmt,
                      ch, int(sr), int(sr) * block, block, bits)
    return hdr + struct.pack("<4sI", b"data", len(body)) + body


def wav_decode(data: bytes) -> tuple[torch.Tensor, int]:
    """``(audio [T] or [C, T] float32, sample rate)`` from float32 / PCM16 WAV bytes."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SynthError("Not a WAV file.", stage="types")
    pos = 12
    fmt = ch = sr = bits = None
    body = b""
    while pos + 8 <= len(data):
        kind, n = struct.unpack("<4sI", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + n]
        if kind == b"fmt ":
            fmt, ch, sr, _, _, bits = struct.unpack("<HHIIHH", chunk[:16])
        elif kind == b"data":
            body = chunk
        pos += 8 + n + (n & 1)
    if fmt == 3 and bits == 32:
        x = torch.frombuffer(bytearray(body), dtype=torch.float32).clone()
    elif fmt == 1 and bits == 16:
        x = torch.frombuffer(bytearray(body), dtype=torch.int16).float() / 32767.0
    else:
        raise SynthError(f"Unsupported WAV format {fmt} / {bits} bits.", stage="types")
    x = x.view(-1, ch).t().contiguous()
    return (x[0] if ch == 1 else x), int(sr)


# ---------------------------------------------------------------- tensors

_TMAGIC = b"TSR1"


def tensor_encode(t: torch.Tensor) -> bytes:
    """``TSR1`` + header length + JSON header (dtype, shape) + little-endian bytes."""
    x = t.detach().cpu().contiguous()
    if x.dtype == torch.bfloat16:
        x = x.float()
    head = json.dumps({"dtype": str(x.dtype).replace("torch.", ""),
                       "shape": list(x.shape)}, separators=(",", ":")).encode()
    body = x.numpy().astype(x.numpy().dtype.newbyteorder("<")).tobytes() if x.numel() else b""
    return _TMAGIC + struct.pack("<I", len(head)) + head + body


def tensor_decode(data: bytes) -> torch.Tensor:
    if data[:4] != _TMAGIC:
        raise SynthError("Not a synth tensor blob.", stage="types")
    (n,) = struct.unpack("<I", data[4:8])
    head = json.loads(data[8:8 + n])
    dtype = getattr(torch, head["dtype"])
    body = data[8 + n:]
    if not body:
        return torch.zeros(head["shape"], dtype=dtype)
    return torch.frombuffer(bytearray(body), dtype=dtype).view(head["shape"]).clone()


def tensor_inline(t: torch.Tensor) -> dict[str, Any]:
    """Small tensors stay in the record as base64 (exact)."""
    x = t.detach().cpu().contiguous()
    return {"$tensor": base64.b64encode(tensor_encode(x)).decode("ascii")}


def tensor_from_inline(d: dict[str, Any]) -> torch.Tensor:
    return tensor_decode(base64.b64decode(d["$tensor"]))


__all__ = ["png_decode", "png_encode", "tensor_decode", "tensor_encode", "tensor_from_inline",
           "tensor_inline", "to_uint8", "wav_decode", "wav_encode"]
