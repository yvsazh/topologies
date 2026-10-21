"""``synth.image``: procedural images with exact labels + domain randomization (torch).

    synth.image.Shapes(size=32, classes=["circle", "square", "triangle"])   # classification
    synth.image.Glyphs(chars="0123456789", size=28)                         # digits / letters
    synth.image.TextLines(height=24, width=160, alphabet="ABC...0123")      # OCR lines
    synth.image.Scenes(size=64, n_objects=(1, 4))                           # detection: boxes,
                                                                            # labels, class mask
    randomize=synth.image.Randomize(background=("solid", "gradient", "noise", "stripes"),
                                     brightness=(0.7, 1.3), contrast=(0.7, 1.3), blur=(0, 1.2),
                                     noise=0.03, jpeg=(30, 95), affine=0.15, perspective=0.1)

Label, bounding boxes and masks are produced together, before and through
the same geometric transform, so they stay exact. Images are float tensors
``[C, H, W]`` in ``[0, 1]``; the store saves them as content-addressed PNG blobs
and ``ds.for_recipe("vision-classify")`` loads them.
The text renderer uses a built-in 5x7 bitmap font (no fonts needed).
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from topologies import synth

Recorded, SynthError, fp_canonical = synth.Recorded, synth.SynthError, synth.canonical
Generator, Raw = synth.Generator, synth.Raw

# ---------------------------------------------------------------- bitmap font (5x7)

_FONT_ROWS = {
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110", "D": "11100 10010 10001 10001 10001 10010 11100",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110", "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 10001 11001 10101 10011 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "-": "00000 00000 00000 11111 00000 00000 00000", ".": "00000 00000 00000 00000 00000 01100 01100",
    " ": "00000 00000 00000 00000 00000 00000 00000", "/": "00001 00010 00010 00100 01000 01000 10000",
    ":": "00000 01100 01100 00000 01100 01100 00000",
}
FONT_CHARS = "".join(sorted(_FONT_ROWS))


def glyph(ch: str) -> torch.Tensor:
    rows = _FONT_ROWS.get(ch.upper())
    if rows is None:
        raise SynthError(f"No bitmap glyph for {ch!r}.", stage="vision",
                         hint=f"Supported: {FONT_CHARS!r}.")
    return torch.tensor([[int(c) for c in r] for r in rows.split()], dtype=torch.float32)


# ---------------------------------------------------------------- domain randomization


class Randomize(Recorded):
    """Domain randomization applied to image + masks (see module docstring)."""

    def __init__(self, *, background: Sequence[str] = ("solid", "gradient", "noise", "stripes",
                                                       "checker"),
                 brightness: tuple[float, float] = (0.8, 1.2),
                 contrast: tuple[float, float] = (0.8, 1.2), gamma: tuple[float, float] =
                 (0.8, 1.25), blur: tuple[float, float] = (0.0, 1.0), noise: float = 0.02,
                 jpeg: tuple[int, int] | None = None, affine: float = 0.1,
                 perspective: float = 0.0) -> None:
        self.background, self.brightness, self.contrast = list(background), brightness, contrast
        self.gamma, self.blur, self.noise, self.jpeg = gamma, blur, float(noise), jpeg
        self.affine, self.perspective = float(affine), float(perspective)

    def to_config(self) -> dict[str, Any]:
        return {k: fp_canonical(v) for k, v in vars(self).items()}


NONE = Randomize(background=("solid",), brightness=(1, 1), contrast=(1, 1), gamma=(1, 1),
                 blur=(0, 0), noise=0.0, affine=0.0)


def background(kind: str, c: int, h: int, w: int, rng: Any) -> torch.Tensor:
    g = rng.torch
    col = torch.rand(c, 1, 1, generator=g) * 0.5
    if kind == "solid":
        return col.expand(c, h, w).clone()
    if kind == "gradient":
        a, b = torch.rand(c, 1, 1, generator=g) * 0.6, torch.rand(c, 1, 1, generator=g) * 0.6
        t = torch.linspace(0, 1, w).view(1, 1, w) if rng.random() < 0.5 else \
            torch.linspace(0, 1, h).view(1, h, 1)
        return (a + (b - a) * t).expand(c, h, w).clone()
    if kind == "noise":
        low = torch.rand(c, max(2, h // 8), max(2, w // 8), generator=g) * 0.6
        return F.interpolate(low[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    if kind == "stripes":
        per = rng.randint(3, 8)
        t = torch.arange(w if rng.random() < 0.5 else h)
        s = ((t // per) % 2).float() * 0.3
        img = col.expand(c, h, w).clone()
        return img + (s.view(1, 1, -1) if s.numel() == w else s.view(1, -1, 1))
    if kind == "checker":
        per = rng.randint(3, 8)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        return col + (((yy // per + xx // per) % 2).float() * 0.3)[None]
    raise SynthError(f"Unknown background {kind!r}.", stage="vision")


def _gauss_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.05:
        return img
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    k = (k / k.sum())
    c = img.shape[0]
    img = F.conv2d(F.pad(img[None], (r, r, 0, 0), mode="replicate"),
                   k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    img = F.conv2d(F.pad(img, (0, 0, r, r), mode="replicate"),
                   k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return img[0]


def _jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError:
        # Emulate blocking: average 8x8 blocks partially, by quality.
        q = 1 - quality / 100
        blocks = F.avg_pool2d(img[None], 8, ceil_mode=True)
        up = F.interpolate(blocks, size=img.shape[-2:], mode="nearest")[0]
        return img * (1 - q * 0.5) + up * (q * 0.5)
    arr = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(arr[..., 0] if arr.shape[-1] == 1 else arr)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    back = torch.tensor(list(Image.open(io.BytesIO(buf.getvalue())).getdata()),
                        dtype=torch.float32)
    back = back.view(img.shape[1], img.shape[2], -1).permute(2, 0, 1) / 255.0
    return back[: img.shape[0]]


def _affine_grid(h: int, w: int, rng: Any, strength: float, persp: float) -> torch.Tensor:
    ang = rng.uniform(-strength, strength) * math.pi
    sc = 1 + rng.uniform(-strength, strength)
    tx, ty = rng.uniform(-strength, strength), rng.uniform(-strength, strength)
    theta = torch.tensor([[math.cos(ang) / sc, -math.sin(ang) / sc, tx],
                          [math.sin(ang) / sc, math.cos(ang) / sc, ty]], dtype=torch.float32)
    grid = F.affine_grid(theta[None], [1, 1, h, w], align_corners=False)
    if persp:
        px, py = rng.uniform(-persp, persp), rng.uniform(-persp, persp)
        denom = 1 + px * grid[..., 0] + py * grid[..., 1]
        grid = grid / denom.unsqueeze(-1).clamp(min=0.2)
    return grid


def randomize(img: torch.Tensor, masks: torch.Tensor | None, rng: Any, dr: Randomize
              ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Geometric transform (image + masks together), then photometric changes."""
    _, h, w = img.shape
    if dr.affine or dr.perspective:
        grid = _affine_grid(h, w, rng, dr.affine, dr.perspective)
        img = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)[0]
        if masks is not None:
            masks = F.grid_sample(masks[None].float(), grid, mode="nearest",
                                  padding_mode="zeros", align_corners=False)[0].round().long()
    b = rng.uniform(*dr.brightness)
    ct = rng.uniform(*dr.contrast)
    mean = img.mean()
    img = ((img - mean) * ct + mean) * b
    img = img.clamp(1e-6, 1).pow(rng.uniform(*dr.gamma))
    img = _gauss_blur(img, rng.uniform(*dr.blur))
    if dr.noise:
        img = img + torch.randn(img.shape, generator=rng.torch) * dr.noise
    img = img.clamp(0, 1)
    if dr.jpeg:
        img = _jpeg(img, rng.randint(*dr.jpeg))
    return img.clamp(0, 1), masks


# ---------------------------------------------------------------- shapes


SHAPES = ("circle", "square", "triangle", "ring", "cross", "diamond")


def shape_mask(kind: str, h: int, w: int, cx: float, cy: float, r: float,
               rot: float = 0.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32) + 0.5,
                            torch.arange(w, dtype=torch.float32) + 0.5, indexing="ij")
    dx, dy = xx - cx, yy - cy
    c, s = math.cos(rot), math.sin(rot)
    u, v = c * dx + s * dy, -s * dx + c * dy
    if kind == "circle":
        return dx * dx + dy * dy <= r * r
    if kind == "ring":
        d = dx * dx + dy * dy
        return (d <= r * r) & (d >= (0.55 * r) ** 2)
    if kind == "square":
        return (u.abs() <= r * 0.8) & (v.abs() <= r * 0.8)
    if kind == "diamond":
        return u.abs() + v.abs() <= r
    if kind == "cross":
        return ((u.abs() <= r) & (v.abs() <= r * 0.3)) | ((v.abs() <= r) & (u.abs() <= r * 0.3))
    if kind == "triangle":
        return (v <= r * 0.6) & (v >= -r + 1.7 * u.abs())
    raise SynthError(f"Unknown shape {kind!r}.", stage="vision", available=list(SHAPES))


def bbox(mask: torch.Tensor) -> list[int] | None:
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _color(rng: Any, c: int, avoid: torch.Tensor | None = None) -> torch.Tensor:
    col = torch.rand(c, 1, 1, generator=rng.torch) * 0.5 + 0.5
    return col


class _VisionGen(Generator):
    evidence = "construction"

    def __init__(self, size: int, channels: int, randomize: Randomize | None) -> None:
        self.size, self.channels = int(size), int(channels)
        self.randomize = randomize if randomize is not None else Randomize()

    def _canvas(self, rng: Any) -> torch.Tensor:
        kind = self.randomize.background[rng.randrange(len(self.randomize.background))]
        return background(kind, self.channels, self.size, self.size, rng)


class Shapes(_VisionGen):
    """One shape per image: ``{"image", "label", "bbox"}``."""

    name = "shapes"
    tags = ("image", "classification")
    example = {"size": 32}

    def __init__(self, *, size: int = 32, classes: Sequence[str] = ("circle", "square",
                                                                    "triangle"),
                 channels: int = 3, randomize: Randomize | None = None,
                 min_scale: float = 0.2, max_scale: float = 0.4) -> None:
        super().__init__(size, channels, randomize)
        bad = [c for c in classes if c not in SHAPES]
        if bad:
            unknown = synth.unknown

            raise unknown("shape", bad[0], SHAPES, stage="vision")
        self.classes, self.min_scale, self.max_scale = list(classes), min_scale, max_scale
        Image, Label, Schema, Tokens = synth.Image, synth.Label, synth.Schema, synth.Tokens

        self.produces = Schema(image=Image(channels=channels, height=size, width=size),
                               label=Label(self.classes), bbox=Tokens(optional=True))

    def generate(self, ctx: Any, attrs: dict[str, Any]) -> Raw:
        rng = ctx.rng
        n = self.size
        label = attrs.get("label") if attrs.get("label") in self.classes else \
            self.classes[rng.randrange(len(self.classes))]
        r = rng.uniform(self.min_scale, self.max_scale) * n
        cx, cy = rng.uniform(r, n - r), rng.uniform(r, n - r)
        m = shape_mask(label, n, n, cx, cy, r, rng.uniform(0, 2 * math.pi))
        img = self._canvas(rng)
        img = torch.where(m[None], _color(rng, self.channels).expand_as(img), img)
        img, mm = randomize(img, m[None].long(), rng, self.randomize)
        return Raw({"image": img, "label": label, "bbox": bbox(mm[0] > 0) or [0, 0, 0, 0]})

    def config(self) -> dict[str, Any]:
        return {"type": "shapes", "size": self.size, "channels": self.channels,
                "classes": self.classes, "scale": [self.min_scale, self.max_scale],
                "randomize": self.randomize.to_config()}


class Glyphs(_VisionGen):
    """A bitmap character per image: ``{"image", "label"}`` (digits, letters)."""

    name = "glyphs"
    tags = ("image", "classification", "ocr")
    example = {"chars": "0123", "size": 16}

    def __init__(self, *, chars: str = "0123456789", size: int = 28, channels: int = 1,
                 randomize: Randomize | None = None) -> None:
        super().__init__(size, channels, randomize)
        for ch in chars:
            glyph(ch)
        self.chars = chars
        Image, Label, Schema = synth.Image, synth.Label, synth.Schema

        self.produces = Schema(image=Image(channels=channels, height=size, width=size),
                               label=Label(list(chars)))

    def generate(self, ctx: Any, attrs: dict[str, Any]) -> Raw:
        rng, n = ctx.rng, self.size
        lab = attrs.get("label")
        ch = str(lab) if lab is not None and len(str(lab)) == 1 and str(lab) in self.chars \
            else self.chars[rng.randrange(len(self.chars))]
        g = glyph(str(ch))
        scale = rng.uniform(0.5, 0.8) * n / 7
        gh, gw = max(1, int(7 * scale)), max(1, int(5 * scale))
        big = F.interpolate(g[None, None], size=(gh, gw), mode="nearest")[0, 0] > 0.5
        m = torch.zeros(n, n, dtype=torch.bool)
        y, x = rng.randint(0, n - gh), rng.randint(0, n - gw)
        m[y:y + gh, x:x + gw] = big
        img = self._canvas(rng)
        img = torch.where(m[None], _color(rng, self.channels).expand_as(img), img)
        img, _ = randomize(img, None, rng, self.randomize)
        return Raw({"image": img, "label": str(ch)})

    def config(self) -> dict[str, Any]:
        return {"type": "glyphs", "chars": self.chars, "size": self.size,
                "channels": self.channels, "randomize": self.randomize.to_config()}


class TextLines(_VisionGen):
    """A rendered line of text (OCR): ``{"image", "text"}``."""

    name = "textlines"
    tags = ("image", "ocr")
    example = {"height": 12, "width": 64}

    def __init__(self, *, height: int = 16, width: int = 128, alphabet: str = FONT_CHARS,
                 min_chars: int = 3, max_chars: int = 16, channels: int = 1,
                 randomize: Randomize | None = None, texts: Sequence[str] | None = None) -> None:
        super().__init__(height, channels, randomize or Randomize(affine=0.02))
        self.height, self.width = int(height), int(width)
        self.alphabet = "".join(c for c in alphabet if c.upper() in _FONT_ROWS)
        self.min_chars, self.max_chars = int(min_chars), int(max_chars)
        self.texts = list(texts) if texts else None
        Image, Schema, Text = synth.Image, synth.Schema, synth.Text

        self.produces = Schema(image=Image(channels=channels, height=height, width=width),
                               text=Text())

    def generate(self, ctx: Any, attrs: dict[str, Any]) -> Raw:
        rng = ctx.rng
        if attrs.get("text"):
            text = str(attrs["text"]).upper()
        elif self.texts:
            text = self.texts[rng.randrange(len(self.texts))].upper()
        else:
            k = rng.randint(self.min_chars, self.max_chars)
            text = "".join(rng.choice(self.alphabet) for _ in range(k)).upper()
        scale = max(1, int(self.height * 0.8 / 7))
        cw = 6 * scale
        text = text[: max(1, (self.width - 2) // cw)]
        m = torch.zeros(self.height, self.width, dtype=torch.bool)
        x0, y0 = rng.randint(0, max(0, self.width - cw * len(text))), \
            rng.randint(0, max(0, self.height - 7 * scale))
        for i, ch in enumerate(text):
            g = glyph(ch).repeat_interleave(scale, 0).repeat_interleave(scale, 1) > 0
            x = x0 + i * cw
            m[y0:y0 + g.shape[0], x:x + g.shape[1]] |= g[: self.height - y0, : self.width - x]
        img = background(self.randomize.background[rng.randrange(len(
            self.randomize.background))], self.channels, self.height, self.width, rng)
        img = torch.where(m[None], _color(rng, self.channels).expand_as(img), img)
        img, _ = randomize(img, None, rng, self.randomize)
        return Raw({"image": img, "text": text})

    def config(self) -> dict[str, Any]:
        return {"type": "textlines", "h": self.height, "w": self.width,
                "alphabet": self.alphabet, "chars": [self.min_chars, self.max_chars],
                "channels": self.channels, "texts": self.texts,
                "randomize": self.randomize.to_config()}


class Scenes(_VisionGen):
    """Several shapes per image: ``{"image", "boxes", "labels", "mask"}`` (class-id map,
    0 = background, k+1 = ``classes[k]``)."""

    name = "scenes"
    tags = ("image", "detection", "segmentation")
    example = {"size": 48}

    def __init__(self, *, size: int = 64, classes: Sequence[str] = ("circle", "square",
                                                                    "triangle"),
                 n_objects: tuple[int, int] = (1, 4), channels: int = 3,
                 randomize: Randomize | None = None) -> None:
        super().__init__(size, channels, randomize)
        self.classes, self.n_objects = list(classes), tuple(n_objects)
        Image, Json, Schema = synth.Image, synth.Json, synth.Schema

        self.produces = Schema(image=Image(channels=channels, height=size, width=size),
                               boxes=Json(), labels=Json(), mask=Image(optional=True))

    def generate(self, ctx: Any, attrs: dict[str, Any]) -> Raw:
        rng, n = ctx.rng, self.size
        img = self._canvas(rng)
        cmask = torch.zeros(len(self.classes) + 0, n, n, dtype=torch.long)
        inst = []
        for _ in range(rng.randint(*self.n_objects)):
            k = rng.randrange(len(self.classes))
            r = rng.uniform(0.08, 0.2) * n
            m = shape_mask(self.classes[k], n, n, rng.uniform(r, n - r), rng.uniform(r, n - r),
                           r, rng.uniform(0, 6.283))
            img = torch.where(m[None], _color(rng, self.channels).expand_as(img), img)
            for prev in inst:  # occlusion: later objects cover earlier ones
                prev[1] &= ~m
            inst.append([k, m.clone()])
        stack = torch.stack([m.long() * (j + 1) for j, (_, m) in enumerate(inst)]) if inst \
            else torch.zeros(1, n, n, dtype=torch.long)
        img, stack = randomize(img, stack, rng, self.randomize)
        boxes, labels = [], []
        class_map = torch.zeros(n, n, dtype=torch.long)
        for j, (k, _) in enumerate(inst):
            m = stack[j] == j + 1
            b = bbox(m)
            if b is None:
                continue
            boxes.append(b)
            labels.append(self.classes[k])
            class_map[m] = k + 1
        _ = cmask
        return Raw({"image": img, "boxes": boxes, "labels": labels,
                    "mask": class_map[None].float() / 255.0})

    def config(self) -> dict[str, Any]:
        return {"type": "scenes", "size": self.size, "classes": self.classes,
                "n_objects": list(self.n_objects), "channels": self.channels,
                "randomize": self.randomize.to_config()}


# ---------------------------------------------------------------- blobs / files


def encode_image(t: torch.Tensor) -> tuple[bytes, str]:
    """PNG bytes (``synth.codecs``, no pillow) or tensor bytes for other channel counts."""
    Image = synth.Image

    return Image().encode(t)


def decode_image(path: str | Path) -> torch.Tensor:
    Image = synth.Image

    p = Path(path)
    return Image().decode(p.read_bytes(), {"type": p.suffix.lstrip(".")})


def save_png(value: Any, path: str | Path) -> None:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    data, ext = encode_image(t)
    p = Path(path)
    if ext != "png":
        p = p.with_suffix(".tensor")
    p.write_bytes(data)


__all__ = ["FONT_CHARS", "NONE", "SHAPES", "Glyphs", "Randomize", "Scenes", "Shapes", "TextLines",
           "bbox", "decode_image", "encode_image", "glyph", "randomize", "save_png", "shape_mask"]


# ---------------------------------------------------------------- composable node additions


def _image_tensor(value: Any) -> torch.Tensor:
    """Normalize image-like inputs without unwrapping typed image metadata."""
    value = getattr(value, "data", value)
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError("images must have shape [C,H,W]")
    return tensor


def _texture(kind: str, h: int, w: int, rng: Any, scale: int = 8) -> torch.Tensor:
    """Generate a single-channel procedural texture."""
    if h <= 0 or w <= 0 or scale <= 0:
        raise ValueError("texture dimensions and scale must be positive")
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    if kind in ("value", "perlin"):
        gh, gw = max(1, math.ceil(h / scale)), max(1, math.ceil(w / scale))
        if kind == "value":
            lattice = torch.rand((1, 1, gh + 1, gw + 1), generator=rng.torch)
            return F.interpolate(lattice, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        # Gradient Perlin: independent unit gradients at lattice vertices, dot products
        # with corner offsets, quintic fade interpolation in the two dimensions.
        angles = torch.rand((gh + 1, gw + 1), generator=rng.torch) * (2 * math.pi)
        grad_y, grad_x = angles.sin(), angles.cos()
        fy = torch.arange(h, dtype=torch.float32) / scale
        fx = torch.arange(w, dtype=torch.float32) / scale
        iy, ix = fy.floor().long(), fx.floor().long()
        dy, dx = fy - iy, fx - ix
        dxg, dyg = torch.meshgrid(dx, dy, indexing="xy")
        iy2, ix2 = torch.meshgrid(iy, ix, indexing="ij")
        def corner(oy: int, ox: int) -> torch.Tensor:
            gy, gx = grad_y[iy2 + oy, ix2 + ox], grad_x[iy2 + oy, ix2 + ox]
            return gx * (dxg - ox) + gy * (dyg - oy)
        u, v = dx * dx * dx * (dx * (dx * 6 - 15) + 10), \
            dy * dy * dy * (dy * (dy * 6 - 15) + 10)
        top = corner(0, 0) * (1 - u[None, :]) + corner(0, 1) * u[None, :]
        bottom = corner(1, 0) * (1 - u[None, :]) + corner(1, 1) * u[None, :]
        return ((top * (1 - v[:, None]) + bottom * v[:, None]) * .5 + .5).clamp(0, 1)
    if kind == "voronoi":
        points = torch.rand(max(1, h * w // (scale * scale)), 2, generator=rng.torch)
        points[:, 0] *= h
        points[:, 1] *= w
        d = (yy[..., None] - points[:, 0]).square() + (xx[..., None] - points[:, 1]).square()
        return d.min(-1).values.sqrt().div(max(h, w)).clamp(0, 1)
    if kind == "stripes":
        return ((xx // scale) % 2).float()
    if kind == "checker":
        return ((xx // scale + yy // scale) % 2).float()
    if kind == "gradient":
        return xx.float() / max(w - 1, 1)
    raise ValueError(f"unknown texture kind: {kind}")


@synth.node(type=synth.Image(), tags=("image", "texture"),
            example={"kind": "perlin", "height": 32, "width": 32})
def texture(r, *, kind: str = "perlin", height: int = 64, width: int = 64,
            scale: int = 8) -> torch.Tensor:
    """Create a seeded value/Perlin-like, Voronoi, stripe, checker or gradient image.

    >>> synth.image.texture(kind="checker", height=32, width=32)
    """
    return _texture(kind, height, width, r, scale)[None].expand(3, -1, -1).contiguous()


@synth.node(tags=("image", "composite"), example={"height": 32, "width": 32,
                                                   "layers": []})
def canvas(*, height: int, width: int, layers: Sequence[dict[str, Any]],
           background: Any = 0.0) -> dict[str, Any]:
    """Composite RGBA/RGB/mask layers; later opaque masks occlude earlier labels.

    Layer mapping has image or color, optional mask, blend (normal/add/multiply/screen),
    label and keypoints. Returned boxes are exclusive [x0,y0,x1,y1] bounds of visible masks.
    >>> synth.image.canvas(height=32, width=32, layers=[])
    """
    if height <= 0 or width <= 0:
        raise ValueError("canvas dimensions must be positive")
    base = torch.as_tensor(background, dtype=torch.float32)
    if base.ndim == 0:
        out = base.expand(3, height, width).clone()
    elif base.ndim == 1:
        out = base[:, None, None].expand(-1, height, width).clone()
    else:
        out = _image_tensor(base).clone()
        if tuple(out.shape[-2:]) != (height, width):
            raise ValueError("background size must match canvas")
    masks, labels, points = [], [], []
    for layer in layers:
        image = layer.get("image")
        mask = layer.get("mask")
        if image is None:
            color = torch.as_tensor(layer.get("color", 1.0), dtype=out.dtype)
            image = color[:, None, None] if color.ndim else color
        image = torch.as_tensor(image, dtype=out.dtype)
        if image.ndim == 0:
            image = image.expand_as(out)
        elif image.ndim == 1:
            image = image[:, None, None].expand_as(out)
        elif image.ndim == 3 and image.shape[-2:] == (1, 1):
            image = image.expand_as(out)
        image = _image_tensor(image)
        if image.shape[-2:] != (height, width):
            raise ValueError("layer image size must match canvas")
        if mask is None:
            mask = torch.ones((height, width), dtype=torch.bool)
        mask = torch.as_tensor(mask).squeeze().bool()
        if tuple(mask.shape) != (height, width):
            raise ValueError("layer mask size must match canvas")
        mode = layer.get("blend", "normal")
        if mode not in ("normal", "add", "multiply", "screen"):
            raise ValueError(f"unknown blend mode {mode}")
        if mode == "normal":
            blended = image
        elif mode == "add":
            blended = out + image
        elif mode == "multiply":
            blended = out * image
        else:
            blended = 1 - (1 - out) * (1 - image)
        alpha = float(layer.get("alpha", 1.0))
        if not 0 <= alpha <= 1:
            raise ValueError("layer alpha must lie in [0,1]")
        effective = mask[None].to(out.dtype) * alpha
        out = (blended * effective + out * (1 - effective)).clamp(0, 1)
        for old in masks:
            old &= ~(mask & (alpha == 1.0))
        masks.append(mask.clone() if alpha == 1.0 else torch.zeros_like(mask))
        labels.append(layer.get("label"))
        points.append(layer.get("keypoints", []))
    keep = [i for i, m in enumerate(masks) if bbox(m) is not None]
    return {"image": out, "masks": torch.stack([masks[i] for i in keep]) if keep else
            torch.zeros((0, height, width), dtype=torch.bool),
            "boxes": [bbox(masks[i]) for i in keep], "labels": [labels[i] for i in keep],
            "keypoints": [points[i] for i in keep]}


def _warp(image: Any, matrix: torch.Tensor, height: int | None = None,
          width: int | None = None, *, mode: str = "bilinear") -> torch.Tensor:
    src = _image_tensor(image)
    _, h, w = src.shape
    oh, ow = height or h, width or w
    if matrix.shape != (3, 3) or not bool(torch.isfinite(matrix).all()) or \
            abs(float(torch.det(matrix))) < 1e-8:
        raise ValueError("transform matrix must be finite and invertible [3,3]")
    y, x = torch.meshgrid(torch.arange(oh, dtype=src.dtype) + .5,
                          torch.arange(ow, dtype=src.dtype) + .5, indexing="ij")
    dest = torch.stack((x, y, torch.ones_like(x)), -1)
    srcp = dest @ torch.linalg.inv(matrix).T
    sx, sy = srcp[..., 0] / srcp[..., 2], srcp[..., 1] / srcp[..., 2]
    grid = torch.stack((2 * sx / w - 1, 2 * sy / h - 1), -1)[None]
    return F.grid_sample(src[None], grid, align_corners=False, padding_mode="zeros", mode=mode)[0]


def _transform_record(value: Any, matrix: torch.Tensor) -> Any:
    """Transform plain image or paired canvas record using one shared sampling grid."""
    if not isinstance(value, dict):
        result = _warp(value, matrix)
        return synth.Image(result) if isinstance(value, synth.Image) else result
    record = dict(value)
    original_image = value["image"]
    warped_image = _warp(original_image, matrix)
    record["image"] = synth.Image(warped_image) if isinstance(original_image, synth.Image) \
        else warped_image
    if "masks" in value:
        old = torch.as_tensor(getattr(value["masks"], "data", value["masks"]))
        warped = _warp(old.float() if old.ndim == 3 else old[None].float(), matrix, mode="nearest")
        masks = warped.round().to(old.dtype)
        if old.ndim == 2:
            masks = masks[0]
        record["masks"] = synth.Mask(masks) if isinstance(value["masks"], synth.Mask) else masks
        if old.ndim == 3 and "labels" in value:
            keep = [i for i, m in enumerate(masks) if bbox(m.bool()) is not None]
            visible_masks = masks[keep]
            record["masks"] = synth.Mask(visible_masks) if isinstance(value["masks"], synth.Mask) \
                else visible_masks
            record["labels"] = [value["labels"][i] for i in keep]
            boxes = [bbox(masks[i].bool()) for i in keep]
            record["boxes"] = synth.Boxes(boxes) if isinstance(value.get("boxes"), synth.Boxes) \
                else boxes
            if "keypoints" in value:
                transformed = []
                for i in keep:
                    pts = torch.as_tensor(value["keypoints"][i], dtype=torch.float32)
                    if pts.numel():
                        hp = torch.cat((pts[:, :2], torch.ones_like(pts[:, :1])), 1) @ matrix.T
                        xy = hp[:, :2] / hp[:, 2:3]
                        pts = torch.cat((xy, pts[:, 2:]), 1) if pts.shape[1] > 2 else xy
                    transformed.append(pts.tolist())
                record["keypoints"] = transformed
    elif "mask" in value:
        mask = torch.as_tensor(getattr(value["mask"], "data", value["mask"]))
        warped = _warp(mask.float()[None], matrix, mode="nearest")[0].round().to(mask.dtype)
        record["mask"] = synth.Mask(warped) if isinstance(mask, synth.Mask) else warped
        if "boxes" in value:
            boxes = [bbox(warped.bool())] if bbox(warped.bool()) else []
            record["boxes"] = synth.Boxes(boxes) if isinstance(value["boxes"], synth.Boxes) else boxes
    if "keypoints" in value and "masks" not in value:
        pts = torch.as_tensor(getattr(value["keypoints"], "data", value["keypoints"]),
                              dtype=torch.float32)
        if pts.numel():
            xy1 = torch.cat((pts[:, :2], torch.ones_like(pts[:, :1])), 1) @ matrix.T
            xy = xy1[:, :2] / xy1[:, 2:3]
            mapped = torch.cat((xy, pts[:, 2:]), 1) if pts.shape[1] > 2 else xy
            record["keypoints"] = synth.Keypoints(mapped) if isinstance(value["keypoints"],
                                                                       synth.Keypoints) else mapped
    return record


@synth.node(tags=("image", "transform", "affine"),
            example={"image": [[[0, 0], [0, 0]]], "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
def affine(*, image: Any, matrix: Any) -> torch.Tensor:
    """Apply a forward pixel-coordinate affine matrix (pixel centers; exclusive boxes).

    Keypoints are transformed analytically when supplied via the companion record API.
    >>> synth.image.affine(image=torch.zeros(1, 16, 16), matrix=[[1,0,1],[0,1,1],[0,0,1]])
    """
    m = torch.as_tensor(matrix, dtype=torch.float32)
    if not torch.allclose(m[2], torch.tensor([0., 0., 1.])):
        raise ValueError("affine matrix last row must be [0,0,1]")
    return _transform_record(image, m)


@synth.node(tags=("image", "transform", "perspective"),
            example={"image": [[[0, 0], [0, 0]]], "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
def perspective(*, image: Any, matrix: Any) -> torch.Tensor:
    """Apply invertible forward homography in pixel coordinates.

    >>> synth.image.perspective(image=torch.zeros(1, 16, 16), matrix=[[1,0,0],[0,1,0],[0,0,1]])
    """
    return _transform_record(image, torch.as_tensor(matrix, dtype=torch.float32))


@synth.node(tags=("image", "transform", "elastic"),
            example={"image": [[[0, 0], [0, 0]]], "amplitude": 1.0})
def elastic(r, *, image: Any, amplitude: float = 2.0, grid: int = 8) -> torch.Tensor:
    """Apply seeded smooth displacement; landmarks are not exact unless rendered from it.

    The transform samples a low-resolution displacement field, interpolated bilinearly.
    >>> synth.image.elastic(image=torch.zeros(1, 16, 16), amplitude=1.0)
    """
    source = image["image"] if isinstance(image, dict) else image
    src = _image_tensor(source)
    if amplitude < 0 or grid < 2:
        raise ValueError("amplitude must be nonnegative and grid at least 2")
    _, h, w = src.shape
    field = torch.rand((1, 2, grid, grid), generator=r.torch) * 2 - 1
    field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False)[0]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    gx = 2 * (xx.float() + .5 + amplitude * field[0]) / w - 1
    gy = 2 * (yy.float() + .5 + amplitude * field[1]) / h - 1
    grid = torch.stack((gx, gy), -1)[None]
    if not isinstance(image, dict):
        return F.grid_sample(src[None], grid, align_corners=False)[0]
    record = dict(image)
    rendered = F.grid_sample(src[None], grid, align_corners=False)[0]
    record["image"] = synth.Image(rendered) if isinstance(image["image"], synth.Image) else rendered
    if "masks" in image:
        raw_masks = image["masks"]
        masks = torch.as_tensor(getattr(raw_masks, "data", raw_masks))
        lead = masks[None] if masks.ndim == 2 else masks[:, None]
        transformed = F.grid_sample(lead.float(), grid.expand(len(lead), -1, -1, -1),
                                    mode="nearest", align_corners=False)[:, 0].round().to(masks.dtype)
        if masks.ndim == 2:
            record["masks"] = synth.Mask(transformed[0]) if isinstance(masks, synth.Mask) else transformed[0]
        else:
            keep = [i for i, m in enumerate(transformed) if bbox(m.bool()) is not None]
            visible_masks = transformed[keep]
            record["masks"] = synth.Mask(visible_masks) if isinstance(masks, synth.Mask) else visible_masks
            record["labels"] = [image["labels"][i] for i in keep]
            boxes = [bbox(transformed[i].bool()) for i in keep]
            record["boxes"] = synth.Boxes(boxes) if isinstance(image.get("boxes"), synth.Boxes) \
                else boxes
            # Elastic keypoint convention: visible-object mask centroid, not an
            # inverse displacement estimate of supplied point coordinates.
            record["keypoints"] = [[[(bbox(transformed[i].bool())[0] + bbox(transformed[i].bool())[2]) / 2,
                                      (bbox(transformed[i].bool())[1] + bbox(transformed[i].bool())[3]) / 2]]
                                    for i in keep]
    return record


@synth.node(type=synth.Image(), tags=("image", "photometric", "noise"),
            example={"image": [[[0, 0], [0, 0]]], "sigma": 0.01})
def photometric_noise(r, *, image: Any, sigma: float = 0.02) -> torch.Tensor:
    """Add reproducible Gaussian pixel noise, clipped to the image range.

    >>> synth.image.photometric_noise(image=torch.zeros(1, 16, 16), sigma=0.01)
    """
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    src = _image_tensor(image)
    return (src + torch.randn(src.shape, generator=r.torch) * sigma).clamp(0, 1)


__all__ += ["affine", "canvas", "elastic", "perspective", "photometric_noise", "texture"]
