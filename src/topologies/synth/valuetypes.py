"""Value types: what a field holds, how it is checked, stored, trained on and shown.

A *type* is a registered kind of value (``Text``, ``Audio``, ``Image``, ``Tensor``,
``Graph``, ...). Each implements a few hooks:

=====================  ======================================================
``validate(value)``    problems (empty list = valid)
``normalize(value)``   the canonical form stored and streamed (e.g. 8-bit image
                       levels, float32 audio) so ``decode(encode(v)) == v``
``encode`` / ``decode``  blob codec (PNG, WAV, tensor bytes, JSON) or inline JSON
``to_torch(value)``    tensor(s) for recipes
``preview(value)``     one console line; ``preview_html(value)``: an HTML snippet
``features(values)``   ``FloatTensor[N, D]`` for quality (C2ST, MMD, near-dups)
``stats(values)``      report statistics
=====================  ======================================================

Types double as schema fields (``synth.Schema(audio=synth.Audio(sr=16_000))``),
as node value types (``@synth.node(type=synth.Audio)``) and as value wrappers
(``synth.Audio(x, sr=16_000)``: the data plus its metadata). A new modality is
one subclass plus ``synth.register_type(MyType)``.

    t = synth.Tensor(shape=(None, 3), dtype="float32")
    t.validate(torch.zeros(5, 3))          # []
    synth.Audio(x, sr=8_000).meta()        # {"sr": 8000, "channels": None}
"""

from __future__ import annotations

import base64
import html
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch

from topologies.synth.ctor import Recorded
from topologies.synth.errors import SynthError, unknown


class _MissingType:
    """No data: a type spec rather than a wrapped value (a picklable singleton)."""

    def __repr__(self) -> str:
        return "<missing>"

    def __reduce__(self) -> str:
        return "_MISSING"


_MISSING: Any = _MissingType()
TYPES: dict[str, type[Type]] = {}
INLINE_NUMEL = 1024
SPARK = "▁▂▃▄▅▆▇█"
SHADES = " .:-=+*#%@"


def register_type(cls: type[Type]) -> type[Type]:
    """Register a value type (``synth.register_type(MyType)``; usable as a decorator)."""
    if not (isinstance(cls, type) and issubclass(cls, Type)):
        raise SynthError(f"register_type needs a synth.Type subclass; got {cls!r}.",
                         stage="types")
    TYPES[cls.kind] = cls
    return cls


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _short(s: str, n: int = 100) -> str:
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[: n - 1] + "…"


def _tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    data = getattr(value, "data", value) if isinstance(value, Type) else value
    t = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    return t if dtype is None else t.to(dtype)


def spark(values: Sequence[float], width: int = 24) -> str:
    vals = [float(v) for v in values]
    if not vals:
        return ""
    if len(vals) > width:
        k = len(vals) / width
        vals = [max(vals[int(i * k):max(int(i * k) + 1, int((i + 1) * k))]) for i in range(width)]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((v - lo) / span * 7.999))] for v in vals)


def thumbnail(img: torch.Tensor, width: int = 24) -> list[str]:
    """Block-character thumbnail lines of an image ``[C, H, W]`` (gray levels)."""
    x = img.float()
    if x.dim() == 3:
        x = x[:3].mean(0)
    h, w = x.shape
    cols = min(width, w)
    rows = max(1, round(h / w * cols / 2))
    small = torch.nn.functional.interpolate(x[None, None], size=(rows, cols), mode="area")[0, 0]
    lo, hi = float(small.min()), float(small.max())
    span = (hi - lo) or 1.0
    return ["".join(SHADES[min(9, int((float(v) - lo) / span * 9.999))] for v in row)
            for row in small]


# ---------------------------------------------------------------- base


class Type(Recorded):
    """Base value type (see module docstring). ``Type(data, **params)`` wraps a value."""

    kind: str = "any"
    #: Values of this type are stored as blobs (files) rather than inline JSON.
    blob: bool = False

    def __init__(self, data: Any = _MISSING, *, optional: bool = False,
                 description: str | None = None) -> None:
        self.data = data
        self.optional = bool(optional)
        self.description = description

    # -- spec vs value -----------------------------------------------------------
    @property
    def is_value(self) -> bool:
        return self.data is not _MISSING

    def params(self) -> dict[str, Any]:
        """Type parameters (``to_config``, ``repr``)."""
        return {}

    def meta(self) -> dict[str, Any]:
        """Parameters that travel with a wrapped value (e.g. the sample rate)."""
        return self.params()

    def spec(self) -> Type:
        """This type without data."""
        if not self.is_value:
            return self
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.data = _MISSING
        return new

    def wrap(self, data: Any) -> Type:
        """A value of this type holding ``data``."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.data = data
        return new

    @staticmethod
    def unwrap(value: Any) -> Any:
        return value.data if isinstance(value, Type) and value.is_value else value

    # -- behaviour ----------------------------------------------------------------
    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        """Problems with ``value`` (empty list = valid)."""
        return []

    def coerce(self, value: Any) -> Any:
        """Light normalization applied before validation (e.g. ``"3"`` -> 3 for numbers)."""
        return value

    def normalize(self, value: Any) -> Any:
        """Canonical stored / streamed form (deterministic; see module docstring)."""
        return value

    def json_schema(self) -> dict[str, Any]:
        return {}

    def example(self, rng: random.Random) -> Any:
        return None

    # -- storage ---------------------------------------------------------------------
    def encode(self, value: Any) -> tuple[bytes, str]:
        """Blob bytes and file extension (blob types)."""
        raise SynthError(f"{type(self).__name__} values are stored inline.", stage="types")

    def decode(self, data: bytes, ref: Mapping[str, Any]) -> Any:
        raise SynthError(f"{type(self).__name__} has no blob decoder.", stage="types")

    def to_json(self, value: Any) -> Any:
        """Inline storage form (JSON-able)."""
        return to_jsonable(self.unwrap(value))

    def from_json(self, value: Any) -> Any:
        return from_jsonable(value)

    # -- training / display / quality --------------------------------------------------
    def to_torch(self, value: Any) -> Any:
        return _tensor(value)

    def preview(self, value: Any) -> str:
        return _short(repr(self.unwrap(value)))

    def preview_html(self, value: Any) -> str:
        return f"<code>{_esc(self.preview(value))}</code>"

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        """``FloatTensor[N, D]`` describing each value (quality metrics)."""
        return torch.zeros(len(values), 1)

    def stats(self, values: Sequence[Any]) -> dict[str, Any]:
        return {"n": len(values)}

    # -- config -------------------------------------------------------------------------
    def to_config(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.kind, **self.params()}
        if self.optional:
            out["optional"] = True
        if self.description:
            out["description"] = self.description
        return out

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items() if v is not None)
        if self.is_value:
            d = self.data
            shown = f"tensor{list(d.shape)}" if isinstance(d, torch.Tensor) else _short(repr(d), 40)
            args = shown + (", " + args if args else "")
        return f"{type(self).__name__}({args})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Type) or type(other) is not type(self):
            return False
        if self.is_value or other.is_value:
            from topologies.synth.core import values_equal

            return self.is_value and other.is_value and self.meta() == other.meta() and \
                values_equal(self.data, other.data)
        return self.to_config() == other.to_config()

    def __hash__(self) -> int:
        return hash(repr(self.to_config()))


Field = Type


def _with_desc(schema: dict[str, Any], f: Type) -> dict[str, Any]:
    if f.description:
        schema["description"] = f.description
    return schema


# ---------------------------------------------------------------- JSON helpers


def to_jsonable(v: Any) -> Any:
    """Inline storage form: tensors become exact base64 blobs-in-record."""
    if isinstance(v, torch.Tensor):
        from topologies.synth.codecs import tensor_inline

        if v.dim() == 0:
            return v.item()
        return tensor_inline(v)
    if isinstance(v, Type) and v.is_value:
        return {"$type": v.kind, "meta": to_jsonable(v.meta()), "data": to_jsonable(v.data)}
    if isinstance(v, Mapping):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def from_jsonable(v: Any) -> Any:
    if isinstance(v, Mapping):
        if "$tensor" in v:
            from topologies.synth.codecs import tensor_from_inline

            return tensor_from_inline(dict(v))
        if "$type" in v:
            t = type_class(v["$type"])(**from_jsonable(v.get("meta") or {}))
            return t.wrap(from_jsonable(v.get("data")))
        return {k: from_jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [from_jsonable(x) for x in v]
    return v


# ---------------------------------------------------------------- text-like


class Text(Type):
    """A string. ``min_chars`` / ``max_chars`` bound its length; ``language`` is an ISO code
    (checked by ``ops.filters.language``, not by ``validate``)."""

    kind = "text"

    def __init__(self, data: Any = _MISSING, *, min_chars: int | None = None,
                 max_chars: int | None = None, language: str | Sequence[str] | None = None,
                 lang: str | Sequence[str] | None = None, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.min_chars, self.max_chars = min_chars, max_chars
        self.language = language if language is not None else lang

    @property
    def lang(self) -> Any:
        return self.language

    def coerce(self, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        value = self.unwrap(value)
        if not isinstance(value, str):
            return [f"expected text, got {type(value).__name__}"]
        n = len(value)
        if self.min_chars is not None and n < self.min_chars:
            return [f"text has {n} chars < min_chars={self.min_chars}"]
        if self.max_chars is not None and n > self.max_chars:
            return [f"text has {n} chars > max_chars={self.max_chars}"]
        return []

    def json_schema(self) -> dict[str, Any]:
        s: dict[str, Any] = {"type": "string"}
        if self.min_chars is not None:
            s["minLength"] = self.min_chars
        if self.max_chars is not None:
            s["maxLength"] = self.max_chars
        return _with_desc(s, self)

    def example(self, rng: random.Random) -> str:
        words = ["the", "order", "arrived", "late", "and", "the", "box", "was", "damaged",
                 "please", "help", "me", "with", "a", "refund"]
        n = max(3, (self.min_chars or 20) // 5)
        text = " ".join(rng.choice(words) for _ in range(n)).capitalize() + "."
        if self.max_chars is not None:
            text = text[: self.max_chars]
        return text

    def to_torch(self, value: Any) -> Any:
        raise SynthError("Text needs a tokenizer to become a tensor.", stage="types",
                         hint="Use ds.for_recipe(..., tokenizer=...) / stream.for_recipe(...).")

    def preview(self, value: Any) -> str:
        return repr(_short(str(self.unwrap(value)), 120))

    def preview_html(self, value: Any) -> str:
        return f"<pre class='t'>{_esc(self.unwrap(value))}</pre>"

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        from topologies.synth.quality.embed import HashedEmbedder

        return HashedEmbedder(1024).encode([str(self.unwrap(v)) for v in values])

    def stats(self, values: Sequence[Any]) -> dict[str, Any]:
        texts = [str(self.unwrap(v)) for v in values]
        words = [len(t.split()) for t in texts]
        return {"n": len(texts), "mean_chars": sum(map(len, texts)) / max(1, len(texts)),
                "mean_words": sum(words) / max(1, len(words)),
                "distinct": len(set(texts)) / max(1, len(texts))}

    def params(self) -> dict[str, Any]:
        return {"min_chars": self.min_chars, "max_chars": self.max_chars,
                "language": self.language}


def _choices(choices: Any) -> tuple[list[str], dict[str, str]]:
    if isinstance(choices, Mapping):
        names = [str(k) for k in choices]
        desc = {str(k): str(v) for k, v in choices.items() if v is not None}
        return names, desc
    if isinstance(choices, (str, bytes)) or not isinstance(choices, Iterable):
        raise SynthError(f"choices must be a list or a {{name: description}} dict; got "
                         f"{choices!r}.", stage="schema",
                         hint='Label(["refund", "shipping"]) or Label({"refund": "..."})')
    names = [str(c) for c in choices]
    if len(set(names)) != len(names):
        raise SynthError("choices contain duplicates.", stage="schema", available=names)
    return names, {}


class Category(Type):
    """One value out of a fixed set of strings (not the training target; see ``Label``)."""

    kind = "category"

    def __init__(self, choices: Sequence[str] | Mapping[str, str] | None = None,
                 data: Any = _MISSING, **kw: Any) -> None:
        super().__init__(data, **kw)
        if choices is None:
            raise SynthError(f"{type(self).__name__} needs its choices.", stage="schema",
                             hint=f"{type(self).__name__}(['a', 'b'])")
        self.names, self.descriptions = _choices(choices)
        if not self.names:
            raise SynthError(f"{type(self).__name__} needs at least one choice.", stage="schema")

    def index(self, value: Any) -> int:
        """Integer id of ``value`` (class index for training)."""
        try:
            return self.names.index(str(self.unwrap(value)))
        except ValueError:
            raise unknown("label", value, self.names, stage="schema") from None

    def coerce(self, value: Any) -> Any:
        if isinstance(value, str):
            v = value.strip()
            if v in self.names:
                return v
            low = {n.lower(): n for n in self.names}
            return low.get(v.lower().strip("\"'. "), v)
        if isinstance(value, int) and not isinstance(value, bool) and \
                0 <= value < len(self.names) and value not in self.names:
            return self.names[value]
        return value

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        value = self.unwrap(value)
        if str(value) not in self.names:
            from topologies.synth.errors import suggest

            sug = suggest(str(value), self.names)
            return [f"{value!r} is not one of {self.names}"
                    + (f" (did you mean {sug!r}?)" if sug else "")]
        return []

    def json_schema(self) -> dict[str, Any]:
        return _with_desc({"type": "string", "enum": list(self.names)}, self)

    def example(self, rng: random.Random) -> str:
        return rng.choice(self.names)

    def to_torch(self, value: Any) -> torch.Tensor:
        return torch.tensor(self.index(value))

    def preview(self, value: Any) -> str:
        return str(self.unwrap(value))

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        out = torch.zeros(len(values), len(self.names))
        for i, v in enumerate(values):
            s = str(self.unwrap(v))
            if s in self.names:
                out[i, self.names.index(s)] = 1.0
        return out

    def stats(self, values: Sequence[Any]) -> dict[str, Any]:
        from collections import Counter

        c = Counter(str(self.unwrap(v)) for v in values)
        return {"n": len(values), "counts": dict(c)}

    def params(self) -> dict[str, Any]:
        return {"choices": dict(self.descriptions) if self.descriptions else list(self.names)}


class Label(Category):
    """The classification target: one of ``choices`` (names or ``{name: description}``).

    Descriptions document the classes (reports, docs).
    """

    kind = "label"

    @property
    def num_classes(self) -> int:
        return len(self.names)


class Labels(Type):
    """Multi-label target: a list of distinct names from ``choices``."""

    kind = "labels"

    def __init__(self, choices: Sequence[str] | Mapping[str, str] | None = None, *,
                 min_count: int = 0, max_count: int | None = None, data: Any = _MISSING,
                 **kw: Any) -> None:
        super().__init__(data, **kw)
        if choices is None:
            raise SynthError("Labels needs its choices.", stage="schema")
        self.names, self.descriptions = _choices(choices)
        self.min_count, self.max_count = int(min_count), max_count

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def coerce(self, value: Any) -> Any:
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        return value

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        value = self.unwrap(value)
        if not isinstance(value, (list, tuple)):
            return [f"expected a list of labels, got {type(value).__name__}"]
        bad = [v for v in value if str(v) not in self.names]
        if bad:
            return [f"unknown labels {bad}; allowed {self.names}"]
        if len(set(map(str, value))) != len(value):
            return ["duplicate labels"]
        if len(value) < self.min_count:
            return [f"{len(value)} labels < min_count={self.min_count}"]
        if self.max_count is not None and len(value) > self.max_count:
            return [f"{len(value)} labels > max_count={self.max_count}"]
        return []

    def json_schema(self) -> dict[str, Any]:
        s: dict[str, Any] = {"type": "array", "items": {"type": "string", "enum": self.names},
                             "uniqueItems": True, "minItems": self.min_count}
        if self.max_count is not None:
            s["maxItems"] = self.max_count
        return _with_desc(s, self)

    def example(self, rng: random.Random) -> list[str]:
        hi = self.max_count if self.max_count is not None else min(3, len(self.names))
        k = rng.randint(min(self.min_count, hi), max(self.min_count, hi))
        return rng.sample(self.names, k)

    def to_torch(self, value: Any) -> torch.Tensor:
        out = torch.zeros(len(self.names))
        for x in self.unwrap(value) or []:
            if str(x) in self.names:
                out[self.names.index(str(x))] = 1.0
        return out

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        return torch.stack([self.to_torch(v) for v in values]) if values else torch.zeros(0, 1)

    def params(self) -> dict[str, Any]:
        return {"choices": dict(self.descriptions) if self.descriptions else list(self.names),
                "min_count": self.min_count, "max_count": self.max_count}


# ---------------------------------------------------------------- numbers / tokens


class Number(Type):
    """A number in ``[min, max]``; ``integer=True`` requires an int."""

    kind = "number"

    def __init__(self, data: Any = _MISSING, *, min: float | None = None,
                 max: float | None = None, integer: bool = False, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.min, self.max, self.integer = min, max, bool(integer)

    def coerce(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                f = float(value.strip().replace(",", ""))
                return int(f) if self.integer and f.is_integer() else f
            except ValueError:
                return value
        return value

    def normalize(self, value: Any) -> Any:
        value = self.unwrap(value)
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            value = value.item()
        return value

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        value = self.normalize(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"expected a number, got {type(value).__name__}"]
        if isinstance(value, float) and not math.isfinite(value):
            return ["number is not finite"]
        if self.integer and not (isinstance(value, int) or float(value).is_integer()):
            return [f"expected an integer, got {value!r}"]
        if self.min is not None and value < self.min:
            return [f"{value} < min={self.min}"]
        if self.max is not None and value > self.max:
            return [f"{value} > max={self.max}"]
        return []

    def json_schema(self) -> dict[str, Any]:
        s: dict[str, Any] = {"type": "integer" if self.integer else "number"}
        if self.min is not None:
            s["minimum"] = self.min
        if self.max is not None:
            s["maximum"] = self.max
        return _with_desc(s, self)

    def example(self, rng: random.Random) -> float | int:
        lo = self.min if self.min is not None else 0
        hi = self.max if self.max is not None else lo + 100
        return rng.randint(int(lo), int(hi)) if self.integer else round(rng.uniform(lo, hi), 3)

    def to_torch(self, value: Any) -> torch.Tensor:
        return torch.tensor(float(self.normalize(value)))

    def preview(self, value: Any) -> str:
        v = self.normalize(value)
        return f"{v:.6g}" if isinstance(v, float) else str(v)

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        return torch.tensor([[float(self.normalize(v) or 0.0)] for v in values])

    def stats(self, values: Sequence[Any]) -> dict[str, Any]:
        xs = [float(self.normalize(v)) for v in values if self.normalize(v) is not None]
        if not xs:
            return {"n": 0}
        t = torch.tensor(xs)
        return {"n": len(xs), "mean": float(t.mean()), "std": float(t.std(unbiased=False)),
                "min": float(t.min()), "max": float(t.max())}

    def params(self) -> dict[str, Any]:
        return {"min": self.min, "max": self.max, "integer": self.integer}


class Tokens(Type):
    """A list of integer token ids in ``[0, vocab)``; ``ignore_index`` (-100) is allowed
    (label positions that do not count)."""

    kind = "tokens"

    def __init__(self, data: Any = _MISSING, *, vocab: int | None = None,
                 min_len: int | None = None, max_len: int | None = None,
                 ignore_index: int | None = -100, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.vocab, self.min_len, self.max_len = vocab, min_len, max_len
        self.ignore_index = ignore_index

    def normalize(self, value: Any) -> Any:
        value = self.unwrap(value)
        return value.tolist() if isinstance(value, torch.Tensor) else value

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        value = self.normalize(value)
        if not isinstance(value, (list, tuple)):
            return [f"expected a list of token ids, got {type(value).__name__}"]
        n = len(value)
        if self.min_len is not None and n < self.min_len:
            return [f"{n} tokens < min_len={self.min_len}"]
        if self.max_len is not None and n > self.max_len:
            return [f"{n} tokens > max_len={self.max_len}"]
        for v in value:
            if isinstance(v, bool) or not isinstance(v, int):
                return [f"token {v!r} is not an int"]
            if v == self.ignore_index:
                continue
            if v < 0 or (self.vocab is not None and v >= self.vocab):
                return [f"token id {v} outside [0, {self.vocab})"]
        return []

    def json_schema(self) -> dict[str, Any]:
        return _with_desc({"type": "array", "items": {"type": "integer"}}, self)

    def example(self, rng: random.Random) -> list[int]:
        n = self.min_len or 8
        return [rng.randrange(self.vocab or 64) for _ in range(n)]

    def to_torch(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(self.normalize(value), dtype=torch.long)

    def preview(self, value: Any) -> str:
        v = self.normalize(value)
        return f"{len(v)} tokens {v[:12]}{'…' if len(v) > 12 else ''}"

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        out = torch.zeros(len(values), 33)
        for i, v in enumerate(values):
            ids = torch.as_tensor(self.normalize(v), dtype=torch.long)
            ids = ids[ids >= 0]
            out[i, 0] = math.log1p(len(ids))
            if len(ids):
                out[i, 1:] = torch.bincount(ids % 32, minlength=32).float() / len(ids)
        return out

    def params(self) -> dict[str, Any]:
        return {"vocab": self.vocab, "min_len": self.min_len, "max_len": self.max_len,
                "ignore_index": self.ignore_index}


# ---------------------------------------------------------------- tensors


_DTYPES = {"float32": torch.float32, "float64": torch.float64, "float16": torch.float16,
           "int64": torch.int64, "int32": torch.int32, "int16": torch.int16,
           "int8": torch.int8, "uint8": torch.uint8, "bool": torch.bool}


def _dtype(d: Any) -> torch.dtype | None:
    if d is None or isinstance(d, torch.dtype):
        return d
    key = str(d).replace("torch.", "")
    if key not in _DTYPES:
        raise unknown("dtype", key, _DTYPES, stage="types")
    return _DTYPES[key]


def _numeric_features(rows: list[torch.Tensor]) -> torch.Tensor:
    from topologies.synth.dsp import numeric_features

    if not rows:
        return torch.zeros(0, 25)
    n = max(r.numel() for r in rows)
    n = max(2, min(n, 4096))
    mat = torch.zeros(len(rows), n)
    for i, r in enumerate(rows):
        f = torch.nan_to_num(r.reshape(-1).float())[:n]
        mat[i, : f.numel()] = f
    return numeric_features(mat)


class Tensor(Type):
    """A tensor of any dimension; ``shape`` may hold ``None`` for free dims."""

    kind = "tensor"
    blob = True

    def __init__(self, data: Any = _MISSING, *, shape: Sequence[int | None] | None = None,
                 dtype: Any = None, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.shape = None if shape is None else [None if s is None else int(s) for s in shape]
        self.dtype = None if dtype is None else str(_dtype(dtype)).replace("torch.", "")

    def normalize(self, value: Any) -> Any:
        t = _tensor(value, _dtype(self.dtype))
        return t.detach().cpu().contiguous()

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        if isinstance(value, Mapping) and ("$blob" in value or "$tensor" in value):
            return []
        try:
            t = _tensor(value)
        except Exception:  # noqa: BLE001
            return [f"expected a tensor, got {type(self.unwrap(value)).__name__}"]
        if self.shape is not None and (t.dim() != len(self.shape) or any(
                want is not None and want != got for want, got in zip(self.shape, t.shape))):
            return [f"shape {list(t.shape)} does not match {self.shape}"]
        if self.dtype is not None and str(t.dtype).replace("torch.", "") != self.dtype:
            return [f"dtype {str(t.dtype).replace('torch.', '')} != {self.dtype}"]
        return []

    def inline(self, value: Any) -> bool:
        return _tensor(value).numel() <= INLINE_NUMEL

    def encode(self, value: Any) -> tuple[bytes, str]:
        from topologies.synth.codecs import tensor_encode

        return tensor_encode(self.normalize(value)), "tensor"

    def decode(self, data: bytes, ref: Mapping[str, Any]) -> Any:
        from topologies.synth.codecs import tensor_decode

        return tensor_decode(data)

    def to_torch(self, value: Any) -> torch.Tensor:
        return _tensor(value)

    def preview(self, value: Any) -> str:
        t = _tensor(value)
        if t.numel() == 0:
            return f"{type(self).__name__}{list(t.shape)} empty"
        f = t.float()
        return (f"{type(self).__name__}{list(t.shape)} {str(t.dtype).replace('torch.', '')} "
                f"μ={float(f.mean()):.3g} σ={float(f.std(unbiased=False)):.3g} "
                f"[{float(f.min()):.3g}, {float(f.max()):.3g}] "
                f"{spark(f.flatten()[:256].tolist(), 16)}")

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        return _numeric_features([_tensor(v).float() for v in values])

    def stats(self, values: Sequence[Any]) -> dict[str, Any]:
        shapes = sorted({tuple(_tensor(v).shape) for v in values})
        return {"n": len(values), "shapes": [list(s) for s in shapes[:5]]}

    def params(self) -> dict[str, Any]:
        return {"shape": self.shape, "dtype": self.dtype}


class Audio(Tensor):
    """Audio: a float tensor ``[T]`` (mono) or ``[C, T]`` at sample rate ``sr``.

    ``synth.Audio(x, sr=16_000)`` wraps a value (the sample rate travels with it).
    Stored as float32 WAV (lossless; ``pcm16=True`` for 16-bit).
    """

    kind = "audio"

    def __init__(self, data: Any = _MISSING, *, sr: int | None = None,
                 channels: int | None = None, pcm16: bool = False, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.sr = None if sr is None else int(sr)
        self.channels = channels
        self.pcm16 = bool(pcm16)

    def rate(self, value: Any) -> int:
        sr = value.sr if isinstance(value, Audio) and value.sr else self.sr
        return int(sr or 16_000)

    def normalize(self, value: Any) -> Any:
        x = _tensor(value).detach().cpu().float().contiguous()
        return self.wrap(x) if isinstance(value, Audio) else x

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        if isinstance(value, Mapping) and "$blob" in value:
            return []
        try:
            t = _tensor(value)
        except Exception:  # noqa: BLE001
            return [f"expected audio (a tensor), got {type(self.unwrap(value)).__name__}"]
        if t.dim() not in (1, 2):
            return [f"audio must be [T] or [C, T]; got shape {list(t.shape)}"]
        if self.channels is not None and (1 if t.dim() == 1 else t.shape[0]) != self.channels:
            return [f"audio has {1 if t.dim() == 1 else t.shape[0]} channels != {self.channels}"]
        if not bool(torch.isfinite(t).all()):
            return ["audio has non-finite samples"]
        return []

    def inline(self, value: Any) -> bool:
        return False

    def encode(self, value: Any) -> tuple[bytes, str]:
        from topologies.synth.codecs import wav_encode

        return wav_encode(_tensor(value), self.rate(value), pcm16=self.pcm16), "wav"

    def decode(self, data: bytes, ref: Mapping[str, Any]) -> Any:
        from topologies.synth.codecs import wav_decode

        x, sr = wav_decode(data)
        return Audio(x, sr=sr)

    def meta(self) -> dict[str, Any]:
        return {"sr": self.sr, "channels": self.channels}

    def preview(self, value: Any) -> str:
        x = _tensor(value).float()
        sr = self.rate(value)
        mono = x if x.dim() == 1 else x.mean(0)
        n = mono.numel()
        env = mono.abs()
        blocks = env[: n - n % 24].view(24, -1).amax(1).tolist() if n >= 24 else env.tolist()
        return (f"Audio {n / sr:.2f}s @{sr}Hz ch{1 if x.dim() == 1 else x.shape[0]} "
                f"rms={float(mono.pow(2).mean().sqrt()):.3f} {spark(blocks)}")

    def preview_html(self, value: Any) -> str:
        data, _ = self.encode(value)
        b64 = base64.b64encode(data).decode("ascii")
        return (f"<audio controls preload='none' src='data:audio/wav;base64,{b64}'></audio>"
                f"<div class='m'>{_esc(self.preview(value))}</div>")

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        from topologies.synth.dsp import audio_features

        rows = []
        for v in values:
            x = _tensor(v).float()
            rows.append(audio_features(x if x.dim() == 1 else x.mean(0), self.rate(v)))
        return torch.stack(rows) if rows else torch.zeros(0, 74)

    def stats(self, values: Sequence[Any]) -> dict[str, Any]:
        durs = [_tensor(v).shape[-1] / self.rate(v) for v in values]
        return {"n": len(values), "mean_seconds": sum(durs) / max(1, len(durs))}

    def params(self) -> dict[str, Any]:
        return {"sr": self.sr, "channels": self.channels}


class Image(Tensor):
    """An image ``[C, H, W]`` float in ``[0, 1]`` (or uint8). Stored as 8-bit PNG; values are
    normalized to 8-bit levels so storage is exact."""

    kind = "image"

    def __init__(self, data: Any = _MISSING, *, channels: int | None = None,
                 height: int | None = None, width: int | None = None, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.channels, self.height, self.width = channels, height, width

    def normalize(self, value: Any) -> Any:
        from topologies.synth.codecs import to_uint8

        t = _tensor(value)
        if t.dim() not in (2, 3):
            return t
        return to_uint8(t).float() / 255.0

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        if isinstance(value, Mapping) and "$blob" in value:
            return []
        try:
            shape = tuple(_tensor(value).shape)
        except Exception:  # noqa: BLE001
            return ["image is not a tensor, nested list or blob"]
        if len(shape) == 2:
            shape = (1, *shape)
        if len(shape) != 3:
            return [f"image must be [C, H, W]; got shape {shape}"]
        for name, want, got in (("channels", self.channels, shape[0]),
                                ("height", self.height, shape[1]),
                                ("width", self.width, shape[2])):
            if want is not None and want != got:
                return [f"image {name} {got} != {want}"]
        return []

    def inline(self, value: Any) -> bool:
        return False

    def encode(self, value: Any) -> tuple[bytes, str]:
        t = _tensor(value)
        if t.dim() == 3 and t.shape[0] not in (1, 2, 3, 4):
            return Tensor.encode(self, t)
        from topologies.synth.codecs import png_encode

        return png_encode(t), "png"

    def decode(self, data: bytes, ref: Mapping[str, Any]) -> Any:
        if ref.get("type") == "tensor":
            return Tensor.decode(self, data, ref)
        from topologies.synth.codecs import png_decode

        return png_decode(data)

    def to_torch(self, value: Any) -> torch.Tensor:
        t = _tensor(value)
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        t = t.float()
        return t[None] if t.dim() == 2 else t

    def preview(self, value: Any) -> str:
        t = self.to_torch(value)
        return f"Image{list(t.shape)} μ={float(t.mean()):.2f} | " + " | ".join(
            thumbnail(t, 16)[:4])

    def preview_html(self, value: Any) -> str:
        data, ext = self.encode(value)
        if ext != "png":
            return f"<code>{_esc(self.preview(value))}</code>"
        b64 = base64.b64encode(data).decode("ascii")
        t = self.to_torch(value)
        scale = max(1, 96 // max(t.shape[-1], 1))
        return (f"<img class='px' width='{t.shape[-1] * scale}' "
                f"src='data:image/png;base64,{b64}'>")

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        rows = []
        for v in values:
            t = self.to_torch(v)
            c = t.shape[0]
            rgb = t[:3] if c >= 3 else t[:1].expand(3, *t.shape[1:])
            hist = torch.cat([torch.histc(rgb[k], bins=8, min=0, max=1) for k in range(3)])
            hist = hist / hist.sum().clamp_min(1)
            g = rgb.mean(0)
            gx = g[:, 1:] - g[:, :-1]
            gy = g[1:, :] - g[:-1, :]
            h_, w_ = min(gx.shape[0], gy.shape[0]), min(gx.shape[1], gy.shape[1])
            gx, gy = gx[:h_, :w_], gy[:h_, :w_]
            mag = (gx ** 2 + gy ** 2).sqrt()
            ang = torch.atan2(gy, gx)
            bins = ((ang + math.pi) / (2 * math.pi) * 8).long().clamp(0, 7)
            hog = torch.zeros(8).index_add_(0, bins.flatten(), mag.flatten())
            hog = hog / hog.sum().clamp_min(1e-8)
            small = torch.nn.functional.interpolate(g[None, None], size=(8, 8),
                                                    mode="area")[0, 0].flatten()
            rows.append(torch.cat([hist, hog, small, torch.tensor([float(mag.mean())])]))
        return torch.stack(rows) if rows else torch.zeros(0, 97)

    def params(self) -> dict[str, Any]:
        return {"channels": self.channels, "height": self.height, "width": self.width}


class Video(Tensor):
    """Frames ``[T, C, H, W]`` (stored as a tensor blob)."""

    kind = "video"

    def __init__(self, data: Any = _MISSING, *, frames: int | None = None,
                 channels: int | None = None, height: int | None = None,
                 width: int | None = None, fps: float | None = None, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.frames, self.channels, self.height, self.width = frames, channels, height, width
        self.fps = fps

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        if isinstance(value, Mapping) and "$blob" in value:
            return []
        t = _tensor(value)
        if t.dim() != 4:
            return [f"video must be [T, C, H, W]; got shape {list(t.shape)}"]
        for name, want, got in (("frames", self.frames, t.shape[0]),
                                ("channels", self.channels, t.shape[1]),
                                ("height", self.height, t.shape[2]),
                                ("width", self.width, t.shape[3])):
            if want is not None and want != got:
                return [f"video {name} {got} != {want}"]
        return []

    def inline(self, value: Any) -> bool:
        return False

    def preview(self, value: Any) -> str:
        t = _tensor(value).float()
        motion = float((t[1:] - t[:-1]).abs().mean()) if t.shape[0] > 1 else 0.0
        return f"Video{list(t.shape)} motion={motion:.3f}"

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        img = Image()
        rows = []
        for v in values:
            t = _tensor(v).float()
            f = img.features(list(t)).mean(0)
            motion = (t[1:] - t[:-1]).abs().mean() if t.shape[0] > 1 else torch.tensor(0.0)
            rows.append(torch.cat([f, motion[None]]))
        return torch.stack(rows) if rows else torch.zeros(0, 98)

    def params(self) -> dict[str, Any]:
        return {"frames": self.frames, "channels": self.channels, "height": self.height,
                "width": self.width, "fps": self.fps}


class Series(Type):
    """A numeric time series: ``[T]`` floats or ``[T][channels]`` (``None`` = missing).
    Tensors are accepted and stored as lists (NaN -> ``None``)."""

    kind = "series"

    def __init__(self, data: Any = _MISSING, *, length: int | None = None,
                 channels: int | None = None, allow_missing: bool = True, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.length, self.channels, self.allow_missing = length, channels, bool(allow_missing)

    def normalize(self, value: Any) -> Any:
        value = self.unwrap(value)
        if isinstance(value, torch.Tensor):
            rows = value.tolist()

            def fix(v: Any) -> Any:
                if isinstance(v, list):
                    return [fix(x) for x in v]
                return None if isinstance(v, float) and math.isnan(v) else v
            return fix(rows)
        return value

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        value = self.normalize(value)
        if not isinstance(value, (list, tuple)):
            return [f"expected a series (list), got {type(value).__name__}"]
        if self.length is not None and len(value) != self.length:
            return [f"series length {len(value)} != {self.length}"]
        for row in value[:4]:
            vals = row if isinstance(row, (list, tuple)) else [row]
            if self.channels is not None and isinstance(row, (list, tuple)) and \
                    len(row) != self.channels:
                return [f"series has {len(row)} channels != {self.channels}"]
            for v in vals:
                if v is None and self.allow_missing:
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return [f"series value {v!r} is not numeric"]
        return []

    def json_schema(self) -> dict[str, Any]:
        return _with_desc({"type": "array", "items": {"type": ["number", "array", "null"]}}, self)

    def example(self, rng: random.Random) -> list[float]:
        return [round(rng.gauss(0, 1), 3) for _ in range(self.length or 16)]

    def to_torch(self, value: Any) -> torch.Tensor:
        v = self.normalize(value)
        t = torch.tensor([[float("nan") if x is None else x for x in
                           (step if isinstance(step, list) else [step])] for step in v],
                         dtype=torch.float32)
        t = torch.nan_to_num(t)
        return t[:, 0] if t.shape[1] == 1 else t

    def preview(self, value: Any) -> str:
        t = self.to_torch(value)
        mono = t if t.dim() == 1 else t[:, 0]
        return f"Series[{list(t.shape)}] {spark(mono.tolist(), 32)}"

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        return _numeric_features([self.to_torch(v) for v in values])

    def params(self) -> dict[str, Any]:
        return {"length": self.length, "channels": self.channels,
                "allow_missing": self.allow_missing}


class Mask(Tensor):
    """A segmentation mask ``[H, W]`` of integer classes (``classes`` optional names)."""

    kind = "mask"

    def __init__(self, data: Any = _MISSING, *, classes: Sequence[str] | None = None,
                 **kw: Any) -> None:
        super().__init__(data, **kw)
        self.classes = list(classes) if classes is not None else None

    def normalize(self, value: Any) -> Any:
        t = _tensor(value)
        small = t.dtype == torch.bool or not t.numel() or (
            not t.is_floating_point() and int(t.min()) >= 0 and int(t.max()) < 256)
        return (t.to(torch.uint8) if small else t.long()).contiguous()

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        if isinstance(value, Mapping) and "$blob" in value:
            return []
        t = _tensor(value)
        if t.dim() != 2:
            return [f"mask must be [H, W]; got {list(t.shape)}"]
        if t.is_floating_point():
            return ["mask values must be integers or booleans"]
        if self.classes is not None and t.numel() and int(t.max()) >= len(self.classes):
            return [f"mask class {int(t.max())} >= {len(self.classes)} classes"]
        return []

    def to_torch(self, value: Any) -> torch.Tensor:
        return _tensor(value).long()

    def preview(self, value: Any) -> str:
        t = _tensor(value).long()
        return f"Mask{list(t.shape)} classes={sorted(set(t.unique().tolist()))[:8]}"

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        k = len(self.classes) if self.classes else 8
        rows = []
        for v in values:
            t = _tensor(v).long().flatten() % k
            rows.append(torch.bincount(t, minlength=k).float() / max(1, t.numel()))
        return torch.stack(rows) if rows else torch.zeros(0, k)

    def params(self) -> dict[str, Any]:
        return {"classes": self.classes}


class PointCloud(Tensor):
    """Points ``[N, D]`` (``D`` = 3 by default; extra columns are features)."""

    kind = "pointcloud"

    def __init__(self, data: Any = _MISSING, *, dims: int = 3, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.dims = int(dims)

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        if isinstance(value, Mapping) and ("$blob" in value or "$tensor" in value):
            return []
        t = _tensor(value)
        if t.dim() != 2 or t.shape[1] < self.dims:
            return [f"point cloud must be [N, >={self.dims}]; got {list(t.shape)}"]
        return []

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        rows = []
        for v in values:
            p = _tensor(v).float()[:, : self.dims]
            c = p - p.mean(0)
            r = c.norm(dim=1)
            hist = torch.histc(r / r.max().clamp_min(1e-8), bins=8, min=0, max=1) / max(1, len(r))
            cov = (c.t() @ c / max(1, len(c) - 1)).flatten()
            rows.append(torch.cat([p.mean(0), p.std(0, unbiased=False), cov, hist,
                                   torch.tensor([math.log1p(len(p))])]))
        return torch.stack(rows) if rows else torch.zeros(0, 2 * self.dims + self.dims ** 2 + 9)

    def preview(self, value: Any) -> str:
        p = _tensor(value).float()
        return f"PointCloud[{p.shape[0]} x {p.shape[1]}] extent={float((p.amax(0) - p.amin(0)).max()):.3g}"

    def params(self) -> dict[str, Any]:
        return {"dims": self.dims}


# ---------------------------------------------------------------- structured


class Boxes(Type):
    """Bounding boxes ``[[x0, y0, x1, y1], ...]`` (pixels; ``format="xyxy"``)."""

    kind = "boxes"

    def __init__(self, data: Any = _MISSING, *, format: str = "xyxy", **kw: Any) -> None:
        super().__init__(data, **kw)
        self.format = format

    def normalize(self, value: Any) -> Any:
        v = self.unwrap(value)
        return v.tolist() if isinstance(v, torch.Tensor) else [list(b) for b in v]

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        boxes = self.normalize(value)
        for i, b in enumerate(boxes):
            if len(b) != 4:
                return [f"box {i} has {len(b)} numbers, not 4"]
            if self.format == "xyxy" and (b[2] < b[0] or b[3] < b[1]):
                return [f"box {i} {b} has x1 < x0 or y1 < y0"]
        return []

    def to_torch(self, value: Any) -> torch.Tensor:
        b = self.normalize(value)
        return torch.tensor(b, dtype=torch.float32).reshape(-1, 4)

    def preview(self, value: Any) -> str:
        b = self.normalize(value)
        return f"{len(b)} boxes {b[:3]}{'…' if len(b) > 3 else ''}"

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        rows = []
        for v in values:
            b = self.to_torch(v)
            if len(b):
                wh = (b[:, 2:] - b[:, :2]).clamp_min(0)
                rows.append(torch.cat([torch.tensor([math.log1p(len(b))]), wh.mean(0),
                                       b[:, :2].mean(0)]))
            else:
                rows.append(torch.zeros(5))
        return torch.stack(rows) if rows else torch.zeros(0, 5)

    def params(self) -> dict[str, Any]:
        return {"format": self.format}


class Keypoints(Type):
    """Keypoints ``[[x, y], ...]`` or ``[[x, y, visible], ...]`` (optionally ``names``)."""

    kind = "keypoints"

    def __init__(self, data: Any = _MISSING, *, names: Sequence[str] | None = None,
                 **kw: Any) -> None:
        super().__init__(data, **kw)
        self.names = list(names) if names is not None else None

    def normalize(self, value: Any) -> Any:
        v = self.unwrap(value)
        return v.tolist() if isinstance(v, torch.Tensor) else [list(p) for p in v]

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        pts = self.normalize(value)
        if self.names is not None and len(pts) != len(self.names):
            return [f"{len(pts)} keypoints != {len(self.names)} names"]
        for i, p in enumerate(pts):
            if len(p) not in (2, 3):
                return [f"keypoint {i} must be [x, y] or [x, y, v]"]
        return []

    def to_torch(self, value: Any) -> torch.Tensor:
        return torch.tensor(self.normalize(value), dtype=torch.float32)

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        rows = []
        for v in values:
            p = self.to_torch(v)
            rows.append(torch.cat([p[:, :2].mean(0), p[:, :2].std(0, unbiased=False)])
                        if len(p) else torch.zeros(4))
        return torch.stack(rows) if rows else torch.zeros(0, 4)

    def params(self) -> dict[str, Any]:
        return {"names": self.names}


class Graph(Type):
    """A graph: ``num_nodes``, ``edges`` ``[[u, v], ...]``, optional node features ``x``,
    node labels ``y`` and ``edge_attr``. ``synth.Graph(num_nodes=5, edges=[[0, 1]])`` is a
    value (as is a dict with those keys). Stored as JSON (a blob when large)."""

    kind = "graph"
    blob = True

    def __init__(self, data: Any = _MISSING, *, num_nodes: int | None = None,
                 edges: Any = None, x: Any = None, y: Any = None, edge_attr: Any = None,
                 directed: bool = False, **kw: Any) -> None:
        if data is _MISSING and (num_nodes is not None or edges is not None):
            data = {"num_nodes": num_nodes, "edges": edges, "x": x, "y": y,
                    "edge_attr": edge_attr}
        super().__init__(data, **kw)
        self.directed = bool(directed)

    @staticmethod
    def parts(value: Any) -> dict[str, Any]:
        d = Type.unwrap(value)
        if not isinstance(d, Mapping):
            raise SynthError(f"A graph value is synth.Graph(...) or a dict; got "
                             f"{type(d).__name__}.", stage="types")
        edges = d.get("edges")
        edges = edges.tolist() if isinstance(edges, torch.Tensor) else [list(e) for e in
                                                                        (edges or [])]
        if edges and len(edges) == 2 and len(edges[0]) != 2:  # edge_index [2, E]
            edges = [list(e) for e in zip(edges[0], edges[1])]
        n = d.get("num_nodes")
        if n is None:
            n = 1 + max((max(e) for e in edges), default=-1)
        return {"num_nodes": int(n), "edges": edges,
                "directed": bool(d.get("directed", getattr(value, "directed", False))),
                "x": d.get("x"), "y": d.get("y"),
                "edge_attr": d.get("edge_attr")}

    def normalize(self, value: Any) -> Any:
        p = self.parts(value)
        p["directed"] = p["directed"] or self.directed
        return Graph(directed=p["directed"]).wrap(
            {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in p.items()})

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        if isinstance(value, Mapping) and "$blob" in value:
            return []
        try:
            p = self.parts(value)
        except SynthError as e:
            return [e.message]
        n = p["num_nodes"]
        for i, e in enumerate(p["edges"]):
            if len(e) != 2 or not all(isinstance(k, int) and 0 <= k < n for k in e):
                return [f"edge {i} {e} is not a pair of node ids < {n}"]
        return []

    def inline(self, value: Any) -> bool:
        return len(self.parts(value)["edges"]) <= 2000

    def encode(self, value: Any) -> tuple[bytes, str]:
        data = json.dumps(to_jsonable(self.parts(self.normalize(value))), sort_keys=True,
                          separators=(",", ":")).encode()
        return data, "json"

    def decode(self, data: bytes, ref: Mapping[str, Any]) -> Any:
        payload = from_jsonable(json.loads(data))
        return Graph(directed=bool(payload.get("directed", ref.get("directed", False)))).wrap(
            payload)

    def to_json(self, value: Any) -> Any:
        return {"$type": "graph", "meta": {"directed": self.parts(value)["directed"] or
                                          self.directed},
                "data": to_jsonable(self.parts(value))}

    def meta(self) -> dict[str, Any]:
        return {"directed": self.directed}

    def to_torch(self, value: Any) -> dict[str, torch.Tensor]:
        p = self.parts(value)
        e = torch.tensor(p["edges"], dtype=torch.long).reshape(-1, 2).t()
        if not (p["directed"] or self.directed) and e.numel():
            e = torch.cat([e, e.flip(0)], 1)
        out = {"edge_index": e, "num_nodes": torch.tensor(p["num_nodes"])}
        if p["x"] is not None:
            out["x"] = torch.as_tensor(p["x"], dtype=torch.float32)
        if p["y"] is not None:
            out["y"] = torch.as_tensor(p["y"])
        return out

    def degree(self, value: Any) -> torch.Tensor:
        p = self.parts(value)
        deg = torch.zeros(p["num_nodes"])
        for u, v in p["edges"]:  # total degree (in + out for directed graphs)
            deg[u] += 1
            deg[v] += 1
        return deg

    def components(self, value: Any) -> list[int]:
        p = self.parts(value)
        parent = list(range(p["num_nodes"]))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for u, v in p["edges"]:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
        from collections import Counter

        return sorted(Counter(find(i) for i in range(p["num_nodes"])).values(), reverse=True)

    def clustering(self, value: Any) -> float:
        p = self.parts(value)
        nb: list[set[int]] = [set() for _ in range(p["num_nodes"])]
        for u, v in p["edges"]:
            if u != v:
                nb[u].add(v)
                nb[v].add(u)
        vals = []
        for i, s in enumerate(nb):
            k = len(s)
            if k < 2:
                continue
            links = sum(1 for a in s for b in s if a < b and b in nb[a])
            vals.append(2 * links / (k * (k - 1)))
            if len(vals) >= 256 and i > 512:
                break
        return sum(vals) / len(vals) if vals else 0.0

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        rows = []
        for v in values:
            p = self.parts(v)
            n = max(1, p["num_nodes"])
            deg = self.degree(v)
            hist = torch.histc(deg, bins=8, min=0, max=float(max(1.0, float(deg.max()))))
            hist = hist / n
            comp = self.components(v)
            m = len(p["edges"])
            rows.append(torch.cat([hist, torch.tensor([
                math.log1p(n), math.log1p(m), 2 * m / (n * max(1, n - 1)),
                float(deg.mean()), float(deg.std(unbiased=False)) if n > 1 else 0.0,
                self.clustering(v), len(comp) / n, comp[0] / n if comp else 0.0])]))
        return torch.stack(rows) if rows else torch.zeros(0, 16)

    def preview(self, value: Any) -> str:
        p = self.parts(value)
        comp = self.components(value)
        return (f"Graph n={p['num_nodes']} e={len(p['edges'])} components={len(comp)} "
                f"mean degree={float(self.degree(value).mean()):.2f}")

    def preview_html(self, value: Any) -> str:
        p = self.parts(value)
        n = p["num_nodes"]
        if n > 200:
            return f"<code>{_esc(self.preview(value))}</code>"
        pos = [(60 + 50 * math.cos(2 * math.pi * i / max(1, n)),
                60 + 50 * math.sin(2 * math.pi * i / max(1, n))) for i in range(n)]
        lines = "".join(f"<line x1='{pos[u][0]:.1f}' y1='{pos[u][1]:.1f}' x2='{pos[v][0]:.1f}' "
                        f"y2='{pos[v][1]:.1f}'/>" for u, v in p["edges"][:2000])
        dots = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.5'/>" for x, y in pos)
        return (f"<svg class='g' viewBox='0 0 120 120' width='120' height='120'>"
                f"<g stroke='currentColor' stroke-opacity='.4'>{lines}</g>"
                f"<g fill='currentColor'>{dots}</g></svg>"
                f"<div class='m'>{_esc(self.preview(value))}</div>")

    def params(self) -> dict[str, Any]:
        return {"directed": self.directed}


class Spans(Type):
    """Entity spans ``[{"start", "end", "label"}]`` into ``record[text_field]``.

    ``must_align=True`` checks offsets are in range (and match ``span["text"]``
    when present); ``labels`` restricts the entity types.
    """

    kind = "spans"

    def __init__(self, data: Any = _MISSING, *, text_field: str = "text",
                 labels: Sequence[str] | None = None, must_align: bool = True,
                 allow_overlap: bool = False, **kw: Any) -> None:
        super().__init__(data, **kw)
        self.text_field = text_field
        self.labels = list(labels) if labels is not None else None
        self.must_align, self.allow_overlap = bool(must_align), bool(allow_overlap)

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        value = self.unwrap(value)
        if not isinstance(value, (list, tuple)):
            return [f"expected a list of spans, got {type(value).__name__}"]
        text = record.get(self.text_field) if record is not None else None
        prev_end = -1
        for i, s in enumerate(sorted(value, key=lambda s: (s.get("start", 0), s.get("end", 0))
                                     if isinstance(s, Mapping) else (0, 0))):
            if not isinstance(s, Mapping) or not {"start", "end", "label"} <= set(s):
                return [f"span {i} needs start, end and label"]
            a, b = s["start"], s["end"]
            if not (isinstance(a, int) and isinstance(b, int)) or a < 0 or b <= a:
                return [f"span {i} has invalid offsets ({a}, {b})"]
            if self.labels is not None and s["label"] not in self.labels:
                return [f"span {i} label {s['label']!r} not in {self.labels}"]
            if not self.allow_overlap and a < prev_end:
                return [f"span {i} overlaps the previous span"]
            prev_end = max(prev_end, b)
            if self.must_align and isinstance(text, str):
                if b > len(text):
                    return [f"span {i} end {b} > text length {len(text)}"]
                if "text" in s and text[a:b] != s["text"]:
                    return [f"span {i} text {s['text']!r} != text[{a}:{b}] {text[a:b]!r}"]
        return []

    def json_schema(self) -> dict[str, Any]:
        label = {"type": "string"} if self.labels is None else {"type": "string",
                                                                 "enum": self.labels}
        return _with_desc({"type": "array", "items": {
            "type": "object", "properties": {"start": {"type": "integer"},
                                             "end": {"type": "integer"}, "label": label,
                                             "text": {"type": "string"}},
            "required": ["start", "end", "label"]}}, self)

    def example(self, rng: random.Random) -> list[dict]:
        return [{"start": 0, "end": 4, "label": (self.labels or ["ENT"])[0]}]

    def to_torch(self, value: Any) -> torch.Tensor:
        v = self.unwrap(value)
        return torch.tensor([[s["start"], s["end"]] for s in v], dtype=torch.long).reshape(-1, 2)

    def preview(self, value: Any) -> str:
        v = self.unwrap(value)
        return f"{len(v)} spans " + ", ".join(
            f"{s.get('label')}{repr(s.get('text', ''))[:20] if s.get('text') else ''}"
            for s in v[:4])

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        names = self.labels or []
        rows = []
        for v in values:
            spans = self.unwrap(v) or []
            row = [math.log1p(len(spans))]
            row += [sum(1 for s in spans if s.get("label") == n) / max(1, len(spans))
                    for n in names]
            rows.append(torch.tensor(row))
        return torch.stack(rows) if rows else torch.zeros(0, 1 + len(names))

    def params(self) -> dict[str, Any]:
        return {"text_field": self.text_field, "labels": self.labels,
                "must_align": self.must_align, "allow_overlap": self.allow_overlap}


class Json(Type):
    """Structured data validated by a pydantic model or a JSON-schema dict.

    ``Json(model=Invoice)`` / ``Json(schema={"type": "object", ...})`` / ``Json()`` (any).
    """

    kind = "json"

    def __init__(self, model: Any = None, *, schema: dict | None = None, data: Any = _MISSING,
                 **kw: Any) -> None:
        super().__init__(data, **kw)
        if isinstance(model, Mapping) and schema is None:
            model, schema = None, dict(model)
        self.model = model
        self._schema = dict(schema) if schema is not None else None
        if model is not None and not hasattr(model, "model_validate"):
            raise SynthError(f"Json(model=...) must be a pydantic model class; got {model!r}.",
                             stage="schema", hint="Json(model=MyModel) or Json(schema={...}).")

    def coerce(self, value: Any) -> Any:
        if isinstance(value, str):
            from topologies.synth.jsonutil import extract_json

            try:
                return extract_json(value)
            except Exception:  # noqa: BLE001
                return value
        if self.model is not None and hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    def validate(self, value: Any, record: Mapping[str, Any] | None = None) -> list[str]:
        value = self.unwrap(value)
        if self.model is not None:
            try:
                self.model.model_validate(value)
            except Exception as e:  # noqa: BLE001
                return [_short_validation(e)]
            return []
        if self._schema is not None:
            from topologies.synth.jsonutil import validate_json_schema

            return validate_json_schema(value, self._schema)
        return []

    def json_schema(self) -> dict[str, Any]:
        if self.model is not None:
            return _with_desc(dict(self.model.model_json_schema()), self)
        return _with_desc(dict(self._schema or {}), self)

    def example(self, rng: random.Random) -> Any:
        from topologies.synth.jsonutil import example_from_schema

        return example_from_schema(self.json_schema(), rng)

    def to_torch(self, value: Any) -> Any:
        raise SynthError("Json needs a text rendering + tokenizer to become a tensor.",
                         stage="types", hint="Use the seq2seq-lm recipe (canonical JSON text).")

    def preview(self, value: Any) -> str:
        return _short(json.dumps(self.unwrap(value), ensure_ascii=False, default=str), 120)

    def features(self, values: Sequence[Any]) -> torch.Tensor:
        from topologies.synth.quality.embed import HashedEmbedder

        return HashedEmbedder(512).encode([json.dumps(self.unwrap(v), sort_keys=True,
                                                      default=str) for v in values])

    def params(self) -> dict[str, Any]:
        if self.model is not None:
            from topologies.experiments.spec import canonical

            return {"model": canonical(self.model, warn=False),
                    "schema": self.model.model_json_schema()}
        return {"schema": self._schema}


def _short_validation(e: Exception) -> str:
    try:
        errs = e.errors()  # type: ignore[attr-defined]
        parts = [f"{'.'.join(str(x) for x in er.get('loc', ()))}: {er.get('msg')}"
                 for er in errs[:3]]
        return "; ".join(parts) + (f" (+{len(errs) - 3} more)" if len(errs) > 3 else "")
    except Exception:  # noqa: BLE001
        return str(e).splitlines()[0][:300]


for _cls in (Text, Category, Label, Labels, Number, Tokens, Tensor, Audio, Image, Video, Series,
             Mask, PointCloud, Boxes, Keypoints, Graph, Spans, Json):
    register_type(_cls)


# ---------------------------------------------------------------- normalization / inference


def type_class(kind: str) -> type[Type]:
    if kind not in TYPES:
        raise unknown("value type", kind, TYPES, stage="types",
                      extra="Register new ones with synth.register_type(MyType).")
    return TYPES[kind]


def as_type(t: Any) -> Type | None:
    """A type spec from a class, an instance or a wrapped value (``None`` stays ``None``)."""
    if t is None:
        return None
    if isinstance(t, type) and issubclass(t, Type):
        try:
            return t()
        except SynthError:
            return None
    if isinstance(t, Type):
        return t.spec()
    raise SynthError(f"{t!r} is not a value type.", stage="types",
                     hint="Use synth.Text, synth.Audio(sr=...), synth.Tensor(shape=...), ...")


def infer_type(value: Any) -> Type:
    """The type of a plain value (wrapped values carry theirs)."""
    if isinstance(value, Type) and value.is_value:
        return value.spec()
    if isinstance(value, bool):
        return Category(["False", "True"])
    if isinstance(value, (int, float)):
        return Number()
    if isinstance(value, str):
        return Text()
    if isinstance(value, torch.Tensor):
        return Number() if value.dim() == 0 else Tensor()
    if isinstance(value, (list, tuple)) and value:
        if all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            return Tokens()
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) or x is None
               for x in value):
            return Series()
    return Json()


def normalize_record(rec: Mapping[str, Any], schema: Any = None) -> dict[str, Any]:
    """Canonical field values (``Type.normalize``) for storage and streams."""
    out = {}
    for k, v in rec.items():
        t = schema.fields.get(k) if schema is not None and k in schema else None
        if isinstance(v, Type) and v.is_value:
            out[k] = v.normalize(v)
        elif t is not None:
            out[k] = t.normalize(v)
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().contiguous() if v.dim() else v.item()
        else:
            out[k] = v
    return out


def encode_value(value: Any, t: Type | None, put_blob: Callable[[bytes, str], dict] | None
                 ) -> Any:
    """Storage form of one field value: inline JSON or a ``{"$blob", "type", "kind"}`` ref."""
    if isinstance(value, Mapping) and "$blob" in value:
        return dict(value)
    if isinstance(value, Type) and value.is_value:
        t = value
    if t is None:
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            t = Image() if value.dim() == 3 and value.shape[0] in (1, 3, 4) and \
                value.is_floating_point() else Tensor()
        elif isinstance(value, torch.Tensor):
            t = Tensor()
        else:
            return to_jsonable(value)
    inline = getattr(t, "inline", None)
    if not t.blob or put_blob is None or (callable(inline) and inline(value)):
        return t.to_json(value)
    data, ext = t.encode(value)
    ref = put_blob(data, ext)
    meta = {k: v for k, v in (value.meta() if isinstance(value, Type) and value.is_value
                              else t.meta()).items() if v is not None}
    return {**ref, "kind": t.kind, **meta}


def decode_value(value: Any, load: Callable[[Mapping[str, Any]], bytes] | None) -> Any:
    """Live value of one stored field value (blobs read through ``load(ref)``)."""
    if isinstance(value, Mapping) and "$blob" in value:
        if load is None:
            return dict(value)
        kind = value.get("kind")
        data = load(value)
        if kind is None:  # older refs: by extension
            kind = {"png": "image", "wav": "audio", "tensor": "tensor", "json": "graph"}.get(
                value.get("type"), "tensor")
            if value.get("type") == "pt":
                return torch.load(__import__("io").BytesIO(data), map_location="cpu",
                                  weights_only=True)
        cls = type_class(kind)
        meta = {k: v for k, v in value.items() if k not in ("$blob", "type", "kind")}
        try:
            t = cls(**meta)
        except TypeError:
            t = cls()
        return t.decode(data, value)
    return from_jsonable(value)


__all__ = [
    "INLINE_NUMEL",
    "TYPES",
    "Audio",
    "Boxes",
    "Category",
    "Field",
    "Graph",
    "Image",
    "Json",
    "Keypoints",
    "Label",
    "Labels",
    "Mask",
    "Number",
    "PointCloud",
    "Series",
    "Spans",
    "Tensor",
    "Text",
    "Tokens",
    "Type",
    "Video",
    "as_type",
    "decode_value",
    "encode_value",
    "from_jsonable",
    "infer_type",
    "normalize_record",
    "register_type",
    "spark",
    "thumbnail",
    "to_jsonable",
    "type_class",
]
