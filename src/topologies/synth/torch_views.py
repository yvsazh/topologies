"""Synthetic records -> torch samples in a recipe's batch format.

A *converter* is fitted once on sample records (tokenizer training, class lists,
column layouts) and then converts one record at a time, so the same code feeds
materialized datasets and on-the-fly streams:

    conv = make_converter("text-classify", tokenizer="char", max_len=64).fit(records[:2000])
    sample = conv(records[0])            # {"tokens": LongTensor[T], "labels": LongTensor[]}
    ds = build_dataset(records, "text-classify", tokenizer="char")   # fit + convert all

``ds.for_recipe(...)`` and ``stream.for_recipe(...)`` use the same converters.
The recipe's ``expected_batch_spec()`` decides the keys; value types decide how
fields feed them (``Label`` -> class index, ``Text`` -> token ids, ``Tokens`` -> ids,
``Image`` -> pixels, ``Audio`` / ``Series`` / ``Tensor`` -> float tensors, ``Json`` ->
canonical JSON text). Recipes without a dedicated converter map record fields
named like their batch keys. New converters register with ``register_converter``.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from topologies.synth.errors import SynthError

CONVERTERS: dict[str, type[Converter]] = {}
_BUILDERS: dict[str, Callable[..., Dataset]] = {}


def register_converter(*recipes: str) -> Callable[[type[Converter]], type[Converter]]:
    """Register a ``Converter`` subclass for recipe names."""
    def deco(cls: type[Converter]) -> type[Converter]:
        for r in recipes:
            CONVERTERS[r] = cls
        return cls

    return deco


def register_builder(recipe: str) -> Callable[[Callable[..., Dataset]], Callable[..., Dataset]]:
    """Register ``fn(records, schema=..., **kw) -> Dataset`` (whole-dataset builders; streams
    need a ``Converter``)."""
    def deco(fn: Callable[..., Dataset]) -> Callable[..., Dataset]:
        _BUILDERS[recipe] = fn
        return fn

    return deco


class ListDataset(Dataset):
    """In-memory dict samples (+ ``collate_fn``, ``classes``, ``targets``, ``tokenizer``)."""

    def __init__(self, samples: list[dict[str, Any]], *, collate_fn: Any = None,
                 classes: Sequence[str] | None = None, tokenizer: Any = None,
                 label_key: str | None = "labels", info: dict[str, Any] | None = None) -> None:
        if not samples:
            raise SynthError("No records to build a torch dataset from.", stage="torch",
                             hint="Check the split name and that the dataset is not empty.")
        self.samples = samples
        if collate_fn is not None:
            self.collate_fn = collate_fn
        self.classes = list(classes) if classes is not None else None
        self.tokenizer = tokenizer
        self.info = dict(info or {})
        self._label_key = label_key

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.samples[i]

    @property
    def targets(self) -> list[int] | None:
        """Integer class labels (for stratified splits / ``dataset_labels``)."""
        k = self._label_key
        if k is None or not self.samples or k not in self.samples[0]:
            return None
        first = self.samples[0][k]
        if not (isinstance(first, torch.Tensor) and first.dim() == 0):
            return None
        return [int(s[k]) for s in self.samples]


# ---------------------------------------------------------------- helpers


def resolve_tokenizer(tokenizer: Any, texts: Sequence[str]) -> Any:
    """A tokenizer instance, or one trained on ``texts`` from a spec (``"bpe:4k"``)."""
    if tokenizer is None:
        raise SynthError("This recipe needs a tokenizer for text fields.", stage="torch",
                         hint="for_recipe(..., tokenizer=tok) or tokenizer='bpe:4k' / 'char' "
                              "(trained on the records).")
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "vocab_size"):
        return tokenizer
    if isinstance(tokenizer, str):
        from topologies.experiments.routing import is_tokenizer_path, parse_tokenizer_spec

        if is_tokenizer_path(tokenizer):
            from topologies.tokenizers import load_tokenizer

            return load_tokenizer(tokenizer)
        spec = parse_tokenizer_spec(tokenizer)
        if spec is None:
            raise SynthError(f"Unknown tokenizer spec {tokenizer!r}.", stage="torch",
                             hint="Use 'char', 'word:8k', 'bpe:4k' or a Tokenizer instance.")
        from topologies.tokenizers import train_tokenizer

        kind = spec.pop("kind")
        if kind == "char":
            spec.pop("vocab_size", None)
        return train_tokenizer(list(texts), kind=kind, **spec)
    raise SynthError(f"Invalid tokenizer {tokenizer!r}.", stage="torch")


def encode(tok: Any, text: str, max_len: int | None = None, *, eos: bool = False,
           bos: bool = False) -> list[int]:
    ids = list(tok.encode(text))
    if bos and getattr(tok, "bos_id", None) is not None:
        ids = [int(tok.bos_id)] + ids
    if eos and getattr(tok, "eos_id", None) is not None:
        ids = ids + [int(tok.eos_id)]
    if max_len is not None:
        ids = ids[:max_len]
    return ids or [int(getattr(tok, "pad_id", 0) or 0)]


def pad_id(tok: Any) -> int:
    p = getattr(tok, "pad_id", None) if tok is not None else None
    return int(p) if p is not None else 0


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def label_field(records: Sequence[Mapping[str, Any]], schema: Any, given: str | None) -> str:
    if given:
        return given
    if schema is not None:
        lf = schema.label_field()
        if lf:
            return lf
    for k in ("label", "labels", "intent", "class", "target", "y"):
        if records and k in records[0]:
            return k
    raise SynthError("Cannot find the label field.", stage="torch",
                     hint="for_recipe(..., label_field='intent') or use a Label field in "
                          "the schema.")


def class_names(records: Sequence[Mapping[str, Any]], schema: Any, field: str,
                given: Sequence[str] | None) -> list[str]:
    if given is not None:
        return [str(x) for x in given]
    if schema is not None and field in schema and hasattr(schema[field], "names"):
        return list(schema[field].names)
    vals = {r.get(field) for r in records}
    if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return [str(i) for i in range(max(vals) + 1)]
    return sorted(str(v) for v in vals)


def label_index(value: Any, names: Sequence[str]) -> int:
    if isinstance(value, bool):
        value = str(value)
    if isinstance(value, int) and str(value) not in names:
        return int(value)
    try:
        return list(names).index(str(value))
    except ValueError:
        raise SynthError(f"Label {value!r} is not one of {list(names)[:20]}.", stage="torch",
                         hint="Pass labels=[...] with every class name (a stream fits its "
                              "class list on a sample; declare a Label type to fix it).") from None


def text_field(records: Sequence[Mapping[str, Any]], schema: Any, given: str | None,
               *, exclude: Sequence[str] = ()) -> str:
    if given:
        return given
    if schema is not None:
        texts = [t for t in schema.text_fields() if t not in exclude]
        if texts:
            return texts[0]
    for k in ("text", "input", "document", "content", "src", "source", "question"):
        if records and k in records[0] and k not in exclude:
            return k
    first = records[0] if records else {}
    for k, v in first.items():
        if isinstance(v, str) and not str(k).startswith("_") and k not in exclude:
            return k
    raise SynthError("Cannot find the text field.", stage="torch",
                     hint="for_recipe(..., text_field='utterance').")


def _tokens(v: Any) -> torch.Tensor:
    return torch.as_tensor(v, dtype=torch.long)


def _f(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def to_float_tensor(value: Any) -> torch.Tensor:
    """Audio / Series / Tensor / Image / nested lists -> a float tensor (NaN -> 0)."""
    from topologies.synth.valuetypes import Series, Type

    v = Type.unwrap(value)
    if isinstance(v, torch.Tensor):
        t = v.float()
    elif isinstance(v, list):
        t = Series().to_torch(v)
    else:
        t = torch.as_tensor(v, dtype=torch.float32)
    return torch.nan_to_num(t)


def load_image(value: Any, blob_loader: Callable[[dict], Any] | None = None) -> torch.Tensor:
    if isinstance(value, Mapping) and "$blob" in value:
        if blob_loader is None:
            raise SynthError("Image blobs need the dataset to load them.", stage="torch",
                             hint="Build from ds.for_recipe(...) rather than plain records.")
        value = blob_loader(dict(value))
    from topologies.synth.valuetypes import Type

    t = Type.unwrap(value)
    t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    t = t.float()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.numel() and float(t.max()) > 1.5:
        t = t / 255.0
    return t


def record_seed(record: Mapping[str, Any], seed: int = 0) -> int:
    """A stable per-record seed (record id or index), for random conversions (masking)."""
    key = record.get("_id", record.get("_index"))
    if key is None:
        key = json.dumps({k: v for k, v in record.items() if isinstance(v, (str, int, float))},
                         sort_keys=True)
    return zlib.crc32(f"{seed}:{key}".encode()) & 0x7FFFFFFF


def _first_key(records: Sequence[Mapping[str, Any]], names: Sequence[str],
               exclude: Sequence[str] = ()) -> str | None:
    first = records[0] if records else {}
    return next((n for n in names if n in first and n not in exclude), None)


def _kind_fields(records: Sequence[Mapping[str, Any]], schema: Any, *kinds: str) -> list[str]:
    if schema is not None:
        found = schema.of_kind(*kinds)
        if found:
            return found
    from topologies.synth.valuetypes import Type

    first = records[0] if records else {}
    out = []
    for k, v in first.items():
        if str(k).startswith("_"):
            continue
        kind = v.kind if isinstance(v, Type) and v.is_value else (
            v.get("kind") if isinstance(v, Mapping) and "$blob" in v else None)
        if kind in kinds:
            out.append(k)
    return out


# ---------------------------------------------------------------- converters


class Converter:
    """Fit on sample records, then convert one record at a time (see module docstring)."""

    recipe: str | None = None
    label_key: str | None = "labels"

    def __init__(self, *, schema: Any = None, decode: Callable[[dict], dict] | None = None,
                 **kw: Any) -> None:
        self.schema = schema
        self._decode = decode
        self.kw = dict(kw)
        self.tokenizer: Any = None
        self.classes: list[str] | None = None
        self.info: dict[str, Any] = {}

    # -- protocol ---------------------------------------------------------------------
    def decode(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Live values (blob refs / inline tensors decoded)."""
        if self._decode is not None:
            return self._decode(dict(record))
        from topologies.synth.valuetypes import from_jsonable

        return {k: from_jsonable(v) for k, v in record.items()}

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        return self

    def convert(self, record: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.convert(self.decode(record))

    @property
    def collate_fn(self) -> Any:
        from topologies.data.collate import PadCollator

        return PadCollator(pad_id=pad_id(self.tokenizer))

    def dataset(self, records: Sequence[Mapping[str, Any]]) -> Dataset:
        """A materialized dataset of converted records."""
        return ListDataset([self(r) for r in records], collate_fn=self.collate_fn,
                           classes=self.classes, tokenizer=self.tokenizer,
                           label_key=self.label_key, info=self.info)


class GenericConverter(Converter):
    """Schema-free tensors: int lists -> LongTensor, numbers -> tensors, others as they are."""

    def convert(self, record: Mapping[str, Any]) -> dict[str, Any]:
        from topologies.synth.valuetypes import Type

        s: dict[str, Any] = {}
        for k, v in record.items():
            if str(k).startswith("_"):
                continue
            v = Type.unwrap(v)
            if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
                s[k] = _tokens(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                s[k] = torch.tensor(v)
            else:
                s[k] = v
        return s


class BatchSpecConverter(Converter):
    """Fallback: record fields named like the recipe's batch keys."""

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        from topologies.data.adapters import recipe_keys

        self.keys = recipe_keys(self.recipe, include_optional=True)
        self.required = recipe_keys(self.recipe)
        return self

    def convert(self, record: Mapping[str, Any]) -> dict[str, Any]:
        from topologies.synth.valuetypes import Type

        s = {}
        for k in self.keys:
            if k in record:
                v = Type.unwrap(record[k])
                s[k] = _tokens(v) if isinstance(v, list) and v and isinstance(v[0], int) else \
                    (torch.as_tensor(v) if isinstance(v, (list, int, float)) else v)
        missing = [k for k in self.required if k not in s]
        if missing:
            raise SynthError(f"Records lack {missing} for recipe {self.recipe!r}.",
                             stage="torch",
                             hint="Name the record fields after the recipe's batch keys, or "
                                  "register a converter with synth.torch_views."
                                  "register_converter.")
        return s


class _TextMixin(Converter):
    def _texts(self, records: Sequence[Mapping[str, Any]], field: str) -> list[str]:
        return [as_text(self.decode(r).get(field, "")) for r in records]


@register_converter("causal-lm")
class CausalLM(_TextMixin):
    """``tokens`` (+ ``labels`` with -100 outside answers) records, or text -> token ids.

    Text: ``fields=("problem", "solution")`` (or ``template="{problem}\\n{solution}"``)
    joins several fields; materialized datasets pack documents into ``max_len`` blocks
    (``pack=True``, the default when ``max_len`` is set); streams yield one sequence per
    record.
    """

    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        kw = self.kw
        self.max_len = kw.get("max_len")
        first = self.decode(records[0]) if records else {}
        self.token_mode = "tokens" in first and isinstance(first["tokens"], list)
        if self.token_mode:
            self.info["vocab_size"] = 1 + max(int(max(self.decode(r)["tokens"]))
                                              for r in records)
            return self
        self.template, self.fields = kw.get("template"), kw.get("fields")
        if self.template is None and not self.fields:
            self.tf = text_field(records, self.schema, kw.get("text_field"))
        self.tokenizer = resolve_tokenizer(kw.get("tokenizer"),
                                           [self._text(self.decode(r)) for r in records])
        self.info["vocab_size"] = int(self.tokenizer.vocab_size)
        return self

    def _text(self, r: Mapping[str, Any]) -> str:
        if self.template is not None:
            return self.template.format(**{k: v for k, v in r.items() if not k.startswith("_")})
        if self.fields:
            return "\n".join(as_text(r.get(f, "")) for f in self.fields)
        return as_text(r.get(self.tf, ""))

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        m = self.max_len
        if self.token_mode:
            s = {"tokens": _tokens(r["tokens"][:m] if m else r["tokens"])}
            if r.get("labels") is not None:
                s["labels"] = _tokens(r["labels"][:m] if m else r["labels"])
            return s
        return {"tokens": _tokens(encode(self.tokenizer, self._text(r), m, eos=True))}

    def dataset(self, records: Sequence[Mapping[str, Any]]) -> Dataset:
        pack = self.kw.get("pack")
        if self.token_mode or not (pack if pack is not None else self.max_len is not None):
            return super().dataset(records)
        from topologies.data.text import TokenBlockDataset, pack_sequences

        seqs = [encode(self.tokenizer, self._text(self.decode(r)), eos=True) for r in records]
        ds = TokenBlockDataset(pack_sequences(seqs, int(self.max_len)))
        ds.tokenizer = self.tokenizer  # type: ignore[attr-defined]
        return ds


@register_converter("permutation-lm")
class PermutationLM(CausalLM):
    def convert(self, record: Mapping[str, Any]) -> dict[str, Any]:
        result = super().convert(record)
        for key in ("mask", "token_type_ids", "perm_mask", "target_mapping"):
            if key in record:
                result[key] = torch.as_tensor(record[key])
        return result


@register_converter("masked-lm", "electra", "electra-pretraining")
class MaskedLM(_TextMixin):
    """Masked tokens (``masked-lm``: masked ids + -100 labels) or replaced tokens
    (``electra``: random replacements + 0/1 labels). ``electra-pretraining`` returns
    original IDs for generator sampling in the Trainer recipe."""

    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        first = self.decode(records[0]) if records else {}
        self.ready = "tokens" in first and ("labels" in first or self.recipe == "electra-pretraining")
        if self.ready:
            return self
        self.tf = text_field(records, self.schema, self.kw.get("text_field"))
        self.tokenizer = resolve_tokenizer(self.kw.get("tokenizer"), self._texts(records, self.tf))
        mask_id = getattr(self.tokenizer, "mask_id", None)
        self.mask_id = int(mask_id) if mask_id is not None else int(
            getattr(self.tokenizer, "unk_id", 1) or 1)
        self.info["vocab_size"] = int(self.tokenizer.vocab_size)
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        if self.ready:
            if self.recipe == "electra-pretraining":
                return {"tokens": _tokens(r["tokens"])}
            return {"tokens": _tokens(r["tokens"]), "labels": _tokens(r["labels"])}
        ids = _tokens(encode(self.tokenizer, as_text(r.get(self.tf, "")), self.kw.get("max_len")))
        if self.recipe == "electra-pretraining":
            return {"tokens": ids}
        g = torch.Generator().manual_seed(record_seed(r, int(self.kw.get("seed", 0))))
        pick = torch.rand(ids.shape, generator=g) < float(self.kw.get("mask_prob", 0.15))
        if not pick.any():
            pick[0] = True
        if self.recipe == "electra":
            repl = torch.randint(0, int(self.tokenizer.vocab_size), ids.shape, generator=g)
            changed = pick & (repl != ids)
            return {"tokens": torch.where(pick, repl, ids), "labels": changed.float()}
        labels = torch.full_like(ids, -100)
        labels[pick] = ids[pick]
        return {"tokens": ids.masked_fill(pick, self.mask_id), "labels": labels}


@register_converter("seq2seq-lm")
class Seq2Seq(_TextMixin):
    """``src`` = input text, ``tgt`` = BOS + target, ``labels`` = target + EOS."""

    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        kw = self.kw
        self.src_f = kw.get("src_field") or _pick_field(records, self.schema, (
            "input", "src", "source", "text", "problem", "question"))
        self.tgt_f = kw.get("tgt_field") or _pick_field(records, self.schema, (
            "output", "tgt", "target", "answer", "solution", "json", "completion"),
            exclude=(self.src_f,))
        texts = self._texts(records, self.src_f) + self._texts(records, self.tgt_f)
        self.tokenizer = resolve_tokenizer(kw.get("tokenizer"), texts)
        bos = getattr(self.tokenizer, "bos_id", None)
        self.bos = int(bos) if bos is not None else int(getattr(self.tokenizer, "eos_id", 0) or 0)
        self.info.update({"src_field": self.src_f, "tgt_field": self.tgt_f,
                          "vocab_size": int(self.tokenizer.vocab_size)})
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        m = self.kw.get("max_len")
        src = encode(self.tokenizer, as_text(r.get(self.src_f, "")), m)
        tgt = encode(self.tokenizer, as_text(r.get(self.tgt_f, "")),
                     self.kw.get("max_target_len") or m, eos=True)
        return {"src": _tokens(src), "tgt": _tokens([self.bos] + tgt[:-1]), "labels": _tokens(tgt)}

    @property
    def collate_fn(self) -> Any:
        from topologies.data.collate import PadCollator

        return PadCollator(pad_id=pad_id(self.tokenizer), mask_key=None)


def _pick_field(records: Sequence[Mapping[str, Any]], schema: Any, names: Sequence[str],
                exclude: Sequence[str] = ()) -> str:
    found = _first_key(records, names, exclude)
    if found:
        return found
    if schema is not None:
        rest = [n for n in schema.names if n not in exclude]
        if rest:
            return rest[0]
    raise SynthError(f"Cannot find a field among {list(names)}.", stage="torch",
                     hint="Pass src_field=... / tgt_field=....")


class _ClassMixin(Converter):
    def _classes(self, records: Sequence[Mapping[str, Any]], field: str) -> list[str]:
        live = [self.decode(r) for r in records]
        return class_names(live, self.schema, field, self.kw.get("labels"))


@register_converter("vision-classify")
class Vision(_ClassMixin):
    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        images = _kind_fields(records, self.schema, "image")
        self.imf = self.kw.get("image_field") or (images[0] if images else _first_key(
            records, ("image", "pixels", "img")) or "image")
        self.lf = label_field(records, self.schema, self.kw.get("label_field"))
        self.classes = self._classes(records, self.lf)
        self.info["num_classes"] = len(self.classes)
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        return {"pixels": load_image(r[self.imf], self.kw.get("blob_loader")),
                "labels": torch.tensor(label_index(r[self.lf], self.classes))}

    @property
    def collate_fn(self) -> Any:
        return None


@register_converter("text-classify")
class TextClassify(_ClassMixin, _TextMixin):
    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        kw = self.kw
        self.lf = label_field(records, self.schema, kw.get("label_field"))
        self.tf = text_field(records, self.schema, kw.get("text_field"), exclude=(self.lf,))
        self.classes = self._classes(records, self.lf)
        self.max_len = kw.get("max_len", 128)
        self.tokenizer = resolve_tokenizer(kw.get("tokenizer"), self._texts(records, self.tf))
        self.info = {"vocab_size": int(self.tokenizer.vocab_size),
                     "num_classes": len(self.classes), "max_seq": self.max_len,
                     "text_field": self.tf, "label_field": self.lf}
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        return {"tokens": _tokens(encode(self.tokenizer, as_text(r.get(self.tf, "")),
                                         self.max_len)),
                "labels": torch.tensor(label_index(r[self.lf], self.classes))}


def bio_tags(text: str, spans: Sequence[Mapping[str, Any]], tok: Any,
             max_len: int | None = None) -> tuple[list[int], list[str]]:
    """Token ids + BIO tags for ``spans`` (each whitespace/word segment encoded on its own,
    so char offsets map exactly to tokens)."""
    import re as _re

    ids: list[int] = []
    tags: list[str] = []
    spans = sorted(spans, key=lambda s: s["start"])
    for m in _re.finditer(r"\s+|[^\s]+", text):
        seg_ids = list(tok.encode(m.group(0)))
        if not seg_ids:
            continue
        a, b = m.start(), m.end()
        hit = next((s for s in spans if s["start"] < b and a < s["end"]), None)
        if hit is None or m.group(0).isspace():
            tags += ["O"] * len(seg_ids)
        else:
            first = a <= hit["start"] or not tags or not tags[-1].endswith(hit["label"])
            tags += [("B-" if first else "I-") + hit["label"]] + \
                ["I-" + hit["label"]] * (len(seg_ids) - 1)
        ids += seg_ids
    if max_len is not None:
        ids, tags = ids[:max_len], tags[:max_len]
    return ids or [0], tags or ["O"]


@register_converter("token-classify")
class TokenClassify(_TextMixin):
    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        kw = self.kw
        self.tf = text_field(records, self.schema, kw.get("text_field"))
        self.sf = kw.get("spans_field", "spans")
        self.max_len = kw.get("max_len", 256)
        texts = self._texts(records, self.tf)
        self.tokenizer = resolve_tokenizer(kw.get("tokenizer"), texts)
        tags = kw.get("tags")
        if tags is None:
            ents: set[str] = set()
            if self.schema is not None and self.sf in self.schema and \
                    getattr(self.schema[self.sf], "labels", None):
                ents = set(self.schema[self.sf].labels)
            for r in records:
                ents |= {s["label"] for s in (self.decode(r).get(self.sf) or [])}
            tags = ["O"] + [f"{p}-{e}" for e in sorted(ents) for p in ("B", "I")]
        self.classes = list(tags)
        self.tag_ix = {t: i for i, t in enumerate(self.classes)}
        self.info = {"vocab_size": int(self.tokenizer.vocab_size),
                     "num_classes": len(self.classes), "max_seq": self.max_len}
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        ids, ts = bio_tags(as_text(r.get(self.tf, "")), r.get(self.sf) or [], self.tokenizer,
                           self.max_len)
        return {"tokens": _tokens(ids), "labels": _tokens([self.tag_ix.get(t, 0) for t in ts])}


def _numeric_columns(records: Sequence[Mapping[str, Any]], schema: Any,
                     exclude: Sequence[str]) -> list[str]:
    if schema is not None:
        cols = [n for n in schema.of_kind("number") if n not in exclude]
        if cols:
            return cols
    first = records[0] if records else {}
    return [k for k, v in first.items() if not str(k).startswith("_") and k not in exclude
            and isinstance(v, (int, float)) and not isinstance(v, bool)]


def _categorical_columns(records: Sequence[Mapping[str, Any]], schema: Any,
                         exclude: Sequence[str]) -> dict[str, list[str]]:
    if schema is not None:
        found = {n: list(schema[n].names) for n in schema.of_kind("category")
                 if n not in exclude}
        if found or schema.of_kind("number"):
            return found
    first = records[0] if records else {}
    out = {}
    for k, v in first.items():
        if str(k).startswith("_") or k in exclude or not isinstance(v, str):
            continue
        vals = sorted({str(r.get(k)) for r in records})
        if len(vals) <= 64:
            out[k] = vals
    return out


@register_converter("multilabel-classify")
class Multilabel(_TextMixin):
    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        kw, schema = self.kw, self.schema
        live = [self.decode(r) for r in records]
        self.lf = kw.get("label_field") or (schema.of_kind("labels")[0] if schema is not None
                                            and schema.of_kind("labels") else "labels")
        given = kw.get("labels")
        self.classes = list(given) if given is not None else (
            list(schema[self.lf].names) if schema is not None and self.lf in schema else
            sorted({str(x) for r in live for x in (r.get(self.lf) or [])}))
        self.max_len = kw.get("max_len", 128)
        try:
            self.tf: str | None = text_field(live, schema, kw.get("text_field"),
                                             exclude=(self.lf,))
        except SynthError:
            self.tf = None
        if self.tf is not None:
            self.tokenizer = resolve_tokenizer(kw.get("tokenizer"), self._texts(live, self.tf))
            self.info = {"vocab_size": int(self.tokenizer.vocab_size),
                         "num_classes": len(self.classes), "max_seq": self.max_len}
        else:
            self.cols = _numeric_columns(live, schema, (self.lf,))
            self.info = {"in_features": len(self.cols), "num_classes": len(self.classes),
                         "columns": self.cols}
        return self

    def _hot(self, vals: Any) -> torch.Tensor:
        v = torch.zeros(len(self.classes))
        for x in vals or []:
            if str(x) in self.classes:
                v[self.classes.index(str(x))] = 1.0
        return v

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        if self.tf is not None:
            return {"tokens": _tokens(encode(self.tokenizer, as_text(r.get(self.tf, "")),
                                             self.max_len)), "labels": self._hot(r.get(self.lf))}
        return {"features": torch.tensor([_f(r.get(c)) for c in self.cols]),
                "labels": self._hot(r.get(self.lf))}

    @property
    def collate_fn(self) -> Any:
        return super().collate_fn if self.tf is not None else None


@register_converter("tabular-classify")
class Tabular(_ClassMixin):
    """Numeric + categorical columns -> ``features`` / ``categorical``; or a series / audio /
    tensor field -> ``features`` ``[T]`` / ``[T, C]`` (sequence classification)."""

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        kw, schema = self.kw, self.schema
        live = [self.decode(r) for r in records]
        self.lf = label_field(live, schema, kw.get("label_field"))
        self.classes = self._classes(records, self.lf)
        seq = kw.get("series_field")
        if seq is None:
            seqs = _kind_fields(records, schema, "series", "audio", "tensor")
            seq = seqs[0] if seqs else None
        if seq is None and schema is None and live and isinstance(live[0].get("series"), list):
            seq = "series"
        cols = kw.get("columns")
        self.seq = seq if seq is not None and (cols is None or list(cols) == [seq]) else None
        if self.seq is not None:
            x = to_float_tensor(live[0][self.seq])
            t, c = (x.shape[0], 1 if x.dim() == 1 else x.shape[1]) if x.dim() else (1, 1)
            self.info = {"in_features": t * c, "in_channels": c, "length": t,
                         "num_classes": len(self.classes), "columns": [self.seq],
                         "categorical": {}, "cardinalities": []}
            return self
        self.cols = list(cols) if cols is not None else _numeric_columns(live, schema, (self.lf,))
        cat = kw.get("categorical")
        self.cats = {k: list(v) for k, v in cat.items()} if cat is not None else \
            _categorical_columns(live, schema, (self.lf, *self.cols))
        self.info = {"in_features": len(self.cols), "num_classes": len(self.classes),
                     "columns": self.cols, "categorical": {k: len(v) for k, v in
                                                           self.cats.items()},
                     "categorical_names": self.cats,
                     "cardinalities": [len(v) for v in self.cats.values()]}
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        y = torch.tensor(label_index(r[self.lf], self.classes))
        if self.seq is not None:
            return {"features": to_float_tensor(r[self.seq]), "labels": y}
        s: dict[str, Any] = {"features": torch.tensor([_f(r.get(c)) for c in self.cols],
                                                      dtype=torch.float32), "labels": y}
        if self.cats:
            s["categorical"] = torch.tensor([v.index(str(r.get(k))) if str(r.get(k)) in v
                                             else 0 for k, v in self.cats.items()])
        return s

    @property
    def collate_fn(self) -> Any:
        return None


@register_converter("regression")
class Regression(Converter):
    """``inputs`` / ``targets``: a sequence field (series, audio, tensor) with ``horizon``
    (forecast the last steps) or a ``target_field``; or numeric columns + a numeric target.
    ``target_field`` may hold a number or a tensor (``[K]`` targets, e.g. an f0 curve)."""

    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        kw, schema = self.kw, self.schema
        live = [self.decode(r) for r in records]
        first = live[0] if live else {}
        self.target = kw.get("target_field") or kw.get("target")
        self.horizon = kw.get("horizon")
        self.seq = None
        seq = kw.get("input_field")
        if seq is None:
            seqs = [s for s in _kind_fields(records, schema, "series", "audio", "tensor", "image")
                    if s != self.target]
            seq = seqs[0] if seqs else next(
                (k for k, v in first.items() if isinstance(v, list) and v and
                 isinstance(v[0], (int, float, list)) and not str(k).startswith("_")
                 and k != self.target), None)
        if seq is not None and (self.horizon or self.target is None or kw.get("input_field")
                                or not isinstance(first.get(self.target), (int, float))
                                or _kind_fields(records, schema, "audio", "tensor", "image")):
            self.seq = seq
            s0 = self.convert(first)
            x = s0["inputs"]
            self.info = {"in_channels": int(x.shape[-1]) if x.dim() > 1 else 1,
                         "out_features": int(s0["targets"].numel()),
                         "max_seq": int(x.shape[0]) if x.dim() else 1}
            return self
        if self.target is None:
            self.target = next((n for n in ("target", "y", "value", "price") if n in first), None)
        if self.target is None:
            raise SynthError("regression needs target_field=... (or a series field).",
                             stage="torch")
        self.cols = list(kw["columns"]) if kw.get("columns") is not None else \
            _numeric_columns(live, schema, (self.target,))
        self.info = {"in_features": len(self.cols), "out_features": 1, "columns": self.cols}
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        if self.seq is not None:
            x = to_float_tensor(r[self.seq])
            if x.dim() == 1 and not self.kw.get("preserve_shape", False):
                x = x[:, None]
            if self.target is None:
                h = int(self.horizon or 1)
                return {"inputs": x[:-h], "targets": x[-h:, 0]}
            t = r[self.target]
            tt = torch.tensor(_f(t)) if isinstance(t, (int, float)) else to_float_tensor(t)
            return {"inputs": x, "targets": tt}
        return {"inputs": torch.tensor([_f(r.get(c)) for c in self.cols], dtype=torch.float32),
                "targets": torch.tensor(_f(r[self.target]))}

    @property
    def collate_fn(self) -> Any:
        return None


@register_converter("autoencoder", "vae", "vqvae", "gan", "diffusion")
class XRecipe(Converter):
    """``x``: a tensor-like field (flattened to ``[D]``; ``diffusion`` keeps ``[C, H, W]``)."""

    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        f = self.kw.get("field")
        if f is None:
            kinds = ("image", "tensor", "audio", "series", "pointcloud") \
                if self.recipe != "diffusion" else ("image", "tensor")
            found = _kind_fields(records, self.schema, *kinds)
            f = found[0] if found else _first_key(records, ("x", "image", "features", "series"))
        if f is None:
            raise SynthError(f"{self.recipe} needs a tensor field.", stage="torch",
                             hint=f"for_recipe({self.recipe!r}, field='...').")
        self.field = f
        x = self.convert(self.decode(records[0]))["x"]
        self.info = {"in_features": int(x.numel()), "shape": list(x.shape)}
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        v = r[self.field]
        preserve_shape = bool(self.kw.get("preserve_shape", False))
        x = load_image(v, self.kw.get("blob_loader")) if self.recipe == "diffusion" else \
            to_float_tensor(v)
        if self.recipe != "diffusion" and not preserve_shape:
            x = x.reshape(-1)
        return {"x": x}

    @property
    def collate_fn(self) -> Any:
        return None


@register_converter("audio-classify")
class AudioClassify(_ClassMixin):
    def fit(self, records):
        self.lf = label_field(records, self.schema, self.kw.get("label_field"))
        self.af = self.kw.get("audio_field", "audio")
        self.classes = self._classes(records, self.lf)
        return self

    def convert(self, record):
        return {"x": to_float_tensor(record[self.af]),
                "labels": torch.tensor(label_index(record[self.lf], self.classes))}


@register_converter("ctc-asr")
class CTC(Converter):
    label_key = None

    def fit(self, records):
        self.af = self.kw.get("feature_field", "features")
        self.tf = self.kw.get("target_field", "targets")
        return self

    def convert(self, record):
        x = to_float_tensor(record[self.af])
        if x.ndim == 1:
            x = x.unsqueeze(0)
        targets = torch.as_tensor(record[self.tf], dtype=torch.long).reshape(-1)
        return {"features": x, "lengths": torch.tensor(x.shape[-1]),
                "targets": targets, "target_lengths": torch.tensor(targets.numel())}

    @property
    def collate_fn(self):
        def collate(samples):
            return {"features": torch.stack([s["features"] for s in samples]),
                    "lengths": torch.stack([s["lengths"] for s in samples]),
                    "targets": torch.cat([s["targets"] for s in samples]),
                    "target_lengths": torch.stack([s["target_lengths"] for s in samples])}
        return collate


@register_converter("contrastive-pairs")
class ContrastivePairs(Converter):
    label_key = None

    def fit(self, records):
        self.image_field = self.kw.get("image_field", "image")
        self.text_field = self.kw.get("text_field", "text")
        self.tokenizer = resolve_tokenizer(self.kw.get("tokenizer", "char"),
                                           [str(r[self.text_field]) for r in records])
        return self

    def convert(self, record):
        return {"images": load_image(record[self.image_field]),
                "tokens": _tokens(encode(self.tokenizer, str(record[self.text_field]),
                                          self.kw.get("max_len", 16))),
                "pair_ids": str(record.get("pair_id", record.get("_id")))}

    @property
    def collate_fn(self):
        from topologies.data.collate import PadCollator

        collate = PadCollator()

        def paired(samples):
            batch = collate(samples)
            if batch.get("mask") is not None:
                batch["mask"] = batch["mask"].bool()
            return batch

        return paired


@register_converter("object-detection")
class ObjectDetection(Converter):
    label_key = None

    def fit(self, records):
        self.ifield = self.kw.get("image_field", "image")
        return self

    def convert(self, record):
        return {"images": load_image(record[self.ifield]),
                "targets": {"boxes": torch.as_tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4),
                            "labels": torch.as_tensor(record["labels"], dtype=torch.long).reshape(-1)}}

    @property
    def collate_fn(self):
        return lambda samples: {"images": [s["images"] for s in samples],
                                "targets": [s["targets"] for s in samples]}


@register_converter("graph-supervised")
class GraphSupervised(Converter):
    label_key = None

    def fit(self, records):
        return self

    def convert(self, record):
        return {"x": to_float_tensor(record["x"]),
                "edge_index": _tokens(record["edge_index"]),
                "y": _tokens(record["y"]), "train_mask": torch.as_tensor(record["train_mask"], dtype=torch.bool)}

    @property
    def collate_fn(self):
        return lambda samples: samples[0]

    def dataset(self, records):
        graphs = [self(record) for record in records]
        xs, edges, ys, masks = [], [], [], []
        offset = 0
        for graph in graphs:
            xs.append(graph["x"])
            edges.append(graph["edge_index"] + offset)
            ys.append(graph["y"])
            masks.append(graph["train_mask"])
            offset += graph["x"].shape[0]
        graph = {"x": torch.cat(xs), "edge_index": torch.cat(edges, dim=1),
                 "y": torch.cat(ys), "train_mask": torch.cat(masks)}
        return ListDataset([graph], collate_fn=self.collate_fn, label_key=None)


@register_converter("audio-reconstruct")
class AudioReconstruct(Converter):
    label_key = None

    def fit(self, records):
        return self

    def convert(self, record):
        return {"audio": to_float_tensor(record["audio"]),
                "pitch": torch.as_tensor(record["pitch"], dtype=torch.float32),
                "loudness": torch.as_tensor(record["loudness"], dtype=torch.float32),
                "sample_rate": int(record["sample_rate"]),
                "lengths": torch.tensor(int(record["lengths"]))}

    @property
    def collate_fn(self):
        def collate(samples):
            return {"audio": torch.stack([s["audio"] for s in samples]),
                    "pitch": torch.stack([s["pitch"] for s in samples]),
                    "loudness": torch.stack([s["loudness"] for s in samples]),
                    "sample_rate": samples[0]["sample_rate"],
                    "lengths": torch.stack([s["lengths"] for s in samples])}
        return collate


@register_converter("recommender-bce")
class RecommenderBCE(Converter):
    label_key = None

    def fit(self, records):
        self.fields = "x" in records[0]
        return self

    def convert(self, record):
        if self.fields:
            return {"x": _tokens(record["x"]), "label": torch.tensor(float(record["label"]))}
        return {"user": torch.tensor(int(record["user"])),
                "item": torch.tensor(int(record["item"])),
                "label": torch.tensor(float(record["label"]))}


@register_converter("td-value")
class TDValue(Converter):
    label_key = None

    def fit(self, records):
        return self

    def convert(self, record):
        return {"state": to_float_tensor(record["state"]),
                "action": torch.tensor(int(record["action"])),
                "reward": torch.tensor(float(record["reward"])),
                "next_state": to_float_tensor(record["next_state"]),
                "terminated": torch.tensor(bool(record["terminated"]))}


@register_converter("tts")
class TTS(_TextMixin):
    """``tokens`` from text, ``mel`` (log mel ``[M, T]``) from audio, ``durations``: mel frames
    spread evenly over the tokens."""

    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        kw = self.kw
        self.tf = text_field(records, self.schema, kw.get("text_field"))
        audio = _kind_fields(records, self.schema, "audio")
        self.af = kw.get("audio_field") or (audio[0] if audio else "audio")
        self.n_mels = int(kw.get("n_mels", 40))
        self.tokenizer = resolve_tokenizer(kw.get("tokenizer", "char"),
                                           self._texts(records, self.tf))
        self.info = {"vocab_size": int(self.tokenizer.vocab_size), "n_mels": self.n_mels}
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        from topologies.synth.dsp import log_mel
        from topologies.synth.valuetypes import Audio

        ids = _tokens(encode(self.tokenizer, as_text(r.get(self.tf, "")), self.kw.get("max_len")))
        a = r[self.af]
        sr = Audio().rate(a)
        m = log_mel(to_float_tensor(a), sr, n_mels=self.n_mels)
        frames = m.shape[-1]
        base = torch.full((len(ids),), float(frames // max(1, len(ids))))
        base[: frames - int(base.sum())] += 1
        return {"tokens": ids, "mel": m, "durations": base}


@register_converter("distillation")
class Distillation(Converter):
    """Dispatches to the vision, causal-lm or text-classify converter by record fields."""

    label_key = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> Converter:
        first = self.decode(records[0]) if records else {}
        if any(k in first for k in ("image", "pixels")) or _kind_fields(records, self.schema,
                                                                         "image"):
            cls: type[Converter] = Vision
        elif "tokens" in first:
            cls = CausalLM
        else:
            cls = TextClassify
        self.inner = cls(schema=self.schema, decode=self._decode, **self.kw)
        self.inner.recipe = "distillation"
        self.inner.fit(records)
        self.tokenizer, self.classes, self.info = self.inner.tokenizer, self.inner.classes, \
            self.inner.info
        self.label_key = self.inner.label_key
        return self

    def convert(self, r: Mapping[str, Any]) -> dict[str, Any]:
        return self.inner.convert(r)

    @property
    def collate_fn(self) -> Any:
        return self.inner.collate_fn


# ---------------------------------------------------------------- entry points


def make_converter(recipe: str | None, *, schema: Any = None,
                   decode: Callable[[dict], dict] | None = None, **kw: Any) -> Converter:
    """An unfitted converter for ``recipe`` (``None``: schema-free tensors)."""
    if recipe is None:
        return GenericConverter(schema=schema, decode=decode, **kw)
    import topologies.training  # noqa: F401  (registers recipes)

    cls = CONVERTERS.get(recipe)
    if cls is None:
        from topologies.training.recipe_base import _RECIPE_REGISTRY

        if recipe not in _RECIPE_REGISTRY:
            from topologies.synth.errors import unknown

            raise unknown("recipe", recipe, list(_RECIPE_REGISTRY), stage="torch")
        cls = BatchSpecConverter
    conv = cls(schema=schema, decode=decode, **kw)
    conv.recipe = recipe
    return conv


def build_dataset(records: Sequence[Mapping[str, Any]], recipe: str | None, *,
                  schema: Any = None, decode: Callable[[dict], dict] | None = None,
                  **kw: Any) -> Dataset:
    """Fit a converter on ``records`` and convert them all (see module docstring)."""
    records = list(records)
    if recipe in _BUILDERS:
        return _BUILDERS[recipe](records, schema=schema, **kw)
    if not records:
        raise SynthError("No records to build a torch dataset from.", stage="torch",
                         hint="Check the split name and that the dataset is not empty.")
    conv = make_converter(recipe, schema=schema, decode=decode, **kw).fit(records)
    return conv.dataset(records)


__all__ = ["CONVERTERS", "BatchSpecConverter", "Converter", "GenericConverter", "ListDataset",
           "as_text", "bio_tags", "build_dataset", "class_names", "encode", "label_index",
           "load_image", "make_converter", "record_seed", "register_builder",
           "register_converter", "resolve_tokenizer", "to_float_tensor"]
