"""Tasks: what data a trial trains on, what gets injected into models, how it is scored.

Built-ins: ``LanguageModeling``, ``ImageClassification``, ``Supervised``.
Subclass ``Task`` for anything else:

    class MyTask(Task):
        type = "my-task"
        recipe = "vision-classify"
        default_goal = ("val_acc", "max")

        def __init__(self, root, *, size=32):
            super().__init__(size=size)       # params: overridable as task.size
            self.root = root

        def fingerprint(self):
            return stable_hash({"root": str(self.root), **self.params})

        def build_data(self, ctx):
            p = ctx.task_params                 # params merged with task.* overrides
            return Datasets(train=..., val=...)

Task parameters (``self.params``) can vary per trial through ``task.<name>``
(alias ``data.<name>``) keys; read them from ``ctx.task_params``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from topologies.experiments.cache import fingerprint_data, stable_hash
from topologies.experiments.errors import ExperimentError

#: Constructor-argument aliases tried in order when injecting geometry.
GEOMETRY_ALIASES: dict[str, tuple[str, ...]] = {
    "vocab_size": ("vocab_size", "vocab", "n_vocab", "num_tokens", "vocab_len"),
    "max_seq": ("max_seq", "context_length", "block_size", "max_len", "max_seq_len",
                "n_positions", "max_position_embeddings"),
    "num_classes": ("num_classes", "n_classes", "classes"),
    "in_channels": ("in_channels", "channels", "in_chans"),
    "img_size": ("img_size", "image_size"),
    "in_features": ("in_features",),
    "out_features": ("out_features",),
    "cardinalities": ("cardinalities",),
}


@dataclass
class Datasets:
    """What ``Task.build_data`` returns (datasets, loaders or iterables)."""

    train: Any
    val: Any = None
    test: Any = None
    info: dict[str, Any] = field(default_factory=dict)


class Task:
    """Base class / protocol for experiment tasks (see module docstring)."""

    type: str = "custom"
    recipe: str | None = None
    default_goal: tuple[str, str] = ("val_loss", "min")
    consumes: frozenset[str] = frozenset()

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    @property
    def name(self) -> str:
        return self.type

    def param_names(self) -> set[str]:
        return set(self.params)

    def fingerprint(self) -> str:
        """Stable hash of data identity + task parameters."""
        return stable_hash({"type": self.type, "params": _canon_params(self.params)})

    def geometry(self, ctx: Any) -> dict[str, Any]:
        """Constructor kwargs to inject (by alias) when a model accepts them."""
        return {}

    def build_data(self, ctx: Any) -> Datasets:
        raise NotImplementedError(f"Task {type(self).__name__} must implement build_data(ctx).")

    def evaluate(self, ctx: Any, trainer: Any) -> dict[str, float]:
        """Final metrics; default: ``val_*`` (and ``test_*``) from ``trainer.evaluate``."""
        ds = ctx.datasets
        out: dict[str, float] = {}
        if ds.val is not None:
            out.update(trainer.evaluate(ds.val, prefix="val_"))
        if ds.test is not None:
            out.update(trainer.evaluate(ds.test, prefix="test_"))
        out.update(self.derive(ctx, out))
        return out

    def derive(self, ctx: Any, metrics: dict[str, float]) -> dict[str, float]:
        """Derived metrics from raw ones (applied to every history row and the final)."""
        return {}

    def tokens_per_sample(self, ctx: Any) -> int | None:
        """Tokens in one training sample (for ``train_for="5M tokens"``); None if n/a."""
        return None

    def bytes_per_token(self, ctx: Any) -> float | None:
        """Training-corpus bytes per token (for ``train_for="20MB"``); None if n/a."""
        return None

    def describe(self) -> dict[str, Any]:
        return {"type": self.type, "recipe": self.recipe, "goal": list(self.default_goal),
                "params": _canon_params(self.params)}


def _canon_params(params: dict[str, Any]) -> dict[str, Any]:
    from topologies.experiments.spec import canonical

    return {k: canonical(v, warn=False) for k, v in params.items()}


# ---------------------------------------------------------------- helpers


def _as_docs(source: Any, what: str) -> list[str] | None:
    """Documents from a list of strings, raw text, or file path(s)."""
    if source is None:
        return None
    from topologies.data.text import read_documents, split_documents
    from topologies.tokenizers.base import is_file_path

    if isinstance(source, Path) or (isinstance(source, str) and is_file_path(source)):
        return read_documents(source)
    if isinstance(source, str):
        return split_documents(source)
    if isinstance(source, (list, tuple)):
        if source and all(isinstance(x, (str, Path)) and (isinstance(x, Path) or is_file_path(x))
                          for x in source):
            return read_documents(list(source))
        docs = [d for d in source if isinstance(d, str) and d]
        if len(docs) != len([d for d in source if d]):
            raise ExperimentError(f"{what} must be documents (strings) or file paths.",
                                  stage="data")
        return docs
    raise ExperimentError(f"Cannot read {what} from {type(source).__name__}.", stage="data",
                          hint="Pass a list of documents, raw text, or a path to a text file.")


def _is_text_corpus(obj: Any) -> bool:
    from topologies.data.corpus import TextCorpus

    return isinstance(obj, TextCorpus)


def _is_token_stream(obj: Any) -> bool:
    from topologies.data.stream import TokenStream

    return isinstance(obj, TokenStream)


def _is_stream_source(obj: Any) -> bool:
    if _is_text_corpus(obj) or _is_token_stream(obj):
        return True
    if isinstance(obj, dict) and obj:
        return all(_is_text_corpus(v) or _is_token_stream(v) for v in obj.values())
    return False


def _source_fingerprint(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return stable_hash({str(k): _source_fingerprint(v) for k, v in sorted(obj.items())})
    finger = getattr(obj, "fingerprint", None)
    if callable(finger):
        return str(finger())
    meta = getattr(obj, "meta", None)
    if isinstance(meta, dict) and "n_tokens" in meta:
        return stable_hash({
            "corpus": meta.get("corpus_fingerprint"),
            "tokenizer": meta.get("tokenizer_fingerprint"),
            "n_tokens": meta.get("n_tokens"),
        })
    return fingerprint_data(obj)


class _LabeledTexts:
    """Dict of corpora, each forced to the dict key as its document label."""

    def __init__(self, parts: dict[str, Any]) -> None:
        self.parts = dict(parts)
        self.label_field = "label"

    def fingerprint(self) -> str:
        return str(_source_fingerprint(self.parts))

    def iter_records(self, fields: Sequence[str] | None = None) -> Any:
        del fields
        for name, corpus in self.parts.items():
            if _is_token_stream(corpus):
                raise ExperimentError(
                    "A mapping of token streams cannot be re-encoded as one stream.",
                    stage="data",
                )
            for record in corpus.iter_records(("text",)):
                yield {"text": record.get("text"), "label": str(name)}

    def iter_texts(self) -> Any:
        for record in self.iter_records(("text",)):
            text = record.get("text")
            if isinstance(text, str) and text:
                yield text

    def sample(self, max_bytes: int, seed: int = 0) -> list[str]:
        chosen: list[str] = []
        remaining = int(max_bytes)
        for corpus in self.parts.values():
            if remaining < 1:
                break
            part = corpus.sample(remaining, seed=seed)
            chosen.extend(part)
            remaining -= sum(len(text.encode("utf-8")) for text in part)
        return chosen


def _monitor_progress(trainer: Any, label: str):
    """``(done, total) -> None`` when ``trainer.monitor`` implements ``note``."""
    monitor = getattr(trainer, "monitor", None)
    note = getattr(monitor, "note", None)
    if not callable(note):
        return None

    def on_progress(done: int, total: int) -> None:
        note(label, done=int(done), total=int(total))

    return on_progress


def _grouped_val_bpb(model: Any, stream: Any, block_size: int,
                     on_progress: Any = None) -> dict[str, float]:
    """Per-label bits/byte on a full token stream, plus their byte-weighted mean.

    ``on_progress(done, total)`` is called once per block. The total is the
    number of blocks. Callers use it to keep a monitor from going quiet.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    from topologies.metrics import bits_per_byte

    if stream.doc_labels is None or not stream.label_names:
        return {}
    block = int(block_size)
    n_tokens = int(stream.tokens.shape[0]) // block * block
    if n_tokens < block:
        return {}
    offsets = np.asarray(stream.doc_offsets)
    doc_labels = np.asarray(stream.doc_labels)
    names = list(stream.label_names)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    n_blocks = n_tokens // block
    if on_progress is not None:
        on_progress(0, n_blocks)
    try:
        with torch.no_grad():
            for done, start in enumerate(range(0, n_tokens, block), start=1):
                ids = np.array(stream.tokens[start:start + block], dtype=np.int64, copy=True)
                tokens = torch.from_numpy(ids).unsqueeze(0).to(device)
                out = model(tokens)
                logits = out["logits"] if isinstance(out, dict) else out
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits[:, :-1]
                target = tokens[:, 1:]
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none"
                ).detach().float().cpu().numpy()
                positions = np.arange(start + 1, start + block)
                docs = np.searchsorted(offsets, positions, side="right") - 1
                labels = doc_labels[np.clip(docs, 0, len(doc_labels) - 1)]
                for label_id in np.unique(labels):
                    mask = labels == label_id
                    name = names[int(label_id)]
                    sums[name] = sums.get(name, 0.0) + float(losses[mask].sum())
                    counts[name] = counts.get(name, 0) + int(mask.sum())
                if on_progress is not None:
                    on_progress(done, n_blocks)
    finally:
        model.train(was_training)
    metrics: dict[str, float] = {}
    weighted = 0.0
    weight = 0.0
    tokens_per = stream.meta.get("tokens_per_label") or {}
    bytes_per = stream.meta.get("bytes_per_label") or {}
    for name, total in sums.items():
        n_lab = int(tokens_per.get(name) or 0)
        n_bytes = int(bytes_per.get(name) or 0)
        n_scored = counts.get(name) or 0
        if n_lab < 1 or n_bytes < 1 or n_scored < 1:
            continue
        bpb = bits_per_byte(total / n_scored, n_bytes / n_lab)
        metrics[f"val_bpb/{name}"] = bpb
        weighted += bpb * n_bytes
        weight += n_bytes
    if weight:
        metrics["val_bpb_grouped"] = weighted / weight
    return metrics


# ---------------------------------------------------------------- language modeling


class LanguageModeling(Task):
    """Causal language modeling on documents, with a tokenizer axis.

    ``train`` / ``val`` / ``test``: list of documents, raw text or file path(s).
    Without ``val`` the training documents are split (``val_fraction``, seed 0).
    The task consumes the ``tokenizer`` axis (default ``tokenizer`` or
    ``"bpe:4k"``), injects ``vocab_size`` (or ``vocab``) and ``max_seq`` (or
    ``context_length`` / ``block_size`` / ``max_len`` / ``max_seq_len``) into
    models, and reports tokenizer-independent ``val_bpb`` / ``val_bpc`` (the
    default goal, lower is better) next to the per-token metrics.

    ``sample_prompts=["Once upon a time"]`` stores short greedy and sampled
    continuations per trial (``extras.samples``). ``disk_cache=True`` keeps
    tokenized blocks under ``<root>/.cache/datasets``.

    ``storage="stream"`` (also selected when ``train`` is a ``TextCorpus`` or
    ``TokenStream``) tokenizes into ``<root>/.cache/streams/<key>/`` and trains
    on ``TokenWindows``. The default ``storage="memory"`` is unchanged.
    ``group_by="label"`` adds ``val_bpb/<label>`` from the full validation
    stream. That pass reports block progress through ``trainer.monitor.note``
    when the monitor implements it, so the terminal is not quiet while the
    stream is scored. ``val`` may be a corpus, a token stream, or a
    ``{label: corpus}`` mapping.
    """

    type = "language-modeling"
    recipe = "causal-lm"
    default_goal = ("val_bpb", "min")
    consumes = frozenset({"tokenizer"})

    def __init__(self, train: Any, val: Any = None, test: Any = None, *,
                 val_fraction: float = 0.1, block_size: int = 128, mode: str = "packed",
                 tokenizer: Any = None, train_subset: float | None = None,
                 sample_prompts: Sequence[str] | None = None, sample_tokens: int = 40,
                 disk_cache: bool = False, storage: str = "memory",
                 samples_per_epoch: int | None = None, val_max_tokens: int = 1_000_000,
                 group_by: str | None = None, tokenizer_sample_bytes: int = 300_000_000,
                 stream_workers: int = 1) -> None:
        if storage not in ("memory", "stream"):
            raise ExperimentError(f"storage must be 'memory' or 'stream'; got {storage!r}.",
                                  stage="data")
        if storage == "memory" and _is_stream_source(train):
            storage = "stream"
        stream_params: dict[str, Any] = {}
        if storage == "stream":
            stream_params = {
                "storage": "stream",
                "samples_per_epoch": samples_per_epoch,
                "val_max_tokens": int(val_max_tokens),
                "group_by": group_by,
                "tokenizer_sample_bytes": int(tokenizer_sample_bytes),
            }
        super().__init__(block_size=int(block_size), mode=mode, train_subset=train_subset,
                         val_fraction=val_fraction, **stream_params)
        if mode not in ("packed", "padded"):
            raise ExperimentError(f"mode must be 'packed' or 'padded'; got {mode!r}.",
                                  stage="data")
        self.storage = storage
        self.samples_per_epoch = samples_per_epoch
        self.val_max_tokens = int(val_max_tokens)
        self.group_by = group_by
        self.tokenizer_sample_bytes = int(tokenizer_sample_bytes)
        if int(stream_workers) < 1:
            raise ExperimentError("stream_workers must be >= 1.", stage="data")
        self.stream_workers = int(stream_workers)
        self.train_source = train
        self.val_source = val
        self.test_source = test
        self.default_tokenizer = tokenizer if tokenizer is not None else "bpe:4k"
        self.sample_prompts = list(sample_prompts or [])
        self.sample_tokens = int(sample_tokens)
        self.disk_cache = bool(disk_cache)
        if storage == "stream":
            if mode != "packed":
                raise ExperimentError("storage='stream' supports mode='packed' only.",
                                      stage="data")
            if not _is_stream_source(train):
                raise ExperimentError(
                    "storage='stream' needs a TextCorpus, TokenStream, or {label: corpus}.",
                    stage="data",
                )
            if val is not None and not _is_stream_source(val):
                raise ExperimentError(
                    "stream val must be a TextCorpus, TokenStream, or {label: corpus}.",
                    stage="data",
                )
            if val is None:
                raise ExperimentError(
                    "storage='stream' needs an explicit val corpus.",
                    stage="data",
                    hint="Pass val=TextCorpus(...). The memory path still splits a document list.",
                )
            self.train_docs = []
            self.val_docs = []
            self.test_docs = None
            self._fp = {"train": _source_fingerprint(train),
                        "val": _source_fingerprint(val),
                        "test": _source_fingerprint(test) if test is not None else None}
        else:
            docs = _as_docs(train, "train")
            if not docs:
                raise ExperimentError("LanguageModeling: no training documents.", stage="data")
            if val is None:
                from topologies.data.splits import train_val_split

                docs, val_docs = train_val_split(docs, val_fraction, seed=0)
            else:
                val_docs = _as_docs(val, "val")
            self.train_docs = list(docs)
            self.val_docs = list(val_docs or [])
            self.test_docs = _as_docs(test, "test")
            self._fp = {"train": fingerprint_data(self.train_docs),
                        "val": fingerprint_data(self.val_docs),
                        "test": fingerprint_data(self.test_docs) if self.test_docs else None}

    def fingerprint(self) -> str:
        return stable_hash({"type": self.type, "data": self._fp,
                            "params": _canon_params(self.params)})

    def tokenizer_corpus(self, ctx: Any) -> list[str]:
        if self.storage != "stream":
            return self._train_docs(ctx)
        source = self.train_source
        if isinstance(source, dict):
            return _LabeledTexts(source).sample(self.tokenizer_sample_bytes, seed=0)
        if _is_text_corpus(source):
            return source.sample(self.tokenizer_sample_bytes, seed=0)
        raise ExperimentError(
            "A TokenStream has no raw text to train a tokenizer on.",
            stage="tokenizer",
            hint="Pass a TextCorpus, or pass a tokenizer that is already trained.",
        )

    def corpus_fingerprint(self, ctx: Any) -> str:
        sub = ctx.task_params.get("train_subset") if ctx is not None else None
        return stable_hash({"train": self._fp["train"], "subset": sub})

    def _train_docs(self, ctx: Any) -> list[str]:
        sub = ctx.task_params.get("train_subset")
        if sub is None:
            return self.train_docs
        from topologies.data.splits import take_subset

        return take_subset(self.train_docs, sub, seed=0)

    def geometry(self, ctx: Any) -> dict[str, Any]:
        tok = ctx.tokenizer
        geo: dict[str, Any] = {"max_seq": int(ctx.task_params["block_size"])}
        if tok is not None:
            geo["vocab_size"] = int(tok.vocab_size)
        return geo

    def _split(self, ctx: Any, split: str) -> list[str] | None:
        return {"train": self._train_docs(ctx), "val": self.val_docs,
                "test": self.test_docs}[split]

    def _dataset(self, ctx: Any, split: str) -> Any:
        if self.storage == "stream":
            return self._stream_dataset(ctx, split)
        from topologies.data.text import TokenBlockDataset, lm_dataset, pack_sequences

        docs = self._split(ctx, split)
        if not docs:
            return None
        p = ctx.task_params
        bs, mode = int(p["block_size"]), p["mode"]
        key = stable_hash({"tok": ctx.tokenizer_key, "split": split,
                           "data": self.corpus_fingerprint(ctx) if split == "train"
                           else self._fp[split], "block_size": bs, "mode": mode})
        tok = ctx.tokenizer
        try:
            if mode == "packed":
                def build_blocks() -> Any:
                    eos = getattr(tok, "eos_id", None)
                    seqs = [list(tok.encode(d)) + ([int(eos)] if eos is not None else [])
                            for d in docs]
                    return pack_sequences(seqs, bs)

                blocks = ctx.cache.tensor(f"lm-{key}", build_blocks, disk=self.disk_cache)
                return TokenBlockDataset(blocks)
            return ctx.cache.get_or_create(f"lm-{key}",
                                           lambda: lm_dataset(docs, tok, bs, mode=mode))
        except ValueError as e:
            raise ExperimentError(
                f"Cannot build the {split} set: {e}", stage="data",
                hint=f"Use a smaller task.block_size (now {bs}) or more {split} text.") from e

    def build_data(self, ctx: Any) -> Datasets:
        if ctx.tokenizer is None:
            raise ExperimentError("LanguageModeling needs a tokenizer.", stage="data")
        return Datasets(train=self._dataset(ctx, "train"), val=self._dataset(ctx, "val"),
                        test=self._dataset(ctx, "test"))

    def _stream_dataset(self, ctx: Any, split: str) -> Any:
        from topologies.data.windows import TokenWindows

        stream = self._ensure_stream(ctx, split)
        if stream is None:
            return None
        block = int(ctx.task_params["block_size"])
        n_windows = int(stream.tokens.shape[0]) // block
        if n_windows < 1:
            raise ExperimentError(
                f"The {split} stream is shorter than one block of {block}.",
                stage="data",
            )
        if split == "train":
            requested = self.samples_per_epoch
            per_epoch = min(15_625, n_windows) if requested is None else int(requested)
            if per_epoch < 1:
                raise ExperimentError("samples_per_epoch must be >= 1.", stage="data")
            return TokenWindows(
                stream, block_size=block, samples_per_epoch=per_epoch,
                shuffle_within_epoch=True, seed=int(getattr(ctx, "seed", 0)),
            )

        class _PinnedWindows(TokenWindows):
            def set_epoch(self, epoch: int) -> None:
                super().set_epoch(0)

        cap = max(1, int(self.val_max_tokens) // block)
        return _PinnedWindows(
            stream, block_size=block, samples_per_epoch=min(n_windows, cap),
            shuffle_within_epoch=False, seed=0,
        )

    def _ensure_stream(self, ctx: Any, split: str) -> Any:
        from topologies.data.stream import _tokenizer_fingerprint, write_token_stream

        source = {"train": self.train_source, "val": self.val_source,
                  "test": self.test_source}[split]
        if source is None:
            return None
        if _is_token_stream(source):
            _check_stream_tokenizer(source, ctx.tokenizer)
            return source
        if isinstance(source, dict):
            source = _LabeledTexts(source)
        cache_dir = getattr(getattr(ctx, "cache", None), "dir", None)
        if cache_dir is None:
            raise ExperimentError(
                "Stream storage needs an experiment cache directory.",
                stage="data",
            )
        key = stable_hash({
            "corpus": source.fingerprint(),
            "tokenizer": _tokenizer_fingerprint(ctx.tokenizer),
            "add_eos": True,
        })
        path = Path(cache_dir) / "streams" / key
        stream = write_token_stream(
            source, ctx.tokenizer, path, add_eos=True, batch_docs=1024,
            workers=self.stream_workers,
        )
        _check_stream_tokenizer(stream, ctx.tokenizer)
        return stream

    def _ratios(self, ctx: Any, split: str) -> tuple[float, float] | None:
        if self.storage == "stream":
            stream = self._ensure_stream(ctx, split)
            if stream is None or int(stream.meta.get("n_tokens") or 0) < 1:
                return None
            n = int(stream.meta["n_tokens"])
            return stream.meta["n_bytes"] / n, stream.meta["n_chars"] / n
        docs = self._split(ctx, split)
        if not docs:
            return None

        def compute() -> tuple[float, float]:
            from topologies.tokenizers.stats import corpus_counts

            c = corpus_counts(ctx.tokenizer, docs)
            t = max(1, c["tokens"])
            return c["bytes"] / t, c["chars"] / t

        key = stable_hash({"tok": ctx.tokenizer_key, "split": split,
                           "data": self.corpus_fingerprint(ctx) if split == "train"
                           else self._fp[split]})
        return ctx.cache.get_or_create(f"ratio-{key}", compute)

    def bytes_per_token(self, ctx: Any) -> float | None:
        r = self._ratios(ctx, "train")
        return r[0] if r else None

    def tokens_per_sample(self, ctx: Any) -> int | None:
        return int(ctx.task_params["block_size"])

    def derive(self, ctx: Any, metrics: dict[str, float]) -> dict[str, float]:
        from topologies.metrics import bits_per_byte

        out: dict[str, float] = {}
        for prefix in ("train_", "val_", "test_"):
            loss = metrics.get(f"{prefix}loss")
            if loss is None or not isinstance(loss, (int, float)) or not math.isfinite(loss):
                continue
            r = self._ratios(ctx, prefix[:-1])
            if r is None:
                continue
            out[f"{prefix}bpb"] = bits_per_byte(loss, r[0])
            out[f"{prefix}bpc"] = bits_per_byte(loss, r[1])
        return out

    def evaluate(self, ctx: Any, trainer: Any) -> dict[str, float]:
        out = super().evaluate(ctx, trainer)
        if self.storage == "stream" and self.group_by == "label":
            stream = self._ensure_stream(ctx, "val")
            if stream is not None:
                out.update(_grouped_val_bpb(
                    trainer.model, stream, int(ctx.task_params["block_size"]),
                    on_progress=_monitor_progress(trainer, "val bpb"),
                ))
        try:
            from topologies.tokenizers.stats import tokenizer_stats

            stats_docs = self.val_docs or (
                self._text_sample(self.val_source or self.train_source, 2000)
                if self.storage == "stream" else self._train_docs(ctx)
            )
            stats = tokenizer_stats(ctx.tokenizer, stats_docs, sample=2000)
            stats["train_s"] = ctx.info.get("tokenizer_train_s")
            ctx.extras["tokenizer"] = stats
        except Exception:  # noqa: BLE001, S110 - stats are informational
            pass
        if self.sample_prompts:
            ctx.extras["samples"] = self._samples(ctx, trainer)
        return out

    def _samples(self, ctx: Any, trainer: Any) -> dict[str, dict[str, str]]:
        from topologies.generation import generate

        out: dict[str, dict[str, str]] = {}
        for prompt in self.sample_prompts:
            row: dict[str, str] = {}
            for kind, kw in (("greedy", {"greedy": True}),
                             ("sampled", {"temperature": 0.8, "top_k": 40, "seed": 0})):
                try:
                    row[kind] = generate(trainer.model, prompt, tokenizer=ctx.tokenizer,
                                         max_new_tokens=self.sample_tokens,
                                         context_length=int(ctx.task_params["block_size"]),
                                         truncate_prompt=True, **kw)
                except Exception as e:  # noqa: BLE001
                    row[kind] = f"<generation failed: {type(e).__name__}: {e}>"
            out[prompt] = row
        return out

    def _text_sample(self, source: Any, limit: int) -> list[str]:
        if source is None or _is_token_stream(source):
            return []
        if isinstance(source, dict):
            source = _LabeledTexts(source)
        if _is_text_corpus(source) or hasattr(source, "iter_texts"):
            texts: list[str] = []
            for text in source.iter_texts():
                texts.append(text)
                if len(texts) >= limit:
                    break
            return texts
        return []

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"train_docs": len(self.train_docs), "val_docs": len(self.val_docs),
                  "test_docs": len(self.test_docs or []),
                  "default_tokenizer": self.default_tokenizer
                  if isinstance(self.default_tokenizer, (str, dict)) else
                  type(self.default_tokenizer).__name__})
        if self.storage == "stream":
            d["storage"] = "stream"
        return d


def _check_stream_tokenizer(stream: Any, tokenizer: Any) -> None:
    from topologies.data.stream import _tokenizer_fingerprint

    expected = stream.meta.get("tokenizer_fingerprint")
    actual = _tokenizer_fingerprint(tokenizer)
    if expected and expected != actual:
        raise ExperimentError(
            "This token stream was written with a different tokenizer.",
            stage="data",
            hint="Pass a TextCorpus so the trial can encode it, or reuse the same tokenizer.",
        )


# ---------------------------------------------------------------- image classification


def _first_image(ds: Any) -> Any:
    try:
        s = ds[0]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(s, dict):
        return s.get("pixels", s.get("images", s.get("image")))
    if isinstance(s, (tuple, list)) and s:
        return s[0]
    return None


def _infer_num_classes(ds: Any) -> int | None:
    classes = getattr(ds, "classes", None)
    if classes is not None and not callable(classes):
        try:
            return len(classes)
        except TypeError:
            pass
    try:
        from topologies.data.splits import dataset_labels

        labels = dataset_labels(ds)
        return (max(labels) + 1) if labels else None
    except Exception:  # noqa: BLE001
        return None


class _Augmented:
    """Applies ``augment(image)`` to training samples (``(x, y)`` or dict)."""

    def __init__(self, base: Any, augment: Callable[[Any], Any]) -> None:
        self.base, self.augment = base, augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Any:
        s = self.base[i]
        if isinstance(s, dict):
            key = "pixels" if "pixels" in s else "images"
            return {**s, key: self.augment(s[key])}
        x, *rest = s
        return (self.augment(x), *rest)

    def __getattr__(self, name: str) -> Any:
        if name in ("base", "augment") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)


class ImageClassification(Task):
    """Image classification on any ``(image, label)`` dataset.

    Infers ``num_classes`` (``classes`` / ``targets`` / max label),
    ``in_channels`` and ``img_size`` from the first sample, and injects them
    (plus ``in_features = C*H*W`` and ``out_features = num_classes`` for
    MLPs) into models that accept them. Recipe ``vision-classify``, goal
    ``("val_acc", "max")``; extras hold a per-class report and the confusion
    matrix on the validation set. ``augment(image) -> image`` applies to
    training samples only.
    """

    type = "image-classification"
    recipe = "vision-classify"
    default_goal = ("val_acc", "max")

    def __init__(self, train: Any, val: Any = None, test: Any = None, *,
                 val_fraction: float = 0.1, stratify: bool = True,
                 num_classes: int | None = None, train_subset: float | None = None,
                 augment: Callable[[Any], Any] | None = None, class_names: Any = None) -> None:
        super().__init__(train_subset=train_subset, val_fraction=val_fraction,
                         augment=augment)
        if val is None:
            from topologies.data.splits import train_val_split

            train, val = train_val_split(train, val_fraction, seed=0, stratify=stratify)
        self.train, self.val, self.test = train, val, test
        self.stratify = stratify
        self.num_classes = num_classes or _infer_num_classes(train)
        img = _first_image(train)
        shape = tuple(getattr(img, "shape", ()) or ())
        if len(shape) == 2:
            shape = (1, *shape)
        self.in_channels = int(shape[0]) if len(shape) == 3 else None
        self.img_size = int(shape[1]) if len(shape) == 3 and shape[1] == shape[2] else None
        self.in_features = int(math.prod(shape)) if shape else None
        self.class_names = list(class_names) if class_names is not None else (
            list(getattr(train, "classes", None) or []) or None)
        self._fp = {"train": fingerprint_data(train), "val": fingerprint_data(val),
                    "test": fingerprint_data(test) if test is not None else None}

    def fingerprint(self) -> str:
        return stable_hash({"type": self.type, "data": self._fp, "stratify": self.stratify,
                            "params": _canon_params(self.params)})

    def geometry(self, ctx: Any) -> dict[str, Any]:
        geo: dict[str, Any] = {}
        if self.num_classes:
            geo["num_classes"] = self.num_classes
            geo["out_features"] = self.num_classes
        if self.in_channels:
            geo["in_channels"] = self.in_channels
        if self.img_size:
            geo["img_size"] = self.img_size
        if self.in_features:
            geo["in_features"] = self.in_features
        return geo

    def build_data(self, ctx: Any) -> Datasets:
        p = ctx.task_params
        train = self.train
        if p.get("train_subset") is not None:
            from topologies.data.splits import take_subset

            train = take_subset(train, p["train_subset"], seed=0, stratify=self.stratify)
        if p.get("augment") is not None:
            train = _Augmented(train, p["augment"])
        return Datasets(train=train, val=self.val, test=self.test)

    def evaluate(self, ctx: Any, trainer: Any) -> dict[str, float]:
        out = super().evaluate(ctx, trainer)
        try:
            import torch

            from topologies.data.splits import dataset_labels
            from topologies.metrics import per_class_report

            logits = trainer.predict(self.val)
            labels = torch.as_tensor(dataset_labels(self.val))
            rep = per_class_report(logits, labels, class_names=self.class_names,
                                   num_classes=self.num_classes)
            ctx.extras["per_class_report"] = rep.as_dict()
            ctx.extras["confusion_matrix"] = rep.matrix.tolist()
        except Exception:  # noqa: BLE001, S110 - informational extras
            pass
        return out

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        n = {}
        for k in ("train", "val", "test"):
            try:
                n[k] = len(getattr(self, k)) if getattr(self, k) is not None else 0
            except Exception:  # noqa: BLE001
                n[k] = None
        d.update({"samples": n, "num_classes": self.num_classes,
                  "in_channels": self.in_channels, "img_size": self.img_size})
        return d


# ---------------------------------------------------------------- generic


class Supervised(Task):
    """Any recipe and dataset; injects only ``geometry`` (dict or ``fn(ctx) -> dict``).

    >>> Supervised(train_ds, val_ds, recipe="vision-classify", goal=("val_acc", "max"),
    ...            geometry={"num_classes": 10})
    """

    type = "supervised"

    def __init__(self, train: Any, val: Any = None, test: Any = None, *, recipe: str,
                 goal: tuple[str, str] | str = ("val_loss", "min"),
                 geometry: dict[str, Any] | Callable[[Any], dict] | None = None) -> None:
        super().__init__()
        self.train, self.val, self.test = train, val, test
        self.recipe = recipe
        if isinstance(goal, str):
            from topologies.experiments.metrics import metric_info

            goal = (goal, metric_info(goal).mode)
        self.default_goal = (str(goal[0]), str(goal[1]))
        self._geometry = geometry
        self._fp = {"train": fingerprint_data(train), "val": fingerprint_data(val),
                    "test": fingerprint_data(test) if test is not None else None}

    def fingerprint(self) -> str:
        from topologies.experiments.spec import canonical

        geo = self._geometry if not callable(self._geometry) else canonical(
            self._geometry, warn=False)
        return stable_hash({"type": self.type, "recipe": self.recipe, "data": self._fp,
                            "geometry": geo})

    def geometry(self, ctx: Any) -> dict[str, Any]:
        if self._geometry is None:
            return {}
        if callable(self._geometry):
            return dict(self._geometry(ctx))
        return dict(self._geometry)

    def build_data(self, ctx: Any) -> Datasets:
        return Datasets(train=self.train, val=self.val, test=self.test)


__all__ = ["GEOMETRY_ALIASES", "Datasets", "ImageClassification", "LanguageModeling",
           "Supervised", "Task"]
