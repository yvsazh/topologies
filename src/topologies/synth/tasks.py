"""Experiment tasks over synthetic datasets: train on synthetic, evaluate on real (TSTR).

    task = synth.SynthTask(spec, eval=real_eval)          # or a SynthDataset / dataset name
    exp = Experiment("intent-data", task=task,
                     axes={"synth.generator.noise.rate": [0.0, 0.05],
                           "synth.size": [2000, 8000],
                           "synth.ops.dedup.near.threshold": [0.8, 0.9],
                           "synth.mix.real": [0, 0.3]},
                     model="TransformerClassifier", size="tiny", train_for="3 epochs")
    res = exp.run(); res.pareto(x="synth_seconds")

The recipe and goal follow the schema: ``Label`` + text -> ``text-classify`` /
``val_acc``; ``Label`` + numeric fields -> ``tabular-classify``; ``Labels`` ->
``multilabel-classify``; ``Spans`` -> ``token-classify``; ``Json`` / text output
-> ``seq2seq-lm`` / ``val_exact_match``; ``Tokens`` + ``labels`` -> ``causal-lm`` /
``val_acc_answer``; ``Series`` -> ``regression``; ``Image`` + ``Label`` ->
``vision-classify``. ``val_*`` metrics are measured on the **real** eval
records; ``synth_val_*`` on the synthetic val split when it exists.

Every distinct spec variant (``synth.*`` keys) is generated once - its version
id is content-addressed, so all trials and seeds that use it share it - and
its generation time lands in each trial's ``cost.synth_seconds``. ``exp.plan()``
reports variants that are not generated yet with their projected time.
"""

from __future__ import annotations

import math
import multiprocessing
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from topologies.experiments.cache import fingerprint_data, stable_hash
from topologies.experiments.tasks import Datasets, Task, _canon_params
from topologies.synth.errors import SynthError
from topologies.synth.records import as_records, fingerprint_records


class _BoundedRecipeStream:
    """Length-bounded view of a recipe stream; generation stays per-item."""

    def __init__(self, base: Any, length: int, fit_seconds: float) -> None:
        self.base, self.length = base, int(length)
        self.fit_seconds = float(fit_seconds)
        self._seconds = multiprocessing.get_context("spawn").Value("d", self.fit_seconds)
        self.classes, self.tokenizer = base.classes, base.tokenizer
        self.info = dict(base.info)
        if hasattr(base, "collate_fn"):
            self.collate_fn = base.collate_fn

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Any:
        if not -self.length <= i < self.length:
            raise IndexError(i)
        if i < 0:
            i += self.length
        start = time.perf_counter()
        value = self.base[i]
        with self._seconds.get_lock():
            self._seconds.value += time.perf_counter() - start
        return value

    @property
    def generation_seconds(self) -> float:
        return self._seconds.value

    def reset_timing(self) -> None:
        with self._seconds.get_lock():
            self._seconds.value = self.fit_seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "base"), name)

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:
        if any(not -self.length <= i < self.length for i in indices):
            raise IndexError("Batch index outside stream training length")
        indices = [i + self.length if i < 0 else i for i in indices]
        start = time.perf_counter()
        values = self.base.__getitems__(indices)
        with self._seconds.get_lock():
            self._seconds.value += time.perf_counter() - start
        return values

KINDS = {
    "text-classify": ("val_acc", "max"),
    "tabular-classify": ("val_acc", "max"),
    "multilabel-classify": ("val_f1", "max"),
    "token-classify": ("val_acc", "max"),
    "seq2seq-lm": ("val_exact_match", "max"),
    "causal-lm": ("val_acc_answer", "max"),
    "regression": ("val_loss", "min"),
    "vision-classify": ("val_acc", "max"),
}
TEXT_RECIPES = {"text-classify", "token-classify", "multilabel-classify", "seq2seq-lm",
                "causal-lm"}


def infer_recipe(schema: Any) -> str:
    """Default recipe for a record schema (see module docstring)."""
    if schema is None:
        raise SynthError("Cannot infer a recipe without a schema.", stage="task",
                         hint="Pass recipe=... to SynthTask.")
    kinds = {f.kind for f in schema.fields.values()}
    if "image" in kinds:
        return "vision-classify"
    if "spans" in kinds:
        return "token-classify"
    if "labels" in kinds:
        return "multilabel-classify"
    if "tokens" in kinds:
        return "causal-lm"
    if "label" in kinds:
        return "text-classify" if "text" in kinds else "tabular-classify"
    if "series" in kinds:
        return "regression"
    if "json" in kinds or len(schema.of_kind("text")) >= 2:
        return "seq2seq-lm"
    if "messages" in kinds:
        return "causal-lm"
    if "number" in kinds:
        return "regression"
    raise SynthError(f"No default recipe for a schema with fields {sorted(kinds)}.",
                     stage="task", hint="Pass recipe=... to SynthTask.")


def _slice_values(r: dict, fields: Sequence[str] | None) -> dict[str, Any]:
    out = {}
    attrs = r.get("_attrs") or {}
    for k, v in attrs.items():
        if isinstance(v, (str, int, bool)) and (fields is None or k in fields):
            out[k] = v
    for k in fields or ():
        if k in r and k not in out:
            out[k] = r[k]
    return out


class SynthTask(Task):
    """Train on a synthetic dataset, evaluate on real records (see module docstring)."""

    type = "synth"
    accepts_synth = True

    def __init__(self, source: Any, eval: Any = None, test: Any = None, *,
                 recipe: str | None = None, tokenizer: Any = "bpe:4k", max_len: int = 128,
                 goal: str | tuple[str, str] | None = None, real_train: Any = None,
                 train_split: str = "train", slices: Sequence[str] | None = None,
                 builder: dict[str, Any] | None = None, generate: bool = True,
                 label_field: str | None = None, train_length: int | None = None,
                 fit: int = 1000) -> None:
        super().__init__(max_len=int(max_len))
        from topologies.synth.dataset import SynthDataset, load
        from topologies.synth.spec import Spec
        from topologies.synth.streaming import Stream

        self.spec: Spec | None = None
        self.fixed_ds: SynthDataset | None = None
        self.stream: Stream | None = None
        if isinstance(source, Spec):
            self.spec = source
        elif isinstance(source, Stream):
            self.stream = source
            if source.n is None and train_length is None:
                raise SynthError("An infinite training stream needs train_length=... .",
                                 stage="task", hint="Set an explicit finite epoch length.")
            if source.n is not None and train_length is not None:
                train_length = min(int(train_length), source.n)
            self.train_length = int(train_length if train_length is not None else source.n)
            self.fit_records = max(1, int(fit))
        elif isinstance(source, SynthDataset):
            self.fixed_ds = source
        elif isinstance(source, str):
            self.fixed_ds = load(source)
        else:
            raise SynthError(f"SynthTask needs a Spec, a SynthDataset or a dataset name; got "
                             f"{type(source).__name__}.", stage="task")
        schema = self.spec.schema if self.spec is not None else (
            self.stream.schema if self.stream is not None else self.fixed_ds.schema)
        self.schema = schema
        if self.stream is None:
            self.train_length = None
            self.fit_records = max(1, int(fit))
        self.recipe = recipe or infer_recipe(schema)
        if goal is None:
            goal = KINDS.get(self.recipe, ("val_loss", "min"))
        elif isinstance(goal, str):
            from topologies.experiments.metrics import metric_info

            goal = (goal, metric_info(goal).mode)
        self.default_goal = (str(goal[0]), str(goal[1]))
        self.eval_records = as_records(eval, what="eval") if eval is not None else []
        self.test_records = as_records(test, what="test") if test is not None else []
        self.real_train = as_records(real_train, what="real_train") if real_train is not None \
            else []
        if not self.eval_records:
            raise SynthError("SynthTask needs real eval records (eval=...).", stage="task",
                             hint="TSTR = train on synthetic, test on real: pass "
                                  "eval=synth.load_records('real.jsonl').")
        self.text = self.recipe in TEXT_RECIPES and not (
            schema is not None and "tokens" in {f.kind for f in schema.fields.values()})
        self.default_tokenizer = tokenizer if self.text else None
        self.consumes = frozenset({"tokenizer"}) if self.text else frozenset()
        self.train_split, self.slices = train_split, list(slices) if slices else None
        self.builder = dict(builder or {})
        if label_field:
            self.builder["label_field"] = label_field
        self.generate = bool(generate)
        self._fp = {"eval": fingerprint_records(self.eval_records),
                    "test": fingerprint_records(self.test_records),
                    "real_train": fingerprint_records(self.real_train)}
        self._datasets: dict[str, Any] = {}
        self._generation_seconds: dict[str, float] = {}

    # -- identity -------------------------------------------------------------------
    def fingerprint(self) -> str:
        src = self.spec.fingerprint() if self.spec is not None else (
            self.fixed_ds.version_id if self.fixed_ds is not None else
            stable_hash({"generator": self.stream.gen.fingerprint(), "seed": self.stream.seed,
                         "attributes": self.stream.attributes.describe(),
                         "sampler": self.stream.sampler_spec.describe(),
                         "schema": self.schema.to_config() if self.schema else None,
                         "n": self.stream.n, "epoch": self.stream._epoch,
                         "train_length": self.train_length}))
        return stable_hash({"type": self.type, "source": src, "recipe": self.recipe,
                            "data": self._fp, "builder": _canon_params(self.builder),
                            "fit": self.fit_records, "train_length": self.train_length,
                            "split": self.train_split, "params": _canon_params(self.params)})

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"source": self.spec.name if self.spec is not None else
                  (self.fixed_ds.name if self.fixed_ds is not None else repr(self.stream)),
                  "eval": len(self.eval_records), "test": len(self.test_records),
                  "real_train": len(self.real_train)})
        return d

    # -- variants -------------------------------------------------------------------
    def overrides(self, params: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """``(spec overrides, real share)`` from a trial's ``synth.*`` keys."""
        ov = {k[len("synth."):]: v for k, v in params.items() if k.startswith("synth.")}
        mix = float(ov.pop("mix.real", 0.0) or 0.0)
        return ov, mix

    def variant(self, params: dict[str, Any]) -> Any:
        ov, _ = self.overrides(params)
        if self.spec is None:
            if ov:
                raise SynthError("synth.* axes need a SynthTask built from a Spec.",
                                 stage="task", hint="SynthTask(spec, eval=...)")
            return None
        return self.spec.with_(**ov) if ov else self.spec

    def _variant_stream(self, params: dict[str, Any]) -> tuple[Any, int]:
        from topologies.synth.errors import unknown
        from topologies.synth.spec import Spec
        from topologies.synth.streaming import Stream

        ov, mix = self.overrides(params)
        if mix:
            raise SynthError("Stream training does not support synth.mix.real.", stage="task")
        allowed = {"size", "seed", "attributes", "sampler", "schema", "generator"}
        for key in ov:
            if key.split(".")[0] not in allowed:
                raise unknown("stream axis", key, sorted(allowed), stage="task")
        source = self.stream
        variant = Spec("stream-variant", generator=source.gen, size=self.train_length,
                       seed=source.seed, attributes=source.attributes, sampler=source.sampler_spec,
                       schema=source.schema, monitor="none").with_(**ov)
        length = int(variant.size)
        if length < 1 or (source.n is not None and length > source.n):
            raise SynthError("Stream training length must be positive and within its finite source.",
                             stage="task", hint="Use a larger source or a smaller synth.size.")
        stream = Stream(variant.generator, source.n, seed=variant.seed,
                        attributes=variant.attributes, sampler=variant.sampler_spec,
                        schema=variant.schema, epoch=source._epoch)
        return stream, length

    def dataset(self, params: dict[str, Any], *, converter_kw: dict[str, Any] | None = None) -> Any:
        if self.stream is not None:
            stream, length = self._variant_stream(params)
            kw = dict(self.builder)
            kw.update(converter_kw or {})
            if self.text and kw.get("tokenizer") is None:
                kw["tokenizer"] = self.default_tokenizer
            if self.text:
                kw.setdefault("max_len", int(self.params.get("max_len", 128)))
            canonical_kw = {k: v.to_dict() if hasattr(v, "to_dict") else v
                            for k, v in kw.items()}
            key = stable_hash({"source": self.fingerprint(), "variant": self.overrides(params)[0],
                               "epoch": stream._epoch, "converter": _canon_params(canonical_kw),
                               "length": length, "recipe": self.recipe,
                               "builder": _canon_params(self.builder)})
            if key not in self._datasets:
                start = time.perf_counter()
                fitted = stream.for_recipe(self.recipe, fit=min(self.fit_records, length),
                                           **kw)
                ds = _BoundedRecipeStream(fitted, length,
                                          time.perf_counter() - start)
                ds.info["synth_version"] = key
                self._datasets[key] = ds
            return self._datasets[key]
        if self.fixed_ds is not None:
            return self.fixed_ds
        spec = self.variant(params)
        vid = spec.version_id()
        ds = self._datasets.get(vid)
        if ds is not None:
            return ds
        store = spec.store()
        if store.has_version(vid):
            from topologies.synth.dataset import SynthDataset

            ds = SynthDataset(store, vid)
        elif self.generate:
            ds = spec.run(monitor="none")
        else:
            raise SynthError(f"Synthetic dataset {spec.name!r} ({vid}) is not generated.",
                             stage="data", hint="Run spec.run() first or SynthTask(..., "
                                                "generate=True).")
        self._datasets[vid] = ds
        return ds

    # -- data -------------------------------------------------------------------------
    def _train_records(self, ctx: Any) -> list[dict]:
        ds = self.dataset(ctx.params)
        split = self.train_split if self.train_split in ds.splits else None
        recs = ds.to_records(split)
        _, mix = self.overrides(ctx.params)
        if mix > 0:
            if not self.real_train:
                raise SynthError("synth.mix.real needs SynthTask(real_train=...).",
                                 stage="data")
            import random

            if mix >= 1:
                return list(self.real_train)
            k = round(len(recs) * mix / (1 - mix))
            rng = random.Random(0)
            real = [rng.choice(self.real_train) for _ in range(k)] if k > len(
                self.real_train) else rng.sample(self.real_train, k)
            recs = recs + real
            rng.shuffle(recs)
        return recs

    def tokenizer_corpus(self, ctx: Any) -> list[str]:
        from topologies.synth.torch_views import as_text, text_field

        if self.stream is not None:
            stream, length = self._variant_stream(ctx.params)
            recs = stream.sample(min(self.fit_records, length))
            lf = self.schema.label_field() if self.schema is not None else None
            fields = [f for f in (self.schema.of_kind("text", "json", "messages")
                                  if self.schema is not None else []) if f != lf]
            if not fields:
                fields = [text_field(recs, self.schema, None)]
            return [as_text(r.get(f, "")) for r in recs for f in fields
                    if r.get(f) is not None]
        recs = self._train_records(ctx)
        lf = self.schema.label_field() if self.schema is not None else None
        fields = [f for f in (self.schema.of_kind("text", "json", "messages") if
                              self.schema is not None else []) if f != lf]
        if not fields:
            fields = [text_field(recs, self.schema, None)]
        return [as_text(r.get(f, "")) for r in recs for f in fields if r.get(f) is not None]

    def corpus_fingerprint(self, ctx: Any) -> str:
        if self.stream is not None:
            return stable_hash({"source": self.fingerprint(),
                                "variant": self.overrides(ctx.params)[0],
                                "fit": self.fit_records})
        ds = self.dataset(ctx.params)
        return stable_hash({"version": ds.version_id, "mix": self.overrides(ctx.params)[1],
                            "real": self._fp["real_train"]})

    def _build(self, records: list[dict], ctx: Any, like: Any = None) -> Any:
        from topologies.synth.torch_views import build_dataset

        kw = dict(self.builder)
        if self.text:
            kw.setdefault("tokenizer", ctx.tokenizer)
            kw.setdefault("max_len", int(ctx.task_params.get("max_len", 128)))
        if like is not None:
            info = getattr(like, "info", {}) or {}
            if getattr(like, "classes", None) is not None and self.recipe != "token-classify":
                kw.setdefault("labels", like.classes)
            if self.recipe == "token-classify" and getattr(like, "classes", None):
                kw.setdefault("tags", like.classes)
            for key in ("columns",):
                if key in info:
                    kw.setdefault(key, info[key])
            if self.recipe == "tabular-classify" and "categorical_names" in info:
                kw.setdefault("categorical", info["categorical_names"])
        return build_dataset(records, self.recipe, schema=self.schema, **kw)

    def build_data(self, ctx: Any) -> Datasets:
        if self.stream is not None:
            train = self.dataset(ctx.params, converter_kw={"tokenizer": ctx.tokenizer,
                                 "max_len": int(ctx.task_params.get("max_len", 128))}
                                 if self.text else {})
            train.reset_timing()
            # Stream samples are converted lazily; only eval/test records are materialized.
            val = self._build(self.eval_records, ctx, like=train)
            test = self._build(self.test_records, ctx, like=train) if self.test_records else None
            return Datasets(train=train, val=val, test=test,
                            info={"synth_version": train.info["synth_version"]})
        recs = self._train_records(ctx)
        train = self._build(recs, ctx)
        if isinstance(getattr(train, "info", None), dict):
            train.info["record_ids"] = [r.get("_id") for r in recs]
        val = self._build(self.eval_records, ctx, like=train)
        test = self._build(self.test_records, ctx, like=train) if self.test_records else None
        ds = self.dataset(ctx.params)
        info: dict[str, Any] = {"synth_version": ds.version_id}
        if "val" in ds.splits and self.train_split != "val":
            try:
                info["synth_val"] = self._build(ds.to_records("val"), ctx, like=train)
            except SynthError:
                pass
        return Datasets(train=train, val=val, test=test, info=info)

    def geometry(self, ctx: Any) -> dict[str, Any]:
        train = ctx.datasets.train
        info = dict(getattr(train, "info", {}) or {})
        geo: dict[str, Any] = {}
        tok = ctx.tokenizer if self.text else None
        if tok is not None:
            geo["vocab_size"] = int(tok.vocab_size)
        elif info.get("vocab_size"):
            geo["vocab_size"] = int(info["vocab_size"])
        if self.text or info.get("max_seq"):
            geo["max_seq"] = int(info.get("max_seq") or ctx.task_params.get("max_len", 128))
        classes = getattr(train, "classes", None)
        if classes:
            geo["num_classes"] = len(classes)
            geo["out_features"] = len(classes)
        for k in ("in_features", "in_channels", "out_features", "cardinalities"):
            if info.get(k) is not None and k not in geo:
                geo[k] = info[k]
        if self.recipe == "vision-classify":
            x = train[0]["pixels"]
            geo.update({"in_channels": int(x.shape[0]), "img_size": int(x.shape[-1]),
                        "in_features": int(x.numel())})
        return geo

    def tokens_per_sample(self, ctx: Any) -> int | None:
        return int(ctx.task_params.get("max_len", 128)) if self.text else None

    # -- evaluation ---------------------------------------------------------------------
    def derive(self, ctx: Any, metrics: dict[str, float]) -> dict[str, float]:
        out = {}
        if self.recipe == "causal-lm":
            for p in ("val_", "test_", "synth_val_"):
                if f"{p}acc" in metrics:
                    out[f"{p}acc_answer"] = metrics[f"{p}acc"]
        return out

    def evaluate(self, ctx: Any, trainer: Any) -> dict[str, float]:
        out = super().evaluate(ctx, trainer)
        sv = ctx.datasets.info.get("synth_val") if hasattr(ctx.datasets, "info") else None
        if sv is not None:
            try:
                out.update(trainer.evaluate(sv, prefix="synth_val_"))
            except Exception:  # noqa: BLE001, S110 - informational
                pass
        out.update(self.derive(ctx, out))
        try:
            self._analysis(ctx, trainer)
        except Exception:  # noqa: BLE001, S110 - informational extras
            pass
        return out

    def _analysis(self, ctx: Any, trainer: Any) -> None:
        """Per-class report, confusion matrix and per-slice error rates on the real eval."""
        if self.recipe not in ("text-classify", "tabular-classify", "vision-classify"):
            return
        import torch

        from topologies.metrics import per_class_report

        val = ctx.datasets.val
        logits = trainer.predict(val)
        labels = torch.tensor([int(s["labels"]) for s in val.samples])
        pred = logits.argmax(-1)
        rep = per_class_report(logits, labels, class_names=val.classes,
                               num_classes=len(val.classes))
        ctx.extras["per_class_report"] = rep.as_dict()
        ctx.extras["confusion_matrix"] = rep.matrix.tolist()
        ctx.extras["classes"] = list(val.classes)
        slices: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r, p, y in zip(self.eval_records, pred.tolist(), labels.tolist()):
            for k, v in _slice_values(r, self.slices).items():
                cell = slices[k][str(v)]
                cell[0] += 1
                cell[1] += int(p != y)
        ctx.extras["slices"] = {k: {v: {"n": n, "errors": e, "error_rate": e / n}
                                    for v, (n, e) in d.items()} for k, d in slices.items()}
        ctx.extras["predictions"] = pred.tolist()
        # Hardest synthetic training records (high loss / low confidence) for the loop.
        train = ctx.datasets.train
        ids = (getattr(train, "info", {}) or {}).get("record_ids")
        if ids and hasattr(train, "samples"):
            n = min(len(train), 5000)
            sub = torch.utils.data.Subset(train, list(range(n)))
            sub.collate_fn = getattr(train, "collate_fn", None)
            lg = trainer.predict(sub)
            y = torch.tensor([int(train.samples[i]["labels"]) for i in range(n)])
            loss = torch.nn.functional.cross_entropy(lg, y, reduction="none")
            conf = torch.softmax(lg, -1).max(-1).values
            top = torch.argsort(loss, descending=True)[:50].tolist()
            ctx.extras["hard_examples"] = [{"id": ids[i], "loss": round(float(loss[i]), 4),
                                            "confidence": round(float(conf[i]), 4)}
                                           for i in top]

    def extra_costs(self, ctx: Any) -> dict[str, Any]:
        """Generation time of the trial's dataset variant (``cost.synth_seconds``)."""
        if self.stream is not None:
            ds = ctx.datasets.train
            secs = float(getattr(ds, "generation_seconds", 0.0))
            return {"synth_seconds": secs, "synth_records": len(ds),
                    "synth_version": ds.info["synth_version"]}
        ds = self.dataset(ctx.params)
        secs = ds.gen_seconds
        return {"synth_seconds": float(secs) if math.isfinite(secs) else None,
                "synth_records": len(ds), "synth_version": ds.version_id}

    # -- planning -----------------------------------------------------------------------
    def plan_check(self, exp: Any, trials: list[Any], plan: Any) -> set[str]:
        """Variants not generated yet: projected time note; model checks deferred."""
        skip: set[str] = set()
        if self.stream is not None:
            for trial in trials:
                try:
                    self._variant_stream(trial.params)
                except SynthError as exc:
                    plan.errors.setdefault(trial.trial_id, f"synth variant: {exc}")
            return skip
        if self.spec is None:
            return skip
        by_vid: dict[str, list[Any]] = defaultdict(list)
        specs: dict[str, Any] = {}
        for t in trials:
            try:
                spec = self.variant(t.params)
            except SynthError as e:
                plan.errors.setdefault(t.trial_id, f"synth variant: {e.message}" +
                                       (f" Hint: {e.hint}" if e.hint else ""))
                continue
            vid = spec.version_id()
            by_vid[vid].append(t)
            specs[vid] = spec
        try:
            infer = infer_recipe(self.schema)
        except SynthError:
            infer = None
        if infer is not None and infer != self.recipe and self.recipe in KINDS and \
                infer in ("text-classify", "tabular-classify") and \
                self.recipe in ("seq2seq-lm", "causal-lm"):
            for t in trials:
                plan.errors.setdefault(t.trial_id, f"recipe {self.recipe!r} does not fit the "
                                       f"schema (a {infer} dataset). Hint: SynthTask(..., "
                                       f"recipe={infer!r}).")
        for vid, members in by_vid.items():
            spec = specs[vid]
            if spec.store().has_version(vid):
                continue
            labels = {k: v for k, v in members[0].labels.items() if k.startswith("synth.")}
            try:
                est = spec.plan(pilot=min(64, spec.size), verbose=False)
                secs = est.projected_seconds
                note = (f"synth variant {labels or spec.name}: not generated yet - "
                        + (f"~{secs:.1f}s" if secs is not None else "time unknown") +
                        f", {est.raw_estimate:,} raw records (generated once, shared by "
                        f"{len(members)} trial(s)); model checks run after generation")
            except Exception as e:  # noqa: BLE001
                note = f"synth variant {labels or spec.name}: plan failed ({e})"
            plan.notes.append(note)
            skip.update(t.trial_id for t in members)
        return skip


class ProbeSuite(Task):
    """Algorithmic probes as one experiments task: the ``task.probe`` axis picks a probe.

        suite = synth.ProbeSuite(["mqar", "induction_heads", "parity"],
                                 difficulties={"mqar": {"n_pairs": [4, 16]}},
                                 length_test={"parity": {"length": 32}})
        exp = Experiment("probes", task=suite,
                         axes={"model": [TransformerLM, "mamba"], "task.probe": suite.probes},
                         size="tiny", train_for="3 epochs")
        res = exp.run(); res.pivot("task.probe", "model")

    Probe ids look like ``"mqar(n_pairs=16)"``. Data is generated in memory
    (vectorized where possible) from fixed seeds, so every model sees the same
    examples. ``vocab_size`` is the suite-wide maximum (identical embeddings
    across probes); ``max_seq`` follows the probe (and its length test). Goal:
    ``val_acc_answer`` - accuracy on answer positions only. ``length_test``
    adds a test split with harder knobs (length generalization: ``test_acc_answer``).
    """

    type = "probe-suite"
    recipe = "causal-lm"
    default_goal = ("val_acc_answer", "max")

    def __init__(self, tasks: Sequence[str], *, difficulties: dict[str, dict[str, list]] |
                 None = None, length_test: dict[str, dict[str, Any]] | None = None,
                 train_size: int = 2000, eval_size: int = 500, seed: int = 0) -> None:
        from topologies.synth.lib import seq as alg

        probes: list[str] = []
        diffs = dict(difficulties or {})
        for t in tasks:
            alg._task(t)
            grid = diffs.get(t)
            if not grid:
                probes.append(t)
                continue
            import itertools

            keys = list(grid)
            for combo in itertools.product(*(grid[k] for k in keys)):
                probes.append(t + "(" + ",".join(f"{k}={v}" for k, v in zip(keys, combo)) + ")")
        self.probes = probes
        self.length_test = {k: dict(v) for k, v in (length_test or {}).items()}
        self.train_size, self.eval_size, self.seed = int(train_size), int(eval_size), int(seed)
        super().__init__(probe=probes[0])
        self._cache: dict[tuple, Any] = {}
        self.vocab = max(alg.vocab_size(*self.parse(p)[:1], **self.parse(p)[1])
                         for p in probes)

    @staticmethod
    def parse(probe: str) -> tuple[str, dict[str, Any]]:
        task, _, rest = str(probe).partition("(")
        rest = rest.rstrip(")")
        knobs: dict[str, Any] = {}
        for part in filter(None, rest.split(",")):
            k, _, v = part.partition("=")
            try:
                knobs[k.strip()] = int(v)
            except ValueError:
                try:
                    knobs[k.strip()] = float(v)
                except ValueError:
                    knobs[k.strip()] = v.strip()
        return task, knobs

    def fingerprint(self) -> str:
        return stable_hash({"type": self.type, "probes": self.probes,
                            "length_test": self.length_test, "train": self.train_size,
                            "eval": self.eval_size, "seed": self.seed,
                            "params": _canon_params(self.params)})

    def _tensors(self, probe: str, split: str) -> Any:
        from topologies.data.adapters import TensorDictDataset
        from topologies.synth.lib import seq as alg

        key = (probe, split)
        if key not in self._cache:
            task, knobs = self.parse(probe)
            if split == "test":
                knobs = {**knobs, **self.length_test.get(task, {})}
            n = self.train_size if split == "train" else self.eval_size
            seed = derive_seed_int(self.seed, probe, split)
            b = alg.batch(task, n, seed=seed, **knobs)
            self._cache[key] = TensorDictDataset(tokens=b["tokens"], labels=b["labels"])
        return self._cache[key]

    def build_data(self, ctx: Any) -> Datasets:
        probe = ctx.task_params["probe"]
        task, _ = self.parse(probe)
        test = self._tensors(probe, "test") if task in self.length_test else None
        return Datasets(train=self._tensors(probe, "train"), val=self._tensors(probe, "val"),
                        test=test, info={"probe": probe})

    def geometry(self, ctx: Any) -> dict[str, Any]:
        probe = ctx.task_params["probe"]
        lens = [int(self._tensors(probe, s).tensors["tokens"].shape[1]) for s in ("train",)]
        task, _ = self.parse(probe)
        if task in self.length_test:
            lens.append(int(self._tensors(probe, "test").tensors["tokens"].shape[1]))
        return {"vocab_size": self.vocab, "max_seq": max(lens)}

    def tokens_per_sample(self, ctx: Any) -> int | None:
        return int(self._tensors(ctx.task_params["probe"], "train").tensors["tokens"].shape[1])

    def derive(self, ctx: Any, metrics: dict[str, float]) -> dict[str, float]:
        return {f"{p}acc_answer": metrics[f"{p}acc"] for p in ("train_", "val_", "test_")
                if f"{p}acc" in metrics}

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"probes": self.probes, "vocab_size": self.vocab,
                  "length_test": self.length_test})
        return d


def derive_seed_int(*parts: Any) -> int:
    from topologies.synth.rng import derive_seed

    return derive_seed(*parts) % (2**31)


def from_synth(name_or_ds: Any, split: str | None = "train", recipe: str | None = None,
               *, root: Any = None, **kw: Any) -> Any:
    """``topologies.data.from_synth``: a torch dataset from a synthetic dataset split."""
    from topologies.synth.dataset import SynthDataset, load

    ds = name_or_ds if isinstance(name_or_ds, SynthDataset) else load(name_or_ds, root=root)
    return ds.for_recipe(recipe or infer_recipe(ds.schema), split=split, **kw)


_ = fingerprint_data

__all__ = ["KINDS", "ProbeSuite", "SynthTask", "from_synth", "infer_recipe"]
