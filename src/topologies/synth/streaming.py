"""Streams: a generator as a torch dataset that renders records on the fly (no storage).

    stream = synth.stream(gen, n=100_000, seed=0, attributes={"label": [...]})
    item = stream[42]                     # deterministic: the raw record 42 of a Spec run
    for batch in stream.loader(batch_size=64, workers=4): ...
    train = stream.for_recipe("regression", target_field="f0")     # fitted on a sample
    endless = synth.stream(gen, n=None)   # IterableDataset; stream.epoch(k) reseeds

``stream.raw(i)`` equals the raw (pre-pipeline) record ``i`` that ``Spec.run()``
generates with the same generator, attributes and seed: both go through the
same evaluation, attribute design and value normalization. Items carry
``_index``, ``_attrs`` and ``_latent``. A ``DataLoader`` batch fetch renders
batched nodes once per batch (``__getitems__``); results equal per-record
rendering. Workers are safe: every index derives its own seeds.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from topologies.synth.errors import SynthError


class Stream(Dataset):
    """Map-style stream of ``n`` records (see module docstring)."""

    def __init__(self, generator: Any, n: int | None = 100_000, *, seed: int = 0,
                 attributes: Any = None, sampler: Any = "auto", schema: Any = None,
                 epoch: int = 0) -> None:
        from topologies.experiments.cache import stable_hash
        from topologies.synth.attributes import AttributeSpace
        from topologies.synth.core import as_generator
        from topologies.synth.pipeline import Generation
        from topologies.synth.sampling import make_sampler
        from topologies.synth.schema import Schema, schema_of

        self.gen = as_generator(generator)
        self.attributes = attributes if isinstance(attributes, AttributeSpace) else \
            AttributeSpace(attributes)
        self.sampler_spec = sampler if not isinstance(sampler, str) else make_sampler(
            sampler, self.attributes)
        self.seed = int(seed)
        self.n = None if n is None else int(n)
        self._epoch = int(epoch)
        if schema is not None and not isinstance(schema, Schema):
            schema = Schema.from_config(schema)
        self.schema = schema if schema is not None else schema_of(self.gen)
        self._sampler_arg = sampler
        gen_fp = stable_hash({"stream": self.gen.fingerprint(), "seed": self.seed})
        self.genr = Generation(self.gen, self.attributes,
                               self.sampler_spec.bind(self.attributes, self.seed), self.seed,
                               self.schema, "raise", gen_fp)

    @classmethod
    def from_spec(cls, spec: Any, n: int | None = None, **kw: Any) -> Stream:
        """The raw generation of a ``Spec`` as a stream (``n`` defaults to ``spec.size``)."""
        return cls(spec.generator, spec.size if n is None else n, seed=spec.seed,
                   attributes=spec.attributes, sampler=spec.sampler_spec, schema=spec.schema,
                   **kw)

    # -- records ---------------------------------------------------------------------
    def _index(self, i: int) -> int:
        if self.n is not None and not -self.n <= i < self.n:
            raise IndexError(f"stream index {i} out of range ({self.n} records)")
        if self.n is not None and i < 0:
            i += self.n
        return i + self._epoch * (self.n or 0)

    def _items(self, indices: Sequence[int], *, batch: bool) -> list[dict[str, Any]]:
        items = self.genr.generate_many(list(indices), batch=batch)
        if len(items) == len(indices) and all(
                it.i == i and it.status == "a" for it, i in zip(items, indices)):
            return [{**it.rec, "_index": i, "_seed": self.seed,
                     "_fanout_index": list(it.j), "_attrs": it.attrs,
                     "_latent": it.lin.get("latent", {})}
                    for it, i in zip(items, indices)]
        by_index: dict[int, list[Any]] = {}
        for it in items:
            by_index.setdefault(it.i, []).append(it)
        out = []
        for i in indices:
            got = by_index.get(i, [])
            if len(got) != 1:
                raise SynthError(f"Record {i} produced {len(got)} records; a stream needs "
                                 "exactly one record per index.", stage="stream",
                                 hint="Fan-out generators (lists of records) and generator "
                                      "rejections work with Spec.run(); use where() to retry "
                                      "draws inside the generator instead.")
            it = got[0]
            if it.status != "a":
                raise SynthError(f"Record {i} was rejected by the generator: "
                                 f"{(it.verdict or {}).get('reason', '')}", stage="stream")
            out.append({**it.rec, "_index": i, "_seed": self.seed,
                        "_fanout_index": list(it.j), "_attrs": it.attrs,
                        "_latent": it.lin.get("latent", {})})
        return out

    def raw(self, i: int) -> dict[str, Any]:
        """Raw record ``i`` (live values + ``_index`` / ``_attrs`` / ``_latent``)."""
        return self._items([self._index(i)], batch=False)[0]

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.raw(i)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Batch fetch (``DataLoader``): batched nodes render once for the whole batch."""
        return self._items([self._index(i) for i in indices], batch=True)

    def __len__(self) -> int:
        if self.n is None:
            raise TypeError("an endless stream (n=None) has no length")
        return self.n

    def sample(self, n: int = 8, *, start: int = 0) -> list[dict[str, Any]]:
        """Records ``start .. start + n - 1`` (batched rendering)."""
        idx = list(range(start, start + n))
        return self._items([self._index(i) for i in idx], batch=True)

    def epoch(self, k: int) -> Stream:
        """The same stream over new records (indices offset by ``k * n``)."""
        if self.n is None:
            raise SynthError("epoch() needs a finite stream (n=...).", stage="stream")
        new = Stream.__new__(Stream)
        new.__dict__.update(self.__dict__)
        new._epoch = int(k)
        return new

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.n is not None:
            for i in range(self.n):
                yield self[i]
            return
        yield from _Endless(self)

    # -- torch -----------------------------------------------------------------------
    def loader(self, batch_size: int = 64, *, workers: int = 0, shuffle: bool = False,
               collate_fn: Any = None, **kw: Any) -> DataLoader:
        """A ``DataLoader`` over the stream (raw records, or pass ``collate_fn``)."""
        if workers:
            import pickle

            try:
                pickle.dumps(self)
            except Exception as e:  # noqa: BLE001
                raise SynthError(f"The stream cannot go to worker processes: {e}",
                                 stage="stream",
                                 hint="Define generator functions at module level (lambdas "
                                      "and local functions do not pickle), or use workers=0."
                                 ) from None
        ds: Any = self if self.n is not None else _Endless(self)
        return DataLoader(ds, batch_size=batch_size, num_workers=workers,
                          shuffle=shuffle if self.n is not None else False,
                          collate_fn=collate_fn if collate_fn is not None else _keep_list,
                          persistent_workers=False, **kw)

    def for_recipe(self, recipe: str | None, *, fit: int = 1000, **kw: Any) -> Any:
        """A torch dataset in ``recipe``'s batch format, converting each record on the fly
        (tokenizer / classes / columns fitted on the first ``fit`` records)."""
        from topologies.synth.torch_views import make_converter

        conv = make_converter(recipe, schema=self.schema, **kw)
        n_fit = fit if self.n is None else min(fit, self.n)
        conv.fit(self.sample(n_fit))
        return RecipeStream(self, conv)

    def __repr__(self) -> str:
        return f"Stream({self.gen.node_kind()}, n={self.n}, seed={self.seed})"


def _keep_list(batch: list[Any]) -> list[Any]:
    return batch


class _Endless(IterableDataset):
    """Infinite iteration over a stream (each worker takes every k-th index)."""

    def __init__(self, stream: Stream, convert: Any = None) -> None:
        self.stream = stream
        self.convert = convert

    def __iter__(self) -> Iterator[Any]:
        info = get_worker_info()
        start, step = (info.id, info.num_workers) if info is not None else (0, 1)
        for i in itertools.count(start, step):
            rec = self.stream._items([i], batch=False)[0]
            yield self.convert(rec) if self.convert is not None else rec


class RecipeStream(Dataset):
    """Stream records converted to a recipe's samples (``stream.for_recipe``)."""

    def __init__(self, stream: Stream, conv: Any) -> None:
        self.stream, self.conv = stream, conv
        self.classes = conv.classes
        self.tokenizer = conv.tokenizer
        self.info = dict(conv.info)
        cf = conv.collate_fn
        if cf is not None:
            self.collate_fn = cf

    def __len__(self) -> int:
        return len(self.stream)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.conv(self.stream[i])

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        return [self.conv(r) for r in self.stream.__getitems__(indices)]

    def endless(self) -> IterableDataset:
        return _Endless(self.stream, self.conv)

    @property
    def targets(self) -> list[int] | None:
        return None


_DEFAULT_N = 100_000


def stream(generator: Any, n: Any = "default", *, seed: int = 0,
           attributes: Mapping[str, Any] | Any = None, sampler: Any = "auto",
           schema: Any = None) -> Stream:
    """A generator (or a ``Spec``) as an on-the-fly torch dataset (see module docstring).

    ``n`` defaults to 100 000 records (``spec.size`` for a spec); ``n=None`` is endless.
    """
    if hasattr(generator, "gen_fingerprint") and hasattr(generator, "generator"):
        return Stream.from_spec(generator, n=None if n == "default" else n) if n is not None \
            else Stream(generator.generator, None, seed=generator.seed,
                        attributes=generator.attributes, sampler=generator.sampler_spec,
                        schema=generator.schema)
    return Stream(generator, _DEFAULT_N if n == "default" else n, seed=seed,
                  attributes=attributes, sampler=sampler, schema=schema)


__all__ = ["RecipeStream", "Stream", "stream"]
