"""Closed loop: train, find where the model fails on real data, generate more of that.

    loop = synth.Loop(spec, real_eval=real_eval, model="TransformerClassifier", size="small",
                      rounds=3, per_round=5_000, strategy="slices+hard")
    history = loop.run()          # resumable per round
    history.show()                # TSTR per round, gains, generation time, slice changes
    history.test(real_test)       # once, at the end: the held-out test set never steers

Each round:

1. train on the current dataset and evaluate on ``real_eval`` (TSTR, per-slice errors with
   tagged attributes, confusion pairs, hardest synthetic training records);
2. update the generation plan - strategies (``"+"``-joined):
   ``slices``   reweight attribute values towards high-error slices (factor bounded by
                ``max_shift``; other values keep their mass);
   ``hard``     anchors: attribute combinations of misclassified *real* records (never
                their text) are required rows of the next round's design;
   ``prune``    confident-learning label issues are dropped;
3. generate ``per_round`` new records (new seed, same pipeline + a contamination check
   against ``real_eval``) and merge them into a new dataset version.

Stops after ``rounds``, when ``max_time`` is spent, when the gain stays below ``plateau``
for two rounds, or when ``target`` is reached. ``contamination_guard=False`` disables the
eval-overlap guard (only for toy data with tiny vocabularies). Only ``real_eval`` steers
decisions; ``history.test(real_test)`` is a separate, single-use call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topologies.experiments.store import now_iso, read_json, write_json_atomic
from topologies.experiments.table import Table
from topologies.synth.errors import SynthError
from topologies.synth.limits import parse_time
from topologies.synth.records import as_records

STRATEGIES = ("slices", "hard", "contrast", "prune")


class LoopHistory:
    def __init__(self, loop: Loop, rounds: list[dict[str, Any]], path: Path) -> None:
        self.loop, self.rounds, self.path = loop, rounds, path

    @property
    def final(self) -> Any:
        from topologies.synth.dataset import load

        last = self.rounds[-1]
        return load(self.loop.spec.name, version=last["version"], root=self.loop.spec.root)

    def table(self) -> Table:
        rows = []
        prev = None
        for r in self.rounds:
            gain = None if prev is None or r["tstr"] is None or prev is None else r["tstr"] - prev
            secs = r.get("gen_seconds") or 0.0
            rows.append([r["round"], r["tstr"], gain, r["records"], secs,
                         ", ".join(r.get("changes", []))[:60]])
            prev = r["tstr"] if r["tstr"] is not None else prev
        return Table(["round", f"TSTR {self.loop.goal}", "gain", "records", "gen s",
                      "plan changes"], rows,
                     meta={"title": f"closed loop  {self.loop.spec.name}",
                           "notes": [("decisions use real_eval only; the held-out test set is "
                                      "used once, by history.test(real_test)")]})

    def slice_changes(self) -> Table:
        if len(self.rounds) < 2:
            return Table(["attribute", "value", "first", "last"], [])
        first = {(s["attribute"], s["value"]): s["error_rate"]
                 for s in self.rounds[0].get("slices", [])}
        last = {(s["attribute"], s["value"]): s["error_rate"]
                for s in self.rounds[-1].get("slices", [])}
        rows = [[a, v, first[(a, v)], last[(a, v)], last[(a, v)] - first[(a, v)]]
                for (a, v) in first if (a, v) in last]
        rows.sort(key=lambda r: r[4])
        return Table(["attribute", "value", "error first", "error last", "Δ"], rows[:15],
                     meta={"title": "slice error change (real eval)"})

    def show(self) -> None:
        print(self.table())
        print(self.slice_changes())

    def test(self, real_test: Any, **kw: Any) -> Any:
        """Evaluate the final dataset on the held-out test set - allowed once."""
        state = read_json(self.path) if self.path.is_file() else {}
        if state.get("test_used"):
            raise SynthError("history.test() was already used for this loop.", stage="loop",
                             hint="The held-out test set must stay untouched by decisions; "
                                  "it is evaluated only once.")
        res = self.final.evaluate(real_test, **{**self.loop.eval_kw, **kw})
        state["test_used"] = now_iso()
        state["test"] = res.tstr
        write_json_atomic(self.path, state)
        return res

    def __repr__(self) -> str:
        return str(self.table())


class Loop:
    """Failure-driven generation (see module docstring)."""

    def __init__(self, spec: Any, *, real_eval: Any, model: Any = "TransformerClassifier",
                 size: Any = "small", rounds: int = 3, per_round: int = 1000,
                 strategy: str = "slices+hard", max_time: Any = None, max_shift: float = 3.0,
                 plateau: float = 0.002, target: float | None = None,
                 train_for: Any = "5 epochs", seeds: Any = 1, tokenizer: Any = "word:8k",
                 max_len: int = 128, name: str | None = None,
                 train: dict | None = None, max_anchors: int = 50,
                 contamination_guard: bool = True) -> None:
        bad = [s for s in strategy.split("+") if s and s not in STRATEGIES]
        if bad:
            from topologies.synth.errors import unknown

            raise unknown("loop strategy", bad[0], STRATEGIES, stage="loop")
        self.spec, self.real_eval = spec, as_records(real_eval, what="real_eval")
        self.strategies = [s for s in strategy.split("+") if s]
        self.rounds, self.per_round = int(rounds), int(per_round)
        self.max_time = parse_time(max_time)
        self.max_shift, self.plateau, self.target = float(max_shift), float(plateau), target
        self.name = name or f"loop-{spec.name}"
        self.max_anchors = int(max_anchors)
        self.guard = bool(contamination_guard)
        self.eval_kw = {"model": model, "size": size, "train_for": train_for, "seeds": seeds,
                        "tokenizer": tokenizer, "max_len": max_len, "train": train,
                        "allow_contamination": not contamination_guard}
        self.goal = "val_acc"

    def _path(self) -> Path:
        return Path(self.spec.store().dir) / "loop" / f"{self.name}.json"

    # -- plan updates ------------------------------------------------------------------
    def _factors(self, res: Any) -> dict[str, dict[Any, float]]:
        """Weight multipliers for high-error attribute values: ``1 + lift / overall error``,
        raised so that a failing value gets at least ``1 / (2K)`` of the mass (rare values
        cannot be fixed by a small multiplier), capped at ``max_shift``."""
        space = self.spec.attributes
        out: dict[str, dict[Any, float]] = {}
        overall = 1 - (res.tstr or {}).get("mean", 0.0) if res.tstr else 0.0
        for s in res.slices:
            a = s["attribute"]
            if a not in space or space[a].kind != "categorical":
                continue
            vals = {str(v): v for v in space.values(a)}
            if s["value"] not in vals or s["lift"] <= 0:
                continue
            v = vals[s["value"]]
            p = max(space[a].prob_of(v), 1e-9)
            floor = 1.0 / (2 * len(vals))
            f = max(1 + s["lift"] / max(overall, 0.02), floor / p)
            out.setdefault(a, {})[v] = min(self.max_shift, max(1.0, f))
        return out

    def _anchors(self, res: Any, ds: Any) -> list[dict[str, Any]]:
        """Attribute combinations of misclassified real records (never their text)."""
        trials = [r for r in res.results.records if r.status == "completed"]
        if not trials or not res.classes:
            return []
        preds = (trials[0].extras or {}).get("predictions") or []
        lf = ds.schema.label_field() if ds.schema is not None else "label"
        from topologies.synth.quality.slices import AttributeTagger

        tagged = AttributeTagger().fit(ds.to_records()).tag(self.real_eval)
        space = self.spec.attributes
        out: list[dict[str, Any]] = []
        for r, p in zip(tagged, preds):
            if res.classes[p] == str(r.get(lf)):
                continue
            combo = {}
            for a, v in (r.get("_attrs") or {}).items():
                if a in space and space[a].kind == "categorical":
                    vals = {str(x): x for x in space.values(a)}
                    if str(v) in vals:
                        combo[a] = vals[str(v)]
            if lf in space and space[lf].kind == "categorical" and r.get(lf) is not None:
                vals = {str(x): x for x in space.values(lf)}
                if str(r.get(lf)) in vals:
                    combo[lf] = vals[str(r.get(lf))]
            if combo and combo not in out:
                out.append(combo)
        return out[: self.max_anchors]

    def _local_pairs(self, ds: Any, *, strategy: str, res: Any = None) -> list[dict[str, Any]]:
        """Replay selected records with one nearby categorical intervention."""
        from topologies.synth.counterfactuals import counterfactual

        rows = ds.to_records()
        scores: dict[str, float] = {}
        for trial in getattr(getattr(res, "results", None), "records", []) or []:
            if trial.status == "completed":
                for hard in ((trial.extras or {}).get("hard_examples") or []):
                    if hard.get("id") is not None:
                        scores[str(hard["id"])] = float(hard.get("loss", 0.0))
        if scores:
            rows.sort(key=lambda row: scores.get(str(row.get("_id")), -1.0), reverse=True)
            rows = [r for r in rows if str(r.get("_id")) in scores]
        elif res is not None:
            anchors = self._anchors(res, ds)
            rows = [r for r in rows if any(all((r.get("_attrs") or {}).get(k) == v
                                               for k, v in anchor.items())
                                           for anchor in anchors)]
            if not rows:
                return []
        rows = rows[:self.max_anchors]
        variants: list[dict[str, Any]] = []
        for row in rows:
            attrs = row.get("_attrs") or {}
            if strategy == "hard":
                latents = row.get("_latent") or {}
                numeric = next(((path, value) for path, value in latents.items()
                                if isinstance(value, (int, float)) and
                                not isinstance(value, bool)), None)
                if numeric is not None:
                    path, value = numeric
                    perturbed = value + max(abs(value) * 0.1, 1 if isinstance(value, int) else 0.1)
                    if isinstance(value, int):
                        perturbed = round(perturbed)
                    try:
                        variant = counterfactual(self.spec, row, **{path: perturbed})
                        variant["_source_id"] = row.get("_id")
                        variant["_intervention"] = {"strategy": strategy, "path": path,
                                                     "from": value, "to": perturbed}
                        variants.append(variant)
                        continue
                    except (SynthError, KeyError, TypeError, ValueError):
                        pass
            candidates = [a for a in self.spec.attributes.names
                          if a in attrs and self.spec.attributes[a].kind == "categorical"]
            if not candidates:
                candidates = [a for a in self.spec.attributes.names if a in attrs and
                              self.spec.attributes[a].kind == "continuous" and
                              isinstance(attrs[a], (int, float)) and not isinstance(attrs[a], bool)]
                if not candidates:
                    continue
                attr = candidates[0]
                old = attrs[attr]
                new = old + max(abs(old) * 0.1, 0.1)
                if isinstance(old, int):
                    new = round(new)
                alternate = new
            else:
                attr = candidates[0]
                options = list(self.spec.attributes.values(attr))
                alternate = next((x for x in options if x != attrs[attr]), None)
            if alternate is None:
                continue
            try:
                variant = counterfactual(self.spec, row, **{attr: alternate})
                variant["_source_id"] = row.get("_id")
                variant["_intervention"] = {"strategy": strategy, "attribute": attr,
                                             "from": attrs[attr], "to": alternate}
                variants.append(variant)
            except (SynthError, KeyError, TypeError, ValueError):
                continue
        return variants

    def _variants_spec(self, variants: list[dict[str, Any]], round_no: int,
                       strategy: str = "contrast") -> Any:
        from topologies.synth.core import generator
        from topologies.synth.generators.base import Raw

        rows = [{k: v for k, v in row.items() if not str(k).startswith("_")}
                for row in variants]
        provenance = [row.get("_intervention", {}) for row in variants]
        sources = [row.get("_source_id") for row in variants]

        @generator(schema=self.spec.schema)
        def emit(r, *, _cf_index):
            i = int(_cf_index)
            return Raw(rows[i], parents=[sources[i]] if sources[i] else [],
                       counterfactual=provenance[i])

        pipeline = list(self.spec.pipeline)
        if self.guard:
            from topologies.synth.ops.contamination import Check

            pipeline.append(Check(self.real_eval, name="contamination.real_eval"))
        return self.spec._copy_with(name=f"{self.spec.name}-loop{round_no}-{strategy}-pairs",
                                    generator=emit, size=len(rows), seed=self.spec.seed +
                                    1000 * round_no + (1 if strategy == "hard" else 2),
                                    attributes={"_cf_index": list(
                                    range(len(rows)))}, pipeline=pipeline, monitor="none")

    def _round_spec(self, r: int, res: Any, ds: Any) -> tuple[Any, list[str]]:
        from topologies.synth.ops.contamination import Check
        from topologies.synth.sampling import adaptive

        changes: list[str] = []
        kwargs: dict[str, Any] = {"name": self.spec.name + f"-loop{r}", "size": self.per_round,
                                  "seed": self.spec.seed + 1000 * r, "monitor": "none"}
        if "slices" in self.strategies:
            factors = self._factors(res)
            if factors:
                kwargs["sampler"] = adaptive(factors, base=self.spec.sampler_spec)
                changes.append("reweight " + "; ".join(
                    f"{a}: " + ", ".join(f"{v}×{f:.1f}" for v, f in fs.items())
                    for a, fs in factors.items()))
        if "hard" in self.strategies:
            anchors = self._anchors(res, ds)
            if anchors:
                kwargs["require"] = anchors
            changes.append("local hard-example perturbations")
        if "contrast" in self.strategies:
            changes.append("counterfactual minimal pairs")
        pipeline = list(self.spec.pipeline)
        if self.guard:
            pipeline.append(Check(self.real_eval, name="contamination.real_eval"))
        kwargs["pipeline"] = pipeline
        return self.spec._copy_with(**kwargs), changes

    # -- run ---------------------------------------------------------------------------------
    def run(self) -> LoopHistory:
        from topologies.synth.quality.utility import evaluate

        path = self._path()
        state = read_json(path) if path.is_file() else {"rounds": []}
        rounds: list[dict[str, Any]] = list(state.get("rounds", []))
        spent = sum(float(r.get("gen_seconds") or 0.0) for r in rounds)
        if rounds:
            from topologies.synth.dataset import load

            ds = load(self.spec.name, version=rounds[-1]["version"], root=self.spec.root)
        else:
            ds = self.spec.run(monitor="none")
            spent += ds.gen_seconds
        entry_cost, changes = 0.0, list(rounds[-1].get("changes", [])) if rounds else []
        while len(rounds) <= self.rounds:
            r = len(rounds)
            res = evaluate(ds, self.real_eval, name=f"{self.name}-r{r}", **{
                k: v for k, v in self.eval_kw.items()})
            self.goal = res.goal
            entry: dict[str, Any] = {"round": r, "version": ds.version_id,
                                     "tstr": (res.tstr or {}).get("mean"), "records": len(ds),
                                     "slices": res.slices[:30], "at": now_iso(),
                                     "gen_seconds": ds.gen_seconds if r == 0 else entry_cost}
            if r > 0:
                entry["changes"] = changes
            rounds.append(entry)
            write_json_atomic(path, {**state, "rounds": rounds, "name": self.name})
            stop = self._stop(rounds, spent)
            if stop or r >= self.rounds:
                if stop:
                    rounds[-1]["stop"] = stop
                    write_json_atomic(path, {**state, "rounds": rounds, "name": self.name})
                break
            if "prune" in self.strategies:
                try:
                    issues = ds.label_issues()
                    if len(issues):
                        ds = issues.drop()
                except SynthError:
                    pass
            spec_r, changes = self._round_spec(r + 1, res, ds)
            new = spec_r.run()
            entry_cost = new.gen_seconds
            generated = [new]
            for strategy in ("hard", "contrast"):
                if strategy not in self.strategies:
                    continue
                variants = self._local_pairs(ds, strategy=strategy, res=res)
                if not variants:
                    changes.append(f"{strategy}: no replayable high-loss anchors")
                    continue
                extra = self._variants_spec(variants[:self.per_round], r + 1,
                                            strategy=strategy).run()
                entry_cost += extra.gen_seconds
                generated.append(extra)
                changes.append(f"{len(variants)} {strategy} interventions")
            spent += entry_cost
            for generated_ds in generated:
                ds = ds.merge(generated_ds, dedup=generated_ds is new)
        return LoopHistory(self, rounds, path)

    def _stop(self, rounds: list[dict], spent: float) -> str | None:
        last = rounds[-1]["tstr"]
        if self.target is not None and last is not None and last >= self.target:
            return f"target {self.target} reached"
        if self.max_time is not None and spent >= self.max_time:
            return f"max_time {self.max_time:g}s of generation spent"
        if len(rounds) >= 3:
            g1 = (rounds[-1]["tstr"] or 0) - (rounds[-2]["tstr"] or 0)
            g2 = (rounds[-2]["tstr"] or 0) - (rounds[-3]["tstr"] or 0)
            if g1 < self.plateau and g2 < self.plateau:
                return "plateau"
        return None


__all__ = ["STRATEGIES", "Loop", "LoopHistory"]
