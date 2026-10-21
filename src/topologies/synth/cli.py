"""``topologies synth ...``: scaffold, plan, run, inspect and evaluate synthetic data.

    topologies synth init text_classification my-data -o spec.yaml
    topologies synth plan spec.yaml [--pilot 200]
    topologies synth run spec.yaml [--max-time 30m] [--limit N] [--rerun all]
    topologies synth ls
    topologies synth show my-data [--split train] [-n 10] [--by label] [--rejected dedup.near]
    topologies synth explain my-data <record_id>
    topologies synth quality my-data [--reference real.jsonl] [--html r.html --open]
    topologies synth eval my-data --real real.jsonl [--model ...] [--seeds 3]
    topologies synth review my-data [-n 100] [--web]
    topologies synth loop spec.yaml --real real.jsonl --rounds 3
    topologies synth diff my-data <v1> <v2>
    topologies synth export my-data out/ --format jsonl|parquet|csv|hf|imagefolder

Every read command accepts ``--json`` for machine-readable output and ``--root``
for a non-default synth root. Output style, TTY detection and error handling
match ``topologies exp``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _common(p: argparse.ArgumentParser, *, json_flag: bool = True) -> None:
    p.add_argument("--root", default=None,
                   help="synth directory (default: $TOPOLOGIES_SYNTH_ROOT or "
                        "./synth next to the experiments root)")
    if json_flag:
        p.add_argument("--json", action="store_true", help="machine-readable output")


def add_parser(sub: Any) -> None:
    """Register ``synth`` and its subcommands on the main ``topologies`` parser."""
    ps = sub.add_parser("synth", help="synthetic data: scaffold, run, inspect, evaluate")
    s = ps.add_subparsers(dest="synth_cmd", required=True)

    p = s.add_parser("init", help="scaffold an editable spec from a blueprint")
    p.add_argument("blueprint", nargs="?", help="blueprint name")
    p.add_argument("name", nargs="?", help="dataset name")
    p.add_argument("--node", metavar="NAME", help="scaffold a Python node module")
    p.add_argument("-o", "--output", default=None, metavar="SPEC.YAML",
                   help="where to write the spec (default: <name>.yaml)")
    _common(p, json_flag=False)

    p = s.add_parser("preview", help="preview a spec or an importable node")
    p.add_argument("spec", help="spec file or pkg.mod:node")
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--html", default=None)
    p.add_argument("--open", action="store_true", dest="open_browser")
    _common(p, json_flag=False)

    p = s.add_parser("lib", help="list library nodes or describe one")
    p.add_argument("name", nargs="?", default=None)
    p.add_argument("--tag", default=None)
    _common(p)

    p = s.add_parser("describe", help="show node paths, types and batching")
    p.add_argument("spec")
    _common(p, json_flag=False)

    p = s.add_parser("plan", help="validate a spec: preview, throughput, pilot pass rates")
    p.add_argument("spec", help="spec file (.yaml / .json)")
    p.add_argument("--pilot", type=int, default=200, help="pilot records (default: 200)")
    p.add_argument("--preview", type=int, default=3, help="pilot records to print")
    _common(p)

    p = s.add_parser("run", help="generate (or resume) a dataset from a spec file")
    p.add_argument("spec", help="spec file (.yaml / .json)")
    p.add_argument("--max-time", default=None, dest="max_time",
                   help="stop cleanly after this long, e.g. '30m' (resumable)")
    p.add_argument("--limit", type=int, default=None, help="cap raw records this call")
    p.add_argument("--rerun", choices=("failed", "all"), default=None,
                   help="all: discard finished chunks")
    p.add_argument("--monitor", choices=("live", "lines", "none"), default=None)
    _common(p)

    p = s.add_parser("ls", help="list datasets, versions, sizes, generation time")
    _common(p)

    p = s.add_parser("show", help="print records (or rejected records) of a dataset")
    p.add_argument("name")
    p.add_argument("--split", default=None)
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--by", default=None, help="sample up to -n per value of this field")
    p.add_argument("--rejected", default=None, metavar="OP",
                   help="show rejected records of one op (or all when given without a value)",
                   nargs="?", const="all")
    p.add_argument("--version", default="latest")
    _common(p)

    p = s.add_parser("explain", help="full lineage of one record")
    p.add_argument("name")
    p.add_argument("record_id")
    p.add_argument("--version", default="latest")
    _common(p)

    p = s.add_parser("quality", help="intrinsic + fidelity diagnostics and HTML report")
    p.add_argument("name")
    p.add_argument("--reference", default=None, help="real reference .jsonl for fidelity")
    p.add_argument("--html", default=None, metavar="PATH")
    p.add_argument("--open", action="store_true", dest="open_browser")
    p.add_argument("--version", default="latest")
    _common(p)

    p = s.add_parser("eval", help="TSTR: train on synthetic, test on real")
    p.add_argument("name")
    p.add_argument("--real", required=True, help="real eval records (.jsonl)")
    p.add_argument("--model", default="TransformerClassifier")
    p.add_argument("--size", default="small")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--version", default="latest")
    _common(p)

    p = s.add_parser("review", help="human review session (terminal or local web page)")
    p.add_argument("name")
    p.add_argument("-n", type=int, default=100)
    p.add_argument("--web", action="store_true", help="serve a local review page")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--version", default="latest")
    _common(p, json_flag=False)

    p = s.add_parser("loop", help="closed loop: train, find failing slices, generate more")
    p.add_argument("spec", help="spec file (.yaml / .json)")
    p.add_argument("--real", required=True, help="real eval records (.jsonl)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--per-round", type=int, default=1000, dest="per_round")
    p.add_argument("--strategy", default="slices+hard")
    p.add_argument("--max-time", default=None, dest="max_time")
    _common(p)

    p = s.add_parser("diff", help="compare two versions of a dataset")
    p.add_argument("name")
    p.add_argument("v1")
    p.add_argument("v2")
    _common(p)

    p = s.add_parser("export", help="write a version to out/ in another format")
    p.add_argument("name")
    p.add_argument("out", help="output directory")
    p.add_argument("--format", choices=("jsonl", "parquet", "csv", "hf", "imagefolder",
                                        "txt"), default="jsonl", dest="format")
    p.add_argument("--version", default="latest")
    _common(p, json_flag=False)



# ---------------------------------------------------------------- helpers


def _print_json(obj: Any) -> None:
    from topologies.experiments.store import jsonable

    print(json.dumps(jsonable(obj), indent=1, ensure_ascii=False))


def _load_spec(path: str, root: str | None = None) -> Any:
    from topologies.synth.spec import Spec

    spec = Spec.from_yaml(path)
    if root is not None:
        spec.root = root
    return spec


def _load_ds(name: str, version: str = "latest", root: str | None = None) -> Any:
    from topologies.synth.dataset import load

    return load(name, version=version, root=root)


# ---------------------------------------------------------------- commands


def _init(args: Any) -> int:
    from topologies.synth import blueprints
    from topologies.synth.yaml_spec import spec_from_dict

    if args.node:
        import keyword

        from topologies.synth.errors import SynthError

        if args.blueprint or args.name or not args.node.isidentifier() or keyword.iskeyword(args.node):
            raise SynthError("Use init --node NAME with a valid Python identifier.", stage="init")
        out = Path(args.output) if args.output else Path(f"{args.node}.py")
        if out.exists():
            raise SynthError(f"Scaffold destination already exists: {out}", stage="init")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            '"""Editable synthetic generator; run this module to preview."""\n'
            'from topologies import synth\n\n\n'
            '@synth.node(type=synth.Number, tags=["custom"], example={})\n'
            f'def {args.node}(r, *, low=0.0, high=1.0):\n'
            '    return r.uniform(low, high)\n\n\n'
            f'gen = synth.record(value={args.node}())\n'
            f'spec = synth.Spec({args.node!r}, generator=gen, size=100, seed=0'
            + (f', root={args.root!r}' if args.root is not None else '') + ')\n\n'
            'if __name__ == "__main__":\n'
            '    synth.preview(spec, n=4)\n', encoding="utf-8")
        print(f"wrote {out} (edit generator, then run python {out})")
        return 0
    if not args.blueprint or not args.name:
        from topologies.synth.errors import SynthError

        raise SynthError("Use init BLUEPRINT NAME or init --node NAME.", stage="init")

    fn = blueprints.get(args.blueprint)
    out = Path(args.output) if args.output else Path(f"{args.name}.yaml")
    try:
        # Examples are YAML-ready (they may contain {ref:...}/{call:...} nodes),
        # so build through the YAML builder which resolves them.
        spec = spec_from_dict({"name": args.name,
                               "blueprint": {"name": args.blueprint, **dict(fn.example)}},
                              base=out.resolve().parent)
    except TypeError as e:
        print(f"error: cannot scaffold {args.blueprint!r}: {e}", file=sys.stderr)
        return 1
    if args.root is not None:
        spec = spec.with_(root=args.root)
    text = spec.to_yaml(out)
    _ = text
    print(f"wrote {out} (blueprint {args.blueprint}, edit it and run "
          f"`topologies synth plan {out}`)")
    return 0


def _preview(args: Any) -> int:
    import importlib

    from topologies import synth

    if ":" in args.spec and not Path(args.spec).is_file():
        module, name = args.spec.rsplit(":", 1)
        try:
            value = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError) as exc:
            raise synth.SynthError(f"Cannot import preview target {args.spec!r}: {exc}",
                                   stage="preview") from exc
        if callable(value) and not isinstance(value, synth.Node):
            value = value()
    else:
        value = _load_spec(args.spec, args.root)
    synth.preview(value, n=args.n, html=args.html, open=args.open_browser)
    return 0


def _lib(args: Any) -> int:
    from topologies import synth

    result = synth.lib(args.name, tag=args.tag, echo=not args.json)
    if args.json:
        _print_json(result)
    return 0


def _describe(args: Any) -> int:
    from topologies import synth

    synth.describe(_load_spec(args.spec, args.root))
    return 0


def _plan(args: Any) -> int:
    spec = _load_spec(args.spec, args.root)
    plan = spec.plan(pilot=args.pilot, preview=args.preview, verbose=False)
    if args.json:
        _print_json({"name": spec.name, "ok": plan.ok, "errors": plan.errors,
                     "warnings": plan.warnings, "raw_estimate": plan.raw_estimate,
                     "acceptance": plan.acceptance, "records_per_s": plan.records_per_s,
                     "bytes_per_record": plan.bytes_per_record,
                     "projected_seconds": plan.projected_seconds,
                     "projected_bytes": plan.projected_bytes,
                     "deterministic": plan.deterministic, "pilot": plan.pilot})
    else:
        print(plan)
    return 0 if plan.ok else 1


def _run(args: Any) -> int:
    spec = _load_spec(args.spec, args.root)
    if args.max_time is not None:
        spec = spec.with_(max_time=args.max_time)
    kw: dict[str, Any] = {"limit": args.limit, "rerun": args.rerun}
    if args.monitor:
        kw["monitor"] = args.monitor
    if args.json:
        kw["monitor"] = "none"
    ds = spec.run(**kw)
    if args.json:
        _print_json({"name": ds.name, "version": ds.version_id, "records": len(ds),
                     "splits": ds.splits, "complete": ds.complete,
                     "gen_seconds": ds.gen_seconds, "dir": str(ds.dir)})
    return 0


def _ls(args: Any) -> int:
    from topologies.synth.store import SynthStore, list_datasets

    rows = []
    for name in list_datasets(args.root):
        store = SynthStore(args.root, name)
        for m in store.versions():
            rows.append({"name": name, "version": m.get("version_id"),
                         "label": m.get("label"), "records": m.get("total"),
                         "splits": m.get("splits"), "complete": m.get("complete"),
                         "gen_seconds": (m.get("time") or {}).get("seconds"),
                         "created": m.get("created_at"),
                         "latest": store.latest() == m.get("version_id")})
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        print("no synth datasets" + (f" under {args.root}" if args.root else ""))
        return 0
    from topologies.experiments.render.console import render_table
    from topologies.experiments.results import Table

    cols = ["dataset", "version", "records", "splits", "gen s", "updated"]
    disp = [[(r["name"] + (" *" if r["latest"] else "")),
             (r["version"] or "")[:12],
             f"{r['records']:,}" if r["records"] is not None else "—",
             ", ".join(f"{k} {v:,}" for k, v in (r["splits"] or {}).items()) or "—",
             f"{r['gen_seconds']:.1f}" if r["gen_seconds"] is not None else "—",
             (r["created"] or "—")[:19].replace("T", " ")] for r in rows]
    print(render_table(Table(cols, [list(r.values()) for r in rows], display=disp)))
    return 0


def _show(args: Any) -> int:
    from topologies.synth.report import format_records

    ds = _load_ds(args.name, args.version, args.root)
    if args.rejected is not None:
        op = None if args.rejected == "all" else args.rejected
        view = ds.rejected(op)
        recs = list(view.head(args.n))
        if args.json:
            _print_json(recs)
        else:
            print(format_records(recs, title=f"{ds.name} rejected({op or 'all'})"))
        return 0
    if args.split is not None and args.split not in ds.splits:
        from topologies.synth.errors import unknown

        raise unknown("split", args.split, list(ds.splits), stage="dataset")
    if args.by:
        recs = list(ds.sample(args.n, by=args.by, split=args.split))
    else:
        recs = list(ds.view(args.split).head(args.n))
    if args.json:
        _print_json(recs)
    else:
        title = f"{ds.name} ({ds.version_id})" + (f" /{args.split}" if args.split else "")
        print(format_records(recs, title=title))
    return 0


def _explain(args: Any) -> int:
    ds = _load_ds(args.name, args.version, args.root)
    info = ds.explain(args.record_id)
    if args.json:
        _print_json(info)
        return 0
    from topologies.synth.report import format_records

    shown = dict(info["record"])
    shown["_id"] = args.record_id
    print(format_records([shown], title=f"{ds.name} {args.record_id}"))
    attrs = info.get("attributes") or {}
    if attrs:
        print("attributes: " + ", ".join(f"{k}={v!r}" for k, v in attrs.items()))
    lin = info.get("lineage") or {}
    interesting = {k: v for k, v in lin.items() if k in (
        "generator", "latent", "verdicts", "evidence", "parents", "seed", "op",
        "rejected", "logs")}
    if interesting:
        print(json.dumps(interesting, indent=1, ensure_ascii=False, default=str))
    elif lin:
        print(json.dumps(lin, indent=1, ensure_ascii=False, default=str)[:2000])
    return 0


def _quality(args: Any) -> int:
    ds = _load_ds(args.name, args.version, args.root)
    rep = ds.quality(reference=args.reference)
    if args.html:
        rep = ds.report(html=args.html, reference=args.reference,
                        open_browser=args.open_browser)
        print(f"report: {args.html}")
    elif args.open_browser:
        rep = ds.report(html="report.html", reference=args.reference, open_browser=True)
        print("report: report.html")
    if args.json:
        _print_json(rep.to_json())
    elif not args.html:
        rep.show()
    return 0


def _eval(args: Any) -> int:
    from topologies.synth.records import load_records

    ds = _load_ds(args.name, args.version, args.root)
    real = load_records(args.real)
    res = ds.evaluate(real, model=args.model, size=args.size, seeds=args.seeds,
                      monitor="none")
    if args.json:
        _print_json(res.to_json())
    else:
        res.show()
    return 0


def _review(args: Any) -> int:
    from topologies.synth.reviewing import review

    ds = _load_ds(args.name, args.version, args.root)
    sess = review(ds, n=args.n)
    if args.web:
        sess.serve(port=args.port, open_browser=False, block=True)
        return 0
    if sys.stdin.isatty():
        sess.run()
    else:
        from topologies.synth.report import format_records

        print(format_records(list(sess.records)[:10],
                             title=f"review {ds.name}: {len(sess)} records "
                                   f"(use --web for the browser UI)"))
        print(f"gold: {sess.path()} (run `topologies synth review {args.name} -n {args.n}` "
              "interactively to label them)")
    return 0


def _loop(args: Any) -> int:
    from topologies.synth.loop import Loop
    from topologies.synth.records import load_records

    spec = _load_spec(args.spec, args.root)
    real = load_records(args.real)
    loop = Loop(spec, real_eval=real, rounds=args.rounds, per_round=args.per_round,
                strategy=args.strategy, max_time=args.max_time)
    history = loop.run()
    if args.json:
        _print_json({"name": loop.name, "rounds": history.rounds, "path": str(history.path)})
    else:
        history.show()
    return 0


def _diff(args: Any) -> int:
    a = _load_ds(args.name, args.v1, args.root)
    b = _load_ds(args.name, args.v2, args.root)
    d = a.diff(b)
    if args.json:
        _print_json(d)
    else:
        print(f"diff {args.name} {args.v1} -> {args.v2}")
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                print(f"  {k}: {json.dumps(v, ensure_ascii=False, default=str)[:300]}")
            else:
                print(f"  {k}: {v}")
    return 0


def _export(args: Any) -> int:
    ds = _load_ds(args.name, args.version, args.root)
    files = ds.export(args.out, format=args.format)
    for p in files[:20]:
        print(f"wrote {p}")
    if len(files) > 20:
        print(f"... {len(files) - 20} more")
    return 0


_COMMANDS = {"init": _init, "plan": _plan, "run": _run, "ls": _ls, "show": _show,
             "explain": _explain, "quality": _quality, "eval": _eval, "review": _review,
             "loop": _loop, "diff": _diff, "export": _export,
             "preview": _preview, "lib": _lib, "describe": _describe}


def main(args: Any) -> int:
    from topologies.experiments.errors import ExperimentError

    try:
        return _COMMANDS[args.synth_cmd](args)
    except ExperimentError as e:
        print(f"error: {e}", file=sys.stderr)
        if getattr(e, "hint", None):
            print(f"hint: {e.hint}", file=sys.stderr)
        return 1


__all__ = ["add_parser", "main"]
