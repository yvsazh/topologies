"""Developer CLI for registries, datasets, models, recipes, experiments and synth."""

from __future__ import annotations

import argparse
import json
import sys


def _unknown(what: str, name: str, known: list[str]) -> int:
    from topologies._util.suggest import did_you_mean

    print(f"error: unknown {what} {name!r}.{did_you_mean(name, known, n=3)}", file=sys.stderr)
    return 1


def _registry(args) -> int:
    from topologies.registry.api import Registry
    from topologies.taxonomy.validate import validate_taxonomy

    reg = Registry.load(getattr(args, "registry", None))
    if args.registry_cmd == "validate":
        offline = getattr(args, "offline", False)
        # By default try store verification when reachable; --offline skips it.
        report = reg.validate(
            check_imports=args.check_imports,
            verify_store=not offline, offline=offline, strict=args.strict)
        treport = validate_taxonomy(reg.index.taxonomy)
        for e in report.errors:
            print(f"ERROR: {e}")
        for w in report.warnings:
            print(f"WARN: {w}")
        for e in treport.errors:
            print(f"TAXONOMY ERROR: {e}")
        for w in treport.warnings:
            print(f"TAXONOMY WARN: {w}")
        stats = reg.stats()
        print(f"entries={stats['total']} by_kind={stats['by_kind']} by_status={stats['by_status']}")
        ok = report.ok and treport.ok
        print("registry: VALID" if ok else "registry: INVALID")
        return 0 if ok else 1
    if args.registry_cmd == "list":
        for r in reg.index.records:
            print(f"{r.slug}\t{r.kind.value}\t{r.status.value}\t{r.name}")
        return 0
    if args.registry_cmd == "search":
        # Parity with website browse (see RegistryIndex.search): same recall over
        # rich fields + has_examples/has_graph filters. Deterministic, no network.
        has_examples = True if args.has_examples else None
        has_graph = True if args.has_graph else None
        results = reg.search(
            kind=args.kind, modality=args.modality,
            architecture_paradigm=args.paradigm,
            mechanisms=args.mechanism, properties=args.property,
            status=args.status, query=args.query, limit=args.limit,
            has_examples=has_examples, has_graph=has_graph,
        )
        for r in results:
            print(f"{r.slug}\t{r.kind.value}\t{r.status.value}\t{r.name}")
        if args.json:
            print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return 0
    if args.registry_cmd == "show":
        try:
            r = reg.get(args.slug)
        except KeyError:
            r = None
        if r is None:
            known = [rec.slug for rec in reg.index.records]
            return _unknown("registry entry", args.slug, known)
        print(json.dumps(r.model_dump(mode="json"), indent=2))
        return 0
    return 2


def _taxonomy(args) -> int:
    from topologies.registry.api import Registry
    from topologies.taxonomy.validate import validate_taxonomy

    reg = Registry.load(getattr(args, "registry", None))
    report = validate_taxonomy(reg.index.taxonomy)
    for e in report.errors:
        print(f"ERROR: {e}")
    for w in report.warnings:
        print(f"WARN: {w}")
    print(f"nodes={len(reg.index.taxonomy)} ok={report.ok}")
    return 0 if report.ok else 1


def _recipes(args) -> int:
    from topologies.training import list_recipes, recipe_schema

    if args.recipes_cmd == "list":
        for r in list_recipes():
            print(f"{r['name']}\t{r['level']}\t{r.get('help', '')}")
        return 0
    if args.recipes_cmd == "show":
        print(json.dumps(recipe_schema(args.name), indent=2))
        return 0
    return 2


def _data(args) -> int:
    import warnings

    from topologies.data.download import download, list_datasets

    print("note: `topologies data` is deprecated; use `topologies datasets` (the declarative "
          "catalog).", file=sys.stderr)
    warnings.simplefilter("ignore", DeprecationWarning)  # the note above says it once

    if args.data_cmd == "list":
        for d in list_datasets():
            extra = f"  [needs topologies[{d.extra}]]" if d.extra else ""
            print(f"{d.name:<14} {d.size:>8}  {d.description}{extra}")
        return 0
    if args.data_cmd == "download":
        try:
            paths = download(args.name, args.root, force=args.force)
        except (ValueError, ImportError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        for path in paths:
            print(f"ready: {path}")
        return 0
    return 2


def _datasets(args) -> int:
    """Operate only on the new declarative dataset catalog."""
    from topologies.datasets import DatasetError, get_dataset, list_datasets

    if args.datasets_cmd == "list":
        for item in list_datasets():
            print(f"{item['id']}\t{item['default_version']}\t{item['name']}")
        return 0
    if args.datasets_cmd == "show":
        items = list_datasets()
        item = next((item for item in items if item["id"] == args.id), None)
        if item is None:
            return _unknown("dataset", args.id, [i["id"] for i in items])
        print(json.dumps(item, indent=2))
        return 0
    try:
        dataset = get_dataset(
            args.id, version=args.version, split=args.split, root=args.root, offline=args.offline,
        )
        paths = dataset.download(force=args.force) if args.datasets_cmd == "download" else dataset.prepare(force=args.force)
    except (DatasetError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


def _models(args) -> int:
    from topologies.models import list_models
    from topologies.nn import __all__ as nn_all

    if args.models_cmd == "list":
        print("catalog (topologies.models):")
        for m in list_models(include_demo=args.demo):
            demo = "  [demo]" if m["is_demo"] else ""
            print(f"  {m['id'] or '-':<24} {m['name']:<22} {m['module']}{demo}")
        print("toolbox stacks (topologies.nn):")
        import topologies.nn as tnn
        from topologies.build import get_presets

        for name in nn_all:
            obj = getattr(tnn, name)
            sizes = get_presets(obj)
            if sizes:
                print(f"  {name:<24} sizes: {', '.join(sizes)}")
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    from topologies import __version__

    p = argparse.ArgumentParser(prog="topologies")
    p.add_argument("--version", action="version", version=f"topologies {__version__}")
    p.add_argument("--registry", default=None, help="path to registry dir")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("registry", help="registry commands")
    prs = pr.add_subparsers(dest="registry_cmd", required=True)
    pv = prs.add_parser("validate", help="validate registry + taxonomy")
    pv.add_argument("--check-imports", action="store_true")
    pv.add_argument("--offline", action="store_true",
                    help="skip artifact-store remote checks (offline deterministic only)")
    pv.add_argument("--strict", action="store_true",
                    help="treat unknown (undeclared) record keys as errors, not warnings")
    prs.add_parser("list", help="list entries")
    ps = prs.add_parser("search", help="search entries")
    ps.add_argument("--kind", default=None)
    ps.add_argument("--modality", default=None)
    ps.add_argument("--paradigm", default=None)
    ps.add_argument("--mechanism", dest="mechanism", action="append", default=None)
    ps.add_argument("--property", dest="property", action="append", default=None)
    ps.add_argument("--status", default=None)
    ps.add_argument("--query", default=None)
    ps.add_argument("--limit", type=int, default=50)
    ps.add_argument("--has-examples", action="store_true",
                    help="only entries with usage examples")
    ps.add_argument("--has-graph", action="store_true",
                    help="only entries with a model graph")
    ps.add_argument("--json", action="store_true")
    psh = prs.add_parser("show", help="show one entry")
    psh.add_argument("slug")

    pt = sub.add_parser("taxonomy", help="taxonomy commands")
    pts = pt.add_subparsers(dest="taxonomy_cmd", required=True)
    pts.add_parser("validate", help="validate taxonomy")

    prc = sub.add_parser("recipes", help="training recipe introspection")
    prcs = prc.add_subparsers(dest="recipes_cmd", required=True)
    prcs.add_parser("list", help="list available training recipes")
    pshw = prcs.add_parser("show", help="show one recipe schema")
    pshw.add_argument("name")

    pda = sub.add_parser("data", help="dataset downloads")
    pdas = pda.add_subparsers(dest="data_cmd", required=True)
    pdas.add_parser("list", help="list downloadable datasets")
    pdd = pdas.add_parser("download", help="download a dataset")
    pdd.add_argument("name")
    pdd.add_argument("--root", default=None, help="target directory (default: data/<name>)")
    pdd.add_argument("--force", action="store_true", help="re-download even if present")

    pds = sub.add_parser("datasets", help="declarative dataset catalog")
    pdss = pds.add_subparsers(dest="datasets_cmd", required=True)
    pdss.add_parser("list", help="list datasets in the packaged catalog")
    pdshow = pdss.add_parser("show", help="show one catalog definition")
    pdshow.add_argument("id")
    for command in ("download", "prepare"):
        pdataset = pdss.add_parser(command, help=f"{command} one catalog dataset")
        pdataset.add_argument("id")
        pdataset.add_argument("--version", default=None)
        pdataset.add_argument("--split", default=None)
        pdataset.add_argument("--root", default=None, help="cache base (default: TOPOLOGIES_DATA_HOME or ~/.cache/topologies)")
        pdataset.add_argument("--offline", action="store_true")
        pdataset.add_argument("--force", action="store_true")

    pmo = sub.add_parser("models", help="model catalog")
    pmos = pmo.add_subparsers(dest="models_cmd", required=True)
    pml = pmos.add_parser("list", help="list catalog architectures and toolbox stacks")
    pml.add_argument("--no-demo", dest="demo", action="store_false")

    from topologies.experiments.cli import add_parser as _add_exp_parser
    from topologies.introspection.cli import add_parser as _add_inspect_parser
    from topologies.synth.cli import add_parser as _add_synth_parser

    _add_exp_parser(sub)
    _add_synth_parser(sub)
    _add_inspect_parser(sub)
    for name, func in (("registry", _registry), ("taxonomy", _taxonomy), ("recipes", _recipes),
                       ("data", _data), ("datasets", _datasets), ("models", _models),
                       ("exp", _exp), ("synth", _synth), ("inspect", _inspect)):
        sub.choices[name].set_defaults(func=func)
    return p


def _exp(args) -> int:
    from topologies.experiments.cli import main as exp_main

    return exp_main(args)


def _synth(args) -> int:
    from topologies.synth.cli import main as synth_main

    return synth_main(args)


def _inspect(args) -> int:
    from topologies.introspection.cli import main as inspect_main

    return inspect_main(args)


def main(argv=None) -> int:
    from topologies.training.errors import TrainingError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, TrainingError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
