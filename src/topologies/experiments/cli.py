"""``topologies exp ...``: run, inspect and report experiments from the terminal.

    topologies exp run spec.py [--workers 2] [--rerun failed] [--limit 4]
    topologies exp plan spec.yaml --smoke
    topologies exp ls
    topologies exp show arch-x-tokenizer --top 10 --filter tokenizer=bpe:4k
    topologies exp pivot arch-x-tokenizer model tokenizer
    topologies exp curves arch-x-tokenizer --by tokenizer
    topologies exp report arch-x-tokenizer --html report.html --open

Every read command accepts ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

READ_COMMANDS = ("ls", "show", "pivot", "marginals", "importance", "insights", "curves",
                 "compare", "trial", "failures", "plan")


def _common(p: argparse.ArgumentParser, *, json_flag: bool = True) -> None:
    p.add_argument("--root", default=None,
                   help="experiments directory (default: $TOPOLOGIES_EXPERIMENTS_DIR or "
                        "./experiments)")
    if json_flag:
        p.add_argument("--json", action="store_true", help="machine-readable output")


def add_parser(sub: Any) -> None:
    """Register ``exp`` and its subcommands on the main ``topologies`` parser."""
    pe = sub.add_parser("exp", help="experiments: run, compare, report")
    pe.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                    help="terminal colors (maps to TOPOLOGIES_COLOR)")
    s = pe.add_subparsers(dest="exp_cmd", required=True)

    p = s.add_parser("run", help="run or resume an experiment spec (.py / .yaml)")
    p.add_argument("spec")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--devices", default=None, help="comma separated, e.g. cuda:0,cuda:1")
    p.add_argument("--rerun", choices=("failed", "all"), default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--monitor", choices=("live", "lines", "plain", "dashboard", "none"),
                   default=None)
    _common(p)

    p = s.add_parser("plan", help="validate a spec: trial matrix, params, compatibility")
    p.add_argument("spec")
    p.add_argument("--smoke", action="store_true", help="2 training steps per unique config")
    _common(p)

    p = s.add_parser("ls", help="list experiments")
    _common(p)

    p = s.add_parser("show", help="leaderboard")
    p.add_argument("name")
    p.add_argument("--metric", default=None)
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--filter", action="append", default=[], metavar="K=V")
    p.add_argument("--by-seed", action="store_true", help="one row per trial")
    p.add_argument("--ci", action="store_true", help="95%% confidence intervals")
    p.add_argument("--all-status", action="store_true", help="include failed configs")
    _common(p)

    p = s.add_parser("pivot", help="heatmap of ROW x COL")
    p.add_argument("name")
    p.add_argument("row")
    p.add_argument("col")
    p.add_argument("--metric", default=None)
    p.add_argument("--agg", choices=("mean", "min", "max", "median"), default="mean")
    p.add_argument("--filter", action="append", default=[], metavar="K=V")
    _common(p)

    p = s.add_parser("marginals", help="mean per axis value over the other axes")
    p.add_argument("name")
    p.add_argument("axis", nargs="?", default=None)
    p.add_argument("--metric", default=None)
    _common(p)

    for cmd, hlp in (("importance", "variance share per axis"),
                     ("insights", "deterministic findings"),
                     ("failures", "failed trials grouped by axis value and error")):
        p = s.add_parser(cmd, help=hlp)
        p.add_argument("name")
        if cmd != "failures":
            p.add_argument("--metric", default=None)
        _common(p)

    p = s.add_parser("curves", help="learning curves (braille chart)")
    p.add_argument("name")
    p.add_argument("--metric", default=None)
    p.add_argument("--by", default=None)
    p.add_argument("--x", choices=("epoch", "step"), default="epoch")
    p.add_argument("--filter", action="append", default=[], metavar="K=V")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=14)
    p.add_argument("--log-y", action="store_true", help="log10 y axis")
    p.add_argument("--no-color", action="store_true", help="markers instead of colors")
    _common(p)

    p = s.add_parser("compare", help="config diff + metric diff + significance")
    p.add_argument("name")
    p.add_argument("a", help="trial id or k=v,k=v")
    p.add_argument("b", help="trial id or k=v,k=v")
    p.add_argument("--metric", default=None)
    _common(p)

    p = s.add_parser("trial", help="full view of one trial")
    p.add_argument("name")
    p.add_argument("id")
    _common(p)

    p = s.add_parser("report", help="self-contained HTML or Markdown report")
    p.add_argument("name")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--html", default=None, metavar="PATH")
    g.add_argument("--md", default=None, metavar="PATH")
    p.add_argument("--open", action="store_true")
    _common(p, json_flag=False)

    p = s.add_parser("export", help="flat export of every trial")
    p.add_argument("name")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", default=None, metavar="PATH")
    g.add_argument("--json", dest="json_path", default=None, metavar="PATH")
    _common(p, json_flag=False)

    p = s.add_parser("cache", help="shared cache (tokenizers, datasets)")
    p.add_argument("action", choices=("ls", "clear"))
    _common(p)

    p = s.add_parser("rm", help="delete experiment data")
    p.add_argument("name")
    p.add_argument("--stale-only", action="store_true",
                   help="only trials that are not in the current plan")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    _common(p, json_flag=False)

    p = s.add_parser("bench", help="component benchmark spec (.py)")
    p.add_argument("spec")
    p.add_argument("--monitor", choices=("live", "lines", "plain", "dashboard", "none"),
                   default=None)
    _common(p)

    p = s.add_parser("ui", help="interactive TUI (not implemented yet)")
    p.add_argument("name")
    _common(p, json_flag=False)


# ---------------------------------------------------------------- helpers


def _print_json(obj: Any) -> None:
    from topologies.experiments.store import jsonable

    print(json.dumps(jsonable(obj), indent=1, ensure_ascii=False))


def _filters(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"error: --filter expects K=V, got {it!r}")
        k, _, v = it.partition("=")
        vals = [x for x in v.split("|") if x]
        out[k.strip()] = vals if len(vals) > 1 else v
    return out


def _load(args: Any) -> Any:
    from topologies.experiments.results import load

    rs = load(args.name, args.root)
    flt = _filters(getattr(args, "filter", []))
    return rs.filter(**flt) if flt else rs


def _lines(lines: list[str]) -> None:
    print("\n".join(lines))


def _load_spec(args: Any) -> Any:
    from topologies.experiments.spec import load_spec
    from topologies.experiments.store import ExperimentStore

    exp = load_spec(args.spec)
    if args.root is not None:
        exp.root = args.root
        exp.store = ExperimentStore(args.root, exp.name)
        exp._cache = None
    return exp


# ---------------------------------------------------------------- commands


def _run(args: Any) -> int:
    exp = _load_spec(args)
    if args.workers is not None:
        exp.workers = max(1, args.workers)
    if args.devices:
        exp.devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if args.monitor:
        exp.monitor = args.monitor
    if args.json:
        exp.monitor = "none"
    try:
        res = exp.run(resume=not args.no_resume, rerun=args.rerun, limit=args.limit)
    except KeyboardInterrupt:
        return 130
    if args.json:
        from collections import Counter

        best = res.best(1)
        _print_json({"name": res.name, "trials": len(res),
                     "statuses": dict(Counter(r.status for r in res)),
                     "goal": {"metric": res.goal[0], "mode": res.goal[1]},
                     "best": {"labels": best[0].labels, "mean": best[0].mean} if best else None,
                     "dir": str(exp.store.dir)})
    return 0


def _plan(args: Any) -> int:
    exp = _load_spec(args)
    plan = exp.plan(smoke=args.smoke, verbose=False)
    if args.json:
        _print_json(plan.to_json())
    else:
        print(plan)
    return 0 if plan.ok else 1


def _ls(args: Any) -> int:
    from topologies.experiments.results import TrialRecord
    from topologies.experiments.store import ExperimentStore, list_experiments

    rows = []
    for name in list_experiments(args.root):
        store = ExperimentStore(args.root, name)
        spec = store.read_spec() or {}
        plan = set(spec.get("plan") or [])
        recs = [TrialRecord.from_dict(r) for r in (store.read_result(t, warn=False)
                                                  for t in store.trial_ids()) if r]
        in_plan = [r for r in recs if r.trial_id in plan] if plan else recs
        done = sum(1 for r in in_plan if r.status == "completed")
        failed = sum(1 for r in in_plan if r.status not in ("completed", "pruned"))
        goal = spec.get("goal") or {}
        vals = [r.value for r in in_plan if r.status == "completed" and r.value is not None]
        best = (min(vals) if goal.get("mode", "min") == "min" else max(vals)) if vals else None
        rows.append({"name": spec.get("name", name), "dir": name, "done": done,
                     "total": len(plan) or len(recs), "failed": failed,
                     "goal": goal.get("metric"), "best": best,
                     "updated": spec.get("updated_at"),
                     "stale": len([t for t in store.trial_ids() if plan and t not in plan])})
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        print("no experiments")
        return 0
    from topologies.experiments.metrics import metric_info
    from topologies.experiments.render.console import render_table
    from topologies.experiments.results import Table

    cols = ["name", "done/total", "failed", "best", "updated", "stale"]
    disp = [[r["name"], f"{r['done']}/{r['total']}", str(r["failed"]),
             f"{r['goal']} {metric_info(r['goal'] or '').format(r['best'])}" if r["goal"]
             else "—", (r["updated"] or "—")[:19].replace("T", " "), str(r["stale"])]
            for r in rows]
    _lines(render_table(Table(cols, [list(r.values()) for r in rows], display=disp)))
    return 0


def _show(args: Any) -> int:
    rs = _load(args)
    t = rs.table(metric=args.metric, top=args.top, group_seeds=not args.by_seed,
                 include_failed=args.all_status, ci=args.ci)
    if args.json:
        _print_json(t.to_json())
    else:
        print(t)
    return 0


def _pivot(args: Any) -> int:
    t = _load(args).pivot(args.row, args.col, metric=args.metric, agg=args.agg)
    if args.json:
        _print_json(t.to_json())
    else:
        print(t)
    return 0


def _marginals(args: Any) -> int:
    tables = _load(args).marginals(args.axis, metric=args.metric)
    if args.json:
        _print_json({a: t.to_json() for a, t in tables.items()})
    else:
        print("\n\n".join(str(t) for t in tables.values()))
    return 0


def _importance(args: Any) -> int:
    from topologies.experiments.analysis import importance_data
    from topologies.experiments.render.console import render_bars

    rs = _load(args)
    if args.json:
        _print_json(importance_data(rs, args.metric))
        return 0
    d = importance_data(rs, args.metric)
    items = sorted((d.get("shares") or {}).items(), key=lambda kv: -kv[1])
    _lines(render_bars(items, inline=False, title="importance"))
    print(rs.importance(args.metric))
    return 0


def _insights(args: Any) -> int:
    from topologies.experiments.render.console import render_insights

    rs = _load(args)
    ins = rs.insights()
    if args.json:
        _print_json([i.to_json() for i in ins])
    else:
        _lines(render_insights(ins) or ["no insights"])
    return 0


def _curves(args: Any) -> int:
    from topologies.experiments.render.console import term_width

    rs = _load(args)
    c = rs.curves(metric=args.metric or rs.goal[0], by=args.by, x=args.x)
    if args.json:
        _print_json(c.to_json())
    else:
        width = args.width or min(100, term_width())
        if getattr(args, "log_y", False) or getattr(args, "no_color", False):
            from topologies.experiments.render.console import render_curves

            color = False if args.no_color else None
            _lines(render_curves(c, width=width, height=args.height, color=color,
                                 log_y=args.log_y))
        else:
            _lines(c.lines(width=width, height=args.height))
    return 0


def _compare(args: Any) -> int:
    rs = _load(args)
    c = rs.compare(args.a, args.b, metric=args.metric)
    if args.json:
        _print_json(c.to_json())
    else:
        print(c)
    return 0


def _trial(args: Any) -> int:
    from topologies.experiments.render.console import render_trial

    rs = _load(args)
    hits = rs.find(args.id)
    if not hits:
        print(f"error: no trial {args.id!r} in {args.name}", file=sys.stderr)
        return 1
    if len(hits) > 1 and not all(h.trial_id == hits[0].trial_id for h in hits):
        print(f"error: {args.id!r} matches {len(hits)} trials: "
              + ", ".join(h.trial_id for h in hits[:8]), file=sys.stderr)
        return 1
    if args.json:
        _print_json(hits[0].raw)
    else:
        _lines(render_trial(hits[0]))
    return 0


def _failures(args: Any) -> int:
    t = _load(args).failures()
    if args.json:
        _print_json(t.to_json())
    else:
        print(t if t.rows else "no failed trials")
    return 0


def _report(args: Any) -> int:
    rs = _load(args)
    if args.md:
        path = rs.report(args.md)
    else:
        default = Path(rs.meta.get("dir") or ".") / "report.html"
        path = rs.report(args.html or default)
    print(f"report: {path}")
    if args.open:
        import webbrowser

        webbrowser.open(Path(path).resolve().as_uri())
    return 0


def _export(args: Any) -> int:
    rs = _load(args)
    if args.csv:
        rs.to_csv(args.csv)
        print(f"wrote {args.csv} ({len(rs)} trials)")
    else:
        rs.to_json(args.json_path)
        print(f"wrote {args.json_path} ({len(rs)} trials)")
    return 0


def _cache(args: Any) -> int:
    from topologies.experiments.cache import Cache
    from topologies.experiments.store import default_root

    cache = Cache(default_root(args.root))
    if args.action == "clear":
        n = cache.clear()
        if args.json:
            _print_json({"removed": n})
        else:
            print(f"removed {n} cache entr{'y' if n == 1 else 'ies'}")
        return 0
    entries = cache.entries()
    if args.json:
        _print_json(entries)
        return 0
    if not entries:
        print("cache is empty")
    for e in entries:
        print(f"{e['kind']:<10} {e['key']:<34} {e['bytes'] / 1e6:8.2f} MB")
    return 0


def _rm(args: Any) -> int:
    import shutil

    from topologies.experiments.store import ExperimentStore

    store = ExperimentStore(args.root, args.name)
    if not store.dir.is_dir():
        print(f"error: no experiment {args.name!r} under {store.root}", file=sys.stderr)
        return 1
    if args.stale_only:
        plan = set((store.read_spec() or {}).get("plan") or [])
        targets = [store.trial_dir(t) for t in store.trial_ids() if plan and t not in plan]
        what = f"{len(targets)} stale trial(s) of {args.name}"
    else:
        targets = [store.dir]
        what = f"experiment {args.name} ({len(store.trial_ids())} trials) at {store.dir}"
    if not targets:
        print("nothing to delete")
        return 0
    if not args.yes:
        try:
            answer = input(f"Permanently delete {what}? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
    print(f"deleted {what}")
    return 0


def _bench(args: Any) -> int:
    from topologies.experiments.spec import load_spec

    exp = load_spec(args.spec)
    if args.root is not None:
        from topologies.experiments.store import ExperimentStore

        exp.store = ExperimentStore(args.root, exp.name)
    if args.monitor:
        exp.monitor = args.monitor
    if args.json:
        exp.monitor = "none"
    from topologies.experiments.bench import BenchResult

    res = BenchResult(exp.run())
    if args.json:
        _print_json({"table": res.table().to_json(), "checks": res.check_matrix().to_json()})
    elif exp.monitor == "none":
        res.show()
    return 0


def _ui(args: Any) -> int:
    try:
        from topologies.experiments.tui import run_ui
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return run_ui(args.name, args.root)


_COMMANDS = {"run": _run, "plan": _plan, "ls": _ls, "show": _show, "pivot": _pivot,
             "marginals": _marginals, "importance": _importance, "insights": _insights,
             "curves": _curves, "compare": _compare, "trial": _trial, "failures": _failures,
             "report": _report, "export": _export, "cache": _cache, "rm": _rm,
             "bench": _bench, "ui": _ui}


def main(args: Any) -> int:
    from topologies.experiments.errors import ExperimentError

    color = getattr(args, "color", "auto")
    if color in ("always", "never"):
        import os

        os.environ["TOPOLOGIES_COLOR"] = color
    try:
        return _COMMANDS[args.exp_cmd](args)
    except ExperimentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


__all__ = ["add_parser", "main"]
