"""``topologies inspect``: capability tables and reports from the command line.

    topologies inspect artifacts/tiny-lm --input "Once upon a time" --preset standard \\
        --html report.html --sites "blocks.*"
    topologies inspect artifacts/tiny-lm --capabilities
"""
from __future__ import annotations

import json
import sys
from typing import Any


def add_parser(sub: Any) -> None:
    p = sub.add_parser("inspect", help="introspect a saved model (report, capabilities)")
    p.add_argument("source", help="path to a save_pretrained directory")
    p.add_argument("--input", action="append", default=None,
                   help="text input (repeat for a batch); needs a bundled tokenizer")
    p.add_argument("--tensor", default=None, help="a .pt file holding one input tensor")
    p.add_argument("--preset", default="quick", choices=("quick", "standard", "deep"))
    p.add_argument("--sites", default=None, help="selector for activation statistics")
    p.add_argument("--html", default=None, help="write a self-contained HTML report here")
    p.add_argument("--markdown", default=None, help="write a Markdown report here")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument("--capabilities", action="store_true", help="print the capability table only")
    p.add_argument("--strict", action="store_true", help="raise on the first technique failure")
    p.add_argument("--device", default=None)
    p.add_argument("--trust", action="append", default=None, metavar="MODULE",
                   help="trust model classes from MODULE when rebuilding the bundle "
                        "(repeatable; only for bundles you created)")


def main(args: Any) -> int:
    import torch

    from topologies.introspection import introspect

    source: Any = args.source
    if args.trust:
        from topologies.pretrained import load_pretrained

        source = load_pretrained(args.source, device=args.device or "cpu",
                                 trusted_modules=list(args.trust))
    insp = introspect(source, device=args.device)
    if args.capabilities:
        table = insp.capabilities()
        print(json.dumps(table.to_dict(), indent=2) if args.json else table)
        return 0
    if args.tensor is not None:
        inputs: Any = torch.load(args.tensor, weights_only=True)
    elif args.input:
        inputs = args.input[0] if len(args.input) == 1 else list(args.input)
    else:
        print("error: pass --input TEXT, --tensor FILE.pt or --capabilities", file=sys.stderr)
        return 2
    params: dict[str, Any] = {}
    if args.sites:
        params["activations"] = {"sites": args.sites}
    report = insp.report(inputs, preset=args.preset, strict=args.strict, **params)
    if args.json:
        print(report.to_result().to_json())
    else:
        report.show()
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(report.to_markdown())
        print(f"markdown: {args.markdown}")
    if args.html:
        print(f"html: {report.to_html(args.html)}")
    return 0
