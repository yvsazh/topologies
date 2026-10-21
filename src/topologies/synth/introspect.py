"""Looking at node graphs: ``synth.describe(node)`` and ``synth.sample(node, i)``.

    print(synth.describe(gen))
    # record                                  record
    # ├─ audio            chirp  Audio  batch  f0=<f0> f1=<audio/chirp.f1> seconds=1.0
    # │  └─ audio/chirp.f1  apply                arg0=<f0> arg1=<audio/chirp.f1/mul.arg1>
    # └─ f0               uniform              low=80 high=800
    synth.sample(gen, 3)            # the raw value of record 3 (seed 0)

``describe`` lists every node with its path, kind, value type, whether it
renders in batches and its constant parameters (child nodes appear as
``<path>``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from topologies.synth.errors import SynthError


def _short(v: Any, n: int = 32) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _params_text(node: Any, paths: Mapping[int, str]) -> str:
    from topologies.synth.core import Node, iter_children

    parts = []
    for k, v in node.params().items():
        kids = iter_children(v, k)
        if isinstance(v, Node):
            parts.append(f"{k}=<{paths.get(id(v), '?')}>")
        elif kids:
            parts.append(f"{k}=[{len(kids)} node(s)]")
        else:
            parts.append(f"{k}={_short(v)}")
    return " ".join(parts)


def describe(node: Any, *, echo: bool = True, seed: int = 0) -> str:
    """The node tree of ``node`` (a generator, a node or a spec) as text."""
    from topologies.synth.core import Node, as_generator, iter_children, type_name
    from topologies.synth.evaluate import Binding

    root = getattr(node, "generator", None) if not isinstance(node, Node) and hasattr(
        node, "gen_fingerprint") else node
    root = as_generator(root)
    b = Binding(root, seed)
    lines: list[str] = []
    seen: set[int] = set()

    def walk(n: Node, prefix: str, last: bool, depth: int) -> None:
        path = b.paths.get(id(n), "?") or "<root>"
        branch = "" if depth == 0 else ("└─ " if last else "├─ ")
        kind = n.node_kind()
        t = type_name(getattr(n, "type", None)) or ""
        batch = "batch" if getattr(n, "batched", False) else ""
        again = id(n) in seen
        head = f"{prefix}{branch}{path}"
        info = "shared (see above)" if again else _params_text(n, b.paths)
        lines.append(f"{head:<40} {kind:<14} {t:<8} {batch:<5} {info}".rstrip())
        if again:
            return
        seen.add(id(n))
        kids = [c for k, v in n.params().items() for _, c in iter_children(v, k)]
        ext = "" if depth == 0 else ("   " if last else "│  ")
        for i, c in enumerate(kids):
            walk(c, prefix + ext, i == len(kids) - 1, depth + 1)

    walk(root, "", True, 0)
    text = "\n".join(lines)
    if echo:
        print(text)
    return text


def sample(node: Any, index: int = 0, *, seed: int = 0, attrs: Mapping[str, Any] | None = None,
           latents: bool = False) -> Any:
    """The raw value of record ``index`` of ``node`` (no spec, no pipeline, no storage)."""
    from topologies.synth.core import as_generator
    from topologies.synth.evaluate import Evaluator

    ev = Evaluator(as_generator(node), seed=seed)
    s = ev.evaluate(index, attrs or {})
    if s.error is not None:
        raise s.error
    if latents:
        return s.value, s.latents
    return s.value


def check_paths(node: Any, seed: int = 0) -> dict[str, str]:
    """``{path: node kind}`` for every node (tests and docs)."""
    from topologies.synth.core import as_generator
    from topologies.synth.evaluate import Binding

    b = Binding(as_generator(node), seed)
    out = {p: b.nodes[k].node_kind() for k, p in b.paths.items()}
    if not out:
        raise SynthError("Empty node graph.", stage="node")
    return out


__all__ = ["check_paths", "describe", "sample"]
