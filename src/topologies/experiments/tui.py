"""Placeholder for the optional interactive UI (``topologies exp ui NAME``).

Not implemented yet: use ``topologies exp show / pivot / curves / report``.
"""

from __future__ import annotations

from pathlib import Path


def run_ui(name: str, root: str | Path | None = None) -> int:
    raise ImportError(
        "topologies exp ui is not implemented yet. Use `topologies exp show/pivot/curves "
        f"{name}` or `topologies exp report {name} --html report.html --open`.")
