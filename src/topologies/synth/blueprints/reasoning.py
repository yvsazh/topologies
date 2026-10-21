"""Probe blueprints: algorithmic sequence tasks with exact labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from topologies.synth.blueprints._base import blueprint, spec_kwargs
from topologies.synth.errors import SynthError, unknown
from topologies.synth.spec import Spec

# -- algorithmic probes ----------------------------------------------------------------------------


@blueprint(example={"tasks": ["mqar", "induction_heads", "parity"], "size": 2000})
def algorithmic_probes(name: str, *, tasks: str | Sequence[str] = "mqar",
                       knobs: Mapping[str, Any] | None = None,
                       vary: Mapping[str, Any] | None = None, size: int = 10000,
                       splits: Any = None, **spec: Any) -> Any:
    """Synthetic sequence tasks with exact labels (scenario 5 data): one task, or an equal
    mixture of several. ``knobs`` fix task parameters, ``vary`` samples them per record
    (``vary={"n_pairs": [4, 8, 16]}``); ``splits=synth.ops.split(by_length=...)`` makes
    length-generalization splits."""
    from topologies.synth.core import Mixture
    from topologies.synth.lib import seq as algorithmic

    names = [tasks] if isinstance(tasks, str) else list(tasks)
    for t in names:
        if t not in algorithmic.TASKS:
            raise unknown("probe task", t, algorithmic.TASKS, stage="blueprint")
    kn = dict(knobs or {})
    gens = [algorithmic.Probe(t, **kn) for t in names]
    gen: Any = gens[0] if len(gens) == 1 else Mixture([(g, 1.0) for g in gens])
    if len(gens) > 1 and kn:
        raise SynthError("knobs= applies to a single task; use vary= (attributes are read by "
                         "the tasks that have those knobs).", stage="blueprint")
    return Spec(name, generator=gen, attributes=dict(vary or {}), size=size,
                splits=splits if splits is not None else {"train": 0.9, "val": 0.1},
                **spec_kwargs(spec))


__all__ = ["algorithmic_probes"]
