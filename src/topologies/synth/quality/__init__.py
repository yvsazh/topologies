"""Quality measurement: intrinsic diagnostics, fidelity vs real data, contamination,
privacy, downstream utility (TSTR) and per-slice error analysis.

    rep = ds.quality(reference=real_records)      # QualityReport (Tables + insights)
    res = ds.evaluate(real_eval, model="TransformerClassifier", size="small", seeds=3)
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "quality_report": ("topologies.synth.quality.report", "quality_report"),
    "QualityReport": ("topologies.synth.quality.report", "QualityReport"),
    "get_embedder": ("topologies.synth.quality.embed", "get_embedder"),
    "evaluate": ("topologies.synth.quality.utility", "evaluate"),
    "scaling_curve": ("topologies.synth.quality.utility", "scaling_curve"),
    "mix_curve": ("topologies.synth.quality.utility", "mix_curve"),
    "ablate": ("topologies.synth.quality.utility", "ablate"),
    "c2st": ("topologies.synth.quality.fidelity", "c2st"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'topologies.synth.quality' has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = list(_LAZY)
