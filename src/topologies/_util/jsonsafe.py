"""One recursive JSON-safe conversion."""

from __future__ import annotations

import math
from typing import Any, Literal

NonFinite = Literal["null", "keep", "str"]


def to_jsonable(obj: Any, *, nonfinite: NonFinite = "null") -> Any:
    """Recursively convert ``obj`` to JSON-compatible values.

    Mappings -> dicts with string keys, sequences/sets -> lists, tensors and
    NumPy scalars/arrays -> Python values, anything else -> ``float(obj)`` or
    ``str(obj)``. Non-finite floats become ``None`` (``"null"``), stay as is
    (``"keep"``; ``json.dumps`` then writes ``NaN``) or become strings (``"str"``).
    """
    if isinstance(obj, float):
        if math.isfinite(obj) or nonfinite == "keep":
            return obj
        return None if nonfinite == "null" else str(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, nonfinite=nonfinite) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, nonfinite=nonfinite) for v in obj]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist) and hasattr(obj, "shape"):  # torch tensors, numpy arrays/scalars
        detach = getattr(obj, "detach", None)
        value = (detach().cpu() if callable(detach) else obj).tolist()
        return to_jsonable(value, nonfinite=nonfinite)
    if callable(getattr(obj, "item", None)):
        try:
            return to_jsonable(obj.item(), nonfinite=nonfinite)
        except (TypeError, ValueError):
            pass
    try:
        return to_jsonable(float(obj), nonfinite=nonfinite)
    except (TypeError, ValueError):
        return str(obj)
