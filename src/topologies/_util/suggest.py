"""Hints of the form "Did you mean ...?" for unknown names."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


def did_you_mean(name: str, choices: Iterable[str], *, n: int = 1, cutoff: float = 0.6) -> str:
    """``" Did you mean 'x'?"`` (several: ``'x', 'y' or 'z'``) or ``""`` when nothing is close.

    Substring matches (``"classify"`` -> ``"text-classify"``) count as close.
    """
    options = [str(c) for c in choices]
    key = str(name).strip().lower()
    close = difflib.get_close_matches(key, options, n=n, cutoff=cutoff)
    for option in options:
        if len(close) >= n:
            break
        if key and key in option.lower() and option not in close:
            close.append(option)
    if not close:
        return ""
    if len(close) == 1:
        return f" Did you mean {close[0]!r}?"
    return " Did you mean " + ", ".join(repr(c) for c in close[:-1]) + f" or {close[-1]!r}?"
