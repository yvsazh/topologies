"""One reusable fail-fast rule for constructor arguments.

Constructors validate their geometry up front and raise ``ConfigError``
(a ``ValueError``) whose message names the argument, the offending value
and the fix — instead of a tensor-shape error deep inside ``forward``.
"""

from __future__ import annotations

from typing import Any


class ConfigError(ValueError):
    """Invalid model/component configuration, with the fix spelled out."""

    def __init__(self, message: str, *, hint: str | None = None, owner: str | None = None):
        prefix = f"{owner}: " if owner else ""
        full = prefix + message + (f"\nHint: {hint}" if hint else "")
        super().__init__(full)
        self.hint = hint
        self.owner = owner


def check_positive(owner: str | None = None, **values: Any) -> None:
    """``check_positive("TransformerLM", dim=dim, layers=layers)``."""
    for name, v in values.items():
        if v is None:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            raise ConfigError(f"{name} must be a positive number; got {v!r}.", owner=owner)


def check_non_negative(owner: str | None = None, **values: Any) -> None:
    for name, v in values.items():
        if v is None:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ConfigError(f"{name} must be >= 0; got {v!r}.", owner=owner)


def check_range(name: str, value: float, low: float, high: float, *,
                owner: str | None = None, inclusive_high: bool = False) -> None:
    ok = low <= value <= high if inclusive_high else low <= value < high
    if not ok:
        bracket = "]" if inclusive_high else ")"
        raise ConfigError(f"{name} must be in [{low}, {high}{bracket}; got {value!r}.",
                          owner=owner)


def check_divisible(numerator: tuple[str, int], denominator: tuple[str, int], *,
                    owner: str | None = None) -> None:
    """``check_divisible(("dim", 100), ("heads", 8))`` -> ConfigError with neighbours."""
    (n_name, n), (d_name, d) = numerator, denominator
    if d <= 0:
        raise ConfigError(f"{d_name} must be positive; got {d}.", owner=owner)
    if n % d != 0:
        lower = (n // d) * d
        options = [x for x in (lower, lower + d) if x > 0]
        near = sorted((k for k in range(1, n + 1) if n > 0 and n % k == 0),
                      key=lambda k: (abs(k - d), k))[:2]
        hint = f"use {n_name} in {options}"
        if near:
            hint += f" or {d_name} in {sorted(near)}"
        raise ConfigError(f"{n_name}={n} must be divisible by {d_name}={d}.", hint=hint,
                          owner=owner)


def check_choice(name: str, value: Any, choices: tuple | list, *, owner: str | None = None):
    if value not in choices:
        import difflib

        sug = difflib.get_close_matches(str(value), [str(c) for c in choices], n=1)
        hint = f"did you mean {sug[0]!r}?" if sug else None
        raise ConfigError(f"{name}={value!r} is not one of {list(choices)}.", hint=hint,
                          owner=owner)
