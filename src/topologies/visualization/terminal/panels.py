"""Dashboard panel registry (separate namespaces for training and experiment dashboards).

A panel is a function ``(state, width, height) -> list[str | Line]`` plus layout hints.
The dashboard draws borders and titles, clips content and lays panels out::

    @panel("tokens", title="tokens/s", min_height=3, priority=55)
    def tokens(state, width, height):
        v = state.latest_value("tokens_per_s")
        return [f"{v:,.0f} tok/s" if v is not None else "n/a"]
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from topologies.visualization.terminal.text import Line, as_line, styled
from topologies.visualization.terminal.widgets import box, debug_enabled

DOMAINS = ("training", "experiment")


@dataclass
class Panel:
    """A dashboard panel: render function plus layout hints."""

    name: str
    fn: Callable[[Any, int, int], Sequence[Any]]
    title: str | None = None
    min_height: int = 3
    min_width: int = 20
    preferred_height: int | None = None
    priority: int = 50
    domain: str = "training"
    border: bool = True
    title_right: Callable[[Any], Any] | None = None

    def __call__(self, state: Any, width: int, height: int) -> Sequence[Any]:
        return self.fn(state, width, height)

    def __post_init__(self) -> None:
        if self.preferred_height is None:
            self.preferred_height = self.min_height

    def render(self, state: Any, width: int, height: int) -> list[Line]:
        """Exactly ``height`` lines of ``width`` cells (boxed unless ``border=False``)."""
        inner_w = width - 4 if self.border else width
        inner_h = height - 2 if self.border else height
        try:
            body = [as_line(x).clip(max(0, inner_w))
                    for x in self.fn(state, max(1, inner_w), max(1, inner_h))]
        except Exception as exc:
            if debug_enabled():
                raise
            body = [styled(f"<panel {self.name} failed: {type(exc).__name__}: {exc}>", "muted")
                    .clip(max(0, inner_w))]
        if len(body) > inner_h:
            body = body[-inner_h:] if self.name == "log" else body[:inner_h]
        if not self.border:
            return [b.pad(width) for b in body] + [Line().pad(width)] * max(0, height - len(body))
        right = None
        if self.title_right is not None:
            try:
                right = self.title_right(state)
            except Exception:  # noqa: BLE001
                right = None
        return box(body, self.title if self.title is not None else self.name, width, height,
                   title_right=right)


_REGISTRY: dict[str, dict[str, Panel]] = {d: {} for d in DOMAINS}


def panel(name: str, *, title: str | None = None, min_height: int = 3, min_width: int = 20,
          preferred_height: int | None = None, priority: int = 50, domain: str = "training",
          border: bool = True, register: bool = True) -> Callable[[Callable], Panel]:
    """Decorator turning a render function into a registered :class:`Panel`."""
    if domain not in _REGISTRY:
        raise ValueError(f"Unknown panel domain {domain!r}; use one of {DOMAINS}")

    def decorator(fn: Callable) -> Panel:
        p = fn if isinstance(fn, Panel) else Panel(
            name, fn, title=title, min_height=min_height, min_width=min_width,
            preferred_height=preferred_height, priority=priority, domain=domain, border=border)
        if register:
            _REGISTRY[domain][name] = p
        return p

    return decorator


def register_panel(p: Panel) -> Panel:
    _REGISTRY[p.domain][p.name] = p
    return p


def get_panel(name: str, domain: str = "training") -> Panel:
    try:
        return _REGISTRY[domain][name]
    except KeyError:
        raise ValueError(f"Unknown {domain} panel {name!r}; available: "
                         f"{', '.join(sorted(_REGISTRY[domain]))}") from None


def panel_names(domain: str = "training") -> list[str]:
    return list(_REGISTRY[domain])


def resolve_panels(items: Sequence[str | Panel | Callable] | None, domain: str,
                   defaults: Sequence[str]) -> list[Panel]:
    """Names, Panel objects or plain functions → Panels (defaults when ``items`` is None)."""
    out = []
    for item in (defaults if items is None else items):
        if isinstance(item, Panel):
            out.append(item)
        elif isinstance(item, str):
            out.append(get_panel(item, domain))
        elif callable(item):
            name = getattr(item, "__name__", "panel")
            out.append(Panel(name, item, domain=domain))
        else:
            raise TypeError(f"panels must be names, Panel objects or callables; got {item!r}")
    return out


__all__ = ["DOMAINS", "Panel", "get_panel", "panel", "panel_names", "register_panel",
           "resolve_panels"]
