"""Technique plugins: each declares a tier, its requirements and the result kind it emits.

Tiers: ``U`` universal, ``F`` functional interception, ``D`` differentiable input,
``A`` adapter / semantic. Third parties register with ``register_technique(cls)``
or the entry-point group ``topologies.introspection.techniques``, which is loaded
lazily the first time the registry is used.
"""
from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

TIERS = ("U", "F", "D", "A")
ENTRY_POINT_GROUP = "topologies.introspection.techniques"


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str = ""
    fix: str | None = None

    def __bool__(self) -> bool:
        return self.available


AVAILABLE = Availability(True)


class Technique:
    """Base class. Subclasses set the class attributes and implement ``run``."""

    name: ClassVar[str] = "technique"
    tier: ClassVar[str] = "U"
    requires: ClassVar[frozenset[str]] = frozenset()
    result_kind: ClassVar[str] = "introspection.result"
    summary: ClassVar[str] = ""

    def check(self, subject: Any) -> Availability:
        """Default availability from ``requires`` (roles, objective, differentiable input)."""
        for need in sorted(self.requires):
            if need.startswith("role:"):
                role = need[5:]
                if not subject.index.resolve_modules(f"role:{role}", allow_empty=True):
                    return Availability(False, f"no module has role {role!r}",
                                        f"insp.declare('<selector>', role='{role}') or an adapter")
            elif need == "adapter" and getattr(subject.adapter, "name", "generic") == "generic":
                return Availability(False, "needs a semantic adapter",
                                    "pass introspect(model, adapter=...) or insp.declare(...)")
        return AVAILABLE

    def check_call(self, subject: Any, params: dict[str, Any]) -> Availability:
        """Availability for one concrete call (explicit parameters may supply what is missing)."""
        return self.check(subject)

    def run(self, subject: Any, inputs: Any = None, **params: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {"name": cls.name, "tier": cls.tier, "requires": sorted(cls.requires),
                "result_kind": cls.result_kind, "summary": cls.summary}


class TechniqueRegistry:
    """Name -> technique class; built-ins are loaded on first access, then entry points."""

    def __init__(self, *, builtins: bool = False):
        self._classes: dict[str, type[Technique]] = {}
        self._load_builtins = builtins
        self._loaded = not builtins

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for module in _BUILTIN_MODULES:
            name = f"topologies.introspection.techniques.{module}"
            if importlib.util.find_spec(name) is not None:
                importlib.import_module(name)
        if self is not DEFAULT_TECHNIQUES:
            for builtin_name in BUILTIN_NAMES:
                self._classes.setdefault(builtin_name, DEFAULT_TECHNIQUES._classes[builtin_name])
        _load_entry_points(self)

    def register(self, cls: type[Technique]) -> type[Technique]:
        if not (isinstance(cls, type) and issubclass(cls, Technique)):
            raise TypeError("register a Technique subclass")
        if cls.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        self._classes[cls.name] = cls
        return cls

    def get(self, name: str) -> type[Technique]:
        self._ensure()
        try:
            return self._classes[name]
        except KeyError:
            import difflib

            close = difflib.get_close_matches(name, list(self._classes), n=3)
            raise KeyError(f"unknown technique {name!r}" + (f"; did you mean {close}?" if close
                                                            else "")) from None

    def names(self) -> list[str]:
        self._ensure()
        return sorted(self._classes)

    def __iter__(self) -> Iterator[type[Technique]]:
        self._ensure()
        return iter(self._classes[n] for n in sorted(self._classes))


_BUILTIN_MODULES = ("structure", "activations", "anomalies", "weights", "gradients", "attribution",
                    "attention", "interventions", "neurons", "probing", "similarity", "geometry",
                    "lens", "process", "maximization", "dictionary")

DEFAULT_TECHNIQUES = TechniqueRegistry(builtins=True)


BUILTIN_NAMES: set[str] = set()


def register_technique(cls: type[Technique]) -> type[Technique]:
    """Register globally (usable as a class decorator); prefer local registries in plugins."""
    return DEFAULT_TECHNIQUES.register(cls)


def unregister_technique(name: str) -> None:
    """Remove a globally registered technique (built-ins cannot be removed)."""
    if name in BUILTIN_NAMES:
        raise ValueError(f"{name!r} is a built-in technique")
    DEFAULT_TECHNIQUES._classes.pop(name, None)


def builtin(cls: type[Technique]) -> type[Technique]:
    """Register a built-in without triggering the lazy loader."""
    DEFAULT_TECHNIQUES._classes[cls.name] = cls
    BUILTIN_NAMES.add(cls.name)
    return cls


def _load_entry_points(registry: TechniqueRegistry) -> None:
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    try:
        selected = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 API
        selected = entry_points().get(ENTRY_POINT_GROUP, [])
    for entry in selected:
        try:
            loaded = entry.load()
            if isinstance(loaded, type) and issubclass(loaded, Technique):
                registry.register(loaded)
        except Exception:  # noqa: BLE001, S112 - a broken plugin never breaks built-ins
            continue


@dataclass
class CapabilityRow:
    name: str
    tier: str
    available: bool
    reason: str = ""
    fix: str | None = None


@dataclass
class CapabilityTable:
    """Which techniques can run on a subject, and why the others cannot."""

    rows: list[CapabilityRow] = field(default_factory=list)
    tiers: dict[str, bool] = field(default_factory=dict)

    def available(self) -> list[str]:
        return [r.name for r in self.rows if r.available]

    def unavailable(self) -> dict[str, str]:
        return {r.name: r.reason for r in self.rows if not r.available}

    def __contains__(self, name: str) -> bool:
        return name in self.available()

    def to_records(self) -> list[dict[str, Any]]:
        return [{"technique": r.name, "tier": r.tier, "available": r.available, "reason": r.reason,
                 "fix": r.fix} for r in self.rows]

    def to_dict(self) -> dict[str, Any]:
        return {"tiers": dict(self.tiers), "techniques": self.to_records()}

    def __str__(self) -> str:
        lines = ["tier  available  technique                         reason / fix"]
        for r in self.rows:
            note = "" if r.available else f"{r.reason}" + (f"  (fix: {r.fix})" if r.fix else "")
            lines.append(f"{r.tier:<5} {'yes' if r.available else 'no':<10} {r.name:<33} {note}")
        tiers = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in self.tiers.items())
        return "\n".join([f"tiers: {tiers}", *lines])


def capabilities(subject: Any, registry: TechniqueRegistry | None = None) -> CapabilityTable:
    registry = registry or DEFAULT_TECHNIQUES
    rows = []
    for cls in registry:
        try:
            status = cls().check(subject)
        except Exception as exc:  # noqa: BLE001 - a failing check reports unavailability
            status = Availability(False, f"check failed: {exc}")
        rows.append(CapabilityRow(cls.name, cls.tier, bool(status), status.reason, status.fix))
    tiers = {tier: any(r.available for r in rows if r.tier == tier) for tier in TIERS}
    return CapabilityTable(rows, tiers)


__all__ = ["AVAILABLE", "BUILTIN_NAMES", "DEFAULT_TECHNIQUES", "Availability", "CapabilityRow",
           "CapabilityTable", "Technique", "TechniqueRegistry", "builtin", "capabilities",
           "register_technique", "unregister_technique"]
