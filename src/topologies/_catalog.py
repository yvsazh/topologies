"""Framework-side discovery of the catalog (``topologies.models`` / ``topologies.layers``).

Catalog files are standalone PyTorch code: stdlib + ``torch`` only, no
``topologies`` imports, plain ``nn.Module`` subclasses. They never know the
framework exists; this module adapts them from the outside.

A catalog class is any top-level class whose body sets
``architecture_id = "..."`` (models) or ``component_id = "..."`` (layers).
Inheritance plays no role. Files are parsed with ``ast`` (never imported), so
``import topologies.models`` stays cheap; the class is imported on first use
and its id attribute is checked against the parsed one.

The registry (``registry/*.yaml``: ``id`` / ``module_path`` / ``class_name``)
is cross-checked against the scan by ``registry_mismatches`` (CI and
``topologies registry validate``). It is not read at runtime because the
YAML files are not shipped in the wheel.

``install()`` (called from ``topologies/__init__.py``) attaches the
catalog API to the two packages: ``list_models`` / ``get_model``,
``list_layers`` / ``get_layer`` and lazy class attributes such as
``topologies.models.ResNetBackbone``.
"""

from __future__ import annotations

import ast
import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# kind -> (package, id attribute)
KINDS: dict[str, tuple[str, str]] = {
    "model": ("topologies.models", "architecture_id"),
    "layer": ("topologies.layers", "component_id"),
}

@dataclass(frozen=True)
class CatalogEntry:
    name: str
    id: str
    module: str
    is_demo: bool
    kind: str = "model"

    def load(self) -> type:
        cls = getattr(importlib.import_module(self.module), self.name)
        attr = KINDS[self.kind][1]
        if getattr(cls, attr, None) != self.id:
            raise ImportError(
                f"{self.module}.{self.name}.{attr} is {getattr(cls, attr, None)!r}, "
                f"expected {self.id!r}")
        return cls


def scan(package: str, directory: Path, kind: str) -> dict[str, CatalogEntry]:
    """``class name -> entry`` for every catalog class defined in ``directory``."""
    attribute = KINDS[kind][1]
    out: dict[str, CatalogEntry] = {}
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8")
        for node in ast.parse(text, filename=str(path)).body:
            if not isinstance(node, ast.ClassDef):
                continue
            values = {}
            for assignment in node.body:
                if isinstance(assignment, ast.Assign):
                    targets = assignment.targets
                elif isinstance(assignment, ast.AnnAssign):
                    targets = [assignment.target]
                else:
                    continue
                if isinstance(assignment.value, ast.Constant):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            values[target.id] = assignment.value.value
            ident = values.get(attribute)
            if not isinstance(ident, str):
                continue
            if node.name in out:
                raise ValueError(f"duplicate catalog class {node.name!r}: {path}")
            out[node.name] = CatalogEntry(
                name=node.name, id=ident, module=f"{package}.{path.stem}",
                is_demo=values.get("is_demo") is True, kind=kind,
            )
    return out


_CACHE: dict[str, dict[str, CatalogEntry]] = {}


def entries(kind: str) -> dict[str, CatalogEntry]:
    """Cached scan of the ``kind`` package (``"model"`` or ``"layer"``)."""
    if kind not in _CACHE:
        package = KINDS[kind][0]
        directory = Path(importlib.import_module(package).__path__[0])
        _CACHE[kind] = scan(package, directory, kind)
    return _CACHE[kind]


def find(entries: dict[str, CatalogEntry], key: str, kind: str) -> CatalogEntry:
    """Look up by class name, registry id, or module name (``llama_decoder``)."""
    if key in entries:
        return entries[key]
    norm = key.strip().lower().replace("_", "-")
    for e in entries.values():
        if norm in (e.id, e.name.lower(), e.module.rsplit(".", 1)[-1].replace("_", "-")):
            return e
    import difflib

    names = sorted({*entries, *(e.id for e in entries.values() if e.id)})
    sug = difflib.get_close_matches(key, names, n=3)
    hint = f" Did you mean {', '.join(map(repr, sug))}?" if sug else ""
    raise KeyError(f"Unknown {kind} {key!r}.{hint} See list_{kind}s().")


def list_entries(kind: str, *, include_demo: bool = True) -> list[dict[str, Any]]:
    return [
        {"name": e.name, "id": e.id, "module": e.module, "is_demo": e.is_demo}
        for e in sorted(entries(kind).values(), key=lambda e: e.id or e.name)
        if include_demo or not e.is_demo
    ]


def get(kind: str, key: str) -> type:
    return find(entries(kind), key, kind).load()


def catalog_id(obj: Any) -> tuple[str, str] | None:
    """``(kind, id)`` if ``obj`` (class or instance) is the canonical catalog class."""
    cls = obj if isinstance(obj, type) else type(obj)
    for kind, (package, attr) in KINDS.items():
        ident = cls.__dict__.get(attr)
        if not isinstance(ident, str) or not cls.__module__.startswith(package + "."):
            continue
        entry = entries(kind).get(cls.__name__)
        if entry is not None and entry.id == ident and entry.module == cls.__module__:
            return kind, ident
    return None


FRAMEWORK_DECORATORS = frozenset({"presets", "register_presets", "budget_knobs",
                                  "register_budget_knobs"})


def standalone_violations(source: str, filename: str = "<catalog>") -> list[str]:
    """Why ``source`` is not a standalone catalog file (empty = stdlib + torch only).

    Flags imports outside the standard library and ``torch``, relative imports
    and framework decorators (``@presets`` / ``@budget_knobs`` ...).
    """
    import ast
    import sys

    from topologies.validation.source_style import comment_and_docstring_violations

    allowed = set(sys.stdlib_module_names) | {"torch", "__future__"}
    out: list[str] = []
    from topologies.validation.implementation import discarded_algorithm_classes

    out.extend(f"{filename}: {name}.forward discards its algorithm by multiplying it by zero"
               for name in discarded_algorithm_classes(source))
    out.extend(f"{filename}:{problem}" for problem in
               comment_and_docstring_violations(source, filename))
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            mods = [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                out.append(f"{filename}:{node.lineno}: relative import "
                           f"'{'.' * node.level}{node.module or ''}'")
                continue
            mods = [(node.module or "", node.lineno)]
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            for d in node.decorator_list:
                target = d.func if isinstance(d, ast.Call) else d
                name = target.attr if isinstance(target, ast.Attribute) else \
                    getattr(target, "id", "")
                if name in FRAMEWORK_DECORATORS:
                    out.append(f"{filename}:{d.lineno}: framework decorator @{name}")
            continue
        else:
            continue
        for mod, line in mods:
            if mod.split(".")[0] not in allowed:
                out.append(f"{filename}:{line}: import of {mod!r} (stdlib + torch only)")
    return out


def registry_mismatches(index: Any) -> list[str]:
    """Problems between registry records and the scanned catalog (empty = consistent).

    ``index`` is a ``RegistryIndex`` (``Registry.load().index``).

    Checks every registry record pointing into ``topologies.models`` /
    ``topologies.layers``: the module imports, the class exists, is an
    ``nn.Module``, and its ``architecture_id`` / ``component_id`` equals the
    registry id; and every scanned catalog class has a registry record.
    """
    import torch.nn as nn

    problems: list[str] = []
    seen: dict[str, set[str]] = {k: set() for k in KINDS}
    for rec in index.records:
        mod = rec.module_path or ""
        kind = next((k for k, (pkg, _) in KINDS.items() if mod.startswith(pkg + ".")), None)
        if kind is None:
            continue
        attr = KINDS[kind][1]
        try:
            cls = getattr(importlib.import_module(mod), rec.class_name)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{rec.id}: cannot import {mod}:{rec.class_name} ({e})")
            continue
        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            problems.append(f"{rec.id}: {mod}:{rec.class_name} is not an nn.Module subclass")
        if getattr(cls, attr, None) != rec.id:
            problems.append(f"{rec.id}: {rec.class_name}.{attr} is "
                            f"{getattr(cls, attr, None)!r}, expected {rec.id!r}")
        entry = entries(kind).get(rec.class_name)
        if entry is None or entry.module != mod:
            problems.append(f"{rec.id}: {mod}:{rec.class_name} not found by catalog discovery")
        seen[kind].add(rec.class_name)
    for kind in KINDS:
        for name, e in entries(kind).items():
            if name not in seen[kind]:
                problems.append(f"{e.id}: {e.module}:{name} has no registry record")
            path = Path(importlib.import_module(KINDS[kind][0]).__path__[0])
            source = (path / f"{e.module.rsplit('.', 1)[1]}.py").read_text(encoding="utf-8")
            problems += [f"{e.id}: {v}" for v in standalone_violations(source, e.module)]
    return problems


# ---------------------------------------------------------------- package API


def _deprecated_base(name: str) -> Any:
    warnings.warn(
        f"{name} is deprecated: catalog classes subclass torch.nn.Module directly. "
        "Use topologies.build.Configurable for automatic constructor recording.",
        DeprecationWarning, stacklevel=3)
    from topologies import _compat

    return getattr(_compat, name)


def _make_namespace(kind: str, module: Any) -> None:
    plural = f"{kind}s"
    compat = "TopologiesModel" if kind == "model" else "TopologiesLayer"

    def list_fn(*, include_demo: bool = True) -> list[dict[str, Any]]:
        return list_entries(kind, include_demo=include_demo)

    def get_fn(key: str) -> type:
        return get(kind, key)

    list_fn.__name__ = list_fn.__qualname__ = f"list_{plural}"
    list_fn.__doc__ = f'``[{{"name", "id", "module", "is_demo"}}, ...]`` for every catalog {kind}.'
    get_fn.__name__ = get_fn.__qualname__ = f"get_{kind}"
    get_fn.__doc__ = f"Catalog {kind} class by class name, registry id or module name."
    for fn in (list_fn, get_fn):
        fn.__module__ = module.__name__

    def __getattr__(name: str) -> Any:
        if name == "__all__":
            return [f"get_{kind}", f"list_{plural}", *sorted(entries(kind))]
        if name == compat:
            return _deprecated_base(name)
        entry = entries(kind).get(name)
        if entry is None:
            raise AttributeError(f"module {module.__name__!r} has no attribute {name!r}")
        cls = entry.load()
        setattr(module, name, cls)
        return cls

    def __dir__() -> list[str]:
        return sorted(set(vars(module)) | set(__getattr__("__all__")))

    setattr(module, f"list_{plural}", list_fn)
    setattr(module, f"get_{kind}", get_fn)
    module.__getattr__ = __getattr__
    module.__dir__ = __dir__


def install() -> None:
    """Attach the catalog API to ``topologies.models`` and ``topologies.layers``."""
    for kind, (package, _) in KINDS.items():
        module = importlib.import_module(package)
        if not hasattr(module, f"get_{kind}"):
            _make_namespace(kind, module)
