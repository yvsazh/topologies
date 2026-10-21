"""Extensible validation framework: composable checks over a candidate/registry entry."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""
    severity: str = "error"  # error | warning


class CheckContext(Protocol):
    slug: str


Check = Callable[[object], CheckResult]


@dataclass
class ValidationSuite:
    name: str
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> ValidationSuite:
        self.checks.append(check)
        return self

    def run(self, target: object) -> list[CheckResult]:
        return [c(target) for c in self.checks]


def check_module_imports(record: object) -> CheckResult:
    slug = getattr(record, "slug", "?")
    module_path = getattr(record, "module_path", "")
    class_name = getattr(record, "class_name", "")
    if not module_path or not class_name:
        return CheckResult("module_imports", False, f"{slug}: missing module_path/class_name")
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            return CheckResult("module_imports", False, f"{slug}: {module_path} has no {class_name}")
        import torch.nn as nn

        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            return CheckResult("module_imports", False, f"{slug}: {class_name} not nn.Module")
        return CheckResult("module_imports", True, f"{slug}: {class_name} imports OK")
    except ImportError as e:
        return CheckResult("module_imports", False, f"{slug}: import failed: {e}")


def check_forward_smoke(record: object) -> CheckResult:
    """Run an explicit offline constructor/input smoke specification (CPU)."""
    import torch

    slug = getattr(record, "slug", "?")
    module_path = getattr(record, "module_path", "")
    class_name = getattr(record, "class_name", "")
    try:
        spec = getattr(record, "smoke_spec", None)
        if spec is None:
            from topologies.validation.catalog_specs import (
                catalog_specs,
                fixture_inputs,
                tensor_leaves,
            )

            approved = catalog_specs().get(slug)
            if approved is None:
                return CheckResult("forward_smoke", False,
                                   f"{slug}: no explicit smoke_spec; not executed", severity="warning")
            mod = importlib.import_module(module_path)
            obj = getattr(mod, class_name)(**approved["constructor"]).eval()
            args, kwargs = fixture_inputs(slug)
            with torch.no_grad():
                output = obj(*args, **kwargs)
            leaves = tensor_leaves(output)
            if not leaves or any(not bool(torch.isfinite(value).all()) for value in leaves
                                 if value.is_floating_point() or value.is_complex()):
                return CheckResult("forward_smoke", False, f"{slug}: invalid structured output")
            return CheckResult("forward_smoke", True, f"{slug}: explicit packaged fixture forward OK")
        if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), dict):
            return CheckResult("forward_smoke", False, f"{slug}: smoke_spec requires inputs mapping")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        obj = cls(**dict(spec.get("constructor", {})))
        obj.eval()
        with __import__("torch").no_grad():
            inputs = {key: torch.as_tensor(value) for key, value in spec["inputs"].items()}
            out = obj(**inputs) if spec.get("call", "kwargs") == "kwargs" else obj(*inputs.values())
        expected = spec.get("output")
        selected = out
        if expected and "path" in expected:
            for part in expected["path"]:
                selected = selected[part]
        if expected and "shape" in expected:
            assert tuple(selected.shape) == tuple(expected["shape"]), f"unexpected shape {tuple(selected.shape)}"
        if expected and "type" in expected:
            assert isinstance(selected, getattr(torch, expected["type"])), f"unexpected output type {type(selected)}"
        return CheckResult("forward_smoke", True, f"{slug}: forward OK")
    except Exception as e:  # noqa: BLE001
        return CheckResult("forward_smoke", False, f"{slug}: forward failed: {e}")


def check_metadata_complete(record: object) -> CheckResult:
    slug = getattr(record, "slug", "?")
    missing = [f for f in ("summary", "description") if not getattr(record, f, "")]
    if missing:
        return CheckResult("metadata_complete", False, f"{slug}: missing {missing}", severity="warning")
    return CheckResult("metadata_complete", True, f"{slug}: metadata present")


def check_catalog_standalone(record: object) -> CheckResult:
    """Catalog files are stdlib + torch only and carry the registry id as a class attribute."""
    from topologies._catalog import KINDS, standalone_violations

    slug = getattr(record, "slug", "?")
    module_path = getattr(record, "module_path", "") or ""
    kind = next((k for k, (pkg, _) in KINDS.items() if module_path.startswith(pkg + ".")), None)
    if kind is None:
        return CheckResult("catalog_standalone", True, f"{slug}: not a catalog module, skipped",
                           severity="warning")
    try:
        spec = importlib.util.find_spec(module_path)
        source = Path(spec.origin).read_text(encoding="utf-8") if spec and spec.origin else None
    except (ImportError, ValueError) as e:
        return CheckResult("catalog_standalone", False, f"{slug}: cannot locate source: {e}")
    if source is None:
        return CheckResult("catalog_standalone", False, f"{slug}: no source for {module_path}")
    problems = standalone_violations(source, module_path)
    attr = KINDS[kind][1]
    cls = getattr(importlib.import_module(module_path), getattr(record, "class_name", ""), None)
    if getattr(cls, attr, None) != getattr(record, "id", slug):
        problems.append(f"{attr} is {getattr(cls, attr, None)!r}, expected registry id")
    if problems:
        return CheckResult("catalog_standalone", False, f"{slug}: " + "; ".join(problems))
    return CheckResult("catalog_standalone", True, f"{slug}: standalone PyTorch")


def default_suite() -> ValidationSuite:
    return ValidationSuite("default", [check_module_imports, check_catalog_standalone,
                                       check_forward_smoke, check_metadata_complete])
