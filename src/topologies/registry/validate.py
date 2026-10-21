"""Deterministic registry validation: schema, uniqueness, slugs, relations, taxonomy refs."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Any

from topologies.registry.index import RegistryIndex

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _validate_rich_fields(index: RegistryIndex, report: ValidationReport) -> None:
    """Offline deterministic checks for rich detail fields (additive optional).

    Hard errors: examples[].code must compile(); graph params >= 0, tree parents
    must resolve, total_params must equal sum of direct per-node params. Shapes are
    best-effort (no hard check beyond basic type sanity).
    """
    from topologies.metadata.schema import graph_param_sum

    for r in index.records:
        for i, ex in enumerate(getattr(r, "examples", []) or []):
            code = getattr(ex, "code", "")
            title = getattr(ex, "title", "") or f"example[{i}]"
            if not code or not code.strip():
                report.error(f"{r.slug}: example {title!r} has empty code")
                continue
            try:
                compile(code, f"<{r.slug}:{title}>", "exec")
            except SyntaxError as e:
                report.error(f"{r.slug}: example {title!r} does not compile: {e}")
                continue
            kind = getattr(getattr(r, "kind", ""), "value", str(getattr(r, "kind", "")))
            if kind in {"architecture", "component"}:
                for problem in _example_content_problems(code):
                    report.error(f"{r.slug}: example {title!r} violates Rule 0 content rule: {problem}")
        graph = getattr(r, "model_graph", None)
        if graph is not None:
            ids = {n.id for n in graph.nodes}
            if len(ids) != len(graph.nodes):
                report.error(f"{r.slug}: model_graph has duplicate node ids")
            for n in graph.nodes:
                if not n.id:
                    report.error(f"{r.slug}: model_graph node with empty id")
                if n.params < 0:
                    report.error(f"{r.slug}: model_graph node {n.id!r} has negative params")
                if n.parent:
                    if n.parent == n.id:
                        report.error(f"{r.slug}: model_graph node {n.id!r} is its own parent")
                    elif n.parent not in ids:
                        report.error(
                            f"{r.slug}: model_graph node {n.id!r} "
                            f"has dangling parent {n.parent!r}")
                if n.output_shape is not None:
                    if not isinstance(n.output_shape, list) or len(n.output_shape) > 8:
                        report.error(
                            f"{r.slug}: model_graph node {n.id!r} has bad output_shape")
                    else:
                        for dim in n.output_shape:
                            if dim is not None and (not isinstance(dim, int) or dim < 0):
                                report.error(
                                    f"{r.slug}: model_graph node {n.id!r} "
                                    f"has bad shape dim {dim!r}")
                                break
            if graph.total_params < 0:
                report.error(f"{r.slug}: model_graph has negative total_params")
            else:
                want = graph_param_sum(graph)
                if graph.total_params != want:
                    report.error(
                        f"{r.slug}: model_graph total_params {graph.total_params} "
                        f"!= sum of node params {want}")
        stats = getattr(r, "model_stats", None)
        if stats is not None:
            if stats.total_params < 0 or stats.trainable_params < 0:
                report.error(f"{r.slug}: model_stats has negative params")
            if stats.trainable_params > stats.total_params:
                report.error(
                    f"{r.slug}: model_stats trainable_params > total_params")
            if stats.depth < 0:
                report.error(f"{r.slug}: model_stats has negative depth")


def _example_content_problems(code: str) -> list[str]:
    """Enforce comment-free, framework-first executable catalog examples."""
    import ast
    import tokenize

    problems: list[str] = []
    tree = ast.parse(code)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for token in tokenize.generate_tokens(iter(code.splitlines(keepends=True)).__next__):
        if token.type == tokenize.COMMENT:
            problems.append(f"comment at line {token.start[0]}")
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and ast.get_docstring(node, clean=False) is not None):
            problems.append(f"docstring at line {getattr(node, 'lineno', 1)}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "tensor_leaves":
            problems.append(f"output-walking helper at line {node.lineno}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"backward", "step"}:
                problems.append(f"manual training operation at line {node.lineno}")
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
                    and node.func.attr in {"randn", "rand", "randint"}):
                parent = node
                in_generator = False
                while parent in parents:
                    parent = parents[parent]
                    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        in_generator = True
                        break
                if not in_generator:
                    problems.append(f"raw torch random fixture outside a generator at line {node.lineno}")
    return problems


def _validate_smoke_fields(index: RegistryIndex, report: ValidationReport) -> None:
    """Offline deterministic checks for smoke evidence (additive optional).

    Hard errors: finite floats, len(train_loss) <= 50, example_title resolves
    to an entry in examples[], passed consistent with sub-results, standard
    component check names present, demos carry no smoke fields.
    """


    for r in index.records:
        is_demo = str(getattr(r, "slug", "")).startswith("demo-")
        tr = getattr(r, "training_report", None)
        cc = getattr(r, "component_check", None)
        if is_demo:
            if tr is not None:
                report.error(f"{r.slug}: demo entries must not carry training_report")
            if cc is not None:
                report.error(f"{r.slug}: demo entries must not carry component_check")
            continue
        kind = getattr(getattr(r, "kind", ""), "value", str(getattr(r, "kind", "")))
        if kind == "architecture":
            if tr is None:
                report.error(f"{r.slug}: architecture missing training_report")
            else:
                _check_training_report(r, tr, report)
            # Architectures must not carry component_check (components only).
            if cc is not None:
                report.error(f"{r.slug}: architecture must not carry component_check")
        elif kind == "component":
            if cc is None:
                report.error(f"{r.slug}: component missing component_check")
            else:
                _check_component_check(r, cc, report)
            if tr is not None:
                report.error(f"{r.slug}: component must not carry training_report")
        # Unknown kinds: validate whatever is present.
        else:
            if tr is not None:
                _check_training_report(r, tr, report)
            if cc is not None:
                _check_component_check(r, cc, report, strict_names=False)


def _check_training_report(r, tr, report: ValidationReport) -> None:
    import math as _math

    titles = {getattr(e, "title", "") for e in (getattr(r, "examples", []) or [])}
    if not getattr(tr, "example_title", ""):
        report.error(f"{r.slug}: training_report missing example_title")
    elif tr.example_title not in titles:
        report.error(
            f"{r.slug}: training_report example_title {tr.example_title!r} "
            f"does not match any entry in examples[]")
    else:
        # The referenced example must be tagged smoke-training (either on the
        # example itself via extra tags or on the entry tags).
        entry_tags = set(getattr(r, "tags", []) or [])
        matched = None
        for e in (getattr(r, "examples", []) or []):
            if getattr(e, "title", "") == tr.example_title:
                matched = e
                break
        ex_tags: set[str] = set()
        if matched is not None:
            try:
                dump = matched.model_dump(mode="json")
                raw_tags = dump.get("tags", [])
                if isinstance(raw_tags, list):
                    ex_tags = {str(t) for t in raw_tags}
            except Exception:  # noqa: BLE001 - tags are best-effort
                ex_tags = set()
        if "smoke-training" not in ex_tags and "smoke-training" not in entry_tags:
            report.error(
                f"{r.slug}: training example {tr.example_title!r} "
                f"must be tagged 'smoke-training' (example tags or entry tags)")
    losses = list(getattr(tr, "train_loss", []) or [])
    if not losses:
        report.error(f"{r.slug}: training_report train_loss is empty")
    if len(losses) > 50:
        report.error(
            f"{r.slug}: training_report train_loss has {len(losses)} points (>50 cap)")
    for x in losses:
        try:
            fx = float(x)
        except Exception:  # noqa: BLE001
            report.error(f"{r.slug}: training_report train_loss has non-float {x!r}")
            break
        if not _math.isfinite(fx):
            report.error(f"{r.slug}: training_report train_loss has non-finite {x!r}")
            break
    if getattr(tr, "val_loss", None) is not None:
        try:
            fv = float(tr.val_loss)
            if not _math.isfinite(fv):
                report.error(f"{r.slug}: training_report val_loss is non-finite")
        except Exception:  # noqa: BLE001
            report.error(f"{r.slug}: training_report val_loss is not a float")
    steps = int(getattr(tr, "steps", 0) or 0)
    if steps <= 0:
        report.error(f"{r.slug}: training_report steps must be > 0")
    if steps > 50:
        report.error(f"{r.slug}: training_report steps {steps} exceeds 50-step budget")
    if losses and steps < len(losses):
        report.error(
            f"{r.slug}: training_report steps {steps} < len(train_loss) {len(losses)}")
    if not getattr(tr, "torch_version", ""):
        report.error(f"{r.slug}: training_report missing torch_version")
    try:
        dur = float(getattr(tr, "duration_s", -1.0))
        if not _math.isfinite(dur) or dur < 0:
            report.error(f"{r.slug}: training_report duration_s must be finite >= 0")
    except Exception:  # noqa: BLE001
        report.error(f"{r.slug}: training_report duration_s is not a float")
    cfg = getattr(tr, "config", {}) or {}
    if not isinstance(cfg, dict):
        report.error(f"{r.slug}: training_report config must be a mapping")
    else:
        for key in ("optimizer", "lr", "batch_size", "loss"):
            if key not in cfg:
                report.error(f"{r.slug}: training_report config missing {key!r}")
        try:
            lr = float(cfg.get("lr", float("nan")))
            if not _math.isfinite(lr) or lr <= 0:
                report.error(f"{r.slug}: training_report config lr must be finite > 0")
        except Exception:  # noqa: BLE001
            report.error(f"{r.slug}: training_report config lr is not a float")
        try:
            bs = int(cfg.get("batch_size", 0))
            if bs <= 0:
                report.error(f"{r.slug}: training_report config batch_size must be > 0")
        except Exception:  # noqa: BLE001
            report.error(f"{r.slug}: training_report config batch_size is not an int")
    # Passed consistency: passed=True requires non-empty finite losses + steps>0.
    if getattr(tr, "passed", False) and (not losses or steps <= 0):
        report.error(f"{r.slug}: training_report passed=True with no losses/steps")
    # Examples count: training is an extra one, keep existing >=2.
    if len(getattr(r, "examples", []) or []) < 3:
        report.error(
            f"{r.slug}: architecture with training_report must keep >=2 existing "
            f"examples plus the smoke-training one (>=3 total)")


def _check_component_check(r, cc, report: ValidationReport, strict_names: bool = True) -> None:
    import math as _math

    from topologies.metadata.schema import COMPONENT_CHECK_NAMES

    checks = list(getattr(cc, "checks", []) or [])
    names = [getattr(c, "name", "") for c in checks]
    if strict_names:
        for want in COMPONENT_CHECK_NAMES:
            if want not in names:
                report.error(f"{r.slug}: component_check missing {want!r} check")
    # No duplicate check names.
    if len(set(names)) != len(names):
        report.error(f"{r.slug}: component_check has duplicate check names")
    for c in checks:
        if not getattr(c, "name", ""):
            report.error(f"{r.slug}: component_check has entry with empty name")
        if not isinstance(getattr(c, "passed", None), bool):
            report.error(f"{r.slug}: component_check {getattr(c, 'name', '')!r} passed is not bool")
    want_overall = all(bool(getattr(c, "passed", False)) for c in checks) if checks else False
    if bool(getattr(cc, "passed", False)) != want_overall:
        report.error(
            f"{r.slug}: component_check passed={getattr(cc, 'passed', None)!r} "
            f"inconsistent with sub-results (all passed={want_overall!r})")
    if not checks:
        report.error(f"{r.slug}: component_check has no checks")
    if not getattr(cc, "torch_version", ""):
        report.error(f"{r.slug}: component_check missing torch_version")
    try:
        dur = float(getattr(cc, "duration_s", -1.0))
        if not _math.isfinite(dur) or dur < 0:
            report.error(f"{r.slug}: component_check duration_s must be finite >= 0")
        if dur > 10.0 + 1e-6:
            # Budget is a generation-time guard; stored durations above it are
            # suspicious but not fatal — warn only to keep validation green on
            # slow CI hosts.
            report.warn(f"{r.slug}: component_check duration_s {dur:.2f}s exceeds 10s budget")
    except Exception:  # noqa: BLE001
        report.error(f"{r.slug}: component_check duration_s is not a float")


def _validate_artifact_pins_offline(index: RegistryIndex, report: ValidationReport) -> None:
    """Deterministic checks: sha format, path traversal, duplicate paths per entry."""
    for r in index.records:
        seen: set[str] = set()
        for pin in getattr(r, "artifacts", []) or []:
            path = getattr(pin, "path", "")
            sha = getattr(pin, "sha256", "")
            if not path or path.startswith("/") or ".." in path.replace("\\", "/").split("/"):
                report.error(f"{r.slug}: bad artifact path {path!r}")
            if path in seen:
                report.error(f"{r.slug}: duplicate artifact path {path!r}")
            seen.add(path)
            if not _SHA_RE.fullmatch(str(sha or "").strip().lower()):
                report.error(f"{r.slug}: invalid artifact sha256 for {path!r}")
            if getattr(pin, "version", 0) < 1:
                report.error(f"{r.slug}: invalid artifact version for {path!r}")
            if getattr(pin, "namespace", "") != "library":
                report.warn(f"{r.slug}: artifact {path!r} not in library namespace")


def _validate_artifact_pins_store(
    index: RegistryIndex, report: ValidationReport, client=None
) -> None:
    """When the store is reachable: manifest exists and digests match."""
    from topologies.artifact_store.exceptions import (
        ArtifactNotFoundError,
        ArtifactStoreError,
        ArtifactUnavailableError,
    )

    if client is None:
        try:
            from topologies.artifact_store import build_client_from_env

            client = build_client_from_env(dry_run=False)
        except Exception as e:  # noqa: BLE001 - no client available, warn only
            report.warn(f"artifact store client unavailable: {e}")
            return
    for r in index.records:
        for pin in getattr(r, "artifacts", []) or []:
            try:
                manifest = client.get_manifest(pin.namespace, pin.name, pin.version)
            except ArtifactUnavailableError as e:
                report.warn(f"artifact store unreachable, skipping remote check: {e}")
                return
            except ArtifactNotFoundError:
                report.error(
                    f"{r.slug}: artifact manifest missing {pin.namespace}/{pin.name}@{pin.version}")
                continue
            except ArtifactStoreError as e:
                report.warn(f"{r.slug}: store check failed for {pin.path!r}: {e}")
                continue
            want = {f["path"]: f["sha256"] for f in manifest.get("files", [])}
            if pin.path not in want:
                report.error(
                    f"{r.slug}: path {pin.path!r} not in manifest "
                    f"{pin.namespace}/{pin.name}@{pin.version}")
            elif want[pin.path].lower() != str(pin.sha256).lower():
                report.error(
                    f"{r.slug}: digest mismatch for {pin.path!r}: "
                    f"registry {pin.sha256[:12]} vs store {want[pin.path][:12]}")


def unknown_keys(model: Any, path: str = "") -> list[tuple[str, str]]:
    """``(dotted_path, hint)`` for every key a schema model does not declare.

    The registry schemas keep ``extra="allow"`` for migration safety, so typos
    pass validation silently; this makes them visible.
    """
    from pydantic import BaseModel

    from topologies._util.suggest import did_you_mean

    found: list[tuple[str, str]] = []
    if isinstance(model, BaseModel):
        declared = list(type(model).model_fields)
        for key in sorted(model.model_extra or {}):
            found.append((f"{path}{key}", did_you_mean(key, declared)))
        for name in declared:
            found.extend(unknown_keys(getattr(model, name, None), f"{path}{name}."))
    elif isinstance(model, (list, tuple)):
        for i, item in enumerate(model):
            found.extend(unknown_keys(item, f"{path[:-1]}[{i}]."))
    elif isinstance(model, dict):
        for key, item in model.items():
            found.extend(unknown_keys(item, f"{path}{key}."))
    return found


def validate_index(
    index: RegistryIndex,
    *,
    check_imports: bool = False,
    check_artifacts: bool = True,
    verify_store: bool = False,
    artifact_client=None,
    offline: bool = False,
    strict: bool = False,
) -> ValidationReport:
    """Validate the registry. ``strict`` turns unknown (undeclared) keys into errors."""
    report = ValidationReport(ok=True)
    for r in index.records:
        for key, hint in unknown_keys(r):
            message = f"{r.slug}: unknown key {key!r}.{hint}"
            if strict:
                report.error(message)
            else:
                report.warn(message)
    dups = index.duplicates()
    for slug in dups["slug"]:
        report.error(f"duplicate slug: {slug}")
    for _id in dups["id"]:
        report.error(f"duplicate id: {_id}")
    for problem in index.dangling_relations():
        report.error(f"dangling relation: {problem}")
    tax_slugs = set(index.taxonomy_by_slug)
    for r in index.records:
        for t in r.taxonomy_paths:
            if t not in tax_slugs:
                report.error(f"{r.slug}: unknown taxonomy path {t!r}")
        if not r.module_path or not r.class_name:
            report.warn(f"{r.slug}: missing module_path/class_name")
        if r.kind.value == "architecture" and not r.architecture_paradigms:
            report.warn(f"{r.slug}: architecture without paradigms")
    _validate_rich_fields(index, report)
    _validate_smoke_fields(index, report)
    if check_artifacts:
        _validate_artifact_pins_offline(index, report)
        if verify_store and not offline:
            _validate_artifact_pins_store(index, report, client=artifact_client)
    if check_imports:
        for r in index.records:
            if not r.module_path or not r.class_name:
                continue
            try:
                module = importlib.import_module(r.module_path)
                cls = getattr(module, r.class_name, None)
                if cls is None:
                    report.error(f"{r.slug}: {r.module_path} has no {r.class_name}")
                else:
                    import torch.nn as nn

                    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
                        report.error(f"{r.slug}: {r.class_name} is not an nn.Module")
            except ImportError as e:
                report.error(f"{r.slug}: cannot import {r.module_path}: {e}")
        # Catalog contract: ids match, files are standalone PyTorch, build specs fit.
        from topologies._catalog import registry_mismatches
        from topologies.build.catalog import catalog_spec_problems

        for problem in dict.fromkeys(registry_mismatches(index) + catalog_spec_problems()):
            report.error(f"catalog: {problem}")
    return report
