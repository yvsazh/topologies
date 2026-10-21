"""Registry loader: read version-controlled YAML files, validate with pydantic."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import ValidationError

from topologies.metadata.schema import ArchitectureRecord, TaxonomyNode


class RegistryLoadError(Exception):
    pass


def _iter_yaml_files(dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        if d.is_dir():
            files.extend(sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")))
    return files


def load_records(dirs: list[Path]) -> tuple[list[ArchitectureRecord], list[str]]:
    """Returns (records, errors). Never raises on a single bad file."""
    records: list[ArchitectureRecord] = []
    errors: list[str] = []
    for path in _iter_yaml_files(dirs):
        try:
            data = yaml.safe_load(path.read_text()) or {}
            records.append(ArchitectureRecord.model_validate(data))
        except ValidationError as e:
            errors.append(f"{path}: {e}")
        except Exception as e:  # noqa: BLE001 - report, don't crash
            errors.append(f"{path}: {e}")
    return records, errors


def load_taxonomy_nodes(dirs: list[Path]) -> tuple[list[TaxonomyNode], list[str]]:
    nodes: list[TaxonomyNode] = []
    errors: list[str] = []
    for path in _iter_yaml_files(dirs):
        try:
            data = yaml.safe_load(path.read_text()) or {}
            items = data.get("nodes", data)  # allow {nodes: [...]} or single node
            if isinstance(items, dict) and "slug" in items:
                items = [items]
            for item in items or []:
                nodes.append(TaxonomyNode.model_validate(item))
        except ValidationError as e:
            errors.append(f"{path}: {e}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{path}: {e}")
    return nodes, errors


def find_registry_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    import os

    env = os.environ.get("TOPOLOGIES_REGISTRY_DIR")
    if env:
        return Path(env)
    # src/topologies/registry -> repo/registry (4 levels up from this file's parent chain)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "registry"
        if (candidate / "architectures").exists() or (candidate / "components").exists():
            return candidate
    raise RegistryLoadError(
        "No registry directory found (the registry YAML is not part of the installed "
        "package). Pass --registry / Registry.load(path) or set TOPOLOGIES_REGISTRY_DIR.")
