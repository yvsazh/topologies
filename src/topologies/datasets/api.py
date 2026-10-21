"""Offline catalog discovery and dataset factory."""

from __future__ import annotations

import json
from importlib.resources import files

from .schema import validate_catalog


class DatasetError(RuntimeError):
    pass


class DatasetOfflineError(DatasetError):
    pass


class DatasetDownloadError(DatasetError):
    pass


def _catalog() -> dict:
    return validate_catalog(
        json.loads(files(__package__).joinpath("catalog.json").read_text(encoding="utf-8"))
    )


def list_datasets() -> list[dict]:
    """Return catalog metadata (no network); records are independent JSON copies."""
    return json.loads(json.dumps(_catalog()["datasets"]))


def search_datasets(
    query: str | None = None,
    *,
    domain: str | None = None,
    task: str | None = None,
    tags: list[str] | None = None,
) -> list[dict]:
    """Filter metadata case-insensitively by text/domain/task/tag."""
    result = []
    for d in list_datasets():
        text = " ".join((d["id"], d["name"], d["description"], *d["tasks"], *d["tags"])).lower()
        if query and query.lower() not in text:
            continue
        if domain and domain.lower() != d["domain"].lower():
            continue
        if task and task.lower() not in [x.lower() for x in d["tasks"]]:
            continue
        if tags and not {tag.lower() for tag in tags}.issubset(
            {tag.lower() for tag in d["tags"]}
        ):
            continue
        result.append(d)
    return result


def get_dataset(
    id: str,
    version: str | None = None,
    split: str | None = None,
    root: str | None = None,
    offline: bool = False,
    **dataset_options,
):
    """Create a dataset object without I/O; call download/prepare/load explicitly."""
    from .core import DeclarativeDataset, TopologiesDataset

    catalog = _catalog()
    definition = next((d for d in catalog["datasets"] if d["id"] == id), None)
    if definition is None:
        raise KeyError(f"unknown dataset {id!r}; use list_datasets()")
    if definition.get("catalog_only"):
        raise DatasetError(
            f"{id!r} is a metadata-only catalog record. "
            f"Obtain data and follow access terms at {definition['project_url']}; "
            "no download or loader contract is declared."
        )
    version = version or definition["default_version"]
    if version not in definition["versions"]:
        raise ValueError(f"unknown version {version!r}")
    cls = DeclarativeDataset
    implementation = definition.get("implementation")
    if implementation:
        import importlib

        module, name = implementation.split(":", 1)
        try:
            cls = getattr(importlib.import_module(module), name)
        except ImportError as exc:
            dependencies = definition.get("dependencies", {"extras": [], "packages": []})
            hints = [f"pip install 'topologies[{extra}]'" for extra in dependencies["extras"]]
            hints.extend(f"pip install {package!r}" for package in dependencies["packages"])
            hint = f" Install optional dependencies with: {'; '.join(hints)}." if hints else ""
            raise ImportError(
                f"could not import dataset implementation {implementation!r}.{hint}"
            ) from exc
        if not isinstance(cls, type) or not issubclass(cls, TopologiesDataset):
            raise TypeError("implementation must be a TopologiesDataset subclass")
    return cls(definition, version, split, root, offline, **dataset_options)
