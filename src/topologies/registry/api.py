"""High-level Registry API used by the CLI and docs."""

from __future__ import annotations

from pathlib import Path

from topologies.registry.index import RegistryIndex
from topologies.registry.loader import find_registry_root, load_records, load_taxonomy_nodes
from topologies.registry.validate import ValidationReport, validate_index


class Registry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = find_registry_root(root)
        self.index = RegistryIndex()
        self.load_errors: list[str] = []

    @classmethod
    def load(cls, root: str | Path | None = None) -> Registry:
        reg = cls(root)
        reg.reload()
        return reg

    def reload(self) -> Registry:
        arch = self.root / "architectures"
        comp = self.root / "components"
        tax = self.root / "taxonomy"
        records, rec_errors = load_records([arch, comp])
        nodes, tax_errors = load_taxonomy_nodes([tax])
        self.index = RegistryIndex(records=records, taxonomy=nodes)
        self.load_errors = rec_errors + tax_errors
        return self

    # convenience proxies
    def get(self, slug_or_id: str):
        return self.index.get(slug_or_id)

    def search(self, **kwargs):
        return self.index.search(**kwargs)

    def validate(
        self,
        *,
        check_imports: bool = False,
        verify_store: bool = False,
        offline: bool = False,
        artifact_client=None,
        strict: bool = False,
    ) -> ValidationReport:
        report = validate_index(
            self.index, check_imports=check_imports,
            verify_store=verify_store, offline=offline,
            artifact_client=artifact_client, strict=strict)
        for e in self.load_errors:
            report.error(f"load error: {e}")
        return report

    def stats(self) -> dict:
        return self.index.stats()
