"""In-memory registry index: lookup, rich filtering, relations, duplicates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from topologies.enums import EntryKind
from topologies.metadata.schema import ArchitectureRecord, TaxonomyNode


@dataclass
class RegistryIndex:
    records: list[ArchitectureRecord] = field(default_factory=list)
    taxonomy: list[TaxonomyNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.by_slug: dict[str, ArchitectureRecord] = {r.slug: r for r in self.records}
        self.by_id: dict[str, ArchitectureRecord] = {r.id: r for r in self.records}
        self.taxonomy_by_slug: dict[str, TaxonomyNode] = {n.slug: n for n in self.taxonomy}

    # -- lookup --
    def get(self, slug_or_id: str) -> ArchitectureRecord:
        if slug_or_id in self.by_slug:
            return self.by_slug[slug_or_id]
        if slug_or_id in self.by_id:
            return self.by_id[slug_or_id]
        raise KeyError(f"unknown registry entry: {slug_or_id!r}")

    # -- rich filtering --
    def search(
        self,
        kind: str | None = None,
        modality: str | None = None,
        modalities: list[str] | None = None,
        architecture_paradigm: str | None = None,
        paradigms: list[str] | None = None,
        mechanisms: list[str] | None = None,
        properties: list[str] | None = None,
        tasks: list[str] | None = None,
        status: str | None = None,
        statuses: list[str] | None = None,
        query: str | None = None,
        taxonomy: str | None = None,
        limit: int = 100,
        has_examples: bool | None = None,
        has_graph: bool | None = None,
    ) -> list[ArchitectureRecord]:
        """Rich filter over registry records.

        Query matches name/slug/summary/description plus model_card,
        examples (title/description/code), model_graph node label/type/id,
        original_source display name/URL. Deterministic, no LLM, no network.
        has_examples/has_graph default None = ignore; True = only with, False = only without.
        """
        want_modalities = set(modalities or ([modality] if modality else []))
        want_paradigms = set(paradigms or ([architecture_paradigm] if architecture_paradigm else []))
        want_statuses = set(statuses or ([status] if status else []))
        q = (query or "").lower()
        out: list[ArchitectureRecord] = []
        for r in self.records:
            if kind and r.kind.value != kind and r.kind != kind:
                continue
            if want_modalities and not (want_modalities & set(r.modalities)):
                continue
            if want_paradigms and not (want_paradigms & set(r.architecture_paradigms)):
                continue
            if mechanisms and not (set(mechanisms) & set(r.mechanisms)):
                continue
            if properties and not (set(properties) & set(r.properties)):
                continue
            if tasks and not (set(tasks) & set(r.tasks)):
                continue
            if want_statuses and r.status.value not in want_statuses and r.status not in want_statuses:
                continue
            if taxonomy and taxonomy not in r.taxonomy_paths:
                continue
            if has_examples is not None:
                has_ex = bool(getattr(r, "examples", None))
                if has_ex != has_examples:
                    continue
            if has_graph is not None:
                graph = getattr(r, "model_graph", None)
                has_g = bool(graph is not None and getattr(graph, "nodes", None))
                if has_g != has_graph:
                    continue
            if q and not self._matches_query(r, q):
                continue
            out.append(r)
            if len(out) >= limit:
                break
        return out

    def _matches_query(self, r: ArchitectureRecord, q: str) -> bool:
        """Substring recall over core + rich fields (lowercased q)."""
        if q in f"{r.name} {r.slug} {r.summary} {r.description}".lower():
            return True
        card = getattr(r, "model_card", None)
        if card is not None and q in f"{card.long_description} {card.design_notes}".lower():
            return True
        for ex in getattr(r, "examples", []) or []:
            if q in f"{ex.title} {ex.description} {ex.code}".lower():
                return True
        graph = getattr(r, "model_graph", None)
        if graph is not None:
            for n in getattr(graph, "nodes", []) or []:
                if q in f"{n.id} {n.label} {n.type}".lower():
                    return True
        src = getattr(r, "original_source", None)
        if src is not None:
            if q in f"{src.display_name} {src.url}".lower():
                return True
        else:
            # fall back to canonical source URLs when no projection stored
            try:
                from topologies.metadata.schema import original_source_for

                proj = original_source_for(r)
                if q in f"{proj.display_name} {proj.url}".lower():
                    return True
            except Exception:  # noqa: BLE001, S110 - recall must not crash
                pass
        return False

    # -- relations --
    def related(self, slug: str) -> list[tuple[str, ArchitectureRecord]]:
        """Return (relation_type, record) for all outgoing relations of entry."""
        record = self.get(slug)
        out: list[tuple[str, ArchitectureRecord]] = []
        for rel in record.relations:
            target = self.by_slug.get(rel.target)
            if target is not None:
                out.append((rel.type.value if hasattr(rel.type, "value") else str(rel.type), target))
        for s in record.successors:
            if s in self.by_slug:
                out.append(("successor_of", self.by_slug[s]))
        for p in record.parents:
            if p in self.by_slug:
                out.append(("derived_from", self.by_slug[p]))
        return out

    # -- integrity --
    def duplicates(self) -> dict[str, list[str]]:
        slug_counts = Counter(r.slug for r in self.records)
        id_counts = Counter(r.id for r in self.records)
        return {
            "slug": [s for s, c in slug_counts.items() if c > 1],
            "id": [i for i, c in id_counts.items() if c > 1],
        }

    def dangling_relations(self) -> list[str]:
        problems: list[str] = []
        for r in self.records:
            for rel in r.relations:
                if rel.target not in self.by_slug:
                    problems.append(f"{r.slug}: relation {rel.type} -> unknown {rel.target}")
            for s in r.successors + r.parents + r.inspired_by + r.related:
                if s and s not in self.by_slug:
                    problems.append(f"{r.slug}: reference -> unknown {s}")
        return problems

    def stats(self) -> dict:
        kinds = Counter(r.kind.value if isinstance(r.kind, EntryKind) else str(r.kind) for r in self.records)
        statuses = Counter(r.status.value if hasattr(r.status, "value") else str(r.status) for r in self.records)
        return {"total": len(self.records), "by_kind": dict(kinds), "by_status": dict(statuses)}
