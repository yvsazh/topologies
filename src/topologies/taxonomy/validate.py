"""Taxonomy helpers: dimensions, tree building, validation (dupes, cycles, parents)."""

from __future__ import annotations

from dataclasses import dataclass, field

from topologies.metadata.schema import TaxonomyNode


@dataclass
class TaxonomyReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_taxonomy(nodes: list[TaxonomyNode]) -> TaxonomyReport:
    report = TaxonomyReport()
    seen_slugs: dict[str, TaxonomyNode] = {}
    seen_names: dict[str, str] = {}
    alias_owner: dict[str, str] = {}
    for n in nodes:
        if n.slug in seen_slugs:
            report.errors.append(f"duplicate taxonomy slug: {n.slug}")
            report.ok = False
        else:
            seen_slugs[n.slug] = n
        key = n.name.strip().lower()
        if key in seen_names and seen_names[key] != n.slug:
            report.warnings.append(f"near-duplicate taxonomy name: {n.name!r} ({n.slug} vs {seen_names[key]})")
        seen_names.setdefault(key, n.slug)
        for alias in n.aliases:
            a = alias.strip().lower()
            if a in alias_owner and alias_owner[a] != n.slug:
                report.warnings.append(f"alias {alias!r} shared by {alias_owner[a]} and {n.slug}")
            alias_owner.setdefault(a, n.slug)
            if a == n.slug or a == key:
                report.warnings.append(f"node {n.slug}: alias repeats own name/slug")
    for n in nodes:
        if n.parent:
            if n.parent not in seen_slugs:
                report.errors.append(f"{n.slug}: unknown parent {n.parent!r}")
                report.ok = False
            elif seen_slugs[n.parent].dimension != n.dimension:
                report.errors.append(f"{n.slug}: parent {n.parent} has different dimension")
                report.ok = False
    # cycle detection per dimension
    parent_of = {n.slug: n.parent for n in nodes if n.parent}
    for slug in parent_of:
        visited: set[str] = set()
        cur = slug
        while parent_of.get(cur):
            if cur in visited:
                report.errors.append(f"taxonomy cycle detected at {cur}")
                report.ok = False
                break
            visited.add(cur)
            cur = parent_of[cur]
    # naming consistency: slugs must be kebab-case
    import re

    for n in nodes:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", n.slug):
            report.errors.append(f"invalid taxonomy slug: {n.slug!r}")
            report.ok = False
    return report


def children_of(nodes: list[TaxonomyNode], parent_slug: str) -> list[TaxonomyNode]:
    return [n for n in nodes if n.parent == parent_slug]


def by_dimension(nodes: list[TaxonomyNode], dimension: str) -> list[TaxonomyNode]:
    return [n for n in nodes if n.dimension.value == dimension or n.dimension == dimension]  # type: ignore[comparison-overlap]
