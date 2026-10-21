"""Typed metadata schemas (pydantic v2). Single source of truth for registry records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from topologies.enums import (
    EntryKind,
    RelationType,
    TaxonomyDimension,
    TaxonomyNodeStatus,
    VerificationStatus,
)

SCHEMA_VERSION = "1.0.0"


class PaperRef(BaseModel):
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    url: str = ""
    arxiv_id: str = ""
    doi: str = ""
    published_at: str = ""  # ISO date, may be empty for demos


class SourceRepoRef(BaseModel):
    url: str = ""
    commit: str = ""
    files_consulted: list[str] = Field(default_factory=list)
    license: str = ""


class VerificationInfo(BaseModel):
    status: VerificationStatus = VerificationStatus.DISCOVERED
    tests: list[str] = Field(default_factory=list)
    source_match: str = "unknown"  # e.g. matches-official / unknown / mismatch
    reviewed: bool = False
    confidence: float = 0.0  # 0..1


class ProvenanceInfo(BaseModel):
    paper_url: str = ""
    paper_id: str = ""
    repository_url: str = ""
    repository_commit: str = ""
    files_consulted: list[str] = Field(default_factory=list)
    researched_at: str = ""
    researcher: str = ""  # e.g. "codex:2026-09-30" or "human"
    generator: str = ""  # model/tool that generated implementation
    validation_evidence: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    target: str  # slug of the other entry
    type: RelationType


class ArtifactPin(BaseModel):
    """Pinned library file produced by materialization from the artifact store."""

    path: str  # in-package relative path, e.g. models/demo_decoder.py
    namespace: str = "library"
    name: str  # artifact name in the store, e.g. demo-decoder
    version: int = 1
    sha256: str  # hex digest of the pinned file bytes

    @field_validator("sha256")
    @classmethod
    def _sha_valid(cls, v: str) -> str:
        import re

        if not re.fullmatch(r"[0-9a-f]{64}", v.strip().lower()):
            raise ValueError(f"invalid sha256: {v!r}")
        return v.strip().lower()

    @field_validator("path")
    @classmethod
    def _path_valid(cls, v: str) -> str:
        p = v.strip().replace("\\", "/")
        if not p or p.startswith("/") or ".." in p.split("/"):
            raise ValueError(f"invalid artifact path: {v!r}")
        return p


class ModelCard(BaseModel):
    """Long-form design notes (additive optional, schema_version stays 1.0.0)."""

    long_description: str = ""
    design_notes: str = ""

    model_config = {"extra": "allow"}


class OriginalSource(BaseModel):
    """Display projection over existing paper/official_implementation/repository/
    provenance fields. Not a second truth: use original_source_for() mapper to
    surface existing data; only display_name is new."""

    display_name: str = ""
    url: str = ""
    commit: str = ""
    files_consulted: list[str] = Field(default_factory=list)
    license: str = ""

    model_config = {"extra": "allow"}


class GraphParameter(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    shape: list[int] = Field(default_factory=list)
    dtype: str = ""
    requires_grad: bool = False
    params: int = Field(default=0, ge=0)

    model_config = {"extra": "forbid"}


class GraphNode(BaseModel):
    id: str  # e.g. "blocks.0"
    label: str = ""  # e.g. "Linear"
    type: str = ""  # e.g. "torch.nn.Linear"
    parent: str = ""  # parent node id, "" for roots
    params: int = Field(default=0, ge=0)
    output_shape: list[int | None] | None = None  # null = dynamic/batch; small, optional
    parameters: list[GraphParameter] | None = None

    model_config = {"extra": "allow"}


class ModelGraph(BaseModel):
    generated_at: str = ""
    generator: str = "orchestrator-introspection/1"
    total_params: int = Field(default=0, ge=0)
    nodes: list[GraphNode] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class ModelStats(BaseModel):
    total_params: int = Field(default=0, ge=0)
    trainable_params: int = Field(default=0, ge=0)
    depth: int = Field(default=0, ge=0)
    input_spec: str = "unknown"
    output_spec: str = "unknown"

    model_config = {"extra": "allow"}


class UsageExample(BaseModel):
    title: str
    description: str = ""
    code: str

    model_config = {"extra": "allow"}

    @field_validator("code")
    @classmethod
    def _code_compiles(cls, v: str) -> str:
        try:
            compile(v, "<example>", "exec")
        except SyntaxError as e:
            raise ValueError(f"example code does not compile: {e}")
        return v


class TrainingReport(BaseModel):
    """Illustrative smoke-training result (trusted offline CPU run, never a benchmark).

    Stored on architecture entries. The website only renders these floats;
    it never computes. `example_title` must match an entry in `examples[]`
    (the training script doubles as a usage example).
    """

    example_title: str
    config: dict[str, Any] = Field(default_factory=dict)
    train_loss: list[float] = Field(default_factory=list)
    val_loss: float | None = None
    steps: int = Field(default=0, ge=0)
    seed: int = 0
    torch_version: str = ""
    duration_s: float = Field(default=0.0, ge=0.0)
    passed: bool = False
    notes: str = ""

    model_config = {"extra": "allow"}

    @field_validator("train_loss")
    @classmethod
    def _loss_finite(cls, v: list[float]) -> list[float]:
        import math as _math

        for x in v:
            if not isinstance(x, (int, float)) or not _math.isfinite(float(x)):
                raise ValueError("train_loss must contain only finite floats")
        return [float(x) for x in v]

    @field_validator("val_loss")
    @classmethod
    def _val_finite(cls, v: float | None) -> float | None:
        import math as _math

        if v is None:
            return None
        if not isinstance(v, (int, float)) or not _math.isfinite(float(v)):
            raise ValueError("val_loss must be a finite float or null")
        return float(v)


class TrainingMeta(BaseModel):
    """Optional training metadata (additive, schema_version stays 1.0.0).

    Example YAML::

        training:
          recipe: causal-lm
          loss: cross-entropy
          defaults:
            learning_rate: 3e-4
    """

    recipe: str = ""
    loss: str | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class ComponentCheckItem(BaseModel):
    name: str
    passed: bool = False
    detail: str = ""

    model_config = {"extra": "allow"}


class ComponentCheck(BaseModel):
    """Standardized component battery (trusted offline CPU run).

    Standard check names: forward-shape, gradient-flow, determinism, param-count.
    """

    checks: list[ComponentCheckItem] = Field(default_factory=list)
    train_loss: list[float] = Field(default_factory=list)
    seed: int = 0
    torch_version: str = ""
    duration_s: float = Field(default=0.0, ge=0.0)
    passed: bool = False

    model_config = {"extra": "allow"}

    @field_validator("train_loss")
    @classmethod
    def _loss_finite_and_bounded(cls, v: list[float]) -> list[float]:
        import math as _math

        if len(v) > 50:
            raise ValueError("train_loss must contain at most 50 values")
        for x in v:
            if not isinstance(x, (int, float)) or not _math.isfinite(float(x)):
                raise ValueError("train_loss must contain only finite floats")
        return [float(x) for x in v]


COMPONENT_CHECK_NAMES = ("forward-shape", "gradient-flow", "determinism", "param-count")


class ArchitectureRecord(BaseModel):
    """Canonical record for one architecture or component."""

    schema_version: str = SCHEMA_VERSION
    id: str
    name: str
    slug: str
    kind: EntryKind
    status: VerificationStatus = VerificationStatus.DISCOVERED

    summary: str = ""
    description: str = ""

    module_path: str = ""  # e.g. topologies.layers.demo_attention
    class_name: str = ""  # e.g. DemoAttention

    architecture_family: str = ""
    architecture_paradigms: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)

    mechanisms: list[str] = Field(default_factory=list)  # e.g. attention, normalization
    properties: list[str] = Field(default_factory=list)  # e.g. causal, decoder-only

    parents: list[str] = Field(default_factory=list)
    inspired_by: list[str] = Field(default_factory=list)
    successors: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    paper: PaperRef = Field(default_factory=PaperRef)
    official_implementation: SourceRepoRef = Field(default_factory=SourceRepoRef)
    repository: str = ""
    license: str = ""

    introduced_at: str = ""
    verification: VerificationInfo = Field(default_factory=VerificationInfo)
    source_provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo)

    tags: list[str] = Field(default_factory=list)
    taxonomy_paths: list[str] = Field(default_factory=list)  # taxonomy node slugs

    # Artifact pins: library implementation files materialized from the store.
    # Additive optional field (schema_version stays 1.0.0); see registry/schemas/README.md.
    artifacts: list[ArtifactPin] = Field(default_factory=list)

    # Rich component detail (additive optional, schema_version stays 1.0.0).
    # Code bytes stay in the artifact store; these fields are display metadata
    # captured at creation time (pipeline), cached in Django DB, rendered + indexed.
    model_card: ModelCard | None = None
    original_source: OriginalSource | None = None
    model_graph: ModelGraph | None = None
    model_stats: ModelStats | None = None
    examples: list[UsageExample] = Field(default_factory=list)

    # Smoke evidence (additive optional, schema_version stays 1.0.0).
    # Architectures carry training_report (mini CPU loop on synthetic data);
    # components carry component_check (standardized forward/grad/determinism
    # battery). The website only renders stored data, never computes.
    training_report: TrainingReport | None = None
    component_check: ComponentCheck | None = None

    # Universal training-engine metadata (additive optional, schema_version stays 1.0.0).
    # Declares the default Recipe for `topologies.training` without embedding any
    # training logic in model classes.
    training: TrainingMeta | None = None

    created_at: str = ""
    updated_at: str = ""

    model_config = {"extra": "allow"}  # extensible without breaking migrations

    @field_validator("slug")
    @classmethod
    def _slug_valid(cls, v: str) -> str:
        import re

        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", v):
            raise ValueError(f"invalid slug: {v!r} (use lowercase-kebab-case)")
        return v


class TaxonomyNode(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    name: str
    slug: str
    dimension: TaxonomyDimension
    parent: str = ""  # parent node slug, empty for roots
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    status: TaxonomyNodeStatus = TaxonomyNodeStatus.CANONICAL
    created_by: str = "human"
    source: str = ""

    model_config = {"extra": "allow"}


class QualityGateResult(BaseModel):
    decision: str
    reasons: list[str] = Field(default_factory=list)
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z")
    evaluator: str = "quality-gate"


class ResearchCandidate(BaseModel):
    """A discovered paper/repo under investigation."""

    schema_version: str = SCHEMA_VERSION
    id: str
    title: str = ""
    arxiv_id: str = ""
    doi: str = ""
    paper_url: str = ""
    repository_url: str = ""
    architecture_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    decision: str = "undecided"  # CandidateDecision value or 'undecided'
    decision_reason: str = ""
    status: str = "discovered"
    discovered_at: str = ""
    discovered_by: str = ""
    rejection_reason: str = ""
    notes: str = ""

    model_config = {"extra": "allow"}


def original_source_for(record: ArchitectureRecord) -> OriginalSource:
    """Display projection over existing source fields (not a second truth).

    Reuses paper / official_implementation / repository / provenance where they
    already exist; only display_name is new. If record.original_source is set,
    missing sub-fields fall back to the canonical sources below.
    """
    existing = record.original_source or OriginalSource()
    off = record.official_implementation
    prov = record.source_provenance
    url = existing.url or off.url or record.repository or prov.repository_url
    commit = existing.commit or off.commit or prov.repository_commit
    files = list(existing.files_consulted or off.files_consulted or prov.files_consulted)
    lic = existing.license or off.license or record.license
    display = existing.display_name
    if not display:
        # Derive a human label from the URL when no explicit display_name was stored.
        display = url or "Original source"
    return OriginalSource(
        display_name=display, url=url, commit=commit,
        files_consulted=files, license=lic)


def leaf_param_sum(graph: ModelGraph) -> int:
    """Legacy leaf-only parameter sum; excludes parameters owned by parents."""
    parents = {n.parent for n in graph.nodes if n.parent}
    return sum(n.params for n in graph.nodes if n.id not in parents)


def graph_param_sum(graph: ModelGraph) -> int:
    """Sum direct parameter counts across nodes, including nonleaf/root modules."""
    return sum(node.params for node in graph.nodes)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def extra_fields(record: BaseModel) -> dict[str, Any]:
    """Return forward-compatible extra fields stored beyond the schema."""
    declared = set(type(record).model_fields.keys())
    return {k: v for k, v in record.model_dump().items() if k not in declared}


OptionalRecord = ArchitectureRecord | None
