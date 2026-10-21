"""Shared enumerations: kinds, verification states, relations, taxonomy dimensions."""

from enum import Enum


class EntryKind(str, Enum):
    ARCHITECTURE = "architecture"
    COMPONENT = "component"


class VerificationStatus(str, Enum):
    DISCOVERED = "discovered"
    RESEARCHING = "researching"
    IMPLEMENTED = "implemented"
    VALIDATION_PENDING = "validation_pending"
    VERIFIED = "verified"
    EXPERIMENTAL = "experimental"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    DEPRECATED = "deprecated"


PUBLIC_STATUSES = {
    VerificationStatus.IMPLEMENTED,
    VerificationStatus.VALIDATION_PENDING,
    VerificationStatus.VERIFIED,
    VerificationStatus.EXPERIMENTAL,
}


class RelationType(str, Enum):
    DERIVED_FROM = "derived_from"
    INSPIRED_BY = "inspired_by"
    VARIANT_OF = "variant_of"
    USES = "uses"
    REPLACES = "replaces"
    EXTENDS = "extends"
    RELATED_TO = "related_to"
    SUCCESSOR_OF = "successor_of"
    PREDECESSOR_OF = "predecessor_of"


class TaxonomyDimension(str, Enum):
    KIND = "kind"
    MODALITY = "modality"
    PARADIGM = "paradigm"
    FAMILY = "family"
    PROPERTY = "property"


class TaxonomyNodeStatus(str, Enum):
    CANONICAL = "canonical"
    PROPOSED = "proposed"
    DEPRECATED = "deprecated"
    MERGED = "merged"


class CandidateDecision(str, Enum):
    NEW_ARCHITECTURE = "new_architecture"
    NEW_COMPONENT = "new_component"
    VARIANT = "variant"
    UPDATE_EXISTING = "update_existing"
    IRRELEVANT = "irrelevant"
    DUPLICATE = "duplicate"


class GateDecisionType(str, Enum):
    ACCEPT = "accept"
    EXPERIMENTAL = "experimental"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    NEEDS_REVIEW = "needs_review"


class RunStageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
