"""Compatibility import; metadata consumers should load training_schema directly."""
from topologies.training_schema import ENTRY_ID_RE, KINDS, SHA256_RE, validate_catalog

__all__ = ["ENTRY_ID_RE", "KINDS", "SHA256_RE", "validate_catalog"]
