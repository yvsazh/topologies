"""Shared artifact-store client: the ONLY module that talks HTTP to the store.

Everyone else (pipeline, materialization, website sync, seed) imports from here.
Supports dry_run=True: logs intended calls, stores blobs in memory, zero network.
"""

from .client import (
    ArtifactStoreClient,
    ArtifactStoreConfig,
    build_client_from_env,
    resolve_store_url,
)
from .exceptions import (
    ArtifactAuthError,
    ArtifactDigestError,
    ArtifactNotFoundError,
    ArtifactStoreError,
    ArtifactUnavailableError,
)

__all__ = [
    "ArtifactAuthError",
    "ArtifactDigestError",
    "ArtifactNotFoundError",
    "ArtifactStoreClient",
    "ArtifactStoreConfig",
    "ArtifactStoreError",
    "ArtifactUnavailableError",
    "build_client_from_env",
    "resolve_store_url",
]
