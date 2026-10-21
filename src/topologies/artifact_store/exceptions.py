"""Artifact-store exceptions (single integration point owns all store HTTP errors)."""


class ArtifactStoreError(Exception):
    pass


class ArtifactAuthError(ArtifactStoreError):
    pass


class ArtifactNotFoundError(ArtifactStoreError):
    pass


class ArtifactDigestError(ArtifactStoreError):
    pass


class ArtifactUnavailableError(ArtifactStoreError):
    """Store unreachable (network down). Callers should degrade gracefully."""
