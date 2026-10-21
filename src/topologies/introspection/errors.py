"""Exceptions raised by ``topologies.introspection``.

Every error message names the concrete fix (a selector, a declaration, a policy
knob) instead of only describing the failure.
"""
from __future__ import annotations


class IntrospectionError(RuntimeError):
    """Base class for introspection failures (misuse, nesting, unsupported calls)."""


class SemanticsRequired(IntrospectionError):
    """A technique needs semantics (axes, roles) that were neither declared nor inferable."""


class CaptureBudgetExceeded(IntrospectionError):
    """Kept tensors would exceed ``CapturePolicy.max_bytes``; raised before allocation."""


class UnsupportedModel(IntrospectionError):
    """The source cannot be introspected (not an ``nn.Module``, opaque script module, ...)."""


class ObjectiveRequired(IntrospectionError):
    """A gradient-based technique has no scalar objective and none can be resolved."""


__all__ = [
    "CaptureBudgetExceeded",
    "IntrospectionError",
    "ObjectiveRequired",
    "SemanticsRequired",
    "UnsupportedModel",
]
