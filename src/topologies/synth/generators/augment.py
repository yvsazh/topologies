"""Compatibility imports for text augmentation pipeline operations.

Implementation lives in ``synth.ops.augment``; use ``synth.ops.augment`` for new code.
"""

from topologies.synth.ops import augment as _augment
from topologies.synth.ops._callable import make_callable

__all__ = list(_augment.__all__)
globals().update({name: getattr(_augment, name) for name in __all__})

make_callable(__name__, "augment_op")
