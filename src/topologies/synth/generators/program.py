"""Compatibility imports for function-backed core nodes.

The implementation lives in ``synth.core``; this module remains for callers of
the pre-split import path.
"""

from topologies.synth.core import Program, correlated_values, from_program

__all__ = ["Program", "correlated_values", "from_program"]
