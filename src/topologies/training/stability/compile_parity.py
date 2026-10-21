"""Helpers for checking eager and compiled execution parity."""


def compile_parity(eager, compiled, tol=1e-5):
    """Return whether eager and compiled values are within absolute tolerance."""
    return abs(eager - compiled) <= tol
