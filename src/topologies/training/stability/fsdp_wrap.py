"""Parameter-count based decision helper for FSDP wrapping."""


def fsdp_wrap(min_params, n_params):
    """Return whether a module meets the minimum parameter count to wrap."""
    return n_params >= min_params
