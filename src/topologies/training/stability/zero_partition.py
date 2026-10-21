"""Simple parameter-count partitioning for ZeRO-style sharding."""


def zero_partition(n_params, world_size):
    """Return the integer parameter count assigned to each rank."""
    return n_params // world_size
