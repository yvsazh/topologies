"""Activation-offload decision based on available memory."""


def activation_offload(bytes_needed, bytes_free):
    """Return whether activation memory should be offloaded."""
    return bytes_needed > bytes_free
