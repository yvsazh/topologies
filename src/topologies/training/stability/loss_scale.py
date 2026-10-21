"""Static loss scaling helper."""


def loss_scale(loss, scale=8.0):
    """Return the loss multiplied by a fixed scaling factor."""
    return loss * scale
