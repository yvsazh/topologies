"""Skip training updates when the loss is NaN."""

import math


def nan_skip(loss):
    """Return whether ``loss`` is NaN."""
    return math.isnan(loss)
