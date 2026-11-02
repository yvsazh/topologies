from numbers import Integral

import torch
from torch import nn


class CutPaste(nn.Module):
    component_id = "cutpaste"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("CutPaste takes no constructor arguments")
        super().__init__()

    def _rectangle(self, rectangle, name, image_height, image_width):
        if not isinstance(rectangle, (tuple, list)) or len(rectangle) != 4:
            raise ValueError(f"{name} must contain top, left, height, and width")
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in rectangle
        ):
            raise ValueError(f"{name} values must be integers")
        top, left, height, width = (int(value) for value in rectangle)
        if top < 0 or left < 0:
            raise ValueError(f"{name} coordinates must be nonnegative")
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} height and width must be positive")
        if top + height > image_height or left + width > image_width:
            raise ValueError(f"{name} must fit within the image")
        return top, left, height, width

    def forward(self, image, source, dest):
        return self.paste(image, source, dest)

    def paste(self, image, source, dest):
        if not torch.is_tensor(image):
            raise ValueError("image must be a torch.Tensor")
        if image.ndim < 2:
            raise ValueError("image must have at least two dimensions")
        if image.layout != torch.strided:
            raise ValueError("image must use strided tensor layout")
        image_height, image_width = image.shape[-2:]
        source_top, source_left, source_height, source_width = self._rectangle(
            source, "source", image_height, image_width
        )
        dest_top, dest_left, dest_height, dest_width = self._rectangle(
            dest, "dest", image_height, image_width
        )
        if (source_height, source_width) != (dest_height, dest_width):
            raise ValueError("source and dest must have equal height and width")
        output = image.clone()
        output[..., dest_top:dest_top + dest_height, dest_left:dest_left + dest_width] = (
            image[
                ...,
                source_top:source_top + source_height,
                source_left:source_left + source_width,
            ]
        )
        return output
