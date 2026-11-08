import math

import torch
from torch import nn


class RISE(nn.Module):
    component_id = "rise"

    def __init__(self, mask_size=(7, 7), keep_probability=0.5, n_masks=4000, seed=None):
        super().__init__()
        if not isinstance(mask_size, (tuple, list)) or len(mask_size) != 2 or any(
            type(size) is not int or size < 1 for size in mask_size
        ):
            raise ValueError("mask_size must contain two positive integers")
        if (
            type(keep_probability) not in (int, float)
            or not math.isfinite(keep_probability)
            or not 0 < keep_probability <= 1
        ):
            raise ValueError("keep_probability must be a finite number in (0, 1]")
        if type(n_masks) is not int or n_masks < 1:
            raise ValueError("n_masks must be a positive integer")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        self.mask_size = tuple(mask_size)
        self.keep_probability = float(keep_probability)
        self.n_masks = n_masks
        self.seed = seed

    def generate_masks(self, height, width, *, device=None, dtype=None):
        if type(height) is not int or type(width) is not int or height < 1 or width < 1:
            raise ValueError("height and width must be positive integers")
        mask_height, mask_width = self.mask_size
        if mask_height >= height or mask_width >= width:
            raise ValueError("mask_size dimensions must be smaller than the input dimensions")
        cell_height = height // mask_height
        cell_width = width // mask_width
        upsampled_height = (mask_height + 1) * cell_height
        upsampled_width = (mask_width + 1) * cell_width
        if upsampled_height < height or upsampled_width < width:
            raise ValueError("mask_size does not leave enough upsampled area for a valid crop")
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
        coarse = torch.rand(
            (self.n_masks, 1, mask_height, mask_width), generator=generator
        ) < self.keep_probability
        resized = nn.functional.interpolate(
            coarse.to(torch.float32),
            size=(upsampled_height, upsampled_width),
            mode="bilinear",
            align_corners=False,
        )
        max_top = min(cell_height, upsampled_height - height)
        max_left = min(cell_width, upsampled_width - width)
        tops = torch.randint(0, max_top + 1, (self.n_masks,), generator=generator).tolist()
        lefts = torch.randint(0, max_left + 1, (self.n_masks,), generator=generator).tolist()
        masks = torch.cat(
            [
                resized[index:index + 1, :, top:top + height, left:left + width]
                for index, (top, left) in enumerate(zip(tops, lefts))
            ],
            dim=0,
        )
        return masks.to(device=device, dtype=dtype if dtype is not None else torch.float32)

    def _estimate(self, scores, masks):
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
            raise ValueError("scores must have shape (batch, n_masks)")
        if not scores.is_floating_point():
            raise ValueError("scores must have a floating-point dtype")
        if not isinstance(masks, torch.Tensor) or masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError("masks must have shape (n_masks, 1, height, width)")
        if scores.shape[1] != self.n_masks or masks.shape[0] != self.n_masks:
            raise ValueError("scores and masks must match the configured n_masks")
        weighted_masks = masks[:, 0].to(device=scores.device, dtype=scores.dtype)
        return torch.einsum("bn,nhw->bhw", scores, weighted_masks) / (
            self.n_masks * self.keep_probability
        )

    def forward(self, images, predictor=None):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must be a tensor with shape (batch, channels, height, width)")
        if not images.is_floating_point():
            raise ValueError("images must have a floating-point dtype")
        if any(dimension < 1 for dimension in images.shape):
            raise ValueError("images must have nonempty batch, channel, and spatial dimensions")
        if predictor is not None and not callable(predictor):
            raise ValueError("predictor must be callable or None")
        batch_size, channels, height, width = images.shape
        masks = self.generate_masks(height, width, device=images.device, dtype=images.dtype)
        masked_images = (images[:, None] * masks[None]).reshape(
            batch_size * self.n_masks, channels, height, width
        )
        if predictor is None:
            scores = masked_images.mean(dim=(1, 2, 3))
        else:
            scores = predictor(masked_images)
        if not isinstance(scores, torch.Tensor):
            raise TypeError("predictor must return one tensor score per masked image")
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        elif scores.ndim == 0 and batch_size * self.n_masks == 1:
            scores = scores.reshape(1)
        if scores.ndim != 1 or scores.shape[0] != batch_size * self.n_masks:
            raise ValueError("predictor must return one scalar score per masked image")
        scores = scores.reshape(batch_size, self.n_masks)
        return self._estimate(scores, masks)
