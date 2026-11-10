import math
from numbers import Integral, Real

import torch
from torch import nn


class Mosaic(nn.Module):
    component_id = "mosaic"

    @staticmethod
    def _value_error(message):
        raise ValueError(message)

    def __init__(self, mosaic_probability=0.5, min_offset=0.2):
        super().__init__()
        if isinstance(mosaic_probability, bool) or not isinstance(mosaic_probability, Real):
            self._value_error("mosaic_probability must be a real number in [0, 1]")
        if not math.isfinite(float(mosaic_probability)) or not 0.0 <= mosaic_probability <= 1.0:
            raise ValueError("mosaic_probability must be a finite number in [0, 1]")
        if isinstance(min_offset, bool) or not isinstance(min_offset, Real):
            self._value_error("min_offset must be a real number in [0, 0.5]")
        if not math.isfinite(float(min_offset)) or not 0.0 <= min_offset <= 0.5:
            raise ValueError("min_offset must be a finite number in [0, 0.5]")
        self.mosaic_probability = float(mosaic_probability)
        self.min_offset = float(min_offset)

    def _images(self, images):
        if isinstance(images, torch.Tensor):
            if images.ndim != 4 or images.shape[0] != 4:
                raise ValueError("images must have shape [4, C, H, W]")
            samples = tuple(images.unbind(0))
            batch = images
            mode = "single"
        elif isinstance(images, (tuple, list)):
            if len(images) != 4 or any(not isinstance(image, torch.Tensor) for image in images):
                raise ValueError("images must contain four image tensors")
            samples = tuple(images)
            dimensions = {image.ndim for image in samples}
            if dimensions == {3}:
                batch = None
                mode = "single"
            elif dimensions == {4}:
                batch = samples
                mode = "batched"
            else:
                raise ValueError("each image must have shape [C, H, W] or [B, C, H, W]")
        else:
            self._value_error("images must be a tensor or a sequence of four tensors")
        first = samples[0]
        if any(any(size < 1 for size in image.shape) for image in samples):
            raise ValueError("images must have positive channel, height, and width sizes")
        if any(image.shape != first.shape for image in samples):
            raise ValueError("all four images must have the same shape")
        if any(image.dtype != first.dtype for image in samples):
            raise ValueError("all four images must have the same dtype")
        if any(image.device != first.device for image in samples):
            raise ValueError("all four images must be on the same device")
        if any(image.layout != torch.strided for image in samples):
            raise ValueError("images must use strided tensor layout")
        if any(not image.is_floating_point() for image in samples):
            raise ValueError("images must have a floating-point dtype")
        if batch is None:
            batch = torch.stack(samples)
        return samples, batch, mode

    def _boxes(self, boxes, images, batched=False, batch_size=None):
        if not isinstance(boxes, (tuple, list)) or len(boxes) != 4:
            raise ValueError("boxes must contain four tensors with shape [N, 5]")
        for index, box in enumerate(boxes):
            valid_shapes = (2, 3) if batched else (2,)
            if not isinstance(box, torch.Tensor) or box.ndim not in valid_shapes or box.shape[-1] != 5:
                raise ValueError(f"boxes[{index}] must have shape [N, 5]")
            if batched and box.shape[0] != batch_size:
                raise ValueError(f"boxes[{index}] must match the image batch size")
            if not box.is_floating_point() or box.dtype != images.dtype:
                raise ValueError(f"boxes[{index}] must use the image floating-point dtype")
            if box.device != images.device:
                raise ValueError(f"boxes[{index}] must be on the image device")
            if not bool(torch.isfinite(box).all()) or bool((box[..., 2:4] < 0).any()):
                raise ValueError(f"boxes[{index}] must contain finite values and nonnegative sizes")
        return tuple(boxes)

    def _shifts(self, shifts, images):
        if shifts is None:
            return ((0, 0, 0, 0),) * 4
        if isinstance(shifts, torch.Tensor):
            integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            if shifts.shape != (4, 4) or shifts.dtype not in integer_dtypes:
                raise ValueError("shifts must be an integer tensor with shape [4, 4]")
            values = shifts.detach().cpu().tolist()
        elif isinstance(shifts, (tuple, list)) and len(shifts) == 4:
            values = shifts
        else:
            raise ValueError("shifts must have shape [4, 4]")
        result = []
        for row in values:
            if not isinstance(row, (tuple, list)) or len(row) != 4:
                raise ValueError("each shift must contain left, right, top, and bottom offsets")
            if any(not isinstance(value, Integral) or isinstance(value, bool) for value in row):
                raise ValueError("shift offsets must be integers")
            if any(value < 0 for value in row):
                raise ValueError("shift offsets must be nonnegative")
            result.append(tuple(int(value) for value in row))
        return tuple(result)

    def _cut(self, cut, size, name):
        if not isinstance(cut, Integral) or isinstance(cut, bool) or not 0 <= cut <= size:
            raise ValueError(f"{name} must be an integer inside the image")
        return int(cut)

    def _sample_cut(self, size):
        lower = int(self.min_offset * size)
        upper = int((1.0 - self.min_offset) * size)
        return int(torch.randint(lower, upper + 1, ()).item())

    def _validate_shift_bounds(self, shifts, cut_x, cut_y, height, width):
        limits = (
            (0, width - cut_x, 0, height - cut_y),
            (cut_x, 0, 0, height - cut_y),
            (0, width - cut_x, cut_y, 0),
            (cut_x, 0, cut_y, 0),
        )
        for index, (shift, maximums) in enumerate(zip(shifts, limits)):
            if any(value > maximum for value, maximum in zip(shift, maximums)):
                raise ValueError(f"shifts[{index}] place a crop outside the image")

    def _mosaic_boxes(self, boxes, offsets, destinations, bounds, height, width):
        output = []
        for box, (source_x, source_y), (dest_x, dest_y), (left_bound, top_bound, right_bound, bottom_bound) in zip(
            boxes, offsets, destinations, bounds
        ):
            if box.shape[0] == 0:
                output.append(box.new_empty((0, 5)))
                continue
            center_x = box[:, 0] * width
            center_y = box[:, 1] * height
            half_width = box[:, 2] * width / 2.0
            half_height = box[:, 3] * height / 2.0
            x1 = (center_x - half_width - source_x + dest_x).clamp(left_bound, right_bound)
            y1 = (center_y - half_height - source_y + dest_y).clamp(top_bound, bottom_bound)
            x2 = (center_x + half_width - source_x + dest_x).clamp(left_bound, right_bound)
            y2 = (center_y + half_height - source_y + dest_y).clamp(top_bound, bottom_bound)
            retained = ((x2 - x1) > 1.0) & ((y2 - y1) > 1.0)
            if bool(retained.any()):
                output.append(
                    torch.stack(
                        (
                            ((x1 + x2) / 2.0 / width)[retained],
                            ((y1 + y2) / 2.0 / height)[retained],
                            ((x2 - x1) / width)[retained],
                            ((y2 - y1) / height)[retained],
                            box[:, 4][retained],
                        ),
                        dim=1,
                    )
                )
            else:
                output.append(box.new_empty((0, 5)))
        return tuple(output)

    def _mosaic_one(self, batch, boxes, shifts, apply, cut_x, cut_y):
        height, width = batch.shape[-2:]
        if not apply:
            return batch[0], boxes[0]
        if cut_x is None:
            cut_x = self._sample_cut(width)
            cut_y = self._sample_cut(height)
        self._validate_shift_bounds(shifts, cut_x, cut_y, height, width)
        right0, bottom0 = shifts[0][1], shifts[0][3]
        left1, bottom1 = shifts[1][0], shifts[1][3]
        right2, top2 = shifts[2][1], shifts[2][2]
        left3, top3 = shifts[3][0], shifts[3][2]
        top_left = batch[0, :, height - cut_y - bottom0:height - bottom0, width - cut_x - right0:width - right0]
        top_right = batch[1, :, height - cut_y - bottom1:height - bottom1, left1:left1 + width - cut_x]
        bottom_left = batch[2, :, top2:top2 + height - cut_y, width - cut_x - right2:width - right2]
        bottom_right = batch[3, :, top3:top3 + height - cut_y, left3:left3 + width - cut_x]
        mosaic = torch.cat((torch.cat((top_left, top_right), dim=2), torch.cat((bottom_left, bottom_right), dim=2)), dim=1)
        offsets = (
            (width - cut_x - right0, height - cut_y - bottom0),
            (left1, height - cut_y - bottom1),
            (width - cut_x - right2, top2),
            (left3, top3),
        )
        destinations = ((0, 0), (cut_x, 0), (0, cut_y), (cut_x, cut_y))
        bounds = (
            (0, 0, cut_x, cut_y),
            (cut_x, 0, width, cut_y),
            (0, cut_y, cut_x, height),
            (cut_x, cut_y, width, height),
        )
        mosaic_boxes = self._mosaic_boxes(boxes, offsets, destinations, bounds, height, width)
        return mosaic, torch.cat(mosaic_boxes, dim=0)

    def forward(self, images, boxes, shifts=None, apply=None, cut_x=None, cut_y=None, return_boxes=True):
        samples, batch, mode = self._images(images)
        reference = batch[0] if mode == "batched" else batch
        batch_size = reference.shape[0] if mode == "batched" else None
        boxes = self._boxes(boxes, reference, batched=mode == "batched", batch_size=batch_size)
        shifts = self._shifts(shifts, reference)
        height, width = reference.shape[-2:]
        if apply is not None and not isinstance(apply, bool):
            raise ValueError("apply must be True, False, or None")
        if not isinstance(return_boxes, bool):
            self._value_error("return_boxes must be a bool")
        if (cut_x is None) != (cut_y is None):
            raise ValueError("cut_x and cut_y must be provided together")
        if cut_x is not None:
            cut_x = self._cut(cut_x, width, "cut_x")
            cut_y = self._cut(cut_y, height, "cut_y")
        if mode == "single":
            should_apply = apply
            if should_apply is None:
                should_apply = bool(torch.rand(()).item() < self.mosaic_probability)
            result = self._mosaic_one(batch, boxes, shifts, should_apply, cut_x, cut_y)
            return result if return_boxes else result[0]
        mosaic_images = []
        mosaic_boxes = []
        for index in range(batch_size):
            image_batch = torch.stack([image[index] for image in samples])
            box_batch = tuple(
                box[index].unsqueeze(0) if box.ndim == 2 else box[index]
                for box in boxes
            )
            should_apply = apply
            if should_apply is None:
                should_apply = bool(torch.rand(()).item() < self.mosaic_probability)
            image_output, boxes_output = self._mosaic_one(
                image_batch, box_batch, shifts, should_apply, cut_x, cut_y
            )
            mosaic_images.append(image_output)
            mosaic_boxes.append(boxes_output)
        output_images = torch.stack(mosaic_images)
        if return_boxes:
            return output_images, tuple(mosaic_boxes)
        return output_images
