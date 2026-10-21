

import torch
from torch import nn


class SpecAugment(nn.Module):
    component_id = "specaugment"
    """Zero random frequency/time bands in 3-D or 4-D mel tensors.

    ``freq_mask_width`` and ``time_mask_width`` are inclusive maximum widths.
    ``adaptive=True`` scales the time width cap by ``adaptive_ratio``.
    """

    def __init__(
        self,
        num_freq_masks=2,
        freq_mask_width=27,
        num_time_masks=2,
        time_mask_width=100,
        adaptive=False,
        adaptive_ratio=0.1,
    ):
        super().__init__()
        if min(num_freq_masks, freq_mask_width, num_time_masks, time_mask_width) < 0:
            raise ValueError("mask counts and maximum widths must be non-negative")
        if not 0.0 <= adaptive_ratio <= 1.0:
            raise ValueError("adaptive_ratio must be between zero and one")
        self.num_freq_masks = num_freq_masks
        self.freq_mask_width = freq_mask_width
        self.num_time_masks = num_time_masks
        self.time_mask_width = time_mask_width
        self.adaptive = adaptive
        self.adaptive_ratio = adaptive_ratio

    @staticmethod
    def _mask_axis(result, axis, count, max_width, generator):
        size = result.shape[axis]
        for batch_index in range(result.shape[0]):
            for _ in range(count):
                width = int(torch.randint(max_width + 1, (), generator=generator).item())
                if width == 0:
                    continue
                start = int(torch.randint(size - width + 1, (), generator=generator).item())
                index = [slice(None)] * result.ndim
                index[0] = batch_index
                index[axis] = slice(start, start + width)
                result[tuple(index)] = 0
        return result

    def forward(self, spectrogram, generator=None):
        if spectrogram.ndim not in (3, 4):
            raise ValueError("expected spectrogram shape (batch, freq, time) or (batch, channel, freq, time)")
        if self.num_freq_masks == 0 and self.num_time_masks == 0:
            return spectrogram
        freq_axis, time_axis = spectrogram.ndim - 2, spectrogram.ndim - 1
        freq_cap = min(spectrogram.shape[freq_axis], self.freq_mask_width)
        time_cap = self.time_mask_width
        if self.adaptive:
            time_cap = min(spectrogram.shape[time_axis], int(spectrogram.shape[time_axis] * self.adaptive_ratio))
        else:
            time_cap = min(spectrogram.shape[time_axis], time_cap)
        result = spectrogram.clone()
        result = self._mask_axis(result, freq_axis, self.num_freq_masks, freq_cap, generator)
        return self._mask_axis(result, time_axis, self.num_time_masks, time_cap, generator)
