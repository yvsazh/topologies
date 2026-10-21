import torch
from torch import nn
import torch.nn.functional as F

class SkipUpsampleBlock(nn.Module):
    component_id = 'skip-upsample-block'

    def __init__(self, in_channels, skip_channels, out_channels, mode='bilinear'):
        super().__init__()
        if min(in_channels, skip_channels, out_channels) <= 0 or mode not in ('nearest', 'bilinear', 'bicubic'):
            raise ValueError('invalid channels or interpolation mode')
        self.mode = mode
        self.skip_channels = skip_channels
        self.project = nn.Sequential(nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        if x.ndim != 4 or skip.ndim != 4 or x.shape[0] != skip.shape[0] or (skip.shape[1] != self.skip_channels):
            raise ValueError('expected compatible NCHW inputs and declared skip channels')
        x = F.interpolate(x, size=skip.shape[-2:], mode=self.mode, **{'align_corners': False} if self.mode != 'nearest' else {})
        return self.project(torch.cat((x, skip), dim=1))
