from typing import NamedTuple
import torch
from torch import nn
import torch.nn.functional as F

class RetinaNetOutput(NamedTuple):
    class_logits: tuple
    box_deltas: tuple
    feature_grids: tuple
    strides: tuple
    anchor_sizes: tuple

class _Block(nn.Module):

    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class RetinaNet(nn.Module):
    architecture_id = 'retinanet'

    def __init__(self, in_channels=3, num_classes=2, feature_width=64, anchors_per_location=9, dropout=0.0, anchor_scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)), anchor_ratios=(0.5, 1.0, 2.0), anchor_sizes=(32.0, 64.0, 128.0)):
        super().__init__()
        if min(in_channels, num_classes, feature_width, anchors_per_location) <= 0:
            raise ValueError('dimensions must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        if not anchor_scales or not anchor_ratios or any((s <= 0 for s in anchor_scales)) or any((r <= 0 for r in anchor_ratios)):
            raise ValueError('anchor scales and ratios must be nonempty and positive')
        if len(anchor_scales) * len(anchor_ratios) != anchors_per_location:
            raise ValueError('anchors_per_location must equal len(anchor_scales) * len(anchor_ratios)')
        if len(anchor_sizes) != 3 or any((size <= 0 for size in anchor_sizes)):
            raise ValueError('anchor_sizes must contain three positive base sizes, one per FPN level')
        self.in_channels, self.num_classes, self.anchors_per_location = (in_channels, num_classes, anchors_per_location)
        self.anchor_scales, self.anchor_ratios = (tuple(map(float, anchor_scales)), tuple(map(float, anchor_ratios)))
        self.anchor_sizes = tuple(map(float, anchor_sizes))
        widths = (feature_width, feature_width * 2, feature_width * 4)
        self.stem = _Block(in_channels, widths[0], 2)
        self.stages = nn.ModuleList([_Block(widths[0], widths[0], 2), _Block(widths[0], widths[1], 2), _Block(widths[1], widths[2], 2)])
        self.lateral = nn.ModuleList([nn.Conv2d(c, feature_width, 1) for c in widths])
        self.output = nn.ModuleList([nn.Conv2d(feature_width, feature_width, 3, padding=1) for _ in widths])

        def head(out):
            return nn.Sequential(*[layer for _ in range(4) for layer in (nn.Conv2d(feature_width, feature_width, 3, padding=1), nn.ReLU(inplace=True))], nn.Conv2d(feature_width, anchors_per_location * out, 3, padding=1))
        self.cls_head, self.box_head = (head(num_classes), head(4))
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == anchors_per_location * num_classes:
                nn.init.constant_(m.bias, -4.595)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError('expected [B,C,H,W]')
        feats = []
        x = self.stem(images)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        pyramid = [None] * len(feats)
        top = None
        for i in range(len(feats) - 1, -1, -1):
            lat = self.lateral[i](feats[i])
            top = lat if top is None else lat + F.interpolate(top, size=lat.shape[-2:], mode='nearest')
            pyramid[i] = self.output[i](top)
        logits = []
        deltas = []
        grids = []
        strides = []
        for level, feat in enumerate(pyramid):
            b, _, h, w = feat.shape
            cls = self.cls_head(self.dropout(feat)).view(b, self.anchors_per_location, self.num_classes, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, self.num_classes)
            box = self.box_head(self.dropout(feat)).view(b, self.anchors_per_location, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
            yy, xx = torch.meshgrid(torch.arange(h, device=feat.device), torch.arange(w, device=feat.device), indexing='ij')
            grids.append(torch.stack((xx, yy), -1))
            logits.append(cls)
            deltas.append(box)
            strides.append((images.shape[-2] / h, images.shape[-1] / w))
        return RetinaNetOutput(tuple(logits), tuple(deltas), tuple(grids), tuple(strides), self.anchor_sizes)

def generate_anchors(height, width, stride, scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)), ratios=(0.5, 1.0, 2.0), device=None, dtype=torch.float32, base_size=None):
    sy, sx = (stride, stride) if isinstance(stride, (int, float)) else stride
    base_size = float(base_size) if base_size is not None else None
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing='ij')
    centers = torch.stack(((x + 0.5) * sx, (y + 0.5) * sy), -1).reshape(-1, 2)
    shapes = []
    for scale in scales:
        for ratio in ratios:
            base = base_size if base_size is not None else (sx * sy) ** 0.5
            w = base * scale / ratio ** 0.5
            h = base * scale * ratio ** 0.5
            shapes.append((w, h))
    wh = torch.tensor(shapes, device=device, dtype=dtype)
    centers = centers[:, None, :].expand(-1, len(shapes), -1).reshape(-1, 2)
    wh = wh[None].expand(height * width, -1, -1).reshape(-1, 2)
    return torch.cat((centers - wh / 2, centers + wh / 2), -1)

def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    inter = (torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:]) - torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])).clamp(min=0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(-1)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-08)

def encode_boxes(anchors, boxes):
    awh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-08)
    ac = (anchors[:, :2] + anchors[:, 2:]) / 2
    bwh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-08)
    bc = (boxes[:, :2] + boxes[:, 2:]) / 2
    return torch.cat(((bc - ac) / awh, torch.log(bwh / awh)), -1)

def decode_boxes(anchors, deltas):
    awh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-08)
    ac = (anchors[:, :2] + anchors[:, 2:]) / 2
    center = ac + deltas[:, :2] * awh
    wh = awh * torch.exp(deltas[:, 2:].clamp(max=8))
    return torch.cat((center - wh / 2, center + wh / 2), -1)
