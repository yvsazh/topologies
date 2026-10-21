import torch
from torch import nn


class YOLOv3(nn.Module):
    architecture_id = "yolov3"
    def __init__(self, in_channels=3, num_classes=80, num_anchors=3, anchors=None):
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or num_anchors <= 0:
            raise ValueError("channel, class, and anchor counts must be positive")
        if anchors is None:
            anchors = torch.full((num_anchors, 2), 0.1)
        else:
            anchors = torch.as_tensor(anchors, dtype=torch.float32)
        if anchors.shape != (num_anchors, 2) or not torch.isfinite(anchors).all() or (anchors <= 0).any():
            raise ValueError("anchors must be positive normalized (width, height) pairs, one per anchor")
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.register_buffer("anchors", anchors)
        self.prediction = nn.Conv2d(in_channels, num_anchors * (num_classes + 5), kernel_size=1)

    def forward(self, x):
        raw = self.prediction(x)
        batch, _, height, width = raw.shape
        raw = raw.view(batch, self.num_anchors, self.num_classes + 5, height, width)
        offsets = raw[:, :, :2]
        sizes = raw[:, :, 2:4]
        objectness_logit = raw[:, :, 4:5]
        class_logits = raw[:, :, 5:]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=x.dtype),
            torch.arange(width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=0).view(1, 1, 2, height, width)
        scale = x.new_tensor((width, height)).view(1, 1, 2, 1, 1)
        centers = (torch.sigmoid(offsets) + grid) / scale
        anchor_sizes = self.anchors.to(device=x.device, dtype=x.dtype).view(1, self.num_anchors, 2, 1, 1)
        box_sizes = anchor_sizes * torch.exp(sizes)
        return {
            "class_logits": class_logits,
            "objectness_logit": objectness_logit,
            "center": centers,
            "size": box_sizes,
        }
