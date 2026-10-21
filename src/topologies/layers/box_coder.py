import torch
from torch import nn


class BoxCoder(nn.Module):
    component_id = "box-coder"
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        if len(weights) != 4 or any(weight <= 0 for weight in weights):
            raise ValueError("weights must contain four positive values")
        self.weights = tuple(float(weight) for weight in weights)

    def encode(self, boxes, anchors):
        if boxes.shape[-1] != 4 or anchors.shape[-1] != 4:
            raise ValueError("boxes and anchors must have a final dimension of four")
        anchor_width = anchors[..., 2] - anchors[..., 0]
        anchor_height = anchors[..., 3] - anchors[..., 1]
        box_width = boxes[..., 2] - boxes[..., 0]
        box_height = boxes[..., 3] - boxes[..., 1]
        if torch.any(anchor_width <= 0) or torch.any(anchor_height <= 0):
            raise ValueError("anchors must have positive width and height")
        if torch.any(box_width <= 0) or torch.any(box_height <= 0):
            raise ValueError("boxes must have positive width and height")
        anchor_x = (anchors[..., 0] + anchors[..., 2]) * 0.5
        anchor_y = (anchors[..., 1] + anchors[..., 3]) * 0.5
        box_x = (boxes[..., 0] + boxes[..., 2]) * 0.5
        box_y = (boxes[..., 1] + boxes[..., 3]) * 0.5
        wx, wy, ww, wh = self.weights
        return torch.stack(((box_x - anchor_x) / anchor_width * wx,
                            (box_y - anchor_y) / anchor_height * wy,
                            torch.log(box_width / anchor_width) * ww,
                            torch.log(box_height / anchor_height) * wh), dim=-1)

    def decode(self, deltas, anchors):
        if deltas.shape[-1] != 4 or anchors.shape[-1] != 4:
            raise ValueError("deltas and anchors must have a final dimension of four")
        anchor_width = anchors[..., 2] - anchors[..., 0]
        anchor_height = anchors[..., 3] - anchors[..., 1]
        if torch.any(anchor_width <= 0) or torch.any(anchor_height <= 0):
            raise ValueError("anchors must have positive width and height")
        wx, wy, ww, wh = self.weights
        dx, dy, dw, dh = deltas.unbind(dim=-1)
        anchor_x = (anchors[..., 0] + anchors[..., 2]) * 0.5
        anchor_y = (anchors[..., 1] + anchors[..., 3]) * 0.5
        center_x = dx / wx * anchor_width + anchor_x
        center_y = dy / wy * anchor_height + anchor_y
        width = torch.exp(dw / ww) * anchor_width
        height = torch.exp(dh / wh) * anchor_height
        return torch.stack((center_x - width * 0.5, center_y - height * 0.5,
                            center_x + width * 0.5, center_y + height * 0.5), dim=-1)

    def forward(self, boxes, anchors):
        return self.encode(boxes, anchors)
