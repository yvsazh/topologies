import torch
from torch import nn
from torch.nn import functional as F


class SigLIP(nn.Module):
    architecture_id = "siglip"
    def __init__(self, image_dim=8, text_dim=8, embed_dim=8):
        super().__init__()
        if image_dim <= 0 or text_dim <= 0 or embed_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        self.image_tower = nn.Linear(image_dim, embed_dim, bias=False)
        self.text_tower = nn.Linear(text_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, images, texts):
        image_features = F.normalize(self.image_tower(images), dim=-1)
        text_features = F.normalize(self.text_tower(texts), dim=-1)
        return self.logit_scale.exp() * image_features @ text_features.transpose(-1, -2) + self.logit_bias

    def loss(self, images, texts, labels=None):
        logits = self(images, texts)
        if labels is None:
            if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
                raise ValueError("labels are required unless image and text batch sizes match")
            labels = torch.eye(logits.shape[0], device=logits.device, dtype=logits.dtype)
        else:
            labels = torch.as_tensor(labels, device=logits.device, dtype=logits.dtype)
            if labels.shape != logits.shape:
                raise ValueError("labels must have the same shape as pairwise logits")
        targets = labels.mul(2).sub(1)
        if logits.ndim != 2 or logits.shape[0] == 0:
            raise ValueError("loss expects a nonempty two-dimensional pair matrix")
        return -F.logsigmoid(targets * logits).sum() / logits.shape[0]
