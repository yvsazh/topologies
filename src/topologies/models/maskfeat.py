import torch
from torch import nn
import torch.nn.functional as F


def hog_target(images, patch_size, hog_bins=9):
    if images.ndim != 4 or patch_size <= 0 or hog_bins <= 0:
        raise ValueError("images must be NCHW and patch_size and hog_bins positive")
    if images.shape[-2] % patch_size or images.shape[-1] % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    gray = images.mean(dim=1, keepdim=True)
    dx = F.pad(gray[..., 1:] - gray[..., :-1], (0, 1, 0, 0))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-12)
    orientation = torch.atan2(dy, dx).remainder(torch.pi)
    bin_index = torch.floor(orientation * (hog_bins / torch.pi)).long().clamp_max(hog_bins - 1)
    histograms = F.one_hot(bin_index[:, 0], num_classes=hog_bins).to(images.dtype)
    histograms = histograms * magnitude[:, 0].unsqueeze(-1)
    batch, height, width, _ = histograms.shape
    cells = histograms.permute(0, 3, 1, 2).reshape(batch, hog_bins, height, width)
    pooled = F.avg_pool2d(cells, patch_size, patch_size) * (patch_size * patch_size)
    patches = pooled.flatten(2).transpose(1, 2)
    return F.normalize(patches, p=2, dim=-1, eps=1e-6)


def masked_hog_loss(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("prediction, target, and mask shapes do not align")
    weights = mask.to(prediction.dtype)
    per_patch = (prediction - target).square().mean(dim=-1)
    return (per_patch * weights).sum() / weights.sum().clamp_min(1.0)


class MaskFeat(nn.Module):
    architecture_id = "maskfeat"
    def __init__(self, in_channels=3, patch_size=4, hidden_channels=8, hog_bins=9, mask_ratio=0.5,
                 num_heads=2, max_patches=4096):
        super().__init__()
        if min(in_channels, patch_size, hidden_channels, hog_bins, num_heads, max_patches) <= 0:
            raise ValueError("dimensions, HOG bins, and attention heads must be positive")
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between zero and one")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_channels = hidden_channels
        self.hog_bins = hog_bins
        self.mask_ratio = mask_ratio
        self.max_patches = max_patches
        self.patch_embed = nn.Conv2d(in_channels, hidden_channels, patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_patches, hidden_channels))
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden_channels, num_heads, hidden_channels * 4,
                                           batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.predictor = nn.Linear(hidden_channels, hog_bins)

    def forward(self, images, mask=None):
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError("images must have shape (batch, in_channels, height, width)")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        count = tokens.shape[1]
        if count > self.max_patches:
            raise ValueError("image produces more patches than max_patches")
        if mask is None:
            mask = torch.rand(images.shape[0], count, device=images.device) < self.mask_ratio
        else:
            mask = torch.as_tensor(mask, device=images.device, dtype=torch.bool)
            if mask.shape != (images.shape[0], count):
                raise ValueError("mask must have shape (batch, number_of_patches)")
        masked = torch.where(mask.unsqueeze(-1), self.mask_token.expand(images.shape[0], count, -1), tokens)
        masked = masked + self.position_embedding[:, :count]
        predictions = self.predictor(self.context(masked))
        targets = hog_target(images, self.patch_size, self.hog_bins)
        loss = masked_hog_loss(predictions, targets, mask)
        return {"predictions": predictions, "targets": targets, "mask": mask, "loss": loss}
