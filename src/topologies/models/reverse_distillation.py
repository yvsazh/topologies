import math

import torch
from torch import nn
from torch.nn import functional as F


def _make_residual_block(in_channels, out_channels, stride):
    skip = nn.Identity()
    if stride != 1 or in_channels != out_channels:
        skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    return nn.ModuleDict(
        {
            "conv1": nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            "bn1": nn.BatchNorm2d(out_channels),
            "conv2": nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            "bn2": nn.BatchNorm2d(out_channels),
            "skip": skip,
        }
    )


def _apply_residual_block(values, block):
    residual = block["skip"](values)
    hidden = F.relu(block["bn1"](block["conv1"](values)))
    hidden = block["bn2"](block["conv2"](hidden))
    return F.relu(hidden + residual)


class ReverseDistillation(nn.Module):
    architecture_id = "reverse-distillation"
    PRESETS = {
        "tiny": {
            "feature_channels": (64, 128, 256),
            "width_multiplier": 0.125,
            "oce_blocks": 2,
            "decoder_blocks_per_stage": (2, 2, 2),
            "score_sigma": 4.0,
        },
        "small": {
            "feature_channels": (64, 128, 256),
            "width_multiplier": 0.5,
            "oce_blocks": 2,
            "decoder_blocks_per_stage": (2, 2, 2),
            "score_sigma": 4.0,
        },
        "base": {
            "feature_channels": (64, 128, 256),
            "width_multiplier": 1.0,
            "oce_blocks": 2,
            "decoder_blocks_per_stage": (2, 2, 2),
            "score_sigma": 4.0,
        },
    }

    def __init__(
        self,
        feature_channels=(64, 128, 256),
        width_multiplier=1.0,
        oce_blocks=2,
        decoder_blocks_per_stage=(2, 2, 2),
        score_sigma=4.0,
    ):
        super().__init__()
        if not isinstance(feature_channels, (tuple, list)) or len(feature_channels) != 3:
            raise ValueError("feature_channels must contain three positive channel counts")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in feature_channels):
            raise ValueError("feature_channels must contain three positive channel counts")
        if isinstance(width_multiplier, bool) or not isinstance(width_multiplier, (int, float)) \
                or not math.isfinite(width_multiplier) or width_multiplier <= 0:
            raise ValueError("width_multiplier must be a finite positive number")
        if isinstance(oce_blocks, bool) or not isinstance(oce_blocks, int) or oce_blocks < 1:
            raise ValueError("oce_blocks must be a positive integer")
        if isinstance(decoder_blocks_per_stage, int) and not isinstance(decoder_blocks_per_stage, bool):
            decoder_blocks_per_stage = (decoder_blocks_per_stage,) * 3
        if not isinstance(decoder_blocks_per_stage, (tuple, list)) \
                or len(decoder_blocks_per_stage) != 3 \
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                       for value in decoder_blocks_per_stage):
            raise ValueError("decoder_blocks_per_stage must contain three positive integers")
        if isinstance(score_sigma, bool) or not isinstance(score_sigma, (int, float)) \
                or not math.isfinite(score_sigma) or score_sigma < 0:
            raise ValueError("score_sigma must be a finite non-negative number")

        self.feature_channels = tuple(feature_channels)
        self.width_multiplier = float(width_multiplier)
        self.oce_blocks_count = oce_blocks
        self.decoder_blocks_per_stage = tuple(decoder_blocks_per_stage)
        self.score_sigma = float(score_sigma)
        code_width = max(8, round(512 * self.width_multiplier))
        fusion_width = max(8, code_width // 2)
        decoder_widths = (
            fusion_width,
            max(8, fusion_width // 2),
            max(8, fusion_width // 4),
        )
        self.code_width = code_width
        self.fusion_width = fusion_width

        first_channel, second_channel, third_channel = self.feature_channels
        self.align_first = nn.Sequential(
            nn.Conv2d(first_channel, fusion_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fusion_width),
            nn.ReLU(inplace=False),
            nn.Conv2d(fusion_width, fusion_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fusion_width),
            nn.ReLU(inplace=False),
        )
        self.align_second = nn.Sequential(
            nn.Conv2d(second_channel, fusion_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fusion_width),
            nn.ReLU(inplace=False),
        )
        self.align_third = nn.Sequential(
            nn.Conv2d(third_channel, fusion_width, 1, bias=False),
            nn.BatchNorm2d(fusion_width),
            nn.ReLU(inplace=False),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_width * 3, code_width, 1, bias=False),
            nn.BatchNorm2d(code_width),
            nn.ReLU(inplace=False),
        )
        oce = []
        for index in range(oce_blocks):
            oce.append(_make_residual_block(code_width, code_width, 2 if index == 0 else 1))
        self.oce = nn.ModuleList(oce)

        decoder = []
        output_projections = []
        in_channels = code_width
        for index, (out_channels, block_count) in enumerate(
            zip(decoder_widths, self.decoder_blocks_per_stage)
        ):
            stage_blocks = nn.ModuleList(
                _make_residual_block(out_channels, out_channels, 1)
                for _ in range(block_count)
            )
            decoder.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                        "blocks": stage_blocks,
                    }
                )
            )
            output_projections.append(
                nn.Conv2d(out_channels, self.feature_channels[2 - index], 1)
            )
            in_channels = out_channels
        self.decoder = nn.ModuleList(decoder)
        self.output_projections = nn.ModuleList(output_projections)

    def _validate_features(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 3:
            raise ValueError("features must be a three-scale tuple or list")
        if any(not torch.is_tensor(value) or value.ndim != 4 for value in features):
            raise ValueError("each feature must be a rank-four tensor")
        first, second, third = features
        if any(value.shape[1] != channels for value, channels in zip(features, self.feature_channels)):
            raise ValueError("feature channels do not match feature_channels")
        if any(not value.is_floating_point() for value in features):
            raise ValueError("features must use floating-point tensors")
        if any(value.shape[0] != first.shape[0] for value in features):
            raise ValueError("all features must have the same batch size")
        if any(value.device != first.device or value.dtype != first.dtype for value in features):
            raise ValueError("all features must share a device and dtype")
        if second.shape[-2] != third.shape[-2] * 2 or second.shape[-1] != third.shape[-1] * 2:
            raise ValueError("the middle feature must be twice the deepest spatial size")
        if first.shape[-2] != third.shape[-2] * 4 or first.shape[-1] != third.shape[-1] * 4:
            raise ValueError("the shallow feature must be four times the deepest spatial size")

    @staticmethod
    def cosine_anomaly_maps(teacher_features, reconstructed_features):
        if not isinstance(teacher_features, (tuple, list)) \
                or not isinstance(reconstructed_features, (tuple, list)) \
                or len(teacher_features) != len(reconstructed_features) or not teacher_features:
            raise ValueError("teacher and reconstructed features must have matching non-empty scales")
        maps = []
        for teacher, reconstructed in zip(teacher_features, reconstructed_features):
            if not torch.is_tensor(teacher) or not torch.is_tensor(reconstructed) \
                    or teacher.ndim != 4 or teacher.shape != reconstructed.shape:
                raise ValueError("paired features must be tensors with matching rank-four shapes")
            teacher_unit = F.normalize(teacher, p=2, dim=1, eps=1e-8)
            reconstructed_unit = F.normalize(reconstructed, p=2, dim=1, eps=1e-8)
            maps.append(1.0 - (teacher_unit * reconstructed_unit).sum(dim=1))
        return tuple(maps)

    @staticmethod
    def distillation_loss(anomaly_maps):
        if not isinstance(anomaly_maps, (tuple, list)) or not anomaly_maps:
            raise ValueError("anomaly_maps must be a non-empty tuple or list")
        if any(not torch.is_tensor(value) or value.ndim != 3 for value in anomaly_maps):
            raise ValueError("each anomaly map must have shape (batch, height, width)")
        batch_sizes = {value.shape[0] for value in anomaly_maps}
        if len(batch_sizes) != 1:
            raise ValueError("all anomaly maps must have the same batch size")
        return sum(value.mean(dim=(-2, -1)).mean() for value in anomaly_maps)

    def aggregate_anomaly_maps(self, anomaly_maps, output_size, smooth=True):
        if not isinstance(anomaly_maps, (tuple, list)) or not anomaly_maps:
            raise ValueError("anomaly_maps must be a non-empty tuple or list")
        if not isinstance(output_size, (tuple, list)) or len(output_size) != 2 \
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                       for value in output_size):
            raise ValueError("output_size must contain two positive integers")
        if any(not torch.is_tensor(value) or value.ndim != 3 for value in anomaly_maps):
            raise ValueError("each anomaly map must have shape (batch, height, width)")
        batch_size = anomaly_maps[0].shape[0]
        if any(value.shape[0] != batch_size for value in anomaly_maps):
            raise ValueError("all anomaly maps must have the same batch size")
        resized = [
            F.interpolate(value.unsqueeze(1), size=tuple(output_size), mode="bilinear", align_corners=False)
            for value in anomaly_maps
        ]
        score_map = torch.stack(resized, dim=0).sum(dim=0)
        if smooth and self.score_sigma > 0:
            radius = math.ceil(3 * self.score_sigma)
            positions = torch.arange(-radius, radius + 1, device=score_map.device, dtype=score_map.dtype)
            kernel = torch.exp(-(positions.square()) / (2 * self.score_sigma ** 2))
            kernel = kernel / kernel.sum()
            score_map = F.pad(score_map, (radius, radius, 0, 0), mode="replicate")
            score_map = F.conv2d(score_map, kernel.reshape(1, 1, 1, -1))
            score_map = F.pad(score_map, (0, 0, radius, radius), mode="replicate")
            score_map = F.conv2d(score_map, kernel.reshape(1, 1, -1, 1))
        image_scores = score_map.flatten(1).amax(dim=1)
        return score_map.squeeze(1), image_scores

    def forward(self, features):
        self._validate_features(features)
        first, second, third = features
        aligned = (self.align_first(first), self.align_second(second), self.align_third(third))
        target_size = third.shape[-2:]
        aligned = tuple(
            value if value.shape[-2:] == target_size
            else F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            for value in aligned
        )
        hidden = self.fusion(torch.cat(aligned, dim=1))
        for block in self.oce:
            hidden = _apply_residual_block(hidden, block)

        decoded = hidden
        reconstructed = []
        target_features = (third, second, first)
        for index, (stage, projection, target) in enumerate(
            zip(self.decoder, self.output_projections, target_features)
        ):
            decoded = F.relu(stage["up"](decoded))
            for block in stage["blocks"]:
                decoded = _apply_residual_block(decoded, block)
            if decoded.shape[-2:] != target.shape[-2:]:
                decoded = F.interpolate(
                    decoded, size=target.shape[-2:], mode="bilinear", align_corners=False
                )
            reconstructed.append(projection(decoded))
        reconstructed = tuple(reversed(reconstructed))
        maps = self.cosine_anomaly_maps(features, reconstructed)
        return {
            "reconstructed_features": reconstructed,
            "anomaly_maps": maps,
            "loss": self.distillation_loss(maps),
        }
