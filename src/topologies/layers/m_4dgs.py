import torch
from torch import nn


class FourDGSResidualDeformation(nn.Module):
    component_id = "4dgs"

    def __init__(self, feature_dim=128, hidden_dim=64, depth=1, output_format="tuple"):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("hidden_dim", hidden_dim), ("depth", depth)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if output_format not in ("tuple", "concatenated"):
            raise ValueError("output_format must be 'tuple' or 'concatenated'")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.output_format = output_format
        projection = []
        for index in range(depth):
            projection.extend((nn.Linear(feature_dim if index == 0 else hidden_dim, hidden_dim), nn.ReLU()))
        self.feature_projection = nn.Sequential(*projection)
        self.position_head = self._make_head(3)
        self.scale_head = self._make_head(3)
        self.rotation_head = self._make_head(4)

    def _make_head(self, output_dim):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, features, positions, log_scales, rotations):
        inputs = (features, positions, log_scales, rotations)
        if any(not torch.is_tensor(value) for value in inputs):
            raise ValueError("features and Gaussian attributes must be tensors")
        if features.ndim < 1 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must end with dimension {self.feature_dim}")
        for name, value, width in (
            ("position", positions, 3),
            ("scale", log_scales, 3),
            ("rotation", rotations, 4),
        ):
            if value.ndim < 1 or value.shape[-1] != width:
                raise ValueError(f"{name} must end with dimension {width}")
            if value.shape[:-1] != features.shape[:-1]:
                raise ValueError(f"{name} leading dimensions must match features")
        projected = self.feature_projection(features)
        outputs = (
            positions + self.position_head(projected),
            log_scales + self.scale_head(projected),
            rotations + self.rotation_head(projected),
        )
        return torch.cat(outputs, dim=-1) if self.output_format == "concatenated" else outputs
