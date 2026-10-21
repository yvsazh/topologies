from torch import nn


class MiniGPT4Projection(nn.Module):
    component_id = "minigpt-4"

    def __init__(self, vision_dim=8, text_dim=8):
        super().__init__()
        if vision_dim <= 0 or text_dim <= 0:
            raise ValueError("dimensions must be positive")
        self.vision_projection = nn.Linear(vision_dim, text_dim, bias=False)

    def forward(self, image_features):
        if image_features.ndim != 3:
            raise ValueError("image_features must have shape (batch, patches, vision_dim)")
        if image_features.shape[-1] != self.vision_projection.in_features:
            raise ValueError("image feature width does not match vision_dim")
        return self.vision_projection(image_features)
