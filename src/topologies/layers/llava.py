from torch import nn


class LLaVAVisualProjector(nn.Module):
    component_id = "llava"

    def __init__(self, vision_dim=8, embed_dim=16):
        super().__init__()
        if min(vision_dim, embed_dim) <= 0:
            raise ValueError("dimensions must be positive")
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim

        self.projection = nn.Linear(vision_dim, embed_dim, bias=False)

    def forward(self, image_features):
        if image_features.ndim != 3 or image_features.shape[-1] != self.vision_dim:
            raise ValueError("image_features must have shape (batch, patches, vision_dim)")
        return self.projection(image_features)
