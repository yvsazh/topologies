from torch import nn


class CPVT(nn.Module):
    component_id = "cpvt"
    def __init__(self, channels=4, kernel_size=3, bias=True):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        self.channels = channels
        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=bias,
        )

    def forward(self, x, spatial_size=None):
        if x.ndim == 4:
            if x.shape[1] != self.channels:
                raise ValueError("channel dimension does not match channels")
            return x + self.proj(x)
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError("expected BCHW feature maps or BNC tokens")
        batch, tokens, channels = x.shape
        if spatial_size is None:
            side = int(tokens**0.5)
            if side * side != tokens:
                raise ValueError("spatial_size is required for non-square token counts")
            height = width = side
        else:
            height, width = spatial_size
        if height * width != tokens:
            raise ValueError("spatial_size must match the token count")
        feature_map = x.transpose(1, 2).reshape(batch, channels, height, width)
        encoded = feature_map + self.proj(feature_map)
        return encoded.flatten(2).transpose(1, 2)
