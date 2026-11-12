from torch import nn


class OSNet(nn.Module):
    component_id = "osnet"
    PRESETS = {
        "tiny": {"width_multiplier": 0.25},
        "small": {"width_multiplier": 0.5},
        "base": {"width_multiplier": 1.0},
    }

    def __init__(self, width_multiplier=1.0, num_classes=1000, feature_dim=512):
        super().__init__()
        if type(width_multiplier) not in (int, float) or width_multiplier not in (0.25, 0.5, 0.75, 1.0):
            raise ValueError("width_multiplier must be one of 0.25, 0.5, 0.75, or 1.0")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        width_multiplier = float(width_multiplier)
        stem_channels = int(64 * width_multiplier)
        stage_channels = [int(channels * width_multiplier) for channels in (256, 384, 512)]
        self.width_multiplier = width_multiplier
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=False),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.ModuleList(
            [
                self._make_block(stem_channels, stage_channels[0]),
                self._make_block(stage_channels[0], stage_channels[0]),
            ]
        )
        self.transition1 = self._make_transition(stage_channels[0])
        self.stage2 = nn.ModuleList(
            [
                self._make_block(stage_channels[0], stage_channels[1]),
                self._make_block(stage_channels[1], stage_channels[1]),
            ]
        )
        self.transition2 = self._make_transition(stage_channels[1])
        self.stage3 = nn.ModuleList(
            [
                self._make_block(stage_channels[1], stage_channels[2]),
                self._make_block(stage_channels[2], stage_channels[2]),
            ]
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(stage_channels[2], feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=False),
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_lite_conv(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )

    def _make_block(self, in_channels, out_channels):
        mid_channels = out_channels // 4
        hidden_channels = max(mid_channels // 16, 1)
        gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(mid_channels, hidden_channels),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_channels, mid_channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(gate[2].weight)
        nn.init.ones_(gate[2].bias)
        return nn.ModuleDict(
            {
                "reduce": nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=False),
                ),
                "streams": nn.ModuleList(
                    [
                        nn.Sequential(
                            *[self._make_lite_conv(mid_channels) for _ in range(depth)]
                        )
                        for depth in range(1, 5)
                    ]
                ),
                "gate": gate,
                "expand": nn.Sequential(
                    nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                ),
                "shortcut": nn.Identity()
                if in_channels == out_channels
                else nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                ),
                "activation": nn.ReLU(inplace=False),
            }
        )

    def _make_transition(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def _aggregate_streams(self, streams, gate):
        result = None
        for stream in streams:
            weights = gate(stream).reshape(stream.shape[0], stream.shape[1], 1, 1)
            weighted = weights * stream
            result = weighted if result is None else result + weighted
        return result

    def _forward_block(self, x, block):
        reduced = block["reduce"](x)
        streams = [stream(reduced) for stream in block["streams"]]
        aggregate = self._aggregate_streams(streams, block["gate"])
        return block["activation"](block["expand"](aggregate) + block["shortcut"](x))

    def forward(self, x, return_featuremaps=False, return_features=False):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("input must have shape (batch, 3, height, width)")
        x = self.maxpool(self.stem(x))
        for block in self.stage1:
            x = self._forward_block(x, block)
        x = self.transition1(x)
        for block in self.stage2:
            x = self._forward_block(x, block)
        x = self.transition2(x)
        for block in self.stage3:
            x = self._forward_block(x, block)
        if return_featuremaps:
            return x
        x = self.conv5(x)
        x = self.global_avgpool(x).flatten(start_dim=1)
        if return_features:
            return x
        return self.classifier(x)
