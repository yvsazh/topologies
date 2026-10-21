import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, branch_channels=32, kernel_widths=(3, 5)):
        super().__init__()
        if branch_channels < 1:
            raise ValueError("branch_channels must be positive")
        if not kernel_widths or any(width < 3 or width % 2 == 0 for width in kernel_widths):
            raise ValueError("kernel_widths must contain odd temporal widths of at least three")

        self.direct_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )
        self.temporal_branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
                nn.LeakyReLU(0.01),
                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=(width, 1),
                    padding=(width // 2, 0),
                ),
                nn.LeakyReLU(0.01),
            )
            for width in kernel_widths
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )

    @property
    def out_channels(self):
        return self.direct_branch[0].out_channels * (len(self.temporal_branches) + 2)

    def forward(self, x):
        branches = [self.direct_branch(x)]
        branches.extend(branch(x) for branch in self.temporal_branches)
        branches.append(self.pool_branch(x))
        return torch.cat(branches, dim=1)


class DeepLOB(nn.Module):
    component_id = "deeplob"

    def __init__(
        self,
        input_features=40,
        conv_channels=32,
        branch_channels=32,
        hidden_size=64,
        num_layers=1,
        kernel_widths=(3, 5),
        inception_blocks=2,
        num_classes=3,
    ):
        super().__init__()
        if input_features < 4 or input_features % 4:
            raise ValueError("input_features must be a positive multiple of four")
        if min(conv_channels, branch_channels, hidden_size, num_layers, inception_blocks) < 1:
            raise ValueError("channel counts and layer counts must be positive")
        if num_classes != 3:
            raise ValueError("DeepLOB predicts exactly three price-movement classes")

        self.input_features = input_features
        self.book_convolutions = nn.ModuleList(
            [
                nn.Conv2d(1, conv_channels, kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=(1, 2),
                    stride=(1, 2),
                ),
                nn.Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=(1, input_features // 4),
                ),
            ]
        )
        self.book_activation = nn.LeakyReLU(0.01)

        blocks = []
        in_channels = conv_channels
        for _ in range(inception_blocks):
            block = InceptionBlock(in_channels, branch_channels, kernel_widths)
            blocks.append(block)
            in_channels = block.out_channels
        self.inception = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_features:
            raise ValueError("input must have shape (batch, time, input_features)")
        if x.shape[1] == 0:
            raise ValueError("time dimension must be non-empty")

        features = x.unsqueeze(1)
        for convolution in self.book_convolutions:
            features = self.book_activation(convolution(features))
        features = self.inception(features)
        sequence = features.squeeze(-1).transpose(1, 2)
        recurrent, _ = self.lstm(sequence)
        return self.classifier(recurrent)
