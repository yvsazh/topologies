import torch
from torch import nn


class DETR(nn.Module):
    architecture_id = "detr"

    def __init__(self, input_channels=4, num_queries=2, num_classes=2, hidden_dim=32, nhead=4, encoder_layers=1, decoder_layers=1):
        super().__init__()
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=hidden_dim * 4, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))

    @staticmethod
    def _axis_position(position, channels):
        if channels == 0:
            return position.new_empty((*position.shape, 0))
        pairs = (channels + 1) // 2
        frequency_index = torch.arange(pairs, device=position.device, dtype=position.dtype)
        scale = 10000 ** (2 * frequency_index / channels)
        phase = position.unsqueeze(-1) / scale
        return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)[..., :channels]

    def position_encoding(self, height, width, device=None, dtype=None):
        device = device or self.query_embed.weight.device
        dtype = dtype or self.query_embed.weight.dtype
        y_channels = (self.hidden_dim + 1) // 2
        x_channels = self.hidden_dim - y_channels
        y = (torch.arange(height, device=device, dtype=torch.float32) + 1) * (2 * torch.pi / height)
        x = (torch.arange(width, device=device, dtype=torch.float32) + 1) * (2 * torch.pi / width)
        y_encoding = self._axis_position(y, y_channels)[:, None, :].expand(height, width, -1)
        x_encoding = self._axis_position(x, x_channels)[None, :, :].expand(height, width, -1)
        return torch.cat((y_encoding, x_encoding), dim=-1).reshape(1, height * width, self.hidden_dim).to(dtype=dtype)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("features must have shape (batch, channels, height, width)")
        batch = features.shape[0]
        tokens = self.input_projection(features).flatten(2).transpose(1, 2)
        height, width = features.shape[-2:]
        position = self.position_encoding(height, width, device=tokens.device, dtype=tokens.dtype)
        memory = self.encoder(tokens + position)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return {"pred_logits": self.class_head(decoded), "pred_boxes": self.box_head(decoded).sigmoid()}
