import torch
from torch import nn


class GIT(nn.Module):
    architecture_id = "git"
    component_id = "git"
    PRESETS = {"tiny": {"image_dim": 8, "vocab_size": 32, "hidden_size": 16, "num_heads": 4, "num_layers": 2}}

    def __init__(self, image_dim=8, vocab_size=32, hidden_size=16, num_heads=4, num_layers=2):
        super().__init__()
        if image_dim <= 0 or vocab_size <= 1 or hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("invalid model dimensions")
        self.image_dim = image_dim
        self.vocab_size = vocab_size
        self.image_projection = nn.Linear(image_dim, hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(2048, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_tokens, caption_tokens):
        image_tokens = torch.as_tensor(image_tokens, dtype=self.image_projection.weight.dtype, device=self.image_projection.weight.device)
        caption_tokens = torch.as_tensor(caption_tokens, dtype=torch.long, device=self.token_embedding.weight.device)
        if image_tokens.ndim != 3 or image_tokens.shape[-1] != self.image_dim:
            raise ValueError("image_tokens must have shape (batch, image_sequence, image_dim)")
        if caption_tokens.ndim != 2 or caption_tokens.shape[0] != image_tokens.shape[0]:
            raise ValueError("caption_tokens must have shape (batch, caption_sequence)")
        if image_tokens.shape[1] < 1 or caption_tokens.shape[1] < 1 or caption_tokens.shape[1] + image_tokens.shape[1] > 2048:
            raise ValueError("invalid sequence length")
        if torch.any(caption_tokens < 0) or torch.any(caption_tokens >= self.vocab_size):
            raise ValueError("caption token outside vocabulary")
        image = self.image_projection(image_tokens)
        text = self.token_embedding(caption_tokens)
        combined = torch.cat((image, text), dim=1)
        positions = torch.arange(combined.shape[1], device=combined.device)
        combined = combined + self.position_embedding(positions).unsqueeze(0)
        image_length = image_tokens.shape[1]
        text_length = caption_tokens.shape[1]
        attention_mask = self.build_attention_mask(
            image_length, text_length, device=combined.device
        )
        hidden = self.decoder(combined, mask=attention_mask)
        return self.output_projection(hidden[:, image_tokens.shape[1]:])

    @staticmethod
    def build_attention_mask(image_length, text_length, device=None):
        if image_length < 1 or text_length < 1:
            raise ValueError("image and text sequences must be nonempty")
        total_length = image_length + text_length
        return torch.triu(
            torch.ones(total_length, total_length, dtype=torch.bool, device=device),
            diagonal=1,
        )
