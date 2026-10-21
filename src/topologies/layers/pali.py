import torch
from torch import nn


class PaLI(nn.Module):
    architecture_id = "pali"
    component_id = "pali"

    def __init__(self, vision_dim=8, hidden_dim=16, vocab_size=32, num_heads=2):
        super().__init__()
        if vision_dim <= 0 or hidden_dim <= 0 or vocab_size <= 1:
            raise ValueError("dimensions and vocabulary size must be positive")
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, dim_feedforward=hidden_dim * 4,
            activation="gelu", batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim, num_heads, dim_feedforward=hidden_dim * 4,
            activation="gelu", batch_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, caption_tokens, query_tokens=None):
        image_features = torch.as_tensor(image_features, dtype=self.vision_projection.weight.dtype, device=self.vision_projection.weight.device)
        caption_tokens = torch.as_tensor(caption_tokens, dtype=torch.long, device=self.token_embedding.weight.device)
        if image_features.ndim != 3 or image_features.shape[-1] != self.vision_projection.in_features or image_features.shape[1] == 0:
            raise ValueError("image_features must have shape (batch, image_tokens, vision_dim)")
        if caption_tokens.ndim != 2 or caption_tokens.shape[0] != image_features.shape[0] or caption_tokens.shape[1] == 0:
            raise ValueError("caption_tokens must have shape (batch, caption_length)")
        if torch.any(caption_tokens < 0) or torch.any(caption_tokens >= self.token_embedding.num_embeddings):
            raise ValueError("caption token id is outside the vocabulary")
        encoder_inputs = self.vision_projection(image_features)
        if query_tokens is not None:
            query_tokens = torch.as_tensor(query_tokens, dtype=torch.long, device=self.token_embedding.weight.device)
            if query_tokens.ndim != 2 or query_tokens.shape[0] != image_features.shape[0]:
                raise ValueError("query_tokens must have shape (batch, query_length)")
            if torch.any(query_tokens < 0) or torch.any(query_tokens >= self.token_embedding.num_embeddings):
                raise ValueError("query token id is outside the vocabulary")
            if query_tokens.shape[1]:
                encoder_inputs = torch.cat((encoder_inputs, self.token_embedding(query_tokens)), dim=1)
        memory = self.text_encoder(encoder_inputs)
        decoder_inputs = self.token_embedding(caption_tokens)
        length = caption_tokens.shape[1]
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=caption_tokens.device), diagonal=1)
        decoded = self.text_decoder(decoder_inputs, memory, tgt_mask=causal_mask)
        return self.output_projection(decoded)
