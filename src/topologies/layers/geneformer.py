import torch
from torch import nn


class Geneformer(nn.Module):
    architecture_id = "geneformer"
    component_id = "geneformer"
    def __init__(self, vocab_size, hidden_size=64, num_layers=2, num_heads=4, max_length=512, mask_token_id=0, dropout=0.1):
        super().__init__()
        if vocab_size < 2 or hidden_size < 1 or num_layers < 1 or num_heads < 1 or hidden_size % num_heads:
            raise ValueError("invalid model dimensions")
        if max_length < 1 or not 0 <= mask_token_id < vocab_size:
            raise ValueError("invalid sequence or mask configuration")
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        block = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, gene_ids, masked_positions=None, padding_mask=None):
        gene_ids = torch.as_tensor(gene_ids, dtype=torch.long, device=self.token_embedding.weight.device)
        if gene_ids.ndim != 2 or gene_ids.shape[1] == 0 or gene_ids.shape[1] > self.max_length:
            raise ValueError("gene_ids must have shape (batch, sequence) within max_length")
        if torch.any(gene_ids < 0) or torch.any(gene_ids >= self.vocab_size):
            raise ValueError("gene identity outside vocabulary")
        if masked_positions is None:
            masked_positions = gene_ids.eq(self.mask_token_id)
        else:
            masked_positions = torch.as_tensor(masked_positions, dtype=torch.bool, device=gene_ids.device)
            if masked_positions.shape != gene_ids.shape:
                raise ValueError("masked_positions must match gene_ids")
        visible_ids = gene_ids.masked_fill(masked_positions, self.mask_token_id)
        positions = torch.arange(gene_ids.shape[1], device=gene_ids.device).unsqueeze(0)
        hidden = self.token_embedding(visible_ids) + self.position_embedding(positions)
        key_padding_mask = None
        if padding_mask is not None:
            key_padding_mask = torch.as_tensor(padding_mask, dtype=torch.bool, device=gene_ids.device)
            if key_padding_mask.shape != gene_ids.shape:
                raise ValueError("padding_mask must match gene_ids")
        hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)
        return self.output(self.norm(hidden))
