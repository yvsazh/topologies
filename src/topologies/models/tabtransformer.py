import torch
from torch import nn


class TabTransformer(nn.Module):
    """Tabular predictor with contextualized per-column categorical tokens."""

    def __init__(
        self,
        cardinalities,
        numeric_features=0,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        output_dim=1,
    ):
        super().__init__()
        cardinalities = tuple(int(size) for size in cardinalities)
        if not cardinalities or any(size <= 0 for size in cardinalities):
            raise ValueError("cardinalities must contain positive sizes")
        if numeric_features < 0 or min(embedding_dim, num_heads, num_layers, hidden_dim, output_dim) <= 0:
            raise ValueError("dimensions must be positive (numeric_features may be zero)")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.cardinalities = cardinalities
        self.numeric_features = int(numeric_features)
        self.embeddings = nn.ModuleList(nn.Embedding(size, embedding_dim) for size in cardinalities)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(len(cardinalities) * embedding_dim + numeric_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, categorical, numeric):
        if not isinstance(categorical, torch.Tensor) or categorical.ndim != 2:
            raise ValueError("categorical must have shape (batch, n_cat)")
        if categorical.shape[1] != len(self.cardinalities):
            raise ValueError("categorical column count does not match cardinalities")
        if categorical.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("categorical ids must be integer tensors")
        for column, size in enumerate(self.cardinalities):
            ids = categorical[:, column]
            if torch.any(ids < 0) or torch.any(ids >= size):
                raise ValueError(f"categorical id out of range for column {column}")

        if not isinstance(numeric, torch.Tensor) or numeric.ndim != 2:
            raise ValueError("numeric must have shape (batch, n_num)")
        if numeric.shape != (categorical.shape[0], self.numeric_features):
            raise ValueError("numeric shape does not match batch and numeric_features")
        if not numeric.is_floating_point():
            raise ValueError("numeric features must be floating point")

        tokens = torch.stack(
            [embedding(categorical[:, index].long()) for index, embedding in enumerate(self.embeddings)],
            dim=1,
        )
        contextual = self.transformer(tokens).flatten(start_dim=1)
        return self.head(torch.cat((contextual, numeric.to(dtype=contextual.dtype)), dim=1))
