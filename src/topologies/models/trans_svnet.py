import torch
from torch import nn


class TransSVNet(nn.Module):
    architecture_id = "trans-svnet"
    PRESETS = {
        "tiny": {
            "input_dim": 2048,
            "num_stages": 7,
            "window_size": 8,
            "num_heads": 4,
            "d_k": 8,
            "d_v": 8,
            "d_ff": 16,
        },
        "small": {
            "input_dim": 2048,
            "num_stages": 7,
            "window_size": 16,
            "num_heads": 8,
            "d_k": 16,
            "d_v": 16,
            "d_ff": 32,
        },
        "base": {
            "input_dim": 2048,
            "num_stages": 7,
            "window_size": 30,
            "num_heads": 8,
            "d_k": 32,
            "d_v": 32,
            "d_ff": 32,
        },
    }

    def __init__(
        self,
        input_dim=2048,
        num_stages=7,
        window_size=30,
        num_heads=8,
        d_k=32,
        d_v=32,
        d_ff=32,
    ):
        super().__init__()
        values = {
            "input_dim": input_dim,
            "num_stages": num_stages,
            "window_size": window_size,
            "num_heads": num_heads,
            "d_k": d_k,
            "d_v": d_v,
            "d_ff": d_ff,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = input_dim
        self.num_stages = num_stages
        self.window_size = window_size
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.spatial_projection = nn.Linear(input_dim, num_stages, bias=False)
        self.temporal_q = nn.Linear(num_stages, num_heads * d_k, bias=False)
        self.temporal_k = nn.Linear(num_stages, num_heads * d_k, bias=False)
        self.temporal_v = nn.Linear(num_stages, num_heads * d_v, bias=False)
        self.temporal_attention_output = nn.Linear(num_heads * d_v, num_stages, bias=False)
        self.temporal_ff1 = nn.Linear(num_stages, d_ff, bias=False)
        self.temporal_ff2 = nn.Linear(d_ff, num_stages, bias=False)
        self.temporal_norm1 = nn.LayerNorm(num_stages, elementwise_affine=False)
        self.temporal_norm2 = nn.LayerNorm(num_stages, elementwise_affine=False)
        self.spatial_q = nn.Linear(num_stages, num_heads * d_k, bias=False)
        self.spatial_k = nn.Linear(num_stages, num_heads * d_k, bias=False)
        self.spatial_v = nn.Linear(num_stages, num_heads * d_v, bias=False)
        self.spatial_attention_output = nn.Linear(num_heads * d_v, num_stages, bias=False)
        self.spatial_ff1 = nn.Linear(num_stages, d_ff, bias=False)
        self.spatial_ff2 = nn.Linear(d_ff, num_stages, bias=False)
        self.spatial_norm1 = nn.LayerNorm(num_stages, elementwise_affine=False)
        self.spatial_norm2 = nn.LayerNorm(num_stages, elementwise_affine=False)

    @staticmethod
    def _attention_weights(query, key):
        scores = torch.matmul(query, key.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
        return torch.softmax(scores, dim=-1)

    @classmethod
    def _scaled_dot_product_attention(cls, query, key, value):
        return torch.matmul(cls._attention_weights(query, key), value)

    def _project_spatial(self, spatial_features):
        return torch.tanh(self.spatial_projection(spatial_features))

    def _temporal_windows(self, temporal_embeddings):
        padded = torch.nn.functional.pad(
            temporal_embeddings.transpose(1, 2), (self.window_size - 1, 0)
        )
        return padded.transpose(1, 2).unfold(1, self.window_size, 1).permute(0, 1, 3, 2).contiguous()

    def _multi_head_attention(self, query, sequence, q_proj, k_proj, v_proj, output_proj):
        batch, query_length, _ = query.shape
        sequence_length = sequence.shape[1]
        projected_query = q_proj(query).view(batch, query_length, self.num_heads, self.d_k).transpose(1, 2)
        projected_key = k_proj(sequence).view(batch, sequence_length, self.num_heads, self.d_k).transpose(1, 2)
        projected_value = v_proj(sequence).view(batch, sequence_length, self.num_heads, self.d_v).transpose(1, 2)
        attended = self._scaled_dot_product_attention(projected_query, projected_key, projected_value)
        attended = attended.transpose(1, 2).contiguous().view(batch, query_length, self.num_heads * self.d_v)
        return output_proj(attended)

    def _transformer(
        self,
        query,
        sequence,
        q_proj,
        k_proj,
        v_proj,
        attention_output,
        ff1,
        ff2,
        norm1,
        norm2,
    ):
        attended = self._multi_head_attention(query, sequence, q_proj, k_proj, v_proj, attention_output)
        hidden = norm1(query + attended)
        return norm2(hidden + ff2(torch.relu(ff1(hidden))))

    def forward(self, spatial_features, temporal_embeddings):
        if not isinstance(spatial_features, torch.Tensor) or not isinstance(temporal_embeddings, torch.Tensor):
            raise TypeError("spatial_features and temporal_embeddings must be tensors")
        if spatial_features.ndim != 3 or temporal_embeddings.ndim != 3:
            raise ValueError("spatial_features and temporal_embeddings must have rank 3")
        if spatial_features.shape[0] <= 0 or spatial_features.shape[1] <= 0:
            raise ValueError("batch and time dimensions must be positive")
        if spatial_features.shape[:2] != temporal_embeddings.shape[:2]:
            raise ValueError("spatial_features and temporal_embeddings must have matching batch and time dimensions")
        if spatial_features.shape[-1] != self.input_dim:
            raise ValueError(f"spatial_features last dimension must equal input_dim={self.input_dim}")
        if temporal_embeddings.shape[-1] != self.num_stages:
            raise ValueError(f"temporal_embeddings last dimension must equal num_stages={self.num_stages}")
        if not spatial_features.is_floating_point() or not temporal_embeddings.is_floating_point():
            raise ValueError("spatial_features and temporal_embeddings must be floating point tensors")
        batch, time, _ = spatial_features.shape
        windows = self._temporal_windows(temporal_embeddings).reshape(
            batch * time, self.window_size, self.num_stages
        )
        temporal = self._transformer(
            windows,
            windows,
            self.temporal_q,
            self.temporal_k,
            self.temporal_v,
            self.temporal_attention_output,
            self.temporal_ff1,
            self.temporal_ff2,
            self.temporal_norm1,
            self.temporal_norm2,
        )
        query = self._project_spatial(spatial_features).reshape(batch * time, 1, self.num_stages)
        fused = self._transformer(
            query,
            temporal,
            self.spatial_q,
            self.spatial_k,
            self.spatial_v,
            self.spatial_attention_output,
            self.spatial_ff1,
            self.spatial_ff2,
            self.spatial_norm1,
            self.spatial_norm2,
        )
        return fused.reshape(batch, time, self.num_stages)
