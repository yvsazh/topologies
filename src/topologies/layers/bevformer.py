import torch
from torch import nn
from torch.nn import functional as F


class BEVFormer(nn.Module):
    component_id = "bevformer"

    def __init__(self, num_queries=16, embed_dim=32, num_cameras=1,
                 num_heads=4, num_points=4):
        super().__init__()
        if num_queries < 1 or embed_dim < 1 or num_cameras < 1 or num_points < 1:
            raise ValueError("query, feature, camera, and sample counts must be positive")
        if num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by a positive num_heads")
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_heads = num_heads
        self.num_points = num_points
        self.bev_queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)
        self.value_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_norm = nn.LayerNorm(embed_dim)

    def spatial_cross_attention(self, query, camera_features, reference_points,
                                visibility_mask):
        if camera_features.ndim != 5:
            raise ValueError("camera_features must have shape (batch, cameras, height, width, dim)")
        batch, cameras, height, width, dim = camera_features.shape
        if cameras != self.num_cameras or dim != self.embed_dim:
            raise ValueError("camera count and feature dimension must match constructor configuration")
        if query.shape != (batch, self.num_queries, self.embed_dim):
            raise ValueError("query must match the configured batch, query count, and feature dimension")
        reference_points = torch.as_tensor(
            reference_points, device=camera_features.device, dtype=camera_features.dtype
        )
        visibility_mask = torch.as_tensor(
            visibility_mask, device=camera_features.device, dtype=torch.bool
        )
        if (reference_points.ndim != 5 or reference_points.shape[:3] !=
                (batch, self.num_queries, cameras) or reference_points.shape[-1] != 2 or
                reference_points.shape[3] < 1):
            raise ValueError("reference_points must have shape (batch, queries, cameras, references, 2)")
        expected_visibility = reference_points.shape[:-1]
        if visibility_mask.shape != expected_visibility:
            raise ValueError("visibility_mask must match the reference-point dimensions")

        references = reference_points.shape[3]
        head_dim = dim // self.num_heads
        offsets = self.sampling_offsets(query).reshape(
            batch, self.num_queries, self.num_heads, self.num_points, 2
        )
        point_weights = self.attention_weights(query).reshape(
            batch, self.num_queries, self.num_heads, self.num_points
        ).softmax(dim=-1)
        offset_scale = camera_features.new_tensor((2.0 / width, 2.0 / height))
        sampling_offsets = offsets * offset_scale
        camera_outputs = []
        camera_hits = []

        for camera in range(cameras):
            feature = camera_features[:, camera]
            values = self.value_projection(feature).permute(0, 3, 1, 2)
            values = values.reshape(batch, self.num_heads, head_dim, height, width)
            values = values[:, None].expand(
                batch, references, self.num_heads, head_dim, height, width
            ).reshape(batch * references * self.num_heads, head_dim, height, width)

            reference_grid = reference_points[:, :, camera]
            grid = reference_grid[:, :, :, None, None, :] * 2.0 - 1.0
            grid = grid + sampling_offsets[:, :, None]
            grid = grid.permute(0, 2, 3, 1, 4, 5).reshape(
                batch * references * self.num_heads,
                self.num_queries,
                self.num_points,
                2,
            )
            sampled = F.grid_sample(
                values, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            sampled = sampled.reshape(
                batch, references, self.num_heads, head_dim,
                self.num_queries, self.num_points,
            ).permute(0, 4, 1, 2, 5, 3)
            attended = (
                sampled * point_weights[:, :, None, :, :, None]
            ).sum(dim=4)
            attended = attended.reshape(batch, self.num_queries, references, dim)
            attended = self.output_projection(attended)

            visible = visibility_mask[:, :, camera]
            camera_outputs.append((attended * visible.unsqueeze(-1)).sum(dim=2))
            camera_hits.append(visible.any(dim=2))

        camera_outputs = torch.stack(camera_outputs, dim=2)
        camera_hits = torch.stack(camera_hits, dim=2)
        hit_count = camera_hits.sum(dim=2, keepdim=True).clamp_min(1)
        return camera_outputs.sum(dim=2) / hit_count

    def forward(self, camera_features, reference_points, visibility_mask):
        if camera_features.ndim != 5:
            raise ValueError("camera_features must have shape (batch, cameras, height, width, dim)")
        batch = camera_features.shape[0]
        query = self.bev_queries.unsqueeze(0).expand(batch, -1, -1)
        spatial = self.spatial_cross_attention(
            query, camera_features, reference_points, visibility_mask
        )
        return self.output_norm(query + spatial)
