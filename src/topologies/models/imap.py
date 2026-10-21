import torch
from torch import nn


class iMAP(nn.Module):
    architecture_id = "imap"
    def __init__(self, hidden_dim=32, depth=3):
        super().__init__()
        if hidden_dim < 1 or depth < 1:
            raise ValueError("hidden_dim and depth must be positive")
        layers = []
        input_dim = 3
        for _ in range(depth - 1):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.extend((nn.Linear(input_dim, 3), nn.Sigmoid()))
        self.color_mlp = nn.Sequential(*layers)

    def forward(self, pixel_coordinates, camera_pose, ray_depth=None):
        if pixel_coordinates.ndim != 2 or pixel_coordinates.shape[-1] != 2:
            raise ValueError("pixel_coordinates must have shape (batch, 2)")
        if camera_pose.ndim != 3 or camera_pose.shape[-2:] != (4, 4):
            raise ValueError("camera_pose must have shape (batch, 4, 4)")
        if camera_pose.shape[0] != pixel_coordinates.shape[0]:
            raise ValueError("pixel_coordinates and camera_pose batch sizes must match")
        pixels = pixel_coordinates.to(dtype=self.color_mlp[0].weight.dtype)
        pose = camera_pose.to(device=pixels.device, dtype=pixels.dtype)
        rays = torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=-1)
        if ray_depth is not None:
            depth_values = torch.as_tensor(ray_depth, device=pixels.device, dtype=pixels.dtype)
            if depth_values.ndim == 0:
                depth_values = depth_values.expand(pixels.shape[0])
            depth_values = depth_values.reshape(-1, 1)
            if depth_values.shape[0] != pixels.shape[0]:
                raise ValueError("ray_depth must be scalar or have one value per batch item")
            rays = rays * depth_values
        world_points = torch.bmm(pose[:, :3, :3], rays.unsqueeze(-1)).squeeze(-1) + pose[:, :3, 3]
        return self.color_mlp(world_points)
