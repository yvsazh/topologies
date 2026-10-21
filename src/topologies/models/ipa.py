import math

import torch
from torch import nn
from torch.nn import functional as F


def point_attention_logits(query_points, key_points, raw_point_weights):
    point_count = query_points.shape[-2]
    point_scale = math.sqrt(1.0 / (3.0 * point_count * 4.5))
    weights = F.softplus(raw_point_weights) * point_scale
    differences = query_points[:, :, None] - key_points[:, None, :]
    squared_distance = differences.square().sum(dim=(-1, -2))
    return -0.5 * squared_distance * weights[None, None, None, :]


class IPA(nn.Module):
    architecture_id = "ipa"
    def __init__(self, c_s, c_z, heads=4, scalar_qk=16, points=4):
        super().__init__()
        if min(c_s, c_z, heads, scalar_qk, points) <= 0:
            raise ValueError("dimensions must be positive")
        self.heads = heads
        self.scalar_qk = scalar_qk
        self.points = points
        self.scalar_scale = math.sqrt(1.0 / (3.0 * scalar_qk))
        self.pair_scale = math.sqrt(1.0 / 3.0)
        self.point_weights = nn.Parameter(torch.full((heads,), math.log(math.expm1(1.0))))
        self.q = nn.Linear(c_s, heads * scalar_qk)
        self.k = nn.Linear(c_s, heads * scalar_qk)
        self.v = nn.Linear(c_s, heads * scalar_qk)
        self.qp = nn.Linear(c_s, heads * points * 3)
        self.kp = nn.Linear(c_s, heads * points * 3)
        self.vp = nn.Linear(c_s, heads * points * 3)
        self.pair_bias = nn.Linear(c_z, heads)
        self.output = nn.Linear(heads * (scalar_qk + points * 4 + c_z), c_s)

    def forward(self, single, pair, rotations, translations, mask=None):
        if single.ndim != 3 or pair.ndim != 4:
            raise ValueError("single and pair must have shapes (B,N,Cs) and (B,N,N,Cz)")
        batch, length, _ = single.shape
        if pair.shape[:3] != (batch, length, length):
            raise ValueError("pair shape does not match single representation")
        if rotations.shape != (batch, length, 3, 3) or translations.shape != (batch, length, 3):
            raise ValueError("frames must have shapes (B,N,3,3) and (B,N,3)")
        q = self.q(single).view(batch, length, self.heads, self.scalar_qk)
        k = self.k(single).view(batch, length, self.heads, self.scalar_qk)
        v = self.v(single).view(batch, length, self.heads, self.scalar_qk)
        q_local = self.qp(single).view(batch, length, self.heads, self.points, 3)
        k_local = self.kp(single).view(batch, length, self.heads, self.points, 3)
        v_local = self.vp(single).view(batch, length, self.heads, self.points, 3)
        q_world = torch.einsum("bnij,bnhpj->bnhpi", rotations, q_local) + translations[:, :, None, None, :]
        k_world = torch.einsum("bnij,bnhpj->bnhpi", rotations, k_local) + translations[:, :, None, None, :]
        v_world = torch.einsum("bnij,bnhpj->bnhpi", rotations, v_local) + translations[:, :, None, None, :]
        scalar_logits = torch.einsum("bihc,bjhc->bhij", q, k) * self.scalar_scale
        point_logits = point_attention_logits(q_world, k_world, self.point_weights).permute(0, 3, 1, 2)
        pair_features = pair
        pair_logits = self.pair_bias(pair_features).permute(0, 3, 1, 2) * self.pair_scale
        logits = scalar_logits + point_logits + pair_logits
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        scalar_out = torch.einsum("bhij,bjhc->bihc", attention, v)
        point_out = torch.einsum("bhij,bjhpc->bihpc", attention, v_world)
        local_out = torch.einsum("bnij,bnhpj->bnhpi", rotations.transpose(-1, -2), point_out - translations[:, :, None, None, :])
        point_norm = torch.linalg.vector_norm(local_out, dim=-1)
        pair_out = torch.einsum("bhij,bijc->bihc", attention, pair_features)
        combined = torch.cat((scalar_out.flatten(2), local_out.flatten(2), point_norm.flatten(2), pair_out.flatten(2)), dim=-1)
        return single + self.output(combined)
