import torch
from torch import nn
from torch.nn import functional as F


class SNGAN(nn.Module):
    architecture_id = "sngan"
    def __init__(self, in_features=4, out_features=4, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = self.weight.shape[1] ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def normalized_weight(self):
        weight = self.weight
        vector = F.normalize(torch.ones(weight.shape[1], device=weight.device, dtype=weight.dtype), dim=0)
        left = F.normalize(torch.mv(weight, vector), dim=0)
        right = F.normalize(torch.mv(weight.t(), left), dim=0)
        left = F.normalize(torch.mv(weight, right), dim=0)
        estimate = torch.dot(left, torch.mv(weight, right)).abs().clamp_min(torch.finfo(weight.dtype).eps)
        return weight / estimate

    def forward(self, inputs):
        return F.linear(inputs, self.normalized_weight(), self.bias)
