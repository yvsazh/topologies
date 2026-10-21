import torch


class InfluenceProduct(torch.nn.Module):
    component_id = "influence"
    def __init__(self, damping=0.0):
        super().__init__()
        if damping < 0:
            raise ValueError("damping must be non-negative")
        self.damping = damping

    def forward(self, grad, hessian_diag):
        grad = torch.as_tensor(grad)
        hessian_diag = torch.as_tensor(hessian_diag, device=grad.device, dtype=grad.dtype)
        if hessian_diag.shape != grad.shape:
            raise ValueError("grad and hessian_diag must have the same shape")
        denominator = hessian_diag + self.damping
        if torch.any(denominator == 0):
            raise ValueError("hessian diagonal plus damping must be nonzero")
        return grad / denominator
