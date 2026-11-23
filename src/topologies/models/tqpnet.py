import math

import torch
from torch import nn
from torch.nn import functional as F


class TQPNet(nn.Module):
    architecture_id = "tqpnet"
    PRESETS = {
        "tiny": {"width": 8, "depth": 2, "refinement_steps": 1, "rho": 1.0},
        "small": {"width": 64, "depth": 3, "refinement_steps": 5, "rho": 1.0},
        "base": {"width": 256, "depth": 4, "refinement_steps": 10, "rho": 1.0},
    }

    def __init__(
        self,
        Q=((1.0,),),
        A=((1.0,),),
        parameter_dim=1,
        width=256,
        depth=4,
        refinement_steps=10,
        rho=1.0,
    ):
        super().__init__()
        for name, value in (
            ("parameter_dim", parameter_dim),
            ("width", width),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 2:
            raise ValueError("depth must be an integer of at least 2")
        if (
            not isinstance(refinement_steps, int)
            or isinstance(refinement_steps, bool)
            or refinement_steps < 0
        ):
            raise ValueError("refinement_steps must be a nonnegative integer")
        if (
            not isinstance(rho, (int, float))
            or isinstance(rho, bool)
            or not math.isfinite(rho)
            or rho <= 0
        ):
            raise ValueError("rho must be a finite positive number")
        try:
            q_matrix = torch.as_tensor(Q)
            a_matrix = torch.as_tensor(A)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("Q and A must be numeric matrices") from error
        if q_matrix.is_complex() or a_matrix.is_complex():
            raise ValueError("Q and A must contain real values")
        if q_matrix.dtype == torch.bool or a_matrix.dtype == torch.bool:
            raise ValueError("Q and A must contain real numeric values")
        q_matrix = q_matrix.to(device="cpu", dtype=torch.get_default_dtype()).detach().clone()
        a_matrix = a_matrix.to(device="cpu", dtype=q_matrix.dtype).detach().clone()
        if q_matrix.ndim != 2 or q_matrix.shape[0] != q_matrix.shape[1] or q_matrix.shape[0] < 1:
            raise ValueError("Q must be a nonempty square matrix")
        n = q_matrix.shape[0]
        if a_matrix.ndim != 2 or a_matrix.shape[0] < 1 or a_matrix.shape[1] != n:
            raise ValueError(f"A must have shape (m, {n}) with m positive")
        if not torch.isfinite(q_matrix).all() or not torch.isfinite(a_matrix).all():
            raise ValueError("Q and A must contain only finite values")
        scale = max(1.0, float(q_matrix.abs().max()))
        tolerance = 1e-6 * scale
        if not torch.allclose(q_matrix, q_matrix.T, atol=tolerance, rtol=1e-6):
            raise ValueError("Q must be symmetric")
        q_matrix = 0.5 * (q_matrix + q_matrix.T)
        if float(torch.linalg.eigvalsh(q_matrix).min()) < -tolerance:
            raise ValueError("Q must be positive semidefinite")
        row_squares = a_matrix.square().sum(dim=1)
        if torch.any(row_squares == 0):
            raise ValueError("every row of A must contain a nonzero entry")
        column_counts = torch.count_nonzero(a_matrix, dim=0).to(q_matrix.dtype)
        if torch.any(column_counts == 0):
            raise ValueError("every column of A must contain a nonzero entry")
        self.parameter_dim = parameter_dim
        self.width = width
        self.depth = depth
        self.refinement_steps = refinement_steps
        self.rho = float(rho)
        self.register_buffer("Q", q_matrix)
        self.register_buffer("A", a_matrix)
        self.register_buffer("H_V", torch.linalg.inv(q_matrix + self.rho * torch.diag(column_counts)))
        self.register_buffer("D_V", torch.diag(column_counts))
        self.register_buffer("inverse_D_U", row_squares.reciprocal())
        self.register_buffer("primal_state_matrix", self.rho * self.H_V @ self.D_V)
        self.register_buffer("dual_state_matrix", self.rho * self.H_V @ self.A.T)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(parameter_dim if index == 0 else width, width)
            for index in range(depth)
        )
        self.x_head = nn.Linear(width, n)
        self.beta_head = nn.Linear(width, a_matrix.shape[0])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _as_batch(self, value, name, width, batch_size=None):
        try:
            value = torch.as_tensor(value, device=self.Q.device, dtype=self.Q.dtype)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{name} must be a numeric tensor") from error
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] < 1:
            raise ValueError(f"{name} must have shape (batch, {width}) with batch positive")
        if batch_size is not None and value.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must match the other inputs")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return value

    def warm_start(self, p):
        p = self._as_batch(p, "p", self.parameter_dim)
        hidden = p
        for layer in self.hidden_layers:
            hidden = F.relu(layer(hidden))
        return self.x_head(hidden), F.softplus(self.beta_head(hidden))

    def admm_relu_step(self, x, beta, q, c):
        x = self._as_batch(x, "x", self.Q.shape[0])
        batch_size = x.shape[0]
        beta = self._as_batch(beta, "beta", self.A.shape[0], batch_size)
        q = self._as_batch(q, "q", self.Q.shape[0], batch_size)
        c = self._as_batch(c, "c", self.A.shape[0], batch_size)
        x_next = (
            x @ self.primal_state_matrix.T
            + beta @ self.dual_state_matrix.T
            - q @ self.H_V.T
        )
        dual_direction = (2.0 * x_next - x) @ self.A.T
        beta_next = torch.relu(
            beta - dual_direction * self.inverse_D_U + c * self.inverse_D_U
        )
        return x_next, beta_next

    def optimality_residuals(self, x, beta, q, c):
        x = self._as_batch(x, "x", self.Q.shape[0])
        batch_size = x.shape[0]
        beta = self._as_batch(beta, "beta", self.A.shape[0], batch_size)
        q = self._as_batch(q, "q", self.Q.shape[0], batch_size)
        c = self._as_batch(c, "c", self.A.shape[0], batch_size)
        r1 = x @ self.Q.T - beta @ self.A + q
        r2 = beta - torch.relu(beta - x @ self.A.T + c)
        return r1, r2

    def residual_loss(self, x, beta, q, c):
        r1, r2 = self.optimality_residuals(x, beta, q, c)
        squared_norm = r1.square().sum(dim=-1) + r2.square().sum(dim=-1)
        return torch.log1p(squared_norm).mean()

    def forward(self, p, q, c):
        p = self._as_batch(p, "p", self.parameter_dim)
        batch_size = p.shape[0]
        q = self._as_batch(q, "q", self.Q.shape[0], batch_size)
        c = self._as_batch(c, "c", self.A.shape[0], batch_size)
        x, beta = self.warm_start(p)
        for _ in range(self.refinement_steps):
            x, beta = self.admm_relu_step(x, beta, q, c)
        return x, beta
