import torch
from torch import nn
from torch.nn import functional as F


class CITRISTransitionPrior(nn.Module):
    component_id = "citris"
    PRESETS = {
        "tiny": {
            "num_latents": 2,
            "num_slots": 2,
            "hidden_per_var": 2,
            "imperfect_interventions": False,
            "lambda_reg": 0.01,
        },
        "small": {
            "num_latents": 8,
            "num_slots": 4,
            "hidden_per_var": 8,
            "imperfect_interventions": False,
            "lambda_reg": 0.01,
        },
        "base": {
            "num_latents": 32,
            "num_slots": 8,
            "hidden_per_var": 16,
            "imperfect_interventions": False,
            "lambda_reg": 0.01,
        },
    }

    def __init__(self, num_latents, num_slots, hidden_per_var=16,
                 imperfect_interventions=False, lambda_reg=0.01):
        super().__init__()
        for name, value in (("num_latents", num_latents), ("num_slots", num_slots),
                            ("hidden_per_var", hidden_per_var)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(imperfect_interventions) is not bool:
            raise ValueError("imperfect_interventions must be a boolean")
        if type(lambda_reg) not in (int, float):
            raise ValueError("lambda_reg must be a number in [0, 1]")
        if not 0.0 <= float(lambda_reg) <= 1.0:
            raise ValueError("lambda_reg must be a number in [0, 1]")
        self.num_latents = num_latents
        self.num_slots = num_slots
        self.hidden_per_var = hidden_per_var
        self.imperfect_interventions = imperfect_interventions
        self.lambda_reg = float(lambda_reg)
        hidden_size = num_latents * hidden_per_var
        self.context_projection = nn.Linear(num_latents, hidden_size)
        self.target_projection = nn.Linear(1, hidden_size)
        self.autoregressive_input = nn.Linear(2 * num_latents, hidden_size)
        self.autoregressive_hidden = nn.Linear(hidden_size, hidden_size)
        self.autoregressive_output = nn.Linear(hidden_size, 2 * num_latents)
        output_coordinates = torch.arange(num_latents).repeat_interleave(hidden_per_var)
        input_coordinates = torch.cat((torch.arange(num_latents), torch.arange(num_latents)))
        input_mask = input_coordinates.unsqueeze(0) < output_coordinates.unsqueeze(1)
        hidden_mask = output_coordinates.unsqueeze(0) <= output_coordinates.unsqueeze(1)
        output_hidden_coordinates = torch.arange(num_latents).repeat_interleave(hidden_per_var)
        output_coordinates = torch.arange(num_latents).repeat_interleave(2)
        output_mask = output_hidden_coordinates.unsqueeze(0) <= output_coordinates.unsqueeze(1)
        self.register_buffer("autoregressive_input_mask", input_mask.to(torch.float32))
        self.register_buffer("autoregressive_hidden_mask", hidden_mask.to(torch.float32))
        self.register_buffer("autoregressive_output_mask", output_mask.to(torch.float32))

    def _validate_assignments(self, assignments, batch_size):
        if not torch.is_tensor(assignments) or assignments.shape != (
                batch_size, self.num_latents, self.num_slots):
            raise ValueError("assignments must have shape (B, num_latents, num_slots)")
        if not assignments.is_floating_point():
            raise ValueError("assignments must be floating point")
        if not torch.all((assignments == 0) | (assignments == 1)):
            raise ValueError("assignments must contain hard one-hot rows")
        if not torch.all(assignments.sum(dim=-1) == 1):
            raise ValueError("assignments must contain hard one-hot rows")

    def forward(self, z_t, z_next, assignments, targets):
        if not torch.is_tensor(z_t) or z_t.ndim != 2 or z_t.shape[-1] != self.num_latents:
            raise ValueError("z_t must have shape (B, num_latents)")
        if not torch.is_tensor(z_next) or z_next.shape != z_t.shape:
            raise ValueError("z_next must have matching shape (B, num_latents)")
        if not z_t.is_floating_point() or not z_next.is_floating_point():
            raise ValueError("z_t and z_next must be floating point")
        if z_t.device != z_next.device or z_t.dtype != z_next.dtype:
            raise ValueError("z_t and z_next must share dtype and device")
        batch_size = z_t.shape[0]
        self._validate_assignments(assignments, batch_size)
        if not torch.is_tensor(targets) or targets.shape != (batch_size, self.num_slots):
            raise ValueError("targets must have shape (B, num_slots)")
        if not targets.is_floating_point():
            raise ValueError("targets must be floating point")
        if targets.device != z_t.device or targets.dtype != z_t.dtype:
            raise ValueError("targets, assignments, and latent tensors must share dtype and device")
        if assignments.device != z_t.device or assignments.dtype != z_t.dtype:
            raise ValueError("targets, assignments, and latent tensors must share dtype and device")
        if not torch.all(torch.isfinite(targets)) or not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("targets must contain binary intervention indicators")
        if not torch.all(targets[:, -1] == 0):
            raise ValueError("the no-target slot target must be zero")
        slot_assignments = assignments.transpose(1, 2)
        context = z_t.unsqueeze(1).expand(-1, self.num_slots, -1)
        if not self.imperfect_interventions:
            context = context * (1.0 - targets.unsqueeze(-1))
        context_hidden = self.context_projection(context).view(
            batch_size, self.num_slots, self.num_latents, self.hidden_per_var)
        target_hidden = self.target_projection(targets.unsqueeze(-1)).view(
            batch_size, self.num_slots, self.num_latents, self.hidden_per_var)
        autoregressive_input = torch.cat((z_next.unsqueeze(1) * slot_assignments,
                                          slot_assignments), dim=-1)
        autoregressive_hidden = F.linear(
            autoregressive_input,
            self.autoregressive_input.weight * self.autoregressive_input_mask,
            self.autoregressive_input.bias,
        ).view(batch_size, self.num_slots, self.num_latents, self.hidden_per_var)
        hidden = F.silu((context_hidden + target_hidden + autoregressive_hidden) / 2.0)
        hidden = F.linear(
            hidden.flatten(start_dim=-2),
            self.autoregressive_hidden.weight * self.autoregressive_hidden_mask,
            self.autoregressive_hidden.bias,
        ).view(batch_size, self.num_slots, self.num_latents, self.hidden_per_var)
        hidden = F.silu(hidden)
        parameters = F.linear(
            hidden.flatten(start_dim=-2),
            self.autoregressive_output.weight * self.autoregressive_output_mask,
            self.autoregressive_output.bias,
        )
        return parameters.view(batch_size, self.num_slots, self.num_latents, 2)

    def weighted_kl(self, q_mean, q_log_std, prior_parameters, assignments):
        if not torch.is_tensor(q_mean) or q_mean.ndim != 2 or q_mean.shape[-1] != self.num_latents:
            raise ValueError("q_mean must have shape (B, num_latents)")
        if not torch.is_tensor(q_log_std) or q_log_std.shape != q_mean.shape:
            raise ValueError("q_log_std must have matching shape (B, num_latents)")
        batch_size = q_mean.shape[0]
        self._validate_assignments(assignments, batch_size)
        expected = (batch_size, self.num_slots, self.num_latents, 2)
        if not torch.is_tensor(prior_parameters) or prior_parameters.shape != expected:
            raise ValueError("prior_parameters must have shape (B, num_slots, num_latents, 2)")
        if (q_mean.device != q_log_std.device or q_mean.device != prior_parameters.device
                or q_mean.device != assignments.device):
            raise ValueError("KL tensors must share a device")
        if (q_mean.dtype != q_log_std.dtype or q_mean.dtype != prior_parameters.dtype
                or q_mean.dtype != assignments.dtype):
            raise ValueError("KL tensors must share a dtype")
        prior_mean = prior_parameters[..., 0]
        prior_log_std = prior_parameters[..., 1]
        q_variance = torch.exp(2.0 * q_log_std).unsqueeze(1)
        difference = q_mean.unsqueeze(1) - prior_mean
        per_coordinate = (prior_log_std - q_log_std.unsqueeze(1)
                          + 0.5 * (q_variance + difference.square())
                          * torch.exp(-2.0 * prior_log_std) - 0.5)
        slot_weights = torch.ones(self.num_slots, dtype=q_mean.dtype, device=q_mean.device)
        slot_weights[-1] = 1.0 - self.lambda_reg
        selected = assignments.transpose(1, 2) * slot_weights.view(1, -1, 1)
        return (per_coordinate * selected).sum()
