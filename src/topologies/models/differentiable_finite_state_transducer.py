import torch
from torch import nn


def _matrix_prefix_scan(matrices):
    result = matrices
    stride = 1
    while stride < matrices.shape[1]:
        result = torch.cat(
            (result[:, :stride], result[:, stride:] @ result[:, :-stride]), dim=1
        )
        stride *= 2
    return result


class DFST(nn.Module):
    architecture_id = "differentiable-finite-state-transducer"
    PRESETS = {
        "tiny": {"input_size": 2, "symbol_size": 2, "motion_size": 2, "hidden_size": 4},
        "small": {"input_size": 4, "symbol_size": 4, "motion_size": 3, "hidden_size": 16},
    }

    def __init__(self, input_size, symbol_size, motion_size, hidden_size, scan="parallel"):
        super().__init__()
        dimensions = (input_size, symbol_size, motion_size, hidden_size)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in dimensions):
            raise ValueError("input_size, symbol_size, motion_size, and hidden_size must be positive integers")
        if not isinstance(scan, str) or scan not in ("parallel", "sequential"):
            raise ValueError("scan must be 'parallel' or 'sequential'")
        self.input_size = input_size
        self.symbol_size = symbol_size
        self.motion_size = motion_size
        self.hidden_size = hidden_size
        self.scan = scan
        self.A = nn.Parameter(torch.eye(hidden_size).expand(input_size, -1, -1).clone())
        self.B = nn.Parameter(torch.zeros(input_size, symbol_size, hidden_size))
        self.C = nn.Parameter(torch.zeros(input_size, motion_size, hidden_size))
        self.h0 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, tokens, state=None, mask=None):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must be a rank-2 integer tensor with shape (batch, time)")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if tokens.dtype not in integer_dtypes:
            raise ValueError("tokens must have an integer dtype")
        batch_size, sequence_length = tokens.shape
        if batch_size < 1:
            raise ValueError("tokens must have a nonempty batch dimension")
        if tokens.device != self.A.device:
            raise ValueError("tokens and model parameters must be on the same device")
        if sequence_length and (torch.any(tokens < 0) or torch.any(tokens >= self.input_size)):
            raise ValueError(f"token IDs must be in [0, {self.input_size})")
        if state is None:
            initial_state = self.h0.unsqueeze(0).expand(batch_size, -1)
        else:
            if not isinstance(state, torch.Tensor) or not state.is_floating_point():
                raise ValueError("state must be a floating-point tensor")
            if state.shape == (self.hidden_size,):
                state = state.unsqueeze(0).expand(batch_size, -1)
            if state.shape != (batch_size, self.hidden_size):
                raise ValueError(f"state must have shape ({self.hidden_size},) or ({batch_size}, {self.hidden_size})")
            initial_state = state.to(device=self.A.device, dtype=self.A.dtype)
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise ValueError("mask must be a boolean tensor with shape (batch, time)")
            if mask.shape != tokens.shape:
                raise ValueError("mask must have the same shape as tokens")
            mask = mask.to(device=self.A.device)
            if sequence_length > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
                raise ValueError("mask must describe a valid prefix followed by padding")
        if sequence_length == 0:
            return (
                self.A.new_empty((batch_size, 0, self.symbol_size)),
                self.A.new_empty((batch_size, 0, self.motion_size)),
                initial_state,
            )
        token_ids = tokens.long()
        transitions = self.A[token_ids]
        symbol_maps = self.B[token_ids]
        motion_maps = self.C[token_ids]
        if mask is not None:
            identity = torch.eye(self.hidden_size, device=self.A.device, dtype=self.A.dtype)
            transitions = torch.where(mask[:, :, None, None], transitions, identity)
            symbol_maps = symbol_maps.masked_fill(~mask[:, :, None, None], 0)
            motion_maps = motion_maps.masked_fill(~mask[:, :, None, None], 0)
        if self.scan == "parallel":
            prefixes = _matrix_prefix_scan(transitions)
            states = (prefixes @ initial_state[:, None, :, None]).squeeze(-1)
            previous_states = torch.cat((initial_state[:, None], states[:, :-1]), dim=1)
            symbols = (symbol_maps @ previous_states.unsqueeze(-1)).squeeze(-1)
            motions = (motion_maps @ previous_states.unsqueeze(-1)).squeeze(-1)
            return symbols, motions, states[:, -1]
        current_state = initial_state
        symbols = []
        motions = []
        for timestep in range(sequence_length):
            symbols.append((symbol_maps[:, timestep] @ current_state.unsqueeze(-1)).squeeze(-1))
            motions.append((motion_maps[:, timestep] @ current_state.unsqueeze(-1)).squeeze(-1))
            current_state = (transitions[:, timestep] @ current_state.unsqueeze(-1)).squeeze(-1)
        return torch.stack(symbols, dim=1), torch.stack(motions, dim=1), current_state
