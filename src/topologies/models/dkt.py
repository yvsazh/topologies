import math
from numbers import Integral, Real

import torch
from torch import nn


class DKT(nn.Module):
    architecture_id = "dkt"
    PRESETS = {
        "tiny": {"num_exercises": 4, "hidden_dim": 8, "dropout": 0.0},
        "small": {"num_exercises": 100, "hidden_dim": 200, "dropout": 0.0},
    }

    def __init__(self, num_exercises, hidden_dim=200, dropout=0.0):
        super().__init__()
        if isinstance(num_exercises, bool) or not isinstance(num_exercises, Integral) or num_exercises <= 0:
            raise ValueError("num_exercises must be a positive integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, Integral) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not math.isfinite(dropout):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1)")
        self.num_exercises = int(num_exercises)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Linear(2 * self.num_exercises, self.hidden_dim)
        self.recurrent_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.initial_state = nn.Parameter(torch.zeros(self.hidden_dim))
        self.readout_projection = nn.Linear(self.hidden_dim, self.num_exercises)
        self.readout_dropout = nn.Dropout(float(dropout))

    def forward(self, x, next_exercise_ids=None):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor with rank 3")
        if x.ndim != 3:
            raise ValueError("x must have rank 3 with shape [batch, time, 2 * num_exercises]")
        if x.shape[-1] != 2 * self.num_exercises:
            raise ValueError("x final dimension must equal 2 * num_exercises")
        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("x batch and time dimensions must be positive")
        if not x.is_floating_point():
            raise ValueError("x must have a floating point dtype")
        hidden = self.initial_state.unsqueeze(0).expand(x.shape[0], -1)
        outputs = []
        for step in range(x.shape[1]):
            hidden = torch.tanh(self.input_projection(x[:, step]) + self.recurrent_projection(hidden))
            outputs.append(torch.sigmoid(self.readout_projection(self.readout_dropout(hidden))))
        probabilities = torch.stack(outputs, dim=1)
        if next_exercise_ids is None:
            return probabilities
        try:
            next_exercise_ids = torch.as_tensor(next_exercise_ids, device=probabilities.device)
        except (TypeError, ValueError) as error:
            raise ValueError("next_exercise_ids must be an integer tensor with shape [batch, time]") from error
        if next_exercise_ids.ndim != 2 or next_exercise_ids.shape != probabilities.shape[:2]:
            raise ValueError("next_exercise_ids must have shape [batch, time]")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if next_exercise_ids.dtype not in integer_dtypes:
            raise ValueError("next_exercise_ids must contain integer exercise IDs")
        if torch.any(next_exercise_ids < 0) or torch.any(next_exercise_ids >= self.num_exercises):
            raise ValueError("next_exercise_ids must be in [0, num_exercises)")
        return probabilities.gather(-1, next_exercise_ids.to(torch.long).unsqueeze(-1)).squeeze(-1)
