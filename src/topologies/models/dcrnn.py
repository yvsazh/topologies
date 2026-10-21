import torch
from torch import nn


class DCRNN(nn.Module):
    architecture_id = "dcrnn"
    PRESETS = {
        "tiny": {"input_size": 1, "hidden_size": 8, "output_size": 1, "max_diffusion_step": 1},
        "small": {"input_size": 1, "hidden_size": 32, "output_size": 1, "max_diffusion_step": 2},
        "base": {"input_size": 1, "hidden_size": 64, "output_size": 1, "max_diffusion_step": 2},
    }

    def __init__(self, input_size, hidden_size, output_size=None, max_diffusion_step=2, adjacency=None):
        super().__init__()
        for name, value in (("input_size", input_size), ("hidden_size", hidden_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if output_size is not None and (
            not isinstance(output_size, int) or isinstance(output_size, bool) or output_size < 1
        ):
            raise ValueError("output_size must be None or a positive integer")
        if (
            not isinstance(max_diffusion_step, int)
            or isinstance(max_diffusion_step, bool)
            or max_diffusion_step < 0
        ):
            raise ValueError("max_diffusion_step must be a nonnegative integer")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_diffusion_step = max_diffusion_step
        self.kernel_size = 1 + 2 * max_diffusion_step
        if adjacency is None:
            self.register_buffer("_adjacency", None)
        else:
            self.register_buffer("_adjacency", self._validated_adjacency(adjacency))
        gate_features = (input_size + hidden_size) * self.kernel_size
        candidate_features = (input_size + hidden_size) * self.kernel_size
        self.gates = nn.Linear(gate_features, 2 * hidden_size)
        self.candidate = nn.Linear(candidate_features, hidden_size)
        with torch.no_grad():
            self.gates.bias.fill_(1.0)
            self.candidate.bias.zero_()
        self.projection = (
            nn.Linear(hidden_size, output_size, bias=False) if output_size is not None else None
        )

    @staticmethod
    def _validated_adjacency(adjacency):
        try:
            adjacency = torch.as_tensor(adjacency)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("adjacency must be a finite square matrix") from exc
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if not torch.is_floating_point(adjacency):
            adjacency = adjacency.to(dtype=torch.get_default_dtype())
        if not torch.isfinite(adjacency).all():
            raise ValueError("adjacency must contain only finite values")
        return adjacency

    @staticmethod
    def random_walk_supports(adjacency):
        adjacency = DCRNN._validated_adjacency(adjacency)
        reverse_adjacency = adjacency.transpose(0, 1)
        forward_degree = adjacency.sum(dim=1, keepdim=True)
        reverse_degree = reverse_adjacency.sum(dim=1, keepdim=True)
        forward_safe_degree = torch.where(forward_degree != 0, forward_degree, 1)
        reverse_safe_degree = torch.where(reverse_degree != 0, reverse_degree, 1)
        forward_inverse = torch.where(forward_degree != 0, forward_safe_degree.reciprocal(), 0)
        reverse_inverse = torch.where(reverse_degree != 0, reverse_safe_degree.reciprocal(), 0)
        return forward_inverse * adjacency, reverse_inverse * reverse_adjacency

    @staticmethod
    def diffusion_features(values, forward_support, reverse_support, max_diffusion_step):
        if not isinstance(values, torch.Tensor) or values.ndim not in (2, 3):
            raise ValueError("values must have shape [N,F] or [B,N,F]")
        if (
            not isinstance(max_diffusion_step, int)
            or isinstance(max_diffusion_step, bool)
            or max_diffusion_step < 0
        ):
            raise ValueError("max_diffusion_step must be a nonnegative integer")
        node_count = values.shape[-2]
        for name, support in (("forward_support", forward_support), ("reverse_support", reverse_support)):
            if not isinstance(support, torch.Tensor) or support.shape != (node_count, node_count):
                raise ValueError(f"{name} must have shape [N,N] matching values")
        features = [values]
        forward_values = values
        reverse_values = values
        for _ in range(max_diffusion_step):
            forward_values = torch.matmul(forward_support, forward_values)
            reverse_values = torch.matmul(reverse_support, reverse_values)
            features.extend((forward_values, reverse_values))
        return torch.cat(features, dim=-1)

    def _resolve_supports(self, adjacency, node_count, device, dtype):
        graph = self._adjacency if adjacency is None else adjacency
        if graph is None:
            raise ValueError("adjacency must be supplied at construction or in forward")
        graph = self._validated_adjacency(graph)
        if graph.shape != (node_count, node_count):
            raise ValueError("adjacency shape must match the input node count")
        forward, reverse = self.random_walk_supports(graph)
        return forward.to(device=device, dtype=dtype), reverse.to(device=device, dtype=dtype)

    def _dcgru_components(self, inputs, previous_state, forward_support, reverse_support):
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("step inputs must have shape [B,N,input_size]")
        if previous_state.shape != (*inputs.shape[:2], self.hidden_size):
            raise ValueError("previous_state must have shape [B,N,hidden_size]")
        gate_input = torch.cat((inputs, previous_state), dim=-1)
        gate_features = self.diffusion_features(
            gate_input, forward_support, reverse_support, self.max_diffusion_step
        )
        reset, update = torch.sigmoid(self.gates(gate_features)).chunk(2, dim=-1)
        candidate_input = torch.cat((inputs, reset * previous_state), dim=-1)
        candidate_features = self.diffusion_features(
            candidate_input, forward_support, reverse_support, self.max_diffusion_step
        )
        candidate = torch.tanh(self.candidate(candidate_features))
        state = update * previous_state + (1.0 - update) * candidate
        return state, reset, update, candidate

    def dcgru_step(self, inputs, previous_state, adjacency=None):
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3:
            raise ValueError("step inputs must have shape [B,N,input_size]")
        forward, reverse = self._resolve_supports(
            adjacency, inputs.shape[1], inputs.device, inputs.dtype
        )
        return self._dcgru_components(inputs, previous_state, forward, reverse)[0]

    def forward(self, inputs, adjacency=None, initial_state=None):
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:
            raise ValueError("inputs must have shape [B,T,N,F]")
        if inputs.shape[-1] != self.input_size:
            raise ValueError("input feature width must match input_size")
        if inputs.shape[0] < 1 or inputs.shape[1] < 1 or inputs.shape[2] < 1:
            raise ValueError("batch, sequence, and node dimensions must be nonempty")
        if not torch.is_floating_point(inputs):
            raise ValueError("inputs must have a floating point dtype")
        forward, reverse = self._resolve_supports(
            adjacency, inputs.shape[2], inputs.device, inputs.dtype
        )
        expected_state_shape = (inputs.shape[0], inputs.shape[2], self.hidden_size)
        if initial_state is None:
            state = inputs.new_zeros(expected_state_shape)
        else:
            if not isinstance(initial_state, torch.Tensor) or initial_state.shape != expected_state_shape:
                raise ValueError("initial_state must have shape [B,N,hidden_size]")
            if not torch.is_floating_point(initial_state):
                raise ValueError("initial_state must have a floating point dtype")
            state = initial_state
        outputs = []
        for time_index in range(inputs.shape[1]):
            state = self._dcgru_components(
                inputs[:, time_index], state, forward, reverse
            )[0]
            outputs.append(self.projection(state) if self.projection is not None else state)
        return torch.stack(outputs, dim=1)
