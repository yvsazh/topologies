import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class BGRL(nn.Module):
    architecture_id = "bgrl"
    PRESETS = {
        "tiny": {"hidden_dim": 16, "output_dim": 16, "predictor_hidden_dim": 16},
        "small": {"hidden_dim": 512, "output_dim": 256, "predictor_hidden_dim": 512},
    }

    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        output_dim=256,
        predictor_hidden_dim=512,
        tau_base=0.99,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("output_dim", output_dim),
            ("predictor_hidden_dim", predictor_hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(tau_base, bool)
            or not isinstance(tau_base, (int, float))
            or not math.isfinite(tau_base)
            or not 0.0 <= tau_base < 1.0
        ):
            raise ValueError("tau_base must be finite and in [0, 1)")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.tau_base = float(tau_base)
        self.online_encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "linear": nn.Linear(input_dim, hidden_dim),
                        "norm": nn.LayerNorm(hidden_dim),
                        "activation": nn.PReLU(hidden_dim),
                    }
                ),
                nn.ModuleDict(
                    {
                        "linear": nn.Linear(hidden_dim, output_dim),
                        "norm": nn.LayerNorm(output_dim),
                        "activation": nn.PReLU(output_dim),
                    }
                ),
            ]
        )
        self.predictor = nn.ModuleDict(
            {
                "hidden": nn.Linear(output_dim, predictor_hidden_dim),
                "hidden_norm": nn.LayerNorm(predictor_hidden_dim),
                "hidden_activation": nn.PReLU(predictor_hidden_dim),
                "output": nn.Linear(predictor_hidden_dim, output_dim),
                "output_norm": nn.LayerNorm(output_dim),
                "output_activation": nn.PReLU(output_dim),
            }
        )
        self.online_encoder.apply(self._initialize)
        self.predictor.apply(self._initialize)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_graph_inputs(x, edge_index):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("x must be a nonempty [num_nodes, input_dim] tensor")
        if not x.is_floating_point():
            raise ValueError("x must have a floating point dtype")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must have an integer dtype")
        if edge_index.device != x.device:
            raise ValueError("edge_index and x must be on the same device")
        edge_index = edge_index.to(dtype=torch.long)
        if edge_index.numel() and (torch.any(edge_index < 0) or torch.any(edge_index >= x.shape[0])):
            raise ValueError("edge_index contains an index outside the node range")
        return edge_index

    @staticmethod
    def normalized_aggregate(x, edge_index):
        edge_index = BGRL._validate_graph_inputs(x, edge_index)
        num_nodes = x.shape[0]
        self_loops = torch.arange(num_nodes, device=x.device)
        sources = torch.cat((edge_index[0], self_loops))
        destinations = torch.cat((edge_index[1], self_loops))
        degree = x.new_zeros(num_nodes).index_add_(0, destinations, x.new_ones(destinations.shape[0]))
        normalization = degree.index_select(0, sources).rsqrt() * degree.index_select(
            0, destinations
        ).rsqrt()
        messages = x.index_select(0, sources) * normalization.unsqueeze(-1)
        return x.new_zeros(x.shape).index_add_(0, destinations, messages)

    @staticmethod
    def cosine_loss(prediction, target):
        if (
            not isinstance(prediction, torch.Tensor)
            or not isinstance(target, torch.Tensor)
            or prediction.ndim != 2
            or target.shape != prediction.shape
            or prediction.shape[0] == 0
        ):
            raise ValueError("prediction and target must have the same nonempty [N, D] shape")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("prediction and target must have floating point dtypes")
        if prediction.device != target.device:
            raise ValueError("prediction and target must be on the same device")
        similarities = F.cosine_similarity(prediction, target.detach(), dim=-1)
        return -2.0 * similarities.sum() / prediction.shape[0]

    def _validate_view(self, x, edge_index):
        edge_index = self._validate_graph_inputs(x, edge_index)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x must have feature width input_dim={self.input_dim}")
        parameter = next(self.online_encoder.parameters())
        if x.device != parameter.device:
            raise ValueError("x and the model parameters must be on the same device")
        if x.dtype != parameter.dtype:
            raise ValueError("x and the model parameters must have the same dtype")
        return edge_index

    def _encode(self, x, edge_index, encoder):
        hidden = x
        for block in encoder:
            aggregated = self.normalized_aggregate(hidden, edge_index)
            hidden = block["linear"](aggregated)
            hidden = block["norm"](hidden)
            hidden = block["activation"](hidden)
        return hidden

    def _predict(self, embeddings):
        hidden = self.predictor["hidden"](embeddings)
        hidden = self.predictor["hidden_norm"](hidden)
        hidden = self.predictor["hidden_activation"](hidden)
        prediction = self.predictor["output"](hidden)
        prediction = self.predictor["output_norm"](prediction)
        return self.predictor["output_activation"](prediction)

    def momentum_at(self, step, total_steps):
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if step > total_steps:
            raise ValueError("step must not exceed total_steps")
        return 1.0 - ((1.0 - self.tau_base) / 2.0) * (
            math.cos(step * math.pi / total_steps) + 1.0
        )

    @torch.no_grad()
    def update_target(self, tau):
        if (
            isinstance(tau, bool)
            or not isinstance(tau, (int, float))
            or not math.isfinite(tau)
            or not 0.0 <= tau <= 1.0
        ):
            raise ValueError("tau must be finite and in [0, 1]")
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target.mul_(tau).add_(online, alpha=1.0 - tau)

    def forward(self, x1, edge_index1, x2, edge_index2, return_details=False):
        edge_index1 = self._validate_view(x1, edge_index1)
        edge_index2 = self._validate_view(x2, edge_index2)
        if x1.shape[0] != x2.shape[0]:
            raise ValueError("the two graph views must have the same number of nodes")
        if x1.device != x2.device or x1.dtype != x2.dtype:
            raise ValueError("the two graph views must share a device and dtype")
        if not isinstance(return_details, bool):
            raise TypeError("return_details must be a boolean")
        online_embeddings_1 = self._encode(x1, edge_index1, self.online_encoder)
        online_embeddings_2 = self._encode(x2, edge_index2, self.online_encoder)
        with torch.no_grad():
            target_embeddings_1 = self._encode(x1, edge_index1, self.target_encoder)
            target_embeddings_2 = self._encode(x2, edge_index2, self.target_encoder)
        predictions_1 = self._predict(online_embeddings_1)
        predictions_2 = self._predict(online_embeddings_2)
        loss = self.cosine_loss(predictions_1, target_embeddings_2) + self.cosine_loss(
            predictions_2, target_embeddings_1
        )
        if not return_details:
            return loss
        return {
            "loss": loss,
            "online_embeddings_1": online_embeddings_1,
            "online_embeddings_2": online_embeddings_2,
            "target_embeddings_1": target_embeddings_1,
            "target_embeddings_2": target_embeddings_2,
            "predictions_1": predictions_1,
            "predictions_2": predictions_2,
        }
