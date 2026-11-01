import math

import torch
from torch import nn
from torch.nn import functional as F

PRESETS = {
    "tiny": {
        "node_classes": 3,
        "edge_classes": 3,
        "n_layers": 1,
        "node_hidden_dim": 32,
        "edge_hidden_dim": 16,
        "global_hidden_dim": 16,
        "n_heads": 4,
        "input_mlp_dim": 32,
        "node_ff_dim": 64,
        "edge_ff_dim": 32,
        "global_ff_dim": 2048,
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
    },
    "small": {
        "node_classes": 3,
        "edge_classes": 3,
        "n_layers": 3,
        "node_hidden_dim": 64,
        "edge_hidden_dim": 32,
        "global_hidden_dim": 32,
        "n_heads": 4,
        "input_mlp_dim": 64,
        "node_ff_dim": 128,
        "edge_ff_dim": 64,
        "global_ff_dim": 2048,
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
    },
    "base": {
        "node_classes": 3,
        "edge_classes": 3,
        "n_layers": 6,
        "node_hidden_dim": 128,
        "edge_hidden_dim": 64,
        "global_hidden_dim": 64,
        "n_heads": 8,
        "input_mlp_dim": 128,
        "node_ff_dim": 256,
        "edge_ff_dim": 128,
        "global_ff_dim": 2048,
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
    },
}


class DiGress(nn.Module):
    architecture_id = "digress"
    PRESETS = PRESETS

    def __init__(
        self,
        node_classes,
        edge_classes,
        n_layers=1,
        node_hidden_dim=32,
        edge_hidden_dim=16,
        global_hidden_dim=16,
        n_heads=4,
        input_mlp_dim=32,
        node_ff_dim=64,
        edge_ff_dim=32,
        global_ff_dim=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        integer_arguments = {
            "node_classes": node_classes,
            "edge_classes": edge_classes,
            "n_layers": n_layers,
            "node_hidden_dim": node_hidden_dim,
            "edge_hidden_dim": edge_hidden_dim,
            "global_hidden_dim": global_hidden_dim,
            "n_heads": n_heads,
            "input_mlp_dim": input_mlp_dim,
            "node_ff_dim": node_ff_dim,
            "edge_ff_dim": edge_ff_dim,
            "global_ff_dim": global_ff_dim,
        }
        for name, value in integer_arguments.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if node_hidden_dim % n_heads != 0:
            raise ValueError("node_hidden_dim must be divisible by n_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if isinstance(layer_norm_eps, bool) or not isinstance(layer_norm_eps, (int, float)) or not math.isfinite(layer_norm_eps) or layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be finite and positive")
        self.node_classes = node_classes
        self.edge_classes = edge_classes
        self.n_layers = n_layers
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.global_hidden_dim = global_hidden_dim
        self.n_heads = n_heads
        self.head_dim = node_hidden_dim // n_heads
        self.dropout_probability = float(dropout)
        self.node_input = nn.Sequential(
            nn.Linear(node_classes, input_mlp_dim),
            nn.ReLU(),
            nn.Linear(input_mlp_dim, node_hidden_dim),
        )
        self.edge_input = nn.Sequential(
            nn.Linear(edge_classes, input_mlp_dim),
            nn.ReLU(),
            nn.Linear(input_mlp_dim, edge_hidden_dim),
        )
        self.time_input = nn.Linear(1, global_hidden_dim)
        self.query_projections = nn.ModuleList(nn.Linear(node_hidden_dim, node_hidden_dim) for _ in range(n_layers))
        self.key_projections = nn.ModuleList(nn.Linear(node_hidden_dim, node_hidden_dim) for _ in range(n_layers))
        self.value_projections = nn.ModuleList(nn.Linear(node_hidden_dim, node_hidden_dim) for _ in range(n_layers))
        self.edge_attention_scales = nn.ModuleList(nn.Linear(edge_hidden_dim, node_hidden_dim) for _ in range(n_layers))
        self.edge_attention_shifts = nn.ModuleList(nn.Linear(edge_hidden_dim, node_hidden_dim) for _ in range(n_layers))
        self.node_attention_outputs = nn.ModuleList(nn.Linear(node_hidden_dim, node_hidden_dim) for _ in range(n_layers))
        self.node_global_scales = nn.ModuleList(nn.Linear(global_hidden_dim, node_hidden_dim) for _ in range(n_layers))
        self.node_global_shifts = nn.ModuleList(nn.Linear(global_hidden_dim, node_hidden_dim) for _ in range(n_layers))
        edge_update_dim = edge_hidden_dim + 2 * node_hidden_dim + global_hidden_dim + 1
        self.edge_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(edge_update_dim, edge_ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_ff_dim, edge_hidden_dim),
            )
            for _ in range(n_layers)
        )
        pooled_dim = 4 * (node_hidden_dim + edge_hidden_dim) + global_hidden_dim
        self.global_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(pooled_dim, global_ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(global_ff_dim, global_hidden_dim),
            )
            for _ in range(n_layers)
        )
        self.node_norms_attention = nn.ModuleList(nn.LayerNorm(node_hidden_dim, eps=layer_norm_eps) for _ in range(n_layers))
        self.node_norms_feedforward = nn.ModuleList(nn.LayerNorm(node_hidden_dim, eps=layer_norm_eps) for _ in range(n_layers))
        self.edge_norms_attention = nn.ModuleList(nn.LayerNorm(edge_hidden_dim, eps=layer_norm_eps) for _ in range(n_layers))
        self.edge_norms_feedforward = nn.ModuleList(nn.LayerNorm(edge_hidden_dim, eps=layer_norm_eps) for _ in range(n_layers))
        self.global_norms_attention = nn.ModuleList(nn.LayerNorm(global_hidden_dim, eps=layer_norm_eps) for _ in range(n_layers))
        self.global_norms_feedforward = nn.ModuleList(nn.LayerNorm(global_hidden_dim, eps=layer_norm_eps) for _ in range(n_layers))
        self.node_feedforwards = nn.ModuleList(
            nn.Sequential(
                nn.Linear(node_hidden_dim, node_ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_ff_dim, node_hidden_dim),
            )
            for _ in range(n_layers)
        )
        self.edge_feedforwards = nn.ModuleList(
            nn.Sequential(
                nn.Linear(edge_hidden_dim, edge_ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_ff_dim, edge_hidden_dim),
            )
            for _ in range(n_layers)
        )
        self.global_feedforwards = nn.ModuleList(
            nn.Sequential(
                nn.Linear(global_hidden_dim, global_ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(global_ff_dim, global_hidden_dim),
            )
            for _ in range(n_layers)
        )
        self.node_dropout = nn.Dropout(dropout)
        self.edge_dropout = nn.Dropout(dropout)
        self.global_dropout = nn.Dropout(dropout)
        self.node_classifier = nn.Linear(node_hidden_dim, node_classes)
        self.edge_classifier = nn.Linear(edge_hidden_dim, edge_classes)
        self.edge_global_output_scale = nn.Linear(global_hidden_dim, edge_hidden_dim)
        self.edge_global_output_shift = nn.Linear(global_hidden_dim, edge_hidden_dim)

    @staticmethod
    def _scalar_probability(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return float(value)

    @staticmethod
    def marginal_transition_probs(clean, alpha_bar, marginals):
        if not isinstance(clean, torch.Tensor) or clean.ndim < 1 or clean.shape[-1] <= 0:
            raise ValueError("clean must be a tensor with a nonempty class dimension")
        if not clean.is_floating_point() or not bool(torch.isfinite(clean).all()):
            raise ValueError("clean must contain finite floating-point probabilities")
        if bool((clean < 0).any()) or not torch.allclose(
            clean.sum(dim=-1), torch.ones_like(clean[..., 0]), atol=1e-6, rtol=1e-6
        ):
            raise ValueError("clean categorical features must be nonnegative and sum to one")
        alpha_bar = DiGress._scalar_probability(alpha_bar, "alpha_bar")
        classes = clean.shape[-1]
        if marginals is None:
            raise ValueError("marginals must be supplied by the caller")
        try:
            marginal = torch.as_tensor(marginals, device=clean.device, dtype=clean.dtype)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("marginals must be a probability vector with one value per class") from exc
        if marginal.shape != (classes,) or not bool(torch.isfinite(marginal).all()) or bool((marginal < 0).any()):
            raise ValueError("marginals must be finite, nonnegative, and have one value per class")
        if not torch.allclose(marginal.sum(), marginal.new_tensor(1.0), atol=1e-6, rtol=1e-6):
            raise ValueError("marginals must sum to one")
        return alpha_bar * clean + (1.0 - alpha_bar) * marginal

    @staticmethod
    def _sample_categories(probabilities):
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
        return F.one_hot(sampled, num_classes=probabilities.shape[-1]).reshape(probabilities.shape).to(probabilities.dtype)

    def noise_graph(self, clean_nodes, clean_edges, alpha_bar, node_marginals, edge_marginals):
        if not isinstance(clean_nodes, torch.Tensor) or clean_nodes.ndim != 3 or clean_nodes.shape[-1] != self.node_classes:
            raise ValueError("clean_nodes must have shape [batch, nodes, node_classes]")
        if not isinstance(clean_edges, torch.Tensor) or clean_edges.ndim != 4 or clean_edges.shape[-1] != self.edge_classes:
            raise ValueError("clean_edges must have shape [batch, nodes, nodes, edge_classes]")
        batch, node_count = clean_nodes.shape[:2]
        if clean_edges.shape[:3] != (batch, node_count, node_count):
            raise ValueError("clean_edges must match the batch and node dimensions of clean_nodes")
        if batch <= 0 or node_count <= 0:
            raise ValueError("clean graphs must have a nonempty batch and at least one node")
        if clean_nodes.device != clean_edges.device or clean_nodes.dtype != clean_edges.dtype:
            raise ValueError("clean_nodes and clean_edges must share device and dtype")
        if not clean_nodes.is_floating_point() or not clean_edges.is_floating_point():
            raise ValueError("clean node and edge features must be floating-point tensors")
        if not bool(torch.isfinite(clean_nodes).all()) or not bool(torch.isfinite(clean_edges).all()):
            raise ValueError("clean node and edge features must be finite")
        if not torch.allclose(clean_edges, clean_edges.transpose(1, 2), atol=1e-6, rtol=1e-5):
            raise ValueError("clean_edges must be symmetric for undirected graph noising")
        alpha_bar = self._scalar_probability(alpha_bar, "alpha_bar")
        node_probs = self.marginal_transition_probs(clean_nodes, alpha_bar, node_marginals)
        noisy_nodes = self._sample_categories(node_probs)
        noisy_edges = clean_edges.clone()
        upper_i, upper_j = torch.triu_indices(node_count, node_count, offset=1, device=clean_edges.device)
        if upper_i.numel():
            upper_clean = clean_edges[:, upper_i, upper_j]
            upper_probs = self.marginal_transition_probs(upper_clean, alpha_bar, edge_marginals)
            noisy_upper = self._sample_categories(upper_probs)
            noisy_edges[:, upper_i, upper_j] = noisy_upper
            noisy_edges[:, upper_j, upper_i] = noisy_upper
        else:
            self.marginal_transition_probs(clean_edges[:, 0, 0], alpha_bar, edge_marginals)
        return noisy_nodes, noisy_edges

    @staticmethod
    def _film(values, global_features, scale_projection, shift_projection):
        scale = scale_projection(global_features)
        shift = shift_projection(global_features)
        while scale.ndim < values.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return shift + (scale + 1.0) * values

    @staticmethod
    def _residual_ff(stream, update, attention_norm, feedforward, feedforward_norm, dropout):
        hidden = attention_norm(stream + dropout(update))
        return feedforward_norm(hidden + dropout(feedforward(hidden)))

    @staticmethod
    def _pool_statistics(values, mask):
        expanded_mask = mask.unsqueeze(-1)
        counts = expanded_mask.sum(dim=1).clamp_min(1).to(values.dtype)
        masked_values = values.masked_fill(~expanded_mask, 0.0)
        mean = masked_values.sum(dim=1) / counts
        variance = ((values - mean.unsqueeze(1)).square() * expanded_mask.to(values.dtype)).sum(dim=1) / counts
        standard_deviation = variance.clamp_min(1e-12).sqrt()
        maximum = values.masked_fill(~expanded_mask, -torch.inf).amax(dim=1)
        minimum = values.masked_fill(~expanded_mask, torch.inf).amin(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
        return torch.cat((maximum, minimum, mean, standard_deviation), dim=-1)

    @staticmethod
    def denoising_loss(node_logits, edge_logits, clean_nodes, clean_edges, edge_loss_weight, node_mask=None):
        if not isinstance(node_logits, torch.Tensor) or node_logits.ndim != 3:
            raise ValueError("node_logits must have shape [batch, nodes, node_classes]")
        if not isinstance(edge_logits, torch.Tensor) or edge_logits.ndim != 4:
            raise ValueError("edge_logits must have shape [batch, nodes, nodes, edge_classes]")
        batch, node_count, _ = node_logits.shape
        if batch <= 0 or node_count <= 0 or node_logits.shape[-1] <= 0 or edge_logits.shape[-1] <= 0:
            raise ValueError("logits must have nonempty batch, node, and class dimensions")
        if edge_logits.shape[:3] != (batch, node_count, node_count):
            raise ValueError("edge_logits must match the batch and node dimensions of node_logits")
        if clean_nodes.shape != node_logits.shape or clean_edges.shape != edge_logits.shape:
            raise ValueError("clean categorical features must match their corresponding logits")
        edge_loss_weight = float(edge_loss_weight) if isinstance(edge_loss_weight, (int, float)) and not isinstance(edge_loss_weight, bool) else math.nan
        if not math.isfinite(edge_loss_weight) or edge_loss_weight <= 0:
            raise ValueError("edge_loss_weight must be finite and positive")
        if node_mask is None:
            valid_nodes = torch.ones((batch, node_count), dtype=torch.bool, device=node_logits.device)
        else:
            if not isinstance(node_mask, torch.Tensor) or node_mask.shape != (batch, node_count) or node_mask.dtype != torch.bool:
                raise ValueError("node_mask must be boolean with shape [batch, nodes]")
            if node_mask.device != node_logits.device or bool((~node_mask.any(dim=1)).any()):
                raise ValueError("node_mask must share the logits device and contain a valid node per graph")
            valid_nodes = node_mask
        node_targets = clean_nodes.argmax(dim=-1)
        node_losses = F.cross_entropy(node_logits.reshape(-1, node_logits.shape[-1]), node_targets.reshape(-1), reduction="none")
        node_losses = node_losses.reshape(batch, node_count) * valid_nodes.to(node_logits.dtype)
        edge_targets = clean_edges.argmax(dim=-1)
        edge_losses = F.cross_entropy(edge_logits.reshape(-1, edge_logits.shape[-1]), edge_targets.reshape(-1), reduction="none")
        edge_losses = edge_losses.reshape(batch, node_count, node_count)
        valid_edges = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        valid_edges = valid_edges & ~torch.eye(node_count, dtype=torch.bool, device=node_logits.device).unsqueeze(0)
        edge_losses = edge_losses * valid_edges.to(edge_logits.dtype)
        return (node_losses.sum(dim=1) + edge_loss_weight * edge_losses.sum(dim=(1, 2))).mean()

    def forward(self, noisy_nodes, noisy_edges, time, node_mask=None):
        if not isinstance(noisy_nodes, torch.Tensor) or noisy_nodes.ndim != 3 or noisy_nodes.shape[-1] != self.node_classes:
            raise ValueError("noisy_nodes must have shape [batch, nodes, node_classes]")
        if not isinstance(noisy_edges, torch.Tensor) or noisy_edges.ndim != 4 or noisy_edges.shape[-1] != self.edge_classes:
            raise ValueError("noisy_edges must have shape [batch, nodes, nodes, edge_classes]")
        batch, node_count = noisy_nodes.shape[:2]
        if batch <= 0 or node_count <= 0 or noisy_edges.shape[:3] != (batch, node_count, node_count):
            raise ValueError("noisy_edges must have shape [batch, nodes, nodes, edge_classes]")
        if not noisy_nodes.is_floating_point() or not noisy_edges.is_floating_point():
            raise ValueError("noisy node and edge features must be floating-point tensors")
        if noisy_nodes.device != noisy_edges.device or noisy_nodes.dtype != noisy_edges.dtype:
            raise ValueError("noisy_nodes and noisy_edges must share device and dtype")
        if not bool(torch.isfinite(noisy_nodes).all()) or not bool(torch.isfinite(noisy_edges).all()):
            raise ValueError("noisy node and edge features must be finite")
        if not torch.allclose(noisy_edges, noisy_edges.transpose(1, 2), atol=1e-6, rtol=1e-5):
            raise ValueError("noisy_edges must be symmetric for undirected graphs")
        if not isinstance(time, torch.Tensor) or time.device != noisy_nodes.device or not time.is_floating_point():
            raise ValueError("time must be a floating-point tensor on the input device")
        if time.shape == (batch,):
            time = time.unsqueeze(-1)
        if time.shape != (batch, 1) or not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
            raise ValueError("time must have shape [batch] or [batch, 1] with values in [0, 1]")
        if node_mask is None:
            valid_nodes = torch.ones((batch, node_count), dtype=torch.bool, device=noisy_nodes.device)
        else:
            if not isinstance(node_mask, torch.Tensor) or node_mask.shape != (batch, node_count) or node_mask.dtype != torch.bool:
                raise ValueError("node_mask must be boolean with shape [batch, nodes]")
            if node_mask.device != noisy_nodes.device:
                raise ValueError("node_mask must share the input device")
            valid_nodes = node_mask
        if bool((~valid_nodes.any(dim=1)).any()):
            raise ValueError("each graph must contain at least one valid node")
        model_dtype = self.node_input[0].weight.dtype
        if noisy_nodes.dtype != model_dtype or noisy_edges.dtype != model_dtype or time.dtype != model_dtype:
            raise ValueError("inputs must use the model parameter dtype")
        valid_pair_mask = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        edge_off_diagonal = ~torch.eye(node_count, dtype=torch.bool, device=noisy_nodes.device).unsqueeze(0)
        valid_edges = valid_pair_mask & edge_off_diagonal
        nodes = self.node_input(noisy_nodes) * valid_nodes.unsqueeze(-1).to(noisy_nodes.dtype)
        edges = self.edge_input(noisy_edges) * valid_pair_mask.unsqueeze(-1).to(noisy_edges.dtype)
        global_features = self.time_input(time)
        head_count = self.n_heads
        head_dim = self.head_dim
        for layer in range(self.n_layers):
            query = self.query_projections[layer](nodes).reshape(batch, node_count, head_count, head_dim)
            key = self.key_projections[layer](nodes).reshape(batch, node_count, head_count, head_dim)
            value = self.value_projections[layer](nodes).reshape(batch, node_count, head_count, head_dim)
            scores = query.unsqueeze(2) * key.unsqueeze(1) / math.sqrt(head_dim)
            edge_scale = self.edge_attention_scales[layer](edges).reshape(batch, node_count, node_count, head_count, head_dim)
            edge_shift = self.edge_attention_shifts[layer](edges).reshape(batch, node_count, node_count, head_count, head_dim)
            scores = scores * (edge_scale + 1.0) + edge_shift
            scores = scores.masked_fill(~valid_nodes[:, None, :, None, None], torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=2)
            aggregate = (attention * value.unsqueeze(1)).sum(dim=2).reshape(batch, node_count, self.node_hidden_dim)
            aggregate = self.node_attention_outputs[layer](aggregate)
            aggregate = self._film(
                aggregate,
                global_features,
                self.node_global_scales[layer],
                self.node_global_shifts[layer],
            )
            node_update = aggregate * valid_nodes.unsqueeze(-1).to(aggregate.dtype)
            nodes = self._residual_ff(
                nodes,
                node_update,
                self.node_norms_attention[layer],
                self.node_feedforwards[layer],
                self.node_norms_feedforward[layer],
                self.node_dropout,
            )
            nodes = nodes * valid_nodes.unsqueeze(-1).to(nodes.dtype)
            attention_features = scores.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
            attention_features = torch.where(valid_pair_mask.unsqueeze(-1), attention_features, torch.zeros_like(attention_features))
            left = nodes.unsqueeze(2).expand(-1, -1, node_count, -1)
            right = nodes.unsqueeze(1).expand(-1, node_count, -1, -1)
            pair_features = torch.cat(
                (
                    edges,
                    left + right,
                    left * right,
                    global_features[:, None, None, :].expand(-1, node_count, node_count, -1),
                    attention_features,
                ),
                dim=-1,
            )
            edge_update = self.edge_updates[layer](pair_features)
            edges = self._residual_ff(
                edges,
                edge_update,
                self.edge_norms_attention[layer],
                self.edge_feedforwards[layer],
                self.edge_norms_feedforward[layer],
                self.edge_dropout,
            )
            edges = 0.5 * (edges + edges.transpose(1, 2))
            edges = edges * valid_pair_mask.unsqueeze(-1).to(edges.dtype)
            node_pool_mask = valid_nodes
            edge_pool_mask = valid_edges
            node_pool = self._pool_statistics(nodes, node_pool_mask)
            edge_pool = self._pool_statistics(edges.reshape(batch, node_count * node_count, self.edge_hidden_dim), edge_pool_mask.reshape(batch, node_count * node_count))
            pooled = torch.cat((node_pool, edge_pool, global_features), dim=-1)
            global_update = self.global_updates[layer](pooled)
            global_features = self._residual_ff(
                global_features.unsqueeze(1),
                global_update.unsqueeze(1),
                self.global_norms_attention[layer],
                self.global_feedforwards[layer],
                self.global_norms_feedforward[layer],
                self.global_dropout,
            ).squeeze(1)
        output_nodes = self._film(nodes, global_features, self.node_global_scales[-1], self.node_global_shifts[-1])
        output_edges = self._film(edges, global_features, self.edge_global_output_scale, self.edge_global_output_shift)
        node_logits = self.node_classifier(output_nodes) * valid_nodes.unsqueeze(-1).to(nodes.dtype)
        edge_logits = self.edge_classifier(output_edges)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        edge_logits = edge_logits * valid_pair_mask.unsqueeze(-1).to(edge_logits.dtype)
        edge_logits = edge_logits.masked_fill(~edge_off_diagonal.unsqueeze(-1), 0.0)
        return node_logits, edge_logits
