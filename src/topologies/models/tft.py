import math
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F


def _raise_value_error(message):
    raise ValueError(message)


class TFT(nn.Module):
    architecture_id = "tft"
    PRESETS = {
        "tiny": {"d_model": 10, "num_heads": 1, "dropout": 0.1, "quantiles": (0.1, 0.5, 0.9)},
        "small": {"d_model": 40, "num_heads": 4, "dropout": 0.1, "quantiles": (0.1, 0.5, 0.9)},
        "base": {"d_model": 160, "num_heads": 4, "dropout": 0.1, "quantiles": (0.1, 0.5, 0.9)},
    }

    class GatedLinearUnit(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.gate = nn.Linear(input_dim, output_dim)
            self.value = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return torch.sigmoid(self.gate(x)) * self.value(x)

    class GatedResidualNetwork(nn.Module):
        def __init__(self, input_dim, output_dim, context_dim=0, hidden_dim=None, dropout=0.0):
            super().__init__()
            hidden_dim = output_dim if hidden_dim is None else hidden_dim
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
            self.hidden_projection = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.gate = nn.Linear(output_dim, output_dim)
            self.value = nn.Linear(output_dim, output_dim)
            self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)

        def forward(self, x, context=None):
            hidden = self.input_projection(x)
            if self.context_projection is not None:
                if context is None:
                    raise ValueError("context is required for this gated residual network")
                while context.ndim < x.ndim:
                    context = context.unsqueeze(-2)
                hidden = hidden + self.context_projection(context)
            elif context is not None:
                raise ValueError("context was supplied to a context-free gated residual network")
            hidden = F.elu(hidden)
            hidden = self.dropout(self.hidden_projection(hidden))
            gated = torch.sigmoid(self.gate(hidden)) * self.value(hidden)
            return self.norm(self.skip(x) + gated)

    class VariableSelectionNetwork(nn.Module):
        def __init__(self, num_variables, d_model, context_dim=0, dropout=0.0):
            super().__init__()
            self.num_variables = num_variables
            self.d_model = d_model
            self.variable_networks = nn.ModuleList(
                TFT.GatedResidualNetwork(d_model, d_model, dropout=dropout)
                for _ in range(num_variables)
            )
            self.weight_network = TFT.GatedResidualNetwork(
                num_variables * d_model, num_variables, context_dim=context_dim, dropout=dropout
            )

        def forward(self, variables, context=None):
            if variables.ndim != 4 or variables.shape[-2:] != (self.num_variables, self.d_model):
                raise ValueError("variables must have shape [batch, time, num_variables, d_model]")
            processed = torch.stack(
                [network(variables[..., index, :]) for index, network in enumerate(self.variable_networks)],
                dim=-2,
            )
            flattened = variables.flatten(start_dim=-2)
            weights = torch.softmax(self.weight_network(flattened, context), dim=-1)
            combined = (processed * weights.unsqueeze(-1)).sum(dim=-2)
            return combined, weights

    class InterpretableMultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.0):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.query_projections = nn.ModuleList(nn.Linear(d_model, self.head_dim) for _ in range(num_heads))
            self.key_projections = nn.ModuleList(nn.Linear(d_model, self.head_dim) for _ in range(num_heads))
            self.value_projection = nn.Linear(d_model, self.head_dim)
            self.output_projection = nn.Linear(self.head_dim, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, query, key, value, mask=None):
            shared_value = self.value_projection(value)
            maps = []
            for query_projection, key_projection in zip(self.query_projections, self.key_projections):
                scores = query_projection(query) @ key_projection(key).transpose(-2, -1)
                scores = scores / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                maps.append(torch.softmax(scores, dim=-1))
            attention = torch.stack(maps, dim=1)
            averaged = attention.mean(dim=1)
            output = self.output_projection(self.dropout(averaged) @ shared_value)
            return output, attention

    def __init__(
        self,
        context_length,
        prediction_length,
        past_features,
        future_features,
        static_features=0,
        categorical_cardinalities=(),
        d_model=160,
        num_heads=4,
        dropout=0.1,
        quantiles=(0.1, 0.5, 0.9),
        lstm_layers=1,
    ):
        super().__init__()
        integer_values = {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "past_features": past_features,
            "future_features": future_features,
            "static_features": static_features,
            "d_model": d_model,
            "num_heads": num_heads,
            "lstm_layers": lstm_layers,
        }
        for name, value in integer_values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                _raise_value_error(f"{name} must be an integer")
        if context_length <= 0 or prediction_length <= 0 or past_features <= 0 or future_features <= 0:
            raise ValueError("context_length, prediction_length, past_features, and future_features must be positive")
        if static_features < 0:
            raise ValueError("static_features must be nonnegative")
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0 or lstm_layers <= 0:
            raise ValueError("d_model and lstm_layers must be positive and d_model must be divisible by num_heads")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(categorical_cardinalities, (tuple, list)):
            _raise_value_error("categorical_cardinalities must be a sequence of positive integers")
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in categorical_cardinalities):
            raise ValueError("categorical_cardinalities must contain only positive integers")
        try:
            quantiles = tuple(float(value) for value in quantiles)
        except (TypeError, ValueError) as error:
            raise ValueError("quantiles must be a nonempty increasing sequence in (0, 1)") from error
        if not quantiles or any(not math.isfinite(value) or not 0 < value < 1 for value in quantiles):
            raise ValueError("quantiles must be a nonempty sequence of finite values in (0, 1)")
        if any(left >= right for left, right in pairwise(quantiles)):
            raise ValueError("quantiles must be strictly increasing")
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.past_features = past_features
        self.future_features = future_features
        self.static_features = static_features
        self.categorical_cardinalities = tuple(categorical_cardinalities)
        self.d_model = d_model
        self.num_heads = num_heads
        self.quantiles = quantiles
        self.lstm_layers = lstm_layers
        self.past_embeddings = nn.ModuleList(nn.Linear(1, d_model) for _ in range(past_features))
        self.future_embeddings = nn.ModuleList(nn.Linear(1, d_model) for _ in range(future_features))
        self.static_embeddings = nn.ModuleList(nn.Linear(1, d_model) for _ in range(static_features))
        self.static_category_embeddings = nn.ModuleList(
            nn.Embedding(size, d_model) for size in self.categorical_cardinalities
        )
        self.past_selection = self.VariableSelectionNetwork(past_features, d_model, d_model, dropout)
        self.future_selection = self.VariableSelectionNetwork(future_features, d_model, d_model, dropout)
        static_count = static_features + len(self.categorical_cardinalities)
        self.static_selection = self.VariableSelectionNetwork(static_count, d_model, dropout=dropout) if static_count else None
        self.static_selection_context = self.GatedResidualNetwork(d_model, d_model, dropout=dropout)
        self.static_hidden_context = self.GatedResidualNetwork(d_model, d_model, dropout=dropout)
        self.static_cell_context = self.GatedResidualNetwork(d_model, d_model, dropout=dropout)
        self.static_enrichment_context = self.GatedResidualNetwork(d_model, d_model, dropout=dropout)
        self.encoder = nn.LSTM(d_model, d_model, num_layers=lstm_layers, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, num_layers=lstm_layers, batch_first=True)
        self.local_gate = self.GatedLinearUnit(d_model, d_model)
        self.local_norm = nn.LayerNorm(d_model)
        self.static_enrichment = self.GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.attention = self.InterpretableMultiHeadAttention(d_model, num_heads, dropout)
        self.attention_gate = self.GatedLinearUnit(d_model, d_model)
        self.attention_norm = nn.LayerNorm(d_model)
        self.positionwise = self.GatedResidualNetwork(d_model, d_model, dropout=dropout)
        self.output_gate = self.GatedLinearUnit(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.quantile_heads = nn.ModuleList(nn.Linear(d_model, 1) for _ in quantiles)

    def _check_inputs(self, past, future, static, static_categoricals):
        if not torch.is_tensor(past) or past.ndim != 3:
            raise ValueError("past must be a tensor with shape [batch, context_length, past_features]")
        if tuple(past.shape[1:]) != (self.context_length, self.past_features):
            raise ValueError(f"past must have trailing shape ({self.context_length}, {self.past_features})")
        if not torch.is_tensor(future) or future.ndim != 3:
            raise ValueError("future must be a tensor with shape [batch, prediction_length, future_features]")
        if tuple(future.shape) != (past.shape[0], self.prediction_length, self.future_features):
            raise ValueError(f"future must have shape [batch, {self.prediction_length}, {self.future_features}]")
        if static is not None:
            if not torch.is_tensor(static) or tuple(static.shape) != (past.shape[0], self.static_features):
                raise ValueError(f"static must have shape [batch, {self.static_features}]")
        elif self.static_features:
            raise ValueError("static is required when static_features is positive")
        if static_categoricals is not None:
            expected = (past.shape[0], len(self.categorical_cardinalities))
            if not torch.is_tensor(static_categoricals) or tuple(static_categoricals.shape) != expected:
                raise ValueError(f"static_categoricals must have shape {expected}")
            if static_categoricals.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("static_categoricals must have an integer dtype")
            for index, size in enumerate(self.categorical_cardinalities):
                values = static_categoricals[:, index]
                if values.numel() and (bool((values < 0).any()) or bool((values >= size).any())):
                    raise ValueError(f"static_categoricals column {index} must be in [0, {size})")
        elif self.categorical_cardinalities:
            raise ValueError("static_categoricals is required when categorical_cardinalities is nonempty")
        return past.shape[0]

    def _static_representation(self, static, static_categoricals, batch, reference):
        if self.static_selection is None:
            zeros = reference.new_zeros((batch, self.d_model))
            return zeros, zeros, zeros, zeros, reference.new_zeros((batch, 0))
        variables = []
        for index, projection in enumerate(self.static_embeddings):
            variables.append(projection(static[:, index:index + 1]))
        for index, embedding in enumerate(self.static_category_embeddings):
            variables.append(embedding(static_categoricals[:, index].long()))
        stacked = torch.stack(variables, dim=1).unsqueeze(1)
        selected, weights = self.static_selection(stacked)
        selected = selected[:, 0]
        selection_context = self.static_selection_context(selected)
        hidden_context = self.static_hidden_context(selected)
        cell_context = self.static_cell_context(selected)
        enrichment_context = self.static_enrichment_context(selected)
        return selection_context, hidden_context, cell_context, enrichment_context, weights[:, 0]

    def forward(self, past, future, static=None, static_categoricals=None, return_intermediates=False):
        batch = self._check_inputs(past, future, static, static_categoricals)
        past_vars = torch.stack(
            [projection(past[..., index:index + 1]) for index, projection in enumerate(self.past_embeddings)], dim=-2
        )
        future_vars = torch.stack(
            [projection(future[..., index:index + 1]) for index, projection in enumerate(self.future_embeddings)], dim=-2
        )
        static_selection, hidden_context, cell_context, enrichment_context, static_weights = self._static_representation(
            static, static_categoricals, batch, past
        )
        past_selected, past_weights = self.past_selection(past_vars, static_selection)
        future_selected, future_weights = self.future_selection(future_vars, static_selection)
        hidden = hidden_context.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()
        cell = cell_context.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()
        past_local, state = self.encoder(past_selected, (hidden, cell))
        future_local, _ = self.decoder(future_selected, state)
        selected = torch.cat((past_selected, future_selected), dim=1)
        local = torch.cat((past_local, future_local), dim=1)
        local = self.local_norm(selected + self.local_gate(local))
        enriched = self.static_enrichment(local, enrichment_context)
        total_length = self.context_length + self.prediction_length
        positions = torch.arange(total_length, device=past.device)
        mask = positions.unsqueeze(1) >= positions.unsqueeze(0)
        attended, attention_weights = self.attention(enriched, enriched, enriched, mask.unsqueeze(0))
        fused = self.attention_norm(enriched + self.attention_gate(attended))
        processed = self.positionwise(fused)
        fused = self.output_norm(local + self.output_gate(processed))
        forecast_features = fused[:, self.context_length:]
        forecasts = torch.cat([head(forecast_features) for head in self.quantile_heads], dim=-1)
        if not return_intermediates:
            return forecasts
        return forecasts, {
            "past_selection_weights": past_weights,
            "future_selection_weights": future_weights,
            "static_selection_weights": static_weights,
            "attention_weights": attention_weights,
            "attention_mask": mask,
            "forecast_features": forecast_features,
        }

    def quantile_loss(self, predictions, targets):
        if not torch.is_tensor(predictions) or tuple(predictions.shape[-2:]) != (
            self.prediction_length, len(self.quantiles)
        ):
            raise ValueError("predictions must end with [prediction_length, number_of_quantiles]")
        if not torch.is_tensor(targets) or targets.shape != predictions.shape[:-1]:
            raise ValueError("targets must have the prediction batch and horizon dimensions")
        levels = predictions.new_tensor(self.quantiles)
        errors = targets.unsqueeze(-1) - predictions
        losses = torch.maximum(levels * errors, (levels - 1) * errors)
        return losses.sum(dim=-1).mean()
