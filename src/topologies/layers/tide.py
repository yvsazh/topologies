import torch
from torch import nn


class TiDEResidualBlock(nn.Module):
    component_id = "tide"

    def __init__(self, in_features, out_features, hidden_size, dropout, layer_norm=True):
        super().__init__()
        if min(in_features, out_features, hidden_size) <= 0:
            raise ValueError("feature dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden = nn.Linear(in_features, hidden_size)
        self.output = nn.Linear(hidden_size, out_features)
        self.skip = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features) if layer_norm else nn.Identity()

    def forward(self, values):
        if values.shape[-1] != self.hidden.in_features:
            raise ValueError("values must end in in_features")
        residual = self.skip(values)
        update = self.output(torch.relu(self.hidden(values)))
        combined = residual + self.dropout(update)
        return self.norm(combined)


class ResidualBlock(TiDEResidualBlock):
    def __init__(self, in_features, out_features, hidden_size, dropout):
        super().__init__(
            in_features, out_features, hidden_size, dropout,
            layer_norm=out_features > 1,
        )

    component_id = None


class TiDE(nn.Module):
    architecture_id = "tide"

    def __init__(self, input_length, horizon, input_channels=1, output_channels=None,
                 covariate_channels=0, hidden_size=128, encoder_layers=2,
                 decoder_layers=2, dropout=0.0, temporal_width=None,
                 decoder_output_dim=None, temporal_decoder_hidden=None):
        super().__init__()
        if min(input_length, horizon, input_channels, hidden_size, encoder_layers, decoder_layers) <= 0:
            raise ValueError("lengths, channel counts, layer counts, and hidden_size must be positive")
        if covariate_channels < 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid covariate channel count or dropout")
        output_channels = input_channels if output_channels is None else output_channels
        if output_channels <= 0:
            raise ValueError("output_channels must be positive")
        temporal_width = (max(1, covariate_channels) if temporal_width is None
                          else temporal_width)
        if covariate_channels and temporal_width <= 0:
            raise ValueError("temporal_width must be positive when covariates are used")
        if not covariate_channels:
            temporal_width = 0
        decoder_output_dim = hidden_size if decoder_output_dim is None else decoder_output_dim
        temporal_decoder_hidden = (hidden_size if temporal_decoder_hidden is None
                                   else temporal_decoder_hidden)
        if min(decoder_output_dim, temporal_decoder_hidden) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.input_length = input_length
        self.horizon = horizon
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.covariate_channels = covariate_channels
        self.temporal_width = temporal_width
        self.decoder_output_dim = decoder_output_dim
        self.feature_projection = (
            ResidualBlock(covariate_channels, temporal_width, temporal_width, dropout)
            if covariate_channels else None
        )
        encoder_input = input_length * input_channels + (input_length + horizon) * temporal_width
        self.encoder = self._stack(encoder_input, hidden_size, hidden_size,
                                   encoder_layers, dropout)
        self.decoder = self._stack(hidden_size, hidden_size, horizon * decoder_output_dim,
                                   decoder_layers, dropout)
        self.temporal_decoder = ResidualBlock(
            decoder_output_dim + temporal_width, output_channels,
            temporal_decoder_hidden, dropout,
        )
        self.temporal_projection = nn.Linear(input_length * input_channels, horizon * self.output_channels)

    @staticmethod
    def _stack(input_size, hidden_size, output_size, layers, dropout):
        blocks = []
        current_size = input_size
        for index in range(layers):
            block_output = output_size if index == layers - 1 else hidden_size
            blocks.append(ResidualBlock(current_size, block_output, hidden_size, dropout))
            current_size = block_output
        return nn.Sequential(*blocks)

    def forward(self, history, past_covariates=None, future_covariates=None):
        history = torch.as_tensor(history, dtype=self.temporal_projection.weight.dtype,
                                  device=self.temporal_projection.weight.device)
        if history.ndim == 2:
            history = history.unsqueeze(-1)
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.input_length, self.input_channels):
            raise ValueError("history must have shape (batch, input_length, input_channels)")
        batch = history.shape[0]
        if self.covariate_channels:
            if past_covariates is None or future_covariates is None:
                raise ValueError("past_covariates and future_covariates are required")
            past_covariates = torch.as_tensor(past_covariates, dtype=history.dtype, device=history.device)
            future_covariates = torch.as_tensor(future_covariates, dtype=history.dtype, device=history.device)
            if tuple(past_covariates.shape) != (batch, self.input_length, self.covariate_channels):
                raise ValueError("past_covariates has an invalid shape")
            if tuple(future_covariates.shape) != (batch, self.horizon, self.covariate_channels):
                raise ValueError("future_covariates has an invalid shape")
            covariates = torch.cat((past_covariates, future_covariates), dim=1)
            projected = self.feature_projection(covariates)
            past_features = projected[:, :self.input_length]
            future_features = projected[:, self.input_length:]
            encoder_input = torch.cat(
                (history.flatten(1), past_features.flatten(1), future_features.flatten(1)),
                dim=1,
            )
        else:
            if past_covariates is not None or future_covariates is not None:
                raise ValueError("construct with covariate_channels to use covariates")
            encoder_input = history.flatten(1)
            future_features = history.new_empty((batch, self.horizon, 0))
        representation = self.encoder(encoder_input)
        decoded = self.decoder(representation).reshape(
            batch, self.horizon, self.decoder_output_dim
        )
        temporal_input = torch.cat((decoded, future_features), dim=-1)
        nonlinear_forecast = self.temporal_decoder(temporal_input)
        residual_forecast = self.temporal_projection(history.flatten(1)).reshape(
            batch, self.horizon, self.output_channels
        )
        return nonlinear_forecast + residual_forecast
