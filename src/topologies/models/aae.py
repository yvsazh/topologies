import torch
from torch import nn


class AAE(nn.Module):
    architecture_id = "aae"
    PRESETS = {
        "tiny": {
            "input_dim": 8,
            "latent_dim": 4,
            "hidden_dim": 8,
            "output_activation": "identity",
        },
        "small": {
            "input_dim": 784,
            "latent_dim": 8,
            "hidden_dim": 128,
            "output_activation": "sigmoid",
        },
        "base": {
            "input_dim": 784,
            "latent_dim": 8,
            "hidden_dim": 1000,
            "output_activation": "sigmoid",
        },
    }

    def __init__(
        self,
        input_dim,
        latent_dim=8,
        hidden_dim=1000,
        output_activation="sigmoid",
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("latent_dim", latent_dim),
            ("hidden_dim", hidden_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(output_activation) is not str or output_activation not in ("sigmoid", "identity"):
            raise ValueError("output_activation must be 'sigmoid' or 'identity'")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_activation = output_activation

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        decoder_layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        ]
        if output_activation == "sigmoid":
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x):
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must have shape [B,{self.input_dim}] with B > 0")
        return self.encoder(x)

    def decode(self, z):
        if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape [B,{self.latent_dim}] with B > 0")
        return self.decoder(z)

    def discriminate(self, z):
        if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape [B,{self.latent_dim}] with B > 0")
        return self.discriminator(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    @staticmethod
    def reconstruction_error(x, reconstruction):
        if (
            x.ndim != 2
            or reconstruction.ndim != 2
            or x.shape[0] == 0
            or x.shape != reconstruction.shape
        ):
            raise ValueError("x and reconstruction must have the same nonempty [B,D] shape")
        return 0.5 * (x - reconstruction).square().sum(dim=1).mean()

    @staticmethod
    def discriminator_objective(prior_logits, encoded_logits):
        if (
            prior_logits.ndim != 2
            or encoded_logits.ndim != 2
            or prior_logits.shape[0] == 0
            or prior_logits.shape != encoded_logits.shape
            or prior_logits.shape[1] != 1
        ):
            raise ValueError("prior_logits and encoded_logits must have the same nonempty [B,1] shape")
        return torch.nn.functional.logsigmoid(prior_logits).mean() + torch.nn.functional.logsigmoid(
            -encoded_logits
        ).mean()

    @staticmethod
    def encoder_objective(encoded_logits):
        if encoded_logits.ndim != 2 or encoded_logits.shape[0] == 0 or encoded_logits.shape[1] != 1:
            raise ValueError("encoded_logits must have nonempty shape [B,1]")
        return torch.nn.functional.logsigmoid(-encoded_logits).mean()
