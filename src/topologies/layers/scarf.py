import math

import torch
from torch import nn
from torch.nn import functional as F


class Scarf(nn.Module):
    component_id = "scarf"
    PRESETS = {
        "tiny": {
            "hidden_dim": 16,
            "projection_dim": 16,
            "encoder_layers": 2,
            "head_layers": 2,
            "corruption_rate": 0.6,
            "temperature": 1.0,
        },
        "small": {
            "hidden_dim": 64,
            "projection_dim": 64,
            "encoder_layers": 4,
            "head_layers": 2,
            "corruption_rate": 0.6,
            "temperature": 1.0,
        },
    }

    def __init__(
        self,
        input_dim,
        projection_dim,
        reference_data=None,
        hidden_dim=256,
        encoder_layers=4,
        head_layers=2,
        corruption_rate=0.6,
        temperature=1.0,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("projection_dim", projection_dim),
            ("hidden_dim", hidden_dim),
            ("encoder_layers", encoder_layers),
            ("head_layers", head_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(corruption_rate, bool)
            or not isinstance(corruption_rate, (int, float))
            or not math.isfinite(corruption_rate)
            or not 0.0 <= corruption_rate <= 1.0
        ):
            raise ValueError("corruption_rate must be a finite number in [0, 1]")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0.0
        ):
            raise ValueError("temperature must be a finite positive number")
        if reference_data is not None and (
            not isinstance(reference_data, torch.Tensor)
            or reference_data.ndim != 2
            or reference_data.shape[0] == 0
            or reference_data.shape[1] != input_dim
            or not reference_data.is_floating_point()
            or not torch.isfinite(reference_data).all()
        ):
            raise ValueError("reference_data must be a finite floating point matrix with width input_dim")
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.head_layers = head_layers
        self.corruption_rate = float(corruption_rate)
        self.temperature = float(temperature)
        stored_reference = (
            reference_data.detach().clone()
            if reference_data is not None
            else torch.empty(0, input_dim)
        )
        self.register_buffer("reference_data", stored_reference)
        encoder = []
        in_features = input_dim
        for index in range(encoder_layers):
            encoder.append(nn.Linear(in_features, hidden_dim))
            if index < encoder_layers - 1:
                encoder.append(nn.ReLU())
            in_features = hidden_dim
        self.encoder = nn.Sequential(*encoder)
        projection_head = []
        for _ in range(head_layers - 1):
            projection_head.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        projection_head.append(nn.Linear(hidden_dim, projection_dim))
        self.projection_head = nn.Sequential(*projection_head)

    def _validate_x(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("x must be a nonempty two-dimensional tensor")
        if x.shape[1] != self.input_dim:
            raise ValueError("x feature width must equal input_dim")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("x must contain finite floating point values")
        parameter = next(self.encoder.parameters())
        if x.device != parameter.device or x.dtype != parameter.dtype:
            raise ValueError("x and model parameters must share device and dtype")

    def corrupt(self, x, reference_data=None):
        self._validate_x(x)
        source = self.reference_data if reference_data is None else reference_data
        if (
            not isinstance(source, torch.Tensor)
            or source.ndim != 2
            or source.shape[0] == 0
            or source.shape[1] != self.input_dim
            or not source.is_floating_point()
            or not torch.isfinite(source).all()
        ):
            raise ValueError("reference_data must be a finite floating point matrix with width input_dim")
        if source.device != x.device or source.dtype != x.dtype:
            raise ValueError("x and reference_data must share device and dtype")
        batch_size = x.shape[0]
        num_corrupted = math.floor(self.corruption_rate * self.input_dim)
        mask = torch.zeros_like(x, dtype=torch.bool)
        if num_corrupted:
            selected = torch.rand(batch_size, self.input_dim, device=x.device).topk(
                num_corrupted, dim=1, sorted=False
            ).indices
            mask.scatter_(1, selected, True)
        sampled_rows = torch.randint(
            source.shape[0], (batch_size, self.input_dim), device=x.device
        )
        columns = torch.arange(self.input_dim, device=x.device)
        replacements = source[sampled_rows, columns]
        return torch.where(mask, replacements, x), mask

    def encode(self, x):
        self._validate_x(x)
        return self.encoder(x)

    def project(self, x):
        return F.normalize(self.projection_head(self.encode(x)), p=2, dim=-1)

    def contrastive_loss(self, z, z_tilde):
        if (
            not isinstance(z, torch.Tensor)
            or not isinstance(z_tilde, torch.Tensor)
            or z.ndim != 2
            or z_tilde.shape != z.shape
            or z.shape[0] == 0
            or z.shape[1] != self.projection_dim
        ):
            raise ValueError("z and z_tilde must have matching nonempty (batch, projection_dim) shapes")
        if not z.is_floating_point() or not z_tilde.is_floating_point():
            raise ValueError("z and z_tilde must have floating point dtype")
        if z.device != z_tilde.device or z.dtype != z_tilde.dtype:
            raise ValueError("z and z_tilde must share device and dtype")
        similarities = F.normalize(z, p=2, dim=1) @ F.normalize(z_tilde, p=2, dim=1).transpose(0, 1)
        logits = similarities / self.temperature
        positive = logits.diagonal()
        return (-positive + torch.logsumexp(logits, dim=1) - math.log(z.shape[0])).mean()

    def forward(self, x, reference_data=None, return_loss=True):
        if not return_loss:
            return self.project(x)
        corrupted, _ = self.corrupt(x, reference_data)
        return self.contrastive_loss(self.project(x), self.project(corrupted))
