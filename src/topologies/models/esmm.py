import torch
from torch import nn
from torch.nn import functional as F


class ESMM(nn.Module):
    architecture_id = "esmm"
    PRESETS = {
        "tiny": {"vocab_sizes": (16, 16, 16, 16), "embedding_dim": 4, "hidden_dims": (16, 8)},
        "small": {"vocab_sizes": (128, 128, 128, 128, 128, 128, 128, 128), "embedding_dim": 8, "hidden_dims": (64, 32)},
    }

    def __init__(self, vocab_sizes, embedding_dim=18, hidden_dims=(360, 200, 80)):
        super().__init__()
        try:
            vocab_sizes = tuple(vocab_sizes)
        except TypeError as error:
            raise ValueError("vocab_sizes must be a non-empty sequence of positive integers") from error
        if not vocab_sizes or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in vocab_sizes):
            raise ValueError("vocab_sizes must be a non-empty sequence of positive integers")
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        try:
            hidden_dims = tuple(hidden_dims)
        except TypeError as error:
            raise ValueError("hidden_dims must be a sequence of positive integers") from error
        if any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must be a sequence of positive integers")
        self.vocab_sizes = vocab_sizes
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.embeddings = nn.ModuleList(nn.Embedding(size, embedding_dim) for size in vocab_sizes)
        input_dim = len(vocab_sizes) * embedding_dim

        def make_tower():
            layers = []
            width = input_dim
            for hidden_width in hidden_dims:
                layers.extend((nn.Linear(width, hidden_width), nn.ReLU()))
                width = hidden_width
            layers.append(nn.Linear(width, 2))
            return nn.Sequential(*layers)

        self.ctr_tower = make_tower()
        self.cvr_tower = make_tower()

    def _validated_indices(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != len(self.vocab_sizes) or x.shape[0] == 0:
            raise ValueError("x must be a non-empty rank-2 tensor with one column per vocabulary")
        if x.dtype == torch.bool or x.is_complex():
            raise ValueError("x must contain integer-coded categorical IDs")
        if x.is_floating_point():
            if not torch.isfinite(x).all() or not torch.equal(x, x.round()):
                raise ValueError("floating-point feature IDs must be finite integers")
        elif x.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("x must contain integer-coded categorical IDs")
        indices = x.to(dtype=torch.long)
        for field, vocab_size in enumerate(self.vocab_sizes):
            field_ids = indices[:, field]
            if torch.any(field_ids < 0) or torch.any(field_ids >= vocab_size):
                raise ValueError(f"feature IDs in column {field} must be in [0, {vocab_size})")
        return indices

    def forward(self, x):
        indices = self._validated_indices(x)
        shared = torch.cat([embedding(indices[:, field]) for field, embedding in enumerate(self.embeddings)], dim=-1)
        pctr = self.ctr_tower(shared).softmax(dim=-1)[:, 1:2]
        pcvr = self.cvr_tower(shared).softmax(dim=-1)[:, 1:2]
        pctcvr = pctr * pcvr
        return {
            "pctr": pctr,
            "pcvr": pcvr,
            "pctcvr": pctcvr,
            "objective_probabilities": torch.cat((pctr, pctcvr), dim=-1),
        }

    @staticmethod
    def joint_loss(objective_probabilities, click, conversion):
        if (
            not isinstance(objective_probabilities, torch.Tensor)
            or objective_probabilities.ndim != 2
            or objective_probabilities.shape[1] != 2
            or objective_probabilities.shape[0] == 0
            or not objective_probabilities.is_floating_point()
        ):
            raise ValueError("objective_probabilities must be a non-empty floating-point tensor with shape [B, 2]")
        if not torch.isfinite(objective_probabilities).all() or torch.any(objective_probabilities < 0) or torch.any(objective_probabilities > 1):
            raise ValueError("objective probabilities must be finite values in [0, 1]")
        expected_shape = (objective_probabilities.shape[0], 1)
        if not isinstance(click, torch.Tensor) or not isinstance(conversion, torch.Tensor):
            raise TypeError("click and conversion must be tensors with shape [B, 1]")
        if click.shape != expected_shape or conversion.shape != expected_shape:
            raise ValueError("click and conversion must have shape [B, 1]")
        if click.is_complex() or conversion.is_complex():
            raise ValueError("click and conversion labels must be binary")
        click = click.to(device=objective_probabilities.device)
        conversion = conversion.to(device=objective_probabilities.device)
        if not torch.isfinite(click).all() or not torch.isfinite(conversion).all():
            raise ValueError("click and conversion labels must be finite binary values")
        if not torch.all((click == 0) | (click == 1)) or not torch.all((conversion == 0) | (conversion == 1)):
            raise ValueError("click and conversion labels must be binary")
        if torch.any(conversion > click):
            raise ValueError("conversion labels require a preceding click")
        click_target = click.to(dtype=objective_probabilities.dtype)
        conversion_target = (click * conversion).to(dtype=objective_probabilities.dtype)
        ctr_loss = F.binary_cross_entropy(objective_probabilities[:, 0:1], click_target, reduction="sum")
        ctcvr_loss = F.binary_cross_entropy(objective_probabilities[:, 1:2], conversion_target, reduction="sum")
        return ctr_loss + ctcvr_loss
