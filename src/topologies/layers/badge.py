from numbers import Integral

import torch
from torch import nn


def _value_error(message):
    return ValueError(message)


class BADGE(nn.Module):
    component_id = "badge"

    def __init__(self, seed=0):
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise _value_error("seed must be an integer")
        if seed < -(2**63) or seed >= 2**64:
            raise _value_error("seed must be between -2**63 and 2**64 - 1")
        self.seed = int(seed)

    def _validate_inputs(self, gradients, k):
        try:
            rows = torch.as_tensor(gradients)
        except (TypeError, ValueError) as error:
            raise _value_error("gradients must be a two-dimensional tensor") from error
        if rows.ndim != 2:
            raise _value_error("gradients must be a two-dimensional tensor")
        if rows.shape[1] == 0:
            raise _value_error("gradients must have at least one feature")
        if not rows.is_floating_point():
            raise _value_error("gradients must have a real floating-point dtype")
        if not torch.isfinite(rows).all().item():
            raise _value_error("gradients must contain only finite values")
        if isinstance(k, torch.Tensor):
            if k.dtype == torch.bool or k.is_floating_point() or k.is_complex():
                raise _value_error("k must be an integer")
            if k.ndim == 0:
                k = k.item()
            elif k.ndim == 2 and k.shape == (rows.shape[0], 1):
                counts = k.detach().cpu().flatten().tolist()
                if any(count < 0 for count in counts):
                    raise _value_error("k must be between zero and the number of gradient rows")
                k = sum(counts)
            else:
                raise _value_error("k must be an integer or one integer count per gradient row")
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise _value_error("k must be an integer")
        k = int(k)
        if k < 0 or k > rows.shape[0]:
            raise _value_error("k must be between zero and the number of gradient rows")
        return rows, k

    def select_indices(self, gradients, k):
        rows, k = self._validate_inputs(gradients, k)
        return self._select_validated(rows, k)

    def _select_validated(self, rows, k):
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=rows.device)

        vectors = rows.detach().to(device="cpu", dtype=torch.float64)
        scale = vectors.abs().max()
        if scale.item() > 0:
            vectors = vectors / scale
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        first = int(torch.randint(rows.shape[0], (1,), generator=generator).item())
        selected = [first]
        selected_set = {first}
        nearest_squared = (vectors - vectors[first]).square().sum(dim=1)

        while len(selected) < k:
            weights = nearest_squared.clone()
            if selected_set:
                weights[list(selected_set)] = 0.0
            if weights.sum().item() > 0:
                next_index = int(torch.multinomial(weights, 1, generator=generator).item())
            else:
                available = torch.ones(rows.shape[0], dtype=torch.bool)
                available[selected] = False
                remaining = torch.nonzero(available, as_tuple=False).flatten()
                position = int(torch.randint(remaining.numel(), (1,), generator=generator).item())
                next_index = int(remaining[position].item())
            selected.append(next_index)
            selected_set.add(next_index)
            distance = (vectors - vectors[next_index]).square().sum(dim=1)
            nearest_squared = torch.minimum(nearest_squared, distance)

        return torch.tensor(selected, dtype=torch.long, device=rows.device)

    def forward(self, gradients, k):
        rows, k = self._validate_inputs(gradients, k)
        indices = self._select_validated(rows, k)
        return rows.index_select(0, indices)
