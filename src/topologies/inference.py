"""Device-safe inference helpers that work with any ``nn.Module``.

After ``Trainer.fit()`` the model may live on ``mps``/``cuda``; these helpers
move inputs to the model's device and results back to the CPU, so the
common "Tensor is on cpu but expected on mps" error cannot happen.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn

from topologies._execution import model_execution, primary_output, transfer


def model_device(model: nn.Module) -> torch.device:
    """Device of the model's first parameter/buffer (CPU for parameter-free modules)."""
    for t in model.parameters():
        return t.device
    for t in model.buffers():
        return t.device
    return torch.device("cpu")


def to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in tensors/dicts/lists/tuples to ``device``.

    Alias of ``topologies._execution.transfer`` (one transfer implementation).
    """
    return transfer(obj, device)


@torch.no_grad()
def predict(
    model: nn.Module,
    inputs: torch.Tensor | Iterable[Any],
    *,
    batch_size: int = 256,
    key: str | None = None,
) -> torch.Tensor:
    """Run ``model`` in eval mode on a tensor (batched) or an iterable of batches.

    Batches may be tensors, ``(x, y)`` tuples (``x`` is used) or dicts
    (``key`` selects the input; default: the first non-label tensor).
    Returns concatenated outputs on the CPU.
    """
    device = model_device(model)
    with model_execution(model, training=False, grad_mode="no_grad", preserve_rng=True):
        if isinstance(inputs, torch.Tensor):
            chunks = inputs.split(batch_size) if inputs.dim() > 0 else [inputs]
        else:
            chunks = inputs
        outs = []
        for batch in chunks:
            if isinstance(batch, dict):
                k = key or next(k for k, v in batch.items() if isinstance(v, torch.Tensor)
                                and k not in ("labels", "label", "targets", "target"))
                x = batch[k]
            elif isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            outs.append(primary_output(model(to_device(x, device))).detach().cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty(0)


@torch.no_grad()
def classify(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    class_names: Sequence[str] | None = None,
    top_k: int = 1,
    batch_size: int = 256,
) -> list[list[tuple[Any, float]]]:
    """Top-``k`` ``(label, probability)`` pairs per input (single images get a batch dim)."""
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
    probs = predict(model, inputs, batch_size=batch_size).softmax(-1)
    k = min(top_k, probs.shape[-1])
    vals, idx = probs.topk(k, dim=-1)
    out = []
    for v_row, i_row in zip(vals.tolist(), idx.tolist()):
        out.append([((class_names[i] if class_names else i), p) for i, p in zip(i_row, v_row)])
    return out
