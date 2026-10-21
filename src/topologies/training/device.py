"""Device and precision abstraction (CPU / CUDA / MPS)."""

from __future__ import annotations

import torch

from topologies._execution import transfer
from topologies.training.errors import TrainingError

_VALID_PRECISIONS = ("auto", "fp32", "fp16", "bf16")


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``device`` with ``auto`` selecting CUDA -> MPS -> CPU."""
    d = (device or "auto").strip().lower()
    if d == "auto":
        try:
            if torch.cuda.is_available():
                return torch.device("cuda")
        except Exception:  # noqa: BLE001, S110 - backend probing must not crash
            pass
        try:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:  # noqa: BLE001, S110 - backend probe; unavailable means not usable
            pass
        return torch.device("cpu")
    try:
        return torch.device(device)
    except Exception as e:
        raise TrainingError(
            f"Invalid device {device!r}.",
            stage="config",
            received=str(device),
            hint="Use 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N'.",
            available=["auto", "cpu", "mps", "cuda", "cuda:0"],
        ) from e


def resolve_precision(precision: str = "auto", device: torch.device | None = None) -> str:
    """Resolve ``precision`` to one of fp32/fp16/bf16."""
    p = (precision or "auto").strip().lower()
    if p not in _VALID_PRECISIONS:
        raise TrainingError(
            f"Unknown precision {precision!r}.",
            stage="config",
            received=str(precision),
            available=list(_VALID_PRECISIONS),
            hint="Use 'auto', 'fp32', 'fp16' or 'bf16'.",
        )
    if p != "auto":
        _check_precision_supported(p, device)
        return p
    dev = device or resolve_device("auto")
    if dev.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return "bf16"
        except Exception:  # noqa: BLE001, S110 - backend probe; unavailable means not supported
            pass
        return "fp16"
    if dev.type == "mps":
        return "fp32"
    return "fp32"


def _check_precision_supported(precision: str, device: torch.device | None) -> None:
    dev = device or torch.device("cpu")
    if precision == "bf16" and dev.type == "mps":
        raise TrainingError(
            "bf16 is not supported on MPS.",
            stage="config",
            device=str(dev),
            precision=precision,
            hint="Use precision='fp32' (or 'auto') on MPS.",
            available=["auto", "fp32", "fp16"],
        )
    if precision in ("fp16", "bf16") and dev.type == "cpu":
        # CPU autocast supports bf16 on recent PyTorch; fp16 CPU autocast is
        # limited. Allow bf16, warn-free fallback handled by engine; fp16 on
        # CPU is allowed but engine will run without scaler.
        return


def autocast_kwargs(device: torch.device, precision: str) -> dict:
    """Return kwargs for ``torch.autocast`` or {} when fp32."""
    if precision == "fp32":
        return {}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return {"device_type": device.type, "dtype": dtype}


def move_batch_to_device(batch: dict, device: torch.device, *,
                         non_blocking: bool = False) -> dict:
    """Strict recursive batch transfer (retains the historical dict contract).

    ``non_blocking`` overlaps host->device copies with compute; it only helps
    from pinned memory and is safe because later kernels on the same stream wait.
    """
    if not isinstance(batch, dict):
        raise TypeError("move_batch_to_device expects a dict batch")
    return transfer(batch, device, non_blocking=non_blocking)
