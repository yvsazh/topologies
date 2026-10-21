"""Checkpoint save/load covering model + all optimizers/schedulers + RNG.

Checkpoints hold only tensors and plain Python containers/scalars, so they load
with ``torch.load(weights_only=True)``: a checkpoint file cannot run code.
Older checkpoints that need full unpickling load only with
``trust_checkpoint=True`` (``Trainer(resume=..., trust_checkpoint=True)``).
"""

from __future__ import annotations

import pickle
import random
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _model_state(model: nn.Module) -> dict:
    from topologies.build.contract import unwrap

    # Unwrapped: DDP ("module.") / torch.compile ("_orig_mod.") prefixes never leak in.
    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in unwrap(model).state_dict().items()}


class UntrustedCheckpointError(ValueError):
    """A checkpoint needs full (code-executing) unpickling and was not trusted."""


def _numpy_state_to_plain(state: Any) -> Any:
    """``numpy.random.get_state()`` without ``ndarray`` (weights_only-safe)."""
    name, keys, pos, has_gauss, cached = state
    return [str(name), [int(k) for k in keys], int(pos), int(has_gauss), float(cached)]


def _numpy_state_from_plain(state: Any) -> Any:
    import numpy as _np

    name, keys, pos, has_gauss, cached = state
    return (name, _np.asarray(keys, dtype=_np.uint32), int(pos), int(has_gauss), float(cached))


def read_checkpoint(path: str | Path, *, map_location: Any = "cpu",
                    trust_checkpoint: bool = False) -> Any:
    """``torch.load`` with ``weights_only=True``; full unpickling only if trusted."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        if not trust_checkpoint:
            raise UntrustedCheckpointError(
                f"{path} contains objects that need full unpickling (it can run code). "
                "It was probably written by an older topologies version. If you created "
                "this file yourself, load it with trust_checkpoint=True "
                "(Trainer(resume=..., trust_checkpoint=True))."
            ) from e
        return torch.load(path, map_location=map_location, weights_only=False)  # noqa: S614 - explicit opt-in


def _optimizer_ownership(model: nn.Module, optimizers: dict[str, Any]) -> dict[str, Any]:
    names = {id(param): name for name, param in model.named_parameters(remove_duplicate=False)}
    return {key: {"class": f"{type(opt).__module__}.{type(opt).__qualname__}",
                  "groups": [[names.get(id(param), "<unowned>") for param in group["params"]]
                             for group in opt.param_groups]}
            for key, opt in optimizers.items()}


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizers: dict[str, Any],
    schedulers: dict[str, Any],
    scaler: Any,
    epoch: int,
    global_step: int,
    recipe_state: dict | None = None,
    trainer_state: dict | None = None,
) -> Path:
    """Write a resumable checkpoint. ``epoch`` is the next epoch to run on resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": _model_state(model),
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "schedulers": {
            k: (s.state_dict() if s is not None else None) for k, s in schedulers.items()
        },
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "recipe_state": recipe_state or {},
        "trainer_state": trainer_state or {},
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        },
    }
    session = getattr(model, "__topologies_optimization_session__", None)
    if session is not None:
        if not callable(getattr(session, "state_dict", None)):
            raise TypeError("attached optimization session must implement state_dict()")
        transformation_state = session.state_dict()
        if transformation_state.get("state") == "converted":
            raise ValueError("Converted inference models cannot be saved as resumable training checkpoints")
        payload["transformation_state"] = transformation_state
        payload["optimization_optimizer_ownership"] = _optimizer_ownership(model, optimizers)
    try:
        import numpy as _np
    except ImportError:  # NumPy is optional
        _np = None
    if _np is not None:
        payload["rng"]["numpy"] = _numpy_state_to_plain(_np.random.get_state())
    if torch.cuda.is_available():
        try:
            payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:  # noqa: BLE001, S110 - CUDA RNG capture is optional (driver issues)
            pass
    torch.save(payload, path)
    return path


def _restore_rng(rng: dict) -> None:
    """Restore torch/CUDA/Python/NumPy RNG streams saved by ``save_checkpoint``."""
    if "torch" in rng:
        torch.set_rng_state(rng["torch"].cpu())
    cuda = rng.get("cuda")
    if cuda is not None and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        import numpy as _np

        state = rng["numpy"]
        _np.random.set_state(_numpy_state_from_plain(state) if isinstance(state, list) else state)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizers: dict[str, Any] | None = None,
    schedulers: dict[str, Any] | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    trust_checkpoint: bool = False,
) -> dict:
    payload = read_checkpoint(path, map_location=map_location,
                              trust_checkpoint=trust_checkpoint)
    from topologies.build.contract import unwrap

    mod = unwrap(model)
    transformation = payload.get("transformation_state")
    session = getattr(model, "__topologies_optimization_session__", None)
    if transformation is not None:
        if session is None or not callable(getattr(session, "load_state_dict", None)):
            raise ValueError("checkpoint requires the same prepared optimization session before resume")
        validate = getattr(session, "validate_state_dict", None)
        if callable(validate):
            validate(transformation)
    elif session is not None:
        raise ValueError("prepared optimization session cannot resume a checkpoint without transformation state")
    strict_optimization = transformation is not None
    if strict_optimization and set(optimizers or {}) != set(payload.get("optimizers", {})):
        raise ValueError("optimization checkpoint optimizer set differs; refusing partial optimizer resume")
    if strict_optimization and set(schedulers or {}) != set(payload.get("schedulers", {})):
        raise ValueError("optimization checkpoint scheduler set differs; refusing partial scheduler resume")
    if strict_optimization:
        expected_ownership = payload.get("optimization_optimizer_ownership")
        actual_ownership = _optimizer_ownership(model, optimizers or {})
        if expected_ownership != actual_ownership:
            raise ValueError("optimization checkpoint optimizer parameter ownership differs")
    mod.load_state_dict(payload["model"])
    if transformation is not None:
        session.load_state_dict(transformation)
    failures: list[str] = []

    def restore(label: str, target: Any, state: Any) -> None:
        try:
            target.load_state_dict(state)
        except Exception as exc:
            if strict_optimization:
                raise
            failures.append(f"{label} ({type(exc).__name__}: {exc})")

    for k, o in (optimizers or {}).items():
        if k in payload.get("optimizers", {}):
            restore(f"optimizer {k!r}", o, payload["optimizers"][k])
    for k, s in (schedulers or {}).items():
        if s is not None and k in payload.get("schedulers", {}) and payload["schedulers"][k]:
            restore(f"scheduler {k!r}", s, payload["schedulers"][k])
    if scaler is not None and payload.get("scaler"):
        restore("grad scaler", scaler, payload["scaler"])
    if failures:
        warnings.warn(f"{path}: these states could not be restored and start fresh: "
                      + "; ".join(failures), UserWarning, stacklevel=2)
    _restore_rng(payload.get("rng", {}))
    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "recipe_state": payload.get("recipe_state", {}),
        "trainer_state": payload.get("trainer_state", {}),
        "transformation_state": payload.get("transformation_state"),
    }
