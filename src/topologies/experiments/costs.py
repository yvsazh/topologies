"""Best-effort cost accounting for a trial (``result.json -> cost``).

Every field is ``None`` when unknown and nothing here ever raises.
``flops_est`` is an *estimate*: forward FLOPs of one real batch (measured with
``torch.utils.flop_counter.FlopCounterMode``) x 3 (forward + backward) x
steps; fallback ``6 x non-embedding params x tokens`` for token models.
"""

from __future__ import annotations

import math
from typing import Any


def _device_type(device: Any) -> str:
    return str(device).split(":")[0] if device is not None else "cpu"


def reset_memory(ctx: Any) -> None:
    ctx.info["peak_mem_mb"] = None
    try:
        import torch

        if _device_type(ctx.device) == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001, S110 - cost probes are best-effort
        pass
    sample_memory(ctx)


def sample_memory(ctx: Any) -> None:
    """Update the sampled peak (MPS driver memory, CPU process RSS)."""
    try:
        dt = _device_type(ctx.device)
        mb = None
        if dt == "mps":
            import torch

            mb = float(torch.mps.driver_allocated_memory()) / 2**20
        elif dt == "cpu":
            from topologies.training.monitoring.system import collect_system

            rss = collect_system(ttl=1.0).get("rss_gb")
            mb = float(rss) * 1024 if rss is not None else None
        if mb is not None:
            prev = ctx.info.get("peak_mem_mb")
            ctx.info["peak_mem_mb"] = mb if prev is None else max(prev, mb)
    except Exception:  # noqa: BLE001, S110 - cost probes are best-effort
        pass


def peak_memory_mb(ctx: Any) -> float | None:
    try:
        import torch

        if _device_type(ctx.device) == "cuda" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 2**20
    except Exception:  # noqa: BLE001, S110 - cost probes are best-effort
        pass
    sample_memory(ctx)
    return ctx.info.get("peak_mem_mb")


def param_split(model: Any) -> dict[str, int | None]:
    try:
        from topologies.build.params import count_params, embedding_params

        total = count_params(model)
        return {"params": total, "trainable_params": count_params(model, trainable_only=True),
                "embedding_params": embedding_params(model)}
    except Exception:  # noqa: BLE001
        return {"params": None, "trainable_params": None, "embedding_params": None}


def _sdpa_cpu_flops(query_shape: Any, key_shape: Any, value_shape: Any, *args: Any,
                    out_shape: Any = None, **kwargs: Any) -> int:
    """QK^T + AV of the fused CPU attention kernel (missing from torch's registry)."""
    b, h, s_q, d_q = query_shape
    s_k, d_v = key_shape[-2], value_shape[-1]
    return 2 * b * h * s_q * s_k * d_q + 2 * b * h * s_q * s_k * d_v


def flop_counter() -> Any:
    """``FlopCounterMode`` that also counts CPU fused attention (``display=False``)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    custom = {}
    op = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu", None)
    if op is not None:
        custom[op] = _sdpa_cpu_flops
    try:
        return FlopCounterMode(display=False, custom_mapping=custom)
    except TypeError:  # older torch without custom_mapping
        return FlopCounterMode(display=False)


def forward_flops(trainer: Any, batch: Any) -> float | None:
    """FLOPs of one forward pass of ``batch`` through the recipe (via ``evaluate``)."""
    try:
        counter = flop_counter()
        with counter:
            trainer.evaluate(iter([batch]))
        total = float(counter.get_total_flops())
        return total if total > 0 else None
    except Exception:  # noqa: BLE001
        return None


def first_batch(loader: Any) -> Any:
    try:
        return next(iter(loader))
    except Exception:  # noqa: BLE001
        return None


def flops_estimate(trainer: Any, *, steps: int | None, tokens_seen: int | None,
                   non_embedding_params: int | None) -> tuple[float | None, str | None]:
    """``(flops, method)`` with method ``"flop_counter"`` or ``"6ND"``."""
    batch = first_batch(getattr(trainer, "train_loader", None))
    if batch is not None and steps:
        fwd = forward_flops(trainer, batch)
        if fwd is not None:
            return fwd * 3.0 * float(steps), "flop_counter"
    if tokens_seen and non_embedding_params:
        return 6.0 * float(non_embedding_params) * float(tokens_seen), "6ND"
    return None, None


def _rate(n: Any, s: Any) -> float | None:
    try:
        if n is None or not s or float(s) <= 0:
            return None
        return float(n) / float(s)
    except (TypeError, ValueError):
        return None


def compute_costs(ctx: Any, trainer: Any, model: Any, *, wall_s: float | None,
                  train_s: float | None, eval_s: float | None,
                  fit_out: dict | None) -> dict[str, Any]:
    cost: dict[str, Any] = {"params": None, "trainable_params": None, "embedding_params": None,
                            "wall_s": wall_s, "train_s": train_s, "eval_s": eval_s,
                            "steps": None, "epochs": None, "samples_seen": None,
                            "tokens_seen": None, "bytes_seen": None, "samples_per_s": None,
                            "tokens_per_s": None, "peak_mem_mb": None, "flops_est": None,
                            "flops_method": None}
    try:
        if model is not None:
            cost.update(param_split(model))
        out = fit_out or {}
        cost["steps"] = int(out.get("global_step", getattr(trainer, "global_step", 0)) or 0) \
            if trainer is not None or out else None
        hist = out.get("history") if out else getattr(trainer, "history", None)
        cost["epochs"] = len(hist) if hist is not None else None
        cost["samples_seen"] = out.get("samples_seen", getattr(trainer, "samples_seen", None))
        tokens = out.get("tokens_seen", getattr(trainer, "tokens_seen", None))
        cost["tokens_seen"] = tokens if tokens else None
        bpt = ctx.info.get("bytes_per_token_train")
        if cost["tokens_seen"] and bpt:
            cost["bytes_seen"] = round(cost["tokens_seen"] * bpt)
        cost["samples_per_s"] = _rate(cost["samples_seen"], train_s)
        cost["tokens_per_s"] = _rate(cost["tokens_seen"], train_s)
        cost["peak_mem_mb"] = peak_memory_mb(ctx)
        if trainer is not None:
            non_emb = None
            if cost["params"] is not None and cost["embedding_params"] is not None:
                non_emb = cost["params"] - cost["embedding_params"]
            flops, method = flops_estimate(trainer, steps=cost["steps"],
                                           tokens_seen=cost["tokens_seen"],
                                           non_embedding_params=non_emb)
            cost["flops_est"], cost["flops_method"] = flops, method
    except Exception:  # noqa: BLE001, S110 - costs are best-effort
        pass
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in cost.items()}


__all__ = ["compute_costs", "flop_counter", "flops_estimate", "param_split", "peak_memory_mb", "reset_memory",
           "sample_memory"]
