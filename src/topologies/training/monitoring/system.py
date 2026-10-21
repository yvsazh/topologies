"""Best-effort system telemetry (never raises, never slows the training loop).

``collect_system()`` returns a flat dict; unavailable values are ``None``
(rendered as ``n/a``). Results are cached for ``ttl`` seconds.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

_CACHE: dict[str, Any] = {"t": 0.0, "data": {}}


def _gib(n: float | None) -> float | None:
    return None if n is None else float(n) / (1024**3)


def _process_rss() -> float | None:
    try:
        import psutil  # optional

        return float(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        with open("/proc/self/statm") as fh:  # Linux: current RSS
            pages = int(fh.read().split()[1])
        return float(pages * os.sysconf("SC_PAGE_SIZE"))
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024.0  # peak, not current
    except Exception:  # noqa: BLE001
        return None


def _cpu_percent() -> float | None:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None))
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        load1 = os.getloadavg()[0]
        n = os.cpu_count() or 1
        return min(100.0, 100.0 * load1 / n)
    except Exception:  # noqa: BLE001
        return None


def _accelerator() -> dict[str, Any]:
    out: dict[str, Any] = {"device": "cpu", "gpu_mem_gb": None, "gpu_mem_reserved_gb": None,
                           "gpu_util": None}
    try:
        import torch

        if torch.cuda.is_available():
            out["device"] = "cuda"
            out["gpu_mem_gb"] = _gib(torch.cuda.memory_allocated())
            out["gpu_mem_reserved_gb"] = _gib(torch.cuda.memory_reserved())
            try:
                out["gpu_util"] = float(torch.cuda.utilization())  # needs pynvml
            except Exception:  # noqa: BLE001, S110
                pass
            return out
        mps = getattr(torch, "mps", None)
        backend = getattr(torch.backends, "mps", None)
        if mps is not None and backend is not None and backend.is_available():
            out["device"] = "mps"
            try:
                out["gpu_mem_gb"] = _gib(mps.current_allocated_memory())
                out["gpu_mem_reserved_gb"] = _gib(mps.driver_allocated_memory())
            except Exception:  # noqa: BLE001, S110
                pass
    except Exception:  # noqa: BLE001, S110
        pass
    return out


def collect_system(ttl: float = 2.0) -> dict[str, Any]:
    """``{"cpu_percent", "cpu_count", "rss_gb", "device", "gpu_mem_gb", ...}``."""
    try:
        now = time.monotonic()
        if now - float(_CACHE["t"]) < ttl and _CACHE["data"]:
            return dict(_CACHE["data"])
        data: dict[str, Any] = {
            "cpu_percent": _cpu_percent(),
            "cpu_count": os.cpu_count(),
            "rss_gb": _gib(_process_rss()),
        }
        data.update(_accelerator())
        _CACHE["t"], _CACHE["data"] = now, data
        return dict(data)
    except Exception:  # noqa: BLE001
        return {}


def device_memory_str() -> str | None:
    """Short accelerator memory readout (``"1.2G"``) for CUDA or MPS; None on CPU."""
    try:
        d = collect_system()
        v = d.get("gpu_mem_gb")
        return None if v is None else f"{v:.1f}G"
    except Exception:  # noqa: BLE001
        return None
