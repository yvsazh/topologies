"""Size presets and budget knobs for catalog classes, declared outside the catalog.

Catalog files (``topologies.models`` / ``topologies.layers``) are standalone
PyTorch code and carry no framework decorators. Their build metadata lives
here, keyed by registry id, and is picked up by ``get_presets`` /
``get_budget_knobs`` for the canonical catalog class only, so

    build("transformer-lm", "small", vocab_size=512, context_length=128)
    build(TransformerLM, target_params="200k", vocab_size=512, context_length=128)

work without the model file knowing about ``topologies.build``. Every key
must be a constructor argument; ``catalog_spec_problems()`` checks that
(tests + ``topologies registry validate --check-imports``).
"""

from __future__ import annotations

import inspect
from typing import Any

# Capacity presets are projected from the reviewed module-local PRESETS.  The
# fallback only varies scalar width/embedding dimensions that are genuine
# constructor parameters; task geometry (vocabulary, image size, classes,
# sequence/window lengths) is never a budget knob.
_SIZE_SCALE = {"tiny": 0.5, "small": 1.0, "base": 2.0}
_CAPACITY_NAMES = (
    "hidden_size",
    "hidden_features",
    "hidden_width",
    "hidden_dim",
    "width",
    "embedding_dim",
    "embed_dim",
    "d_model",
    "feature_width",
    "conv_channels",
    "residual_channels",
)


def _catalog_build_specs() -> dict[str, dict[str, dict[str, Any]]]:
    from topologies._catalog import entries

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in ("model", "layer"):
        for entry in entries(kind).values():
            cls = entry.load()
            params = inspect.signature(cls).parameters
            local = getattr(cls, "PRESETS", {})
            presets: dict[str, dict[str, Any]] = {}
            for size in ("tiny", "small", "base"):
                value = local.get(size) if isinstance(local, dict) else None
                if isinstance(value, dict):
                    presets[size] = dict(value)
                elif value is not None:
                    # Tuple-valued family presets conventionally describe a
                    # hidden/features stack; named-preset constructors own
                    # interpretation when they expose `preset`.
                    if "preset" in params:
                        presets[size] = {"preset": size}
                    else:
                        field = next(
                            (n for n in ("features", "hidden_dims", "widths") if n in params), None
                        )
                        presets[size] = {field: value} if field else {}
                else:
                    field = next(
                        (
                            n
                            for n in _CAPACITY_NAMES
                            if n in params
                            and params[n].default is not inspect.Parameter.empty
                            and isinstance(params[n].default, int)
                            and not isinstance(params[n].default, bool)
                        ),
                        None,
                    )
                    presets[size] = (
                        {field: max(1, int(params[field].default * _SIZE_SCALE[size]))}
                        if field
                        else {}
                    )
            knobs: dict[str, range] = {}
            if kind == "model":
                for name in _CAPACITY_NAMES:
                    p = params.get(name)
                    if (
                        p
                        and p.default is not inspect.Parameter.empty
                        and isinstance(p.default, int)
                        and not isinstance(p.default, bool)
                        and p.default > 1
                    ):
                        knobs[name] = range(
                            max(1, p.default // 2), p.default * 4 + 1, max(1, p.default // 4)
                        )
                        break
                declared_knobs = getattr(cls, "BUDGET_KNOBS", None)
                if declared_knobs is not None:
                    if not isinstance(declared_knobs, dict):
                        raise ValueError(f"{entry.id}: BUDGET_KNOBS must map constructor fields to ranges")
                    for name, values in declared_knobs.items():
                        if (name not in params or not isinstance(values, range) or not values
                                or any(value <= 0 for value in values)):
                            raise ValueError(f"{entry.id}: invalid budget candidates for {name!r}")
                    knobs = dict(declared_knobs)
            result[entry.id] = {"presets": presets, "budget_knobs": knobs}
    # These families expose a sequence of hidden widths, with a supported
    # scalar spelling for a single hidden stage. Keep their task feature/label
    # dimensions fixed while searching capacity.
    for ident in ("mlp", "tabular-mlp"):
        result[ident] = {
            "presets": {"tiny": {"hidden_dims": [32]}, "small": {"hidden_dims": [128, 64]},
                        "base": {"hidden_dims": [512, 256, 128]}},
            "budget_knobs": {"hidden_dims": range(8, 2049, 8)},
        }
    for ident in ("actor-critic", "q-network"):
        result[ident]["presets"] = {"tiny": {"hidden_dims": [16]},
                                    "small": {"hidden_dims": [64, 64]},
                                    "base": {"hidden_dims": [256, 256]}}
    result["graph-network"]["presets"] = {
        size: {"hidden_features": width} for size, width in (("tiny", 16), ("small", 64), ("base", 128))
    }
    result["graph-network"]["budget_knobs"] = {"hidden_features": range(4, 257, 4)}
    result["charformer"] = {
        "presets": {
            "tiny": {"d_model": 64, "num_layers": 2, "num_heads": 4, "intermediate_size": 128},
            "small": {"d_model": 256, "num_layers": 6, "num_heads": 8, "intermediate_size": 1024},
            "base": {"d_model": 768, "num_layers": 12, "num_heads": 12, "intermediate_size": 3072},
        },
        "budget_knobs": {"d_model": range(32, 193, 16)},
    }
    result["gpt-2"] = {
        "presets": {name: {"preset": name} for name in ("tiny", "small", "base")},
        "budget_knobs": {"d_model": range(64, 257, 8)},
    }
    result["gpt-j-neox-pythia"] = {
        "presets": {
            **{name: {"preset": name} for name in ("tiny", "small", "base")},
            **{name: {"native_config": name} for name in (
                "gpt-j-6b", "gpt-neox-20b", "pythia-70m", "pythia-160m")},
        },
        "budget_knobs": {"d_model": range(32, 257, 8)},
    }
    result["gpt-decoder-family"] = {
        "presets": {
            **{name: {"family": "gpt3", "preset": name} for name in ("tiny", "small", "base")},
            **{f"{family}-{name}": {"family": family, "preset": name}
               for family in ("gpt3", "opt", "bloom", "falcon")
               for name in ("tiny", "small", "base")},
        },
        "budget_knobs": {"d_model": range(64, 257, 8)},
    }
    result["llama"] = {
        "presets": {
            **{name: {"preset": name} for name in ("tiny", "small", "base")},
            **{name: {"size": name} for name in (
                "llama1-7b", "llama1-13b", "llama1-30b", "llama1-65b",
                "llama2-7b", "llama2-13b", "llama2-70b", "llama3-8b", "llama3-70b",
            )},
        },
        "budget_knobs": {"d_model": range(16, 257, 8)},
    }
    return result


CATALOG_BUILD: dict[str, dict[str, dict[str, Any]]] = _catalog_build_specs()


def catalog_spec(target: Any) -> dict[str, dict[str, Any]]:
    """Build metadata for ``target`` if it is a canonical catalog class, else ``{}``."""
    if not isinstance(target, type):
        return {}
    from topologies._catalog import catalog_id

    found = catalog_id(target)
    return CATALOG_BUILD.get(found[1], {}) if found else {}


def catalog_spec_problems() -> list[str]:
    """Spec entries that do not match a catalog class or its constructor."""
    from topologies._catalog import entries

    by_id = {e.id: e for kind in ("model", "layer") for e in entries(kind).values()}
    problems: list[str] = []
    for ident, spec in CATALOG_BUILD.items():
        entry = by_id.get(ident)
        if entry is None:
            problems.append(f"{ident}: no catalog class with this id")
            continue
        cls = entry.load()
        params = inspect.signature(cls).parameters
        for size, kw in spec.get("presets", {}).items():
            unknown = sorted(set(kw) - set(params))
            if unknown:
                problems.append(f"{ident}: preset {size!r} sets unknown arguments {unknown}")
        for knob, values in spec.get("budget_knobs", {}).items():
            if knob not in params:
                problems.append(f"{ident}: budget knob {knob!r} is not a constructor argument")
            elif not list(values) or not all(isinstance(v, int) for v in values):
                problems.append(f"{ident}: budget knob {knob!r} needs integer candidates")
    return problems
