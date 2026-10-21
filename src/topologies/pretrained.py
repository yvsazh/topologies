"""``save_pretrained`` / ``load_pretrained``: one directory = one reusable model.

Layout::

    my-model/
      config.json        # format, model class + constructor kwargs, recipe, metadata
      model.pt           # state_dict (loaded with weights_only=True)
      tokenizer/         # optional, any topologies tokenizer

The model is rebuilt from the recorded constructor call. It is recorded for

- framework-built models: anything made with ``topologies.build(...)``
  (``build("llama-decoder", ...)``, ``build(LLaMADecoder, ...)``) and every
  ``Experiment`` trial that uses the default model stage;
- ``topologies.nn`` stacks and other ``Configurable`` subclasses, even when
  constructed directly.

Directly constructed plain ``nn.Module`` classes, including catalog classes
(``LLaMADecoder(vocab=64)``) and models returned by a custom
``build_model(ctx)``, carry no record: pass ``model_kwargs=`` when saving,
construct them with ``build()``, or load with ``load_pretrained(path,
model=instance_or_class)``. Such a save emits a ``UserWarning`` and the
config says ``"complete": false``; ``load_pretrained`` refuses to guess.
Importable callables and ``slot(...)`` factories among the arguments are
recorded by import path; lambdas / closures are not.

Trust model: ``config.json`` names classes to import and call, so a
downloaded bundle is code, not data. ``load_pretrained`` only imports
objects from ``topologies.*`` and ``torch.*``, and the model target must be
an ``nn.Module`` subclass. Bundles that name your own classes need
``trusted_modules=["my_pkg"]``; ``trust_remote_code=True`` lifts the check
entirely and should only be used for files you created. Weights are always
loaded with ``torch.load(weights_only=True)``. See ``docs/security.md``.
"""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from topologies._util.fs import atomic_dir, atomic_save, atomic_write_json
from topologies._util.jsonsafe import to_jsonable
from topologies.build.config import (
    UntrustedTargetError,
    class_path,
    decode_value,
    init_config,
    make_init_record,
    safe_import_path,
    trusted_imports,
)

FORMAT = "topologies-pretrained"
VERSION = 1
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.pt"
TOKENIZER_DIR = "tokenizer"


@dataclass
class Pretrained:
    """Everything ``load_pretrained`` restores. Unpacks as ``model, tokenizer``."""

    model: nn.Module
    tokenizer: Any = None
    recipe: str | None = None
    recipe_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    path: Path | None = None

    def __iter__(self):
        yield self.model
        yield self.tokenizer


def _unwrap(model: nn.Module) -> nn.Module:
    from topologies.build.contract import unwrap

    return unwrap(model)


def save_pretrained(
    model: nn.Module,
    path: str | Path,
    *,
    tokenizer: Any = None,
    recipe: str | None = None,
    recipe_config: dict | None = None,
    model_kwargs: dict | None = None,
    metadata: dict | None = None,
    optimization_result: Any = None,
    model_factory: Any = None,
) -> Path:
    """Save ``model`` (+ tokenizer, recipe info) into directory ``path``."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    inner = _unwrap(model)
    if model_kwargs is not None:
        record = make_init_record(type(inner), dict(model_kwargs))
    else:
        record = init_config(inner) or {"target": class_path(inner), "kwargs": None,
                                        "complete": False}
    if not self_contained(record) and not (optimization_result is not None and model_factory is not None):
        warnings.warn(
            f"{record['target']}: constructor arguments were not "
            + ("recorded" if record.get("kwargs") is None else "fully recorded")
            + f", so {out} only loads with load_pretrained(path, model=...). Build the model "
            "with topologies.build(...), subclass topologies.build.Configurable, or pass "
            "model_kwargs={...} to save a self-contained checkpoint.",
            UserWarning, stacklevel=2)
    if optimization_result is not None:
        if optimization_result.model is not model:
            raise ValueError("optimization_result.model must be the model being saved")
        if model_factory is None:
            raise ValueError("Representation-aware pretrained bundles require model_factory=... for reconstruction")
        from topologies.optimization.artifacts import save_artifact
        save_artifact(optimization_result, out / "optimization", model_factory=model_factory,
                      metadata={"source_model": record, **(metadata or {})})
        version = 2
    else:
        state = {k: v.detach().cpu() for k, v in inner.state_dict().items()}
        atomic_save(out / WEIGHTS_FILE, lambda tmp: torch.save(state, tmp))
        version = VERSION
    tok_dir = None
    if tokenizer is not None:
        save = getattr(tokenizer, "save", None)
        if not callable(save):
            raise TypeError("tokenizer must provide save(path) to be bundled.")
        atomic_dir(out / TOKENIZER_DIR, save, keep_existing=False)
        tok_dir = TOKENIZER_DIR
    elif (out / TOKENIZER_DIR).is_dir():
        shutil.rmtree(out / TOKENIZER_DIR)  # a stale tokenizer must not load with new weights
    config = {
        "format": FORMAT,
        "version": version,
        "model": record,
        "tokenizer": tok_dir,
        "recipe": recipe,
        "recipe_config": recipe_config or {},
        "params": sum(p.numel() for p in inner.parameters()),
        "torch_version": torch.__version__,
        "metadata": _jsonable(metadata or {}),
    }
    if optimization_result is not None:
        config["optimization_artifact"] = "optimization"
        config["representation"] = _jsonable(optimization_result.config)
    # config.json last and atomically: load_pretrained never sees a half-written bundle.
    atomic_write_json(out / CONFIG_FILE, config, indent=2)
    return out


def _jsonable(obj: Any) -> Any:
    return to_jsonable(obj, nonfinite="keep")


def self_contained(record: dict | None) -> bool:
    """Whether a saved ``model`` record can rebuild the model on its own."""
    return bool(record) and record.get("kwargs") is not None and \
        record.get("complete", True) is not False


def read_config(path: str | Path) -> dict:
    cfg_file = Path(path) / CONFIG_FILE
    if not cfg_file.is_file():
        raise FileNotFoundError(f"No {CONFIG_FILE} in {path}; was it saved with save_pretrained?")
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    if cfg.get("format") != FORMAT:
        raise ValueError(f"{cfg_file} is not a {FORMAT} config.")
    return cfg


def load_pretrained(
    path: str | Path,
    *,
    model: nn.Module | type | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    device: str | torch.device | None = "cpu",
    strict: bool = True,
    trusted_modules: Iterable[str] = (),
    trust_remote_code: bool = False,
) -> Pretrained:
    """Rebuild a model saved with ``save_pretrained`` (weights, tokenizer, recipe).

    ``model``: optional instance (weights are loaded into it) or class
    (called with the saved kwargs) to override the recorded constructor.

    ``trusted_modules``: extra module prefixes the saved config may import
    from (default: only ``topologies`` and ``torch``). ``trust_remote_code``
    disables that check; use it only for bundles you created yourself.
    """
    with trusted_imports(trusted_modules, trust_remote_code=trust_remote_code):
        return _load_pretrained(path, model=model, model_factory=model_factory,
                                device=device, strict=strict)


def _model_factory(target: Any) -> Any:
    factory = safe_import_path(target)
    if not (isinstance(factory, type) and issubclass(factory, nn.Module)):
        raise UntrustedTargetError(
            f"untrusted constructor target {target!r}: not a torch.nn.Module subclass")
    return factory


def _load_pretrained(
    path: str | Path,
    *,
    model: nn.Module | type | None,
    model_factory: Callable[[], nn.Module] | None,
    device: str | torch.device | None,
    strict: bool,
) -> Pretrained:
    root = Path(path)
    cfg = read_config(root)
    rec = cfg.get("model") or {}
    if model_factory is not None and model is not None:
        raise ValueError("Pass only one of model= or model_factory=")
    if isinstance(model, nn.Module):
        net = model
    elif model_factory is not None:
        net = model_factory()
    else:
        factory = model if model is not None else _model_factory(rec.get("target", ""))
        kwargs = rec.get("kwargs")
        if kwargs is None:
            raise ValueError(
                f"{root} does not record how to construct {rec.get('target')} (the model "
                "was constructed directly, not with topologies.build). Pass "
                "load_pretrained(path, model=MyModel(...)), or save with model_kwargs={...}."
            )
        if model is None and rec.get("complete") is False:
            raise ValueError(
                f"{root} records only some constructor arguments of {rec.get('target')} "
                f"({sorted(kwargs)}); the others (e.g. lambdas) could not be saved. Pass "
                "load_pretrained(path, model=MyModel(...)), or save with model_kwargs={...}."
            )
        net = factory(**{k: decode_value(v) for k, v in kwargs.items()})
    if cfg.get("version") == 2:
        artifact_name = cfg.get("optimization_artifact")
        if not artifact_name:
            raise ValueError("Version-2 pretrained bundle has no representation codec reference")
        requested_device = str(device) if device is not None else "cpu"
        if requested_device != "cpu":
            raise ValueError("Optimized pretrained runtime is CPU-targeted; generic device movement is prohibited")
        from topologies.optimization.artifacts import load_artifact
        net = load_artifact(root / artifact_name, model_factory=lambda: net,
                            map_location="cpu", target="cpu")
        tokenizer = None
        if cfg.get("tokenizer"):
            from topologies.tokenizers import load_tokenizer
            tokenizer = load_tokenizer(root / cfg["tokenizer"])
        return Pretrained(model=net, tokenizer=tokenizer, recipe=cfg.get("recipe"),
                          recipe_config=cfg.get("recipe_config") or {},
                          metadata=cfg.get("metadata") or {}, path=root)
    if cfg.get("version") not in (None, VERSION):
        raise ValueError(f"Unsupported pretrained bundle version {cfg.get('version')!r}")
    state = torch.load(root / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    _unwrap(net).load_state_dict(state, strict=strict)
    if device is not None:
        net.to(device)
    net.eval()
    tokenizer = None
    if cfg.get("tokenizer"):
        from topologies.tokenizers import load_tokenizer

        tokenizer = load_tokenizer(root / cfg["tokenizer"])
    return Pretrained(
        model=net, tokenizer=tokenizer, recipe=cfg.get("recipe"),
        recipe_config=cfg.get("recipe_config") or {}, metadata=cfg.get("metadata") or {},
        path=root,
    )
