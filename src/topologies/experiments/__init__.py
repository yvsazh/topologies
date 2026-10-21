"""Experiments: declare what varies, run every trial (resumably), compare rigorously.

    from topologies import Experiment, data
    from topologies.experiments import LanguageModeling, Variant
    from topologies.nn import TransformerLM

    docs = data.read_documents("data/tinystories/tinystories_mini_train.txt")
    exp = Experiment("arch-x-tokenizer",
                     task=LanguageModeling(docs, val_fraction=0.1, block_size=128),
                     axes={"model": [TransformerLM,
                                     Variant("llama-ish", TransformerLM, pos="rope",
                                             norm="rms", activation="swiglu")],
                           "tokenizer": ["char", "bpe:1k", "bpe:4k"]},
                     target_params="1M", train_for="2 epochs", seeds=2)
    exp.run().show()

Then ``topologies exp show arch-x-tokenizer`` (and ``pivot``, ``curves``,
``report`` ...). Three levels of control share everything: ``axes`` + ``task``
(Level 1), stage overrides ``build_model`` / ``build_data`` / ``make_trainer`` /
``evaluate`` (Level 2), and ``@experiment`` trial functions (Level 3).
See docs/experiments.md.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from topologies.experiments.results import ResultSet

_LAZY: dict[str, tuple[str, str | None]] = {
    "ExperimentError": ("topologies.experiments.errors", "ExperimentError"),
    "Pruned": ("topologies.experiments.errors", "Pruned"),
    "TrialTimeout": ("topologies.experiments.errors", "TrialTimeout"),
    "Experiment": ("topologies.experiments.core", "Experiment"),
    "experiment": ("topologies.experiments.core", "experiment"),
    "MetricInfo": ("topologies.experiments.metrics", "MetricInfo"),
    "metric_info": ("topologies.experiments.metrics", "metric_info"),
    "register_metric": ("topologies.experiments.metrics", "register_metric"),
    "Plan": ("topologies.experiments.planning", "Plan"),
    "MedianPruner": ("topologies.experiments.pruners", "MedianPruner"),
    "Pruner": ("topologies.experiments.pruners", "Pruner"),
    "ConfigSummary": ("topologies.experiments.results", "ConfigSummary"),
    "ResultSet": ("topologies.experiments.results", "ResultSet"),
    "Table": ("topologies.experiments.results", "Table"),
    "TrialRecord": ("topologies.experiments.results", "TrialRecord"),
    "load": ("topologies.experiments.results", "load"),
    "load_all": ("topologies.experiments.results", "load_all"),
    "Fixed": ("topologies.experiments.space", "Fixed"),
    "Variant": ("topologies.experiments.space", "Variant"),
    "choice": ("topologies.experiments.space", "choice"),
    "fixed": ("topologies.experiments.space", "fixed"),
    "intloguniform": ("topologies.experiments.space", "intloguniform"),
    "intuniform": ("topologies.experiments.space", "intuniform"),
    "loguniform": ("topologies.experiments.space", "loguniform"),
    "uniform": ("topologies.experiments.space", "uniform"),
    "Datasets": ("topologies.experiments.tasks", "Datasets"),
    "ImageClassification": ("topologies.experiments.tasks", "ImageClassification"),
    "LanguageModeling": ("topologies.experiments.tasks", "LanguageModeling"),
    "Supervised": ("topologies.experiments.tasks", "Supervised"),
    "Task": ("topologies.experiments.tasks", "Task"),
    "TrialContext": ("topologies.experiments.trial", "TrialContext"),
    "TrialOutput": ("topologies.experiments.trial", "TrialOutput"),
    "validate_result": ("topologies.experiments.trial", "validate_result"),
    "Insight": ("topologies.experiments.insights", "Insight"),
    "bench": ("topologies.experiments.bench", None),
}

#: Routing keys whose ``{label: value}`` dict values become axes in ``compare()``.
_DICT_AXIS_KEYS = frozenset({"model", "tokenizer", "optimizer", "scheduler", "loss",
                             "recipe", "size", "target_params", "train_for", "search"})


def _options() -> set[str]:
    from topologies.experiments.core import Experiment

    return set(inspect.signature(Experiment.__init__).parameters) - {"self", "name"}


def compare(*, name: str | None = None, **kwargs: Any) -> ResultSet:
    """One call: lists become axes, everything else is a setting; runs and returns results.

    >>> res = compare(task=LanguageModeling(train_docs, val_docs),
    ...               model=[TransformerLM, "my_pkg.nets:MyLM"], tokenizer=["bpe:1k", "bpe:4k"],
    ...               target_params="1M", train_for="2 epochs", seeds=2)

    A list (or a ``{label: value}`` dict for routing keys such as ``model``) is an
    axis; wrap list-valued constants with ``fixed([...])``. Experiment options
    (``seeds``, ``goal``, ``train``, ``workers``, ...) keep their meaning. The name
    is ``compare-<axes>-<hash>``.
    """
    from topologies.experiments.cache import stable_hash
    from topologies.experiments.core import Experiment
    from topologies.experiments.errors import ExperimentError
    from topologies.experiments.space import Fixed
    from topologies.experiments.spec import canonical

    options = _options()
    axes: dict[str, Any] = dict(kwargs.pop("axes", None) or {})
    fixed_: dict[str, Any] = dict(kwargs.pop("fixed", None) or {})
    opts: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, Fixed):
            (opts if k in options else fixed_)[k] = v.value
        elif k in options and k != "model":
            opts[k] = v
        elif isinstance(v, (list, tuple)):
            axes[k] = list(v)
        elif isinstance(v, dict) and (k in _DICT_AXIS_KEYS or "." in k):
            axes[k] = v
        elif k in options:
            opts[k] = v
        else:
            fixed_[k] = v
    if not axes:
        raise ExperimentError("compare() needs at least one list-valued argument (an axis).",
                              stage="config", hint="compare(model=[A, B], ...)")
    digest = stable_hash({"axes": canonical(axes, warn=False),
                          "fixed": canonical(fixed_, warn=False),
                          "opts": canonical({k: v for k, v in opts.items()
                                             if k not in ("monitor", "workers", "root")},
                                            warn=False)}, n=6)
    exp = Experiment(name or f"compare-{'-x-'.join(axes)}-{digest}", axes=axes,
                     fixed=fixed_, **opts)
    return exp.run()


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target[0])
    value = module if target[1] is None else getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ConfigSummary",
    "Datasets",
    "Experiment",
    "ExperimentError",
    "Fixed",
    "ImageClassification",
    "Insight",
    "LanguageModeling",
    "MedianPruner",
    "MetricInfo",
    "Plan",
    "Pruned",
    "Pruner",
    "ResultSet",
    "Supervised",
    "Table",
    "Task",
    "TrialContext",
    "TrialOutput",
    "TrialRecord",
    "TrialTimeout",
    "Variant",
    "bench",
    "choice",
    "compare",
    "experiment",
    "fixed",
    "intloguniform",
    "intuniform",
    "load",
    "load_all",
    "loguniform",
    "metric_info",
    "register_metric",
    "uniform",
    "validate_result",
]
