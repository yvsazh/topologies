"""Synthetic data that trains real models: generate, check, measure, improve.

Data is generated algorithmically: programs, grammars, templates, simulators and
renderers produce every value, and labels come from the latent values that
produced the observation.

    from topologies import synth

    spec = synth.Spec("support-intents", generator=make_record, size=20_000,
                      attributes={"label": ["refund", "shipping", "warranty"]})
    spec.plan(pilot=200)               # validate, preview, throughput, determinism
    ds = spec.run()                    # chunked, resumable, content-addressed
    ds.report()                        # diversity, coverage, balance, fidelity
    ds.evaluate(real_eval, model="TransformerClassifier", size="small", seeds=3).show()

Mental model: Spec -> Plan -> Run -> Dataset -> Quality -> (Loop). See docs/synth.md.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY: dict[str, tuple[str, str | None]] = {
    # nodes: the core
    "Node": ("topologies.synth.core", "Node"),
    "Ctx": ("topologies.synth.evaluate", "Ctx"),
    "Rand": ("topologies.synth.rng", "Rand"),
    "node": ("topologies.synth.core", "node"),
    "fn": ("topologies.synth.core", "fn"),
    "as_generator": ("topologies.synth.core", "as_generator"),
    "generator": ("topologies.synth.core", "generator"),
    "record": ("topologies.synth.core", "record"),
    "latent": ("topologies.synth.core", "latent"),
    "const": ("topologies.synth.core", "const"),
    "apply": ("topologies.synth.core", "apply"),
    "switch": ("topologies.synth.core", "switch"),
    "when": ("topologies.synth.core", "when"),
    "repeat": ("topologies.synth.core", "repeat"),
    "iid": ("topologies.synth.core", "iid"),
    "chain": ("topologies.synth.core", "chain"),
    "where": ("topologies.synth.core", "where"),
    "attr": ("topologies.synth.core", "attr"),
    "mixture": ("topologies.synth.core", "mixture"),
    "Mixture": ("topologies.synth.core", "Mixture"),
    "Evaluator": ("topologies.synth.evaluate", "Evaluator"),
    "counterfactual": ("topologies.synth.counterfactuals", "counterfactual"),
    "NodeFactory": ("topologies.synth.core", "NodeFactory"),
    "logged": ("topologies.synth.ctor", "logged"),
    "Recorded": ("topologies.synth.ctor", "Recorded"),
    "unknown": ("topologies.synth.errors", "unknown"),
    "canonical": ("topologies.synth.fingerprint", "fp_canonical"),
    "callable_fingerprint": ("topologies.synth.fingerprint", "callable_fingerprint"),
    "fingerprint_of": ("topologies.synth.fingerprint", "fingerprint"),
    "derive_seed": ("topologies.synth.rng", "derive_seed"),
    "as_records": ("topologies.synth.records", "as_records"),
    "callable_module": ("topologies.synth.ops._callable", "make_callable"),
    "lib": ("topologies.synth.lib", None),
    "random": ("topologies.synth.lib.random", None),
    "process": ("topologies.synth.lib.process", None),
    "graph": ("topologies.synth.lib.graph", None),
    "text": ("topologies.synth.lib.text", None),
    "audio": ("topologies.synth.lib.audio", None),
    "image": ("topologies.synth.lib.image", None),
    "series": ("topologies.synth.lib.series", None),
    "tabular": ("topologies.synth.lib.tabular", None),
    "seq": ("topologies.synth.lib.seq", None),
    "math": ("topologies.synth.lib.math", None),
    "describe": ("topologies.synth.introspect", "describe"),
    "sample": ("topologies.synth.introspect", "sample"),
    # random nodes
    "uniform": ("topologies.synth.dists", "uniform"),
    "normal": ("topologies.synth.dists", "normal"),
    "lognormal": ("topologies.synth.dists", "lognormal"),
    "truncnorm": ("topologies.synth.dists", "truncnorm"),
    "beta": ("topologies.synth.dists", "beta"),
    "gamma": ("topologies.synth.dists", "gamma"),
    "exponential": ("topologies.synth.dists", "exponential"),
    "poisson": ("topologies.synth.dists", "poisson"),
    "intuniform": ("topologies.synth.dists", "intuniform"),
    "loguniform": ("topologies.synth.dists", "loguniform"),
    "choice": ("topologies.synth.dists", "choice"),
    "bernoulli": ("topologies.synth.dists", "bernoulli"),
    "categorical": ("topologies.synth.dists", "categorical"),
    "zipf": ("topologies.synth.dists", "zipf"),
    "dirichlet": ("topologies.synth.dists", "dirichlet"),
    "mvnormal": ("topologies.synth.dists", "mvnormal"),
    # spec / schema / attributes
    "Spec": ("topologies.synth.spec", "Spec"),
    "Schema": ("topologies.synth.schema", "Schema"),
    "Field": ("topologies.synth.valuetypes", "Field"),
    "Text": ("topologies.synth.valuetypes", "Text"),
    "Label": ("topologies.synth.valuetypes", "Label"),
    "Labels": ("topologies.synth.valuetypes", "Labels"),
    "Json": ("topologies.synth.valuetypes", "Json"),
    "validate_json_schema": ("topologies.synth.jsonutil", "validate_json_schema"),
    "Number": ("topologies.synth.valuetypes", "Number"),
    "Category": ("topologies.synth.valuetypes", "Category"),
    "Tokens": ("topologies.synth.valuetypes", "Tokens"),
    "Image": ("topologies.synth.valuetypes", "Image"),
    "Series": ("topologies.synth.valuetypes", "Series"),
    "Spans": ("topologies.synth.valuetypes", "Spans"),
    "Type": ("topologies.synth.valuetypes", "Type"),
    "Tensor": ("topologies.synth.valuetypes", "Tensor"),
    "Audio": ("topologies.synth.valuetypes", "Audio"),
    "Video": ("topologies.synth.valuetypes", "Video"),
    "Mask": ("topologies.synth.valuetypes", "Mask"),
    "Boxes": ("topologies.synth.valuetypes", "Boxes"),
    "Keypoints": ("topologies.synth.valuetypes", "Keypoints"),
    "PointCloud": ("topologies.synth.valuetypes", "PointCloud"),
    "Graph": ("topologies.synth.valuetypes", "Graph"),
    "register_type": ("topologies.synth.valuetypes", "register_type"),
    "infer_type": ("topologies.synth.valuetypes", "infer_type"),
    "codecs": ("topologies.synth.codecs", None),
    "dsp": ("topologies.synth.dsp", None),
    "preview": ("topologies.synth.previews", "preview"),
    "stream": ("topologies.synth.streaming", "stream"),
    "Stream": ("topologies.synth.streaming", "Stream"),
    "AttributeSpace": ("topologies.synth.attributes", "AttributeSpace"),
    "pool": ("topologies.synth.attributes", "pool"),
    "personas": ("topologies.synth.lib.text.pools", "persona_pool"),
    "topics": ("topologies.synth.lib.text.pools", "topic_pool"),
    "sampling": ("topologies.synth.sampling", None),
    "quota": ("topologies.synth.sampling", "quota"),
    "pairwise": ("topologies.synth.sampling", "pairwise"),
    "stratified": ("topologies.synth.sampling", "stratified"),
    "lhs": ("topologies.synth.sampling", "lhs"),
    "adaptive": ("topologies.synth.sampling", "adaptive"),
    # class-based generators
    "Generator": ("topologies.synth.generators.base", "Generator"),
    "GenContext": ("topologies.synth.generators.base", "GenContext"),
    "Raw": ("topologies.synth.generators.base", "Raw"),
    "from_program": ("topologies.synth.core", "from_program"),
    "Program": ("topologies.synth.core", "Program"),
    "Grammar": ("topologies.synth.lib.text.legacy", "Grammar"),
    "SCM": ("topologies.synth.lib.tabular", "SCM"),
    "ProbeSuite": ("topologies.synth.tasks", "ProbeSuite"),
    "augment": ("topologies.synth.ops.augment", None),
    # pipeline / checks
    "ops": ("topologies.synth.ops", None),
    "Verdict": ("topologies.synth.verdict", "Verdict"),
    "Reject": ("topologies.synth.ops.base", "Reject"),
    # datasets
    "SynthDataset": ("topologies.synth.dataset", "SynthDataset"),
    "load": ("topologies.synth.dataset", "load"),
    "mix": ("topologies.synth.dataset", "mix"),
    "load_records": ("topologies.synth.records", "load_records"),
    "split_records": ("topologies.synth.records", "split_records"),
    "list_datasets": ("topologies.synth.store", "list_datasets"),
    # quality / loop / experiments
    "quality": ("topologies.synth.quality", None),
    "scaling_curve": ("topologies.synth.quality.utility", "scaling_curve"),
    "mix_curve": ("topologies.synth.quality.utility", "mix_curve"),
    "ablate": ("topologies.synth.quality.utility", "ablate"),
    "Loop": ("topologies.synth.loop", "Loop"),
    "SynthTask": ("topologies.synth.tasks", "SynthTask"),
    "tasks": ("topologies.synth.tasks", None),
    "review": ("topologies.synth.reviewing", "review"),
    "blueprints": ("topologies.synth.blueprints", None),
    "errors": ("topologies.synth.errors", None),
    "SynthError": ("topologies.synth.errors", "SynthError"),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'topologies.synth' has no attribute {name!r}")
    module = importlib.import_module(target[0])
    value = module if target[1] is None else getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
