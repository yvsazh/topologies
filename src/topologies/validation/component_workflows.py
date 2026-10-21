"""Synth-backed smoke workflows for standalone catalog components."""

from __future__ import annotations

import math
from typing import Any

from topologies import analysis, synth
from topologies.validation.catalog_specs import catalog_specs, decode


def component_ids() -> tuple[str, ...]:
    from topologies._catalog import entries

    return tuple(sorted(entry.id for entry in entries("layer").values()))


def _entry(ident: str):
    try:
        spec = catalog_specs()[ident]
    except KeyError:
        raise ValueError(f"No component fixture registered for {ident!r}") from None
    if ident not in component_ids():
        raise ValueError(f"{ident!r} is not an active catalog component")
    return spec


def build_component(ident: str):
    """Construct a component with its explicit catalog-sized constructor."""
    from topologies import layers

    return layers.get_layer(ident)(**dict(_entry(ident)["constructor"]))


def component_generator(ident: str):
    """Return a synth generator whose fixture tensors are decoded inside generation."""
    def generate(ctx):
        if ident == "top-2-router":
            import torch

            features = torch.randn((3, 3), generator=ctx.rng.torch)
            expert_outputs = torch.randn((3, 3, 2), generator=ctx.rng.torch)
            return {"features": features, "expert_outputs": expert_outputs,
                    "targets": expert_outputs.mean(1)}
        specification = _entry(ident)
        args = decode(specification.get("args", []), seed=41 + int(ctx.index))
        kwargs = decode(specification.get("kwargs", {}), seed=41 + int(ctx.index))
        if ident == "longformer-attention":
            import torch

            global_mask = torch.zeros(args[0].shape[:-1], dtype=torch.bool)
            global_mask[..., 0] = True
            global_mask[..., -1] = True
            kwargs["global_attention_mask"] = global_mask
        def sample_geometry(value):
            import torch

            if torch.is_tensor(value) and value.ndim > 1 and ident not in {"graph-conv", "graph-readout"}:
                return value[0]
            if isinstance(value, list):
                return [sample_geometry(item) for item in value]
            if isinstance(value, tuple):
                return tuple(sample_geometry(item) for item in value)
            if isinstance(value, dict):
                return {key: sample_geometry(item) for key, item in value.items()}
            return value

        args = sample_geometry(args)
        kwargs = sample_geometry(kwargs)
        if ident == "audio-frontend" and "waveform" in kwargs:
            import torch

            # The packaged smoke input is silence; with the log-mel floor it
            # has zero input gradient. Use deterministic non-silent synthetic
            # signals for the downstream workflow's independent gradient check.
            length = kwargs["waveform"].shape[-1]
            time = torch.arange(length, dtype=kwargs["waveform"].dtype)
            frequency = 0.15 + 0.01 * (int(ctx.index) % 5)
            phase = 0.2 * int(ctx.index)
            kwargs["waveform"] = torch.sin(time * frequency + phase)
        return {"args": args, "kwargs": kwargs, "target": float((int(ctx.index) % 5) / 5)}

    return generate


def component_spec(ident: str, *, root: str = "component-evidence", seed: int = 19, size: int = 12):
    """Create deterministic component evidence data using synth.Spec."""
    return synth.Spec(ident, generator=component_generator(ident), size=size,
                      splits={"train": 0.75, "val": 0.25}, seed=seed, root=root,
                      monitor="none")


def _stack(value: Any, template: Any, torch: Any):
    if torch.is_tensor(template):
        return torch.stack(value)
    if isinstance(template, dict):
        return {key: _stack([sample[key] for sample in value], child, torch)
                for key, child in template.items()}
    if isinstance(template, (list, tuple)):
        values = [_stack([sample[index] for sample in value], child, torch)
                  for index, child in enumerate(template)]
        return type(template)(values)
    return template


def _component_collate(ident: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if ident == "top-2-router":
        return {key: torch.stack([sample[key] for sample in samples])
                for key in ("features", "expert_outputs", "targets")}
    template = samples[0]
    args = _stack([sample["args"] for sample in samples], template["args"], torch)
    if ident == "auprc":
        # AveragePrecision's API consumes one flat ranking, not a batch-by-item
        # matrix. A batch of one is used by its workflow trainer below; retain
        # the fixture's vector dimensions here rather than adding a batch axis.
        if len(samples) != 1:
            raise ValueError("auprc workflow batches must contain exactly one ranking")
        kwargs = template["kwargs"]
    else:
        kwargs = _stack([sample["kwargs"] for sample in samples], template["kwargs"], torch)
    target = torch.tensor([sample["target"] for sample in samples], dtype=torch.float32).unsqueeze(-1)
    if ident == "graph-conv":
        x = args[0]
        edges = args[1]
        offsets = torch.arange(len(samples), device=x.device, dtype=edges.dtype).reshape(-1, 1, 1) * x.shape[1]
        args = [x.reshape(-1, x.shape[-1]), (edges + offsets).permute(1, 0, 2).reshape(2, -1)]
    elif ident == "graph-readout":
        x, assignments = args
        graph_stride = int(assignments.max()) + 1 if assignments.numel() else 1
        offsets = torch.arange(len(samples), device=assignments.device, dtype=assignments.dtype).reshape(-1, 1) * graph_stride
        args = [x.reshape(-1, *x.shape[2:]), (assignments + offsets).reshape(-1)]
    component = build_component(ident)
    if not any(parameter.requires_grad for parameter in component.parameters()):
        args = _mark_input_gradients(args, torch)
        kwargs = _mark_input_gradients(kwargs, torch)
    return {"inputs": {"args": args, "kwargs": kwargs}, "targets": target}


def _mark_input_gradients(value: Any, torch: Any):
    if torch.is_tensor(value):
        return value.detach().requires_grad_(value.is_floating_point() or value.is_complex())
    if isinstance(value, list):
        return [_mark_input_gradients(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_mark_input_gradients(item, torch) for item in value)
    if isinstance(value, dict):
        return {key: _mark_input_gradients(item, torch) for key, item in value.items()}
    return value


from topologies.synth.torch_views import Converter, ListDataset, register_converter


@register_converter("component-smoke", "router-probe")
class ComponentSmoke(Converter):
    label_key = None

    def fit(self, records):
        self.ident = self.kw["component_id"]
        _entry(self.ident)
        return self

    def convert(self, record):
        if self.ident == "top-2-router":
            return {key: record[key] for key in ("features", "expert_outputs", "targets")}
        return {key: record[key] for key in ("args", "kwargs", "target")}

    @property
    def collate_fn(self):
        return lambda samples: _component_collate(self.ident, samples)

    def dataset(self, records):
        return ListDataset([self(record) for record in records], collate_fn=self.collate_fn,
                           label_key=None, info={"component_id": self.ident})


def _select_component_features(ident: str, output: Any):
    import torch

    auxiliary_loss = None
    if ident == "feature-pyramid":
        if not isinstance(output, (list, tuple)) or not output:
            raise TypeError("feature-pyramid binding expects a non-empty multilevel sequence")
        features = torch.cat([level.flatten(start_dim=2).mean(dim=-1) for level in output], dim=-1)
    elif ident == "autoformer":
        if (not isinstance(output, (tuple, list)) or len(output) != 2
                or not all(torch.is_tensor(value) for value in output)):
            raise TypeError("autoformer binding expects (seasonal, trend) tensors")
        # Preserve both terms of Autoformer's X = X_s + X_t decomposition as
        # pooled feature channels for the downstream regression readout.
        return torch.cat([value.flatten(start_dim=1).mean(dim=1, keepdim=True)
                          for value in output], dim=-1), auxiliary_loss
    elif ident == "vector-quantizer":
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise TypeError("vector-quantizer binding expects (quantized, indices, auxiliary_loss)")
        auxiliary_loss = output[2]
        output = output[0]
    elif ident in {"diagonal-ssm", "recurrent-backbone", "patch-embedding"}:
        output = output[0]
    if ident != "feature-pyramid":
        if not torch.is_tensor(output):
            raise TypeError(f"{ident} requires an explicit tensor output binding")
        if output.is_complex():
            output = torch.view_as_real(output)
        if ident == "graph-conv":
            output = output.reshape(-1, 3, output.shape[-1]).mean(dim=1)
        elif ident == "graph-readout":
            output = output.reshape(-1, 2, output.shape[-1]).mean(dim=1)
        elif output.ndim > 2:
            output = output.flatten(start_dim=2).mean(dim=-1)
        if output.ndim == 1:
            output = output.unsqueeze(-1)
        elif output.ndim == 0:
            output = output.reshape(1, 1)
        features = output
    return features, auxiliary_loss


def compose_component(ident: str, component: Any = None):
    """Compose frozen-or-trainable component with a trainable scalar regression readout."""
    _ensure_component_recipe()
    import torch
    from torch import nn

    component = component or build_component(ident)

    class Composition(nn.Module):
        def __init__(self):
            super().__init__()
            self.component = component
            self.readout = nn.LazyLinear(1)
            self._parameter_gradient_events = {name: [] for name, parameter in component.named_parameters()
                                               if parameter.requires_grad}
            self._input_gradient_events: dict[str, list[dict[str, bool]]] = {}
            for name, parameter in component.named_parameters():
                if parameter.requires_grad:
                    parameter.register_hook(lambda gradient, key=name: self._record_gradient(
                        self._parameter_gradient_events, key, gradient))

        @staticmethod
        def _record_gradient(store, key, gradient):
            store.setdefault(key, []).append({"finite": bool(torch.isfinite(gradient).all()),
                                               "nonzero": bool(gradient.abs().sum() > 0)})
            return gradient

        def _watch_inputs(self, inputs):
            def watch(value, path):
                if torch.is_tensor(value):
                    if value.requires_grad:
                        value.register_hook(lambda gradient, key=path: self._record_gradient(
                            self._input_gradient_events, key, gradient))
                elif isinstance(value, dict):
                    for key, item in value.items():
                        watch(item, f"{path}.{key}")
                elif isinstance(value, (tuple, list)):
                    for index, item in enumerate(value):
                        watch(item, f"{path}.{index}")

            watch(inputs, "input")

        def forward(self, inputs):
            if not any(parameter.requires_grad for parameter in self.component.parameters()):
                self._watch_inputs(inputs)
            output = self.component(*inputs["args"], **inputs["kwargs"])
            features, auxiliary_loss = _select_component_features(ident, output)
            prediction = self.readout(features)
            self.observed_output_shape = tuple(prediction.shape)
            result = {"prediction": prediction}
            if auxiliary_loss is not None:
                result["auxiliary_loss"] = auxiliary_loss
            return result

        def gradient_evidence(self):
            if self._parameter_gradient_events:
                return {"kind": "parameters", "items": self._parameter_gradient_events}
            return {"kind": "inputs", "items": self._input_gradient_events}

    return Composition()


def make_trainer(ident: str, dataset: Any, *, max_epochs: int = 2, component: Any = None):
    """Create Trainer with component-smoke recipe and real downstream readout training."""
    _ensure_component_recipe()
    from topologies import Trainer

    if ident == "top-2-router":
        train = dataset.for_recipe("router-probe", split="train", component_id=ident)
        val = dataset.for_recipe("router-probe", split="val", component_id=ident)
        return Trainer(component or build_component(ident), train, val,
                       recipe="router-probe", batch_size=2, max_epochs=max_epochs,
                       device="cpu", monitor="none", logger="none")

    train = dataset.for_recipe("component-smoke", split="train", component_id=ident)
    val = dataset.for_recipe("component-smoke", split="val", component_id=ident)
    return Trainer(compose_component(ident, component=component), train, val,
                   recipe="component-smoke-regression",
                   batch_size=1 if ident == "auprc" else 2,
                   max_epochs=max_epochs, device="cpu", monitor="none", logger="none")


def _ensure_component_recipe():
    import torch.nn.functional as functional

    from topologies.training.adapters import call_with_adapter
    from topologies.training.recipe_base import (
        Recipe,
        RecipeComponents,
        RecipeConfig,
        register_recipe,
    )

    try:
        from topologies.training.recipe_base import _RECIPE_REGISTRY
        if "component-smoke-regression" in _RECIPE_REGISTRY:
            return
    except ImportError:
        pass

    @register_recipe("component-smoke-regression", help="Regression smoke objective with optional component auxiliary loss.")
    class ComponentSmokeRecipe(Recipe):
        ConfigSchema = RecipeConfig
        ComponentsSchema = RecipeComponents

        def expected_batch_spec(self):
            return {"inputs": "structured component inputs", "targets": "float[B,1]"}

        def batch_to_loss(self, model, batch, cfg, components, adapter=None):
            result = call_with_adapter(adapter, model, {"inputs": batch["inputs"]}, primary="inputs")
            prediction = result["prediction"]
            prediction_loss = functional.mse_loss(prediction, batch["targets"])
            auxiliary = result.get("auxiliary_loss")
            loss = prediction_loss if auxiliary is None else prediction_loss + auxiliary
            return loss, {"prediction_mse": float(prediction_loss.detach()),
                          "auxiliary_loss": float(auxiliary.detach()) if auxiliary is not None else 0.0}


class _ComponentRuntimeAnalysis:
    name = "component_runtime"

    def run(self, data, context):
        ident, module, args, kwargs, seed = data
        import torch

        was_training = module.training
        module.eval()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                first = module(*args, **kwargs)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                second = module(*args, **kwargs)
        finally:
            module.train(was_training)
        first_outputs = _bound_component_outputs(ident, first)
        second_outputs = _bound_component_outputs(ident, second)
        shapes = {name: tuple(value.shape) for name, value in first_outputs}
        finite = all(bool(torch.isfinite(value).all()) for _, value in first_outputs
                     if value.is_floating_point() or value.is_complex())
        deterministic = len(first_outputs) == len(second_outputs) and all(
            name_a == name_b and torch.equal(value_a, value_b)
            for (name_a, value_a), (name_b, value_b) in zip(first_outputs, second_outputs)
        )
        return analysis.AnalysisResult({"output_shapes": shapes, "finite": finite,
                                        "deterministic": bool(deterministic)},
                                       kind="measured", name=self.name,
                                       provenance={"analyzer": type(self).__name__, "seed": seed})


def _bound_component_outputs(ident: str, output: Any) -> list[tuple[str, Any]]:
    if ident == "top-2-router":
        return [("indices", output[0]), ("weights", output[1]), ("auxiliary_loss", output[2])]
    if ident == "feature-pyramid":
        return [(f"level_{index}", tensor) for index, tensor in enumerate(output)]
    if ident == "autoformer":
        return [("seasonal", output[0]), ("trend", output[1])]
    if ident == "vector-quantizer":
        return [("quantized", output[0]), ("indices", output[1]), ("auxiliary_loss", output[2])]
    if ident in {"diagonal-ssm", "recurrent-backbone", "patch-embedding"}:
        return [("features", output[0])]
    return [("output", output)]


def measure_component(ident: str, *, root: str = "component-measure", size: int = 12,
                      component: Any = None) -> dict[str, Any]:
    """Train and report measured component/readout properties using Trainer and analysis."""
    from topologies._execution import execution_rng

    with execution_rng(seed=11):
        return _measure_component(ident, root=root, size=size, component=component)


def _measure_component(ident: str, *, root: str, size: int, component: Any) -> dict[str, Any]:
    import torch

    if ident == "top-2-router":
        dataset = component_spec(ident, root=root, size=size).run()
        router = component or build_component(ident)
        trainer = make_trainer(ident, dataset, max_epochs=2, component=router)
        fit = trainer.fit()
        evaluation = trainer.evaluate(dataset.for_recipe("router-probe", split="val", component_id=ident))
        report = analysis.analyze_model(router)
        val_view = dataset.for_recipe("router-probe", split="val", component_id=ident)
        batch = next(iter(torch.utils.data.DataLoader(val_view, batch_size=2, collate_fn=val_view.collate_fn)))
        features = batch["features"]
        router.eval()
        with torch.no_grad():
            first, second = router(features), router(features)
        outputs = _bound_component_outputs(ident, first)
        items = {name: [{"finite": p.grad is not None and bool(torch.isfinite(p.grad).all()),
                         "nonzero": p.grad is not None and bool(p.grad.abs().sum() > 0)}]
                 for name, p in router.named_parameters()}
        train_loss = [float(item["train_loss"]) for item in fit["history"] if "train_loss" in item]
        return {"id": ident, "fit": fit, "evaluation": evaluation,
                "parameter_count": report.payload["structure"]["parameters"]["total"],
                "trainable_parameters": report.payload["structure"]["parameters"]["trainable"],
                "analysis": report, "composition_analysis": report,
                "component_trainable": True, "readout_trainable": False,
                "forward_finite": all(bool(torch.isfinite(value).all()) for _, value in outputs)
                                  and math.isfinite(float(evaluation["loss"])),
                "output_shapes": {name: tuple(value.shape) for name, value in outputs},
                "gradient_flow": bool(items) and all(ev[0]["finite"] and ev[0]["nonzero"] for ev in items.values())
                                  and fit["global_step"] > 0,
                "gradient_evidence": {"kind": "parameters", "items": items},
                "train_loss": train_loss,
                "deterministic": component_spec(ident, root=root, size=size).fingerprint() ==
                                 component_spec(ident, root=root + "-determinism", size=size).fingerprint()
                                 and all(torch.equal(a, b) for a, b in zip(first, second)),
                "composition_note": "The router gate was optimized through selected-expert outputs weighted by top-2 probabilities and the auxiliary balance loss."}


    dataset = component_spec(ident, root=root, size=size).run()
    trainer = make_trainer(ident, dataset, max_epochs=2, component=component)
    fit = trainer.fit()
    component_analysis = analysis.analyze_model(trainer.model.component)
    composition_analysis = analysis.analyze_model(trainer.model)
    evaluated = trainer.evaluate(dataset.for_recipe("component-smoke", split="val", component_id=ident))
    losses = [entry["train_loss"] for entry in fit["history"]]
    second_dataset = component_spec(ident, root=root + "-determinism", size=size).run()
    first_data_fingerprint = synth.fingerprint_of(dataset.view().to_records(meta=False))
    second_data_fingerprint = synth.fingerprint_of(second_dataset.view().to_records(meta=False))
    import torch
    eval_view = dataset.for_recipe("component-smoke", split="val", component_id=ident)
    eval_batch_size = 1 if ident == "auprc" else 2
    eval_batch = next(iter(torch.utils.data.DataLoader(
        eval_view, batch_size=eval_batch_size, collate_fn=eval_view.collate_fn)))
    runtime = analysis.analyze((ident, trainer.model.component, eval_batch["inputs"]["args"],
                                eval_batch["inputs"]["kwargs"], 97),
                               _ComponentRuntimeAnalysis())
    gradient_evidence = trainer.model.gradient_evidence()
    gradient_items = gradient_evidence["items"]
    component_trainable = any(parameter.requires_grad for parameter in trainer.model.component.parameters())
    if component_trainable:
        gradient_flow = bool(gradient_items) and all(
            events and any(event["finite"] and event["nonzero"] for event in events)
            for events in gradient_items.values())
    else:
        gradient_flow = bool(gradient_items) and any(
            events and any(event["finite"] and event["nonzero"] for event in events)
            for events in gradient_items.values())
    return {"id": ident, "fit": fit, "evaluation": evaluated,
            "parameter_count": component_analysis.payload["structure"]["parameters"]["total"],
            "trainable_parameters": component_analysis.payload["structure"]["parameters"]["trainable"],
            "analysis": component_analysis, "composition_analysis": composition_analysis,
            "component_trainable": component_trainable,
            "readout_trainable": any(p.requires_grad for p in trainer.model.readout.parameters()),
            "forward_finite": bool(runtime.payload["finite"]) and math.isfinite(float(evaluated["loss"])),
            "output_shapes": runtime.payload["output_shapes"],
            "gradient_flow": gradient_flow and fit["global_step"] > 0 and bool(losses) and
                             all(math.isfinite(float(v)) for v in losses),
             "gradient_evidence": gradient_evidence,
             "train_loss": [float(value) for value in losses[:50]],
            "deterministic": (component_spec(ident, root=root, size=size).fingerprint() ==
                              component_spec(ident, root=root + "-determinism", size=size).fingerprint()
                              and first_data_fingerprint == second_data_fingerprint
                              and bool(runtime.payload["deterministic"])),
             "composition_note": ("The composed objective is readout MSE plus any component auxiliary objective; the catalog component has trainable parameters and was optimized jointly with the readout."
                                  if any(p.requires_grad for p in trainer.model.component.parameters()) else
                                  "The composed objective is readout MSE plus any component auxiliary objective; the catalog operation is parameter-free/fixed, so only the downstream readout was optimized.")}


def _example_prefix(ident: str) -> str:
    _entry(ident)
    return ("import tempfile\n"
            "import torch\n"
            "from topologies import synth\n"
            "from topologies.validation.catalog_specs import catalog_specs, decode\n"
            "from topologies.validation.component_workflows import compose_component\n"
            f"component_id = {ident!r}\n"
            "def generate(ctx):\n"
            "    specification = catalog_specs()[component_id]\n"
            "    args = decode(specification['args'], seed=41 + int(ctx.index))\n"
            "    kwargs = decode(specification.get('kwargs', {}), seed=41 + int(ctx.index))\n"
            "    def sample_geometry(value):\n"
            "        if torch.is_tensor(value) and value.ndim > 1 and component_id not in {'graph-conv', 'graph-readout'}:\n"
            "            return value[0]\n"
            "        if isinstance(value, list):\n"
            "            return [sample_geometry(item) for item in value]\n"
            "        if isinstance(value, tuple):\n"
            "            return tuple(sample_geometry(item) for item in value)\n"
            "        if isinstance(value, dict):\n"
            "            return {key: sample_geometry(item) for key, item in value.items()}\n"
            "        return value\n"
            "    args = sample_geometry(args)\n"
            "    kwargs = sample_geometry(kwargs)\n"
            "    return {'args': args, 'kwargs': kwargs, 'target': float((int(ctx.index) % 5) / 5)}\n")


def example_source(ident: str) -> str:
    """Return a comment-free executable synth.Spec + for_recipe + Trainer example."""
    import textwrap

    if ident == "top-2-router":
        return ("import tempfile\n"
                "from topologies.validation.component_workflows import component_spec, make_trainer\n"
                "with tempfile.TemporaryDirectory() as root:\n"
                "    dataset = component_spec('top-2-router', root=root, seed=19, size=12).run()\n"
                "    trainer = make_trainer('top-2-router', dataset, max_epochs=1)\n"
                "    trainer.fit()\n"
                "    print(trainer.evaluate(dataset.for_recipe('router-probe', split='val', component_id='top-2-router')))\n")

    batch_size = 1 if ident == "auprc" else 2
    body = ("with tempfile.TemporaryDirectory() as root:\n"
            "    dataset = synth.Spec(component_id, generator=generate, size=12, splits={'train': 0.75, 'val': 0.25}, seed=19, root=root, monitor='none').run()\n"
            "    train = dataset.for_recipe('component-smoke', split='train', component_id=component_id)\n"
            "    val = dataset.for_recipe('component-smoke', split='val', component_id=component_id)\n"
            "    model = compose_component(component_id)\n"
            "    from topologies import Trainer\n"
            f"    trainer = Trainer(model, train, val, recipe='regression', batch_size={batch_size}, max_epochs=1, device='cpu', monitor='none', logger='none', forward_map=lambda module, batch: module(batch['inputs']))\n"
            "    trainer.fit()\n"
            "    print(trainer.evaluate(val))\n")
    return _example_prefix(ident) + textwrap.indent(body, "")


def inspection_source(ident: str) -> str:
    """Return executable synth batching and analysis inspection code."""
    import textwrap

    if ident == "top-2-router":
        return ("import torch\n"
                "import tempfile\n"
                "from topologies.validation.component_workflows import build_component, component_spec\n"
                "with tempfile.TemporaryDirectory() as root:\n"
                "    dataset = component_spec('top-2-router', root=root, seed=19, size=4).run()\n"
                "    view = dataset.for_recipe('router-probe', split='train', component_id='top-2-router')\n"
                "    batch = next(iter(torch.utils.data.DataLoader(view, batch_size=2, collate_fn=view.collate_fn)))\n"
                "    router = build_component('top-2-router').eval()\n"
                "    with torch.no_grad():\n"
                "        indices, weights, auxiliary = router(batch['features'])\n"
                "    assert indices.shape[-2:] == (3, 2) and weights.shape == indices.shape\n"
                "    assert auxiliary.ndim == 0 and torch.isfinite(auxiliary)\n"
                "    print({'indices': tuple(indices.shape), 'weights': tuple(weights.shape), 'auxiliary': float(auxiliary)})\n")

    batch_size = 1 if ident == "auprc" else 2
    expected_shape = "(1, 1)" if ident == "auprc" else "(2, 1)"
    body = ("with tempfile.TemporaryDirectory() as root:\n"
            "    dataset = synth.Spec(component_id, generator=generate, size=12, splits={'train': 0.75, 'val': 0.25}, seed=19, root=root, monitor='none').run()\n"
            "    train = dataset.for_recipe('component-smoke', split='train', component_id=component_id)\n"
            "    from torch.utils.data import DataLoader\n"
            f"    batch = next(iter(DataLoader(train, batch_size={batch_size}, collate_fn=train.collate_fn)))\n"
            "    model = compose_component(component_id)\n"
            "    report = analysis.analyze_model(model.component)\n"
            "    prediction = model(batch['inputs'])\n"
            f"    assert tuple(prediction['prediction'].shape) == {expected_shape}\n"
            "    print(report.payload['structure'])\n")
    prefix = _example_prefix(ident).replace("from topologies import synth\n", "from topologies import synth, analysis\n")
    return prefix + textwrap.indent(body, "")


def component_check_examples(ident: str) -> tuple[str, str]:
    """Return the two executable examples required by the component record contract."""
    return example_source(ident), inspection_source(ident)


__all__ = [
    "build_component",
    "component_check_examples",
    "component_generator",
    "component_ids",
    "component_spec",
    "compose_component",
    "example_source",
    "inspection_source",
    "make_trainer",
    "measure_component",
]
