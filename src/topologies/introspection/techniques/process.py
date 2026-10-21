"""Process trajectories: generation steps, diffusion denoising, any user loop (U via occurrences, A).

Occurrence k of the stepped module is step k. That covers any model, through the
callable-input form. Sampler state (x_t and the predicted x0 for diffusion, the
logits for generation) comes from the optional ``callback=`` of
``topologies.generation`` and ``inference_tasks.diffusion``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies.introspection.payloads import Payload, make_result
from topologies.introspection.sites import ROOT
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.views import finish, heatmap, lines, table
from topologies.visualization.views import ViewSpec


def _site_series(trace: Any, step_path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    series: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(trace.records.values(), key=lambda r: (r.site, r.occurrence)):
        if record.summary is None or record.port != "output":
            continue
        series.setdefault(record.site, []).append({"step": record.occurrence, **record.summary})
    return series


def _norm_matrix(series: dict[str, list[dict[str, Any]]]) -> tuple[list[str], torch.Tensor]:
    names = list(series)
    steps = max((len(v) for v in series.values()), default=0)
    matrix = torch.full((len(names), steps), float("nan"))
    for i, name in enumerate(names):
        for j, entry in enumerate(series[name]):
            value = entry.get("l2")
            matrix[i, j] = float(value) if isinstance(value, (int, float)) else float("nan")
    return names, matrix


# --------------------------------------------------------------------------- generation


@dataclass
class GenerationPayload(Payload):
    prompt: Any
    text: Any
    steps: list[dict[str, Any]]
    site_series: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        return [{k: (", ".join(f"{t}:{p:.3f}" for t, p in v) if k == "top" else v)
                 for k, v in step.items()} for step in self.steps]

    def views(self, **options: Any) -> list[ViewSpec]:
        views = [lines("Next-token entropy per step", [{"y": [s["entropy"] for s in self.steps],
                                                        "label": "entropy"}],
                       x_ticks=[s["token_text"] for s in self.steps], y_label="nats"),
                 table("Decoding steps", self.to_records())]
        if self.site_series:
            names, matrix = _norm_matrix(self.site_series)
            views.append(heatmap("Site L2 norm per decoding step", matrix, rows=names,
                                 x_label="step", y_label="site"))
        return finish(views, **options)


@builtin
class Generation(Technique):
    name = "process.generation"
    tier = "U"
    result_kind = "introspection.generation"
    summary = "per-step next-token distributions and site summaries during generation"

    def run(self, subject: Any, inputs: Any = None, *, max_new_tokens: int = 20, sites: Any = None,
            top_k: int = 5, steer: Any = None, interventions: tuple = (), keep: str = "summary",
            greedy: bool = True, **gen_kwargs: Any) -> Any:
        from topologies.generation import generate

        tokenizer = subject.tokenizer
        steps: list[dict[str, Any]] = []

        def callback(step: int, ids: list[int], logits: torch.Tensor) -> None:
            probs = logits.float().softmax(-1)[0]
            top = probs.topk(min(top_k, probs.numel()))
            chosen = ids[-1]
            steps.append({"step": step, "token": chosen,
                          "token_text": _text(tokenizer, chosen),
                          "probability": float(probs[chosen]),
                          "entropy": float(-(probs.clamp_min(1e-12).log() * probs).sum()),
                          "top": [(_text(tokenizer, int(i)), float(p)) for p, i in
                                  zip(top.values, top.indices)]})
        specs = tuple(interventions) + ((steer,) if steer is not None else ())
        policy = {"slice": {"position": -1}} if keep == "tensors" else {}
        session = subject.inspector.session(sites=sites, interventions=specs, call_tree=False,
                                            keep=keep, **policy)
        out_holder: dict[str, Any] = {}

        def drive(model: Any) -> Any:
            out_holder["text"] = generate(model, inputs, tokenizer=tokenizer,
                                          max_new_tokens=max_new_tokens, greedy=greedy,
                                          callback=callback, **gen_kwargs)
            return out_holder["text"]
        trace = session.run(drive)
        payload = GenerationPayload(inputs, out_holder.get("text"), steps, _site_series(trace))
        assumptions = [("occurrence k of the model = decoding step k (no KV cache: each step "
                       "re-runs the full window)")]
        if keep == "tensors":
            assumptions.append("kept tensors sliced to the last position (slice={'position': -1})")
        if specs:
            assumptions.append(f"interventions active during generation: {specs}")
        payload.trace = trace  # not a dataclass field: kept out of JSON
        return make_result(self, payload, subject, warnings=trace.warnings,
                           assumptions=[*assumptions, *trace.assumptions])


def _text(tokenizer: Any, token: int) -> str:
    if tokenizer is None:
        return str(token)
    try:
        return tokenizer.decode([token])
    except Exception:  # noqa: BLE001
        return str(token)


# --------------------------------------------------------------------------- diffusion


@dataclass
class DiffusionPayload(Payload):
    steps: list[int]
    timesteps: list[int]
    x_t: torch.Tensor          # [S, B, ...]
    x0: torch.Tensor           # [S, B, ...]
    final: torch.Tensor
    site_series: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def site_norms(self) -> dict[str, list[float]]:
        return {s: [float(e["l2"]) if isinstance(e.get("l2"), (int, float)) else float("nan")
                    for e in v] for s, v in self.site_series.items()}

    def to_records(self) -> list[dict[str, Any]]:
        return [{"step": s, "t": t, "x0_std": float(self.x0[i].std()),
                 "x_t_std": float(self.x_t[i].std())}
                for i, (s, t) in enumerate(zip(self.steps, self.timesteps))]

    def views(self, **options: Any) -> list[ViewSpec]:
        labels = [f"t={t}" for t in self.timesteps]
        views = [ViewSpec("image_grid", "Predicted x0 over steps (example 0)",
                          {"images": [x[0].tolist() for x in self.x0], "labels": labels}),
                 ViewSpec("image_grid", "x_t over steps (example 0)",
                          {"images": [x[0].tolist() for x in self.x_t], "labels": labels})]
        if self.site_series:
            names, matrix = _norm_matrix(self.site_series)
            views.append(heatmap("Site L2 norm per denoising step", matrix, rows=names,
                                 x_label="denoiser call", y_label="site"))
        return finish(views, **options)


@builtin
class Diffusion(Technique):
    name = "process.diffusion"
    tier = "U"
    result_kind = "introspection.diffusion"
    summary = "x_t / predicted x0 trajectories and per-step site summaries during sampling"

    def run(self, subject: Any, inputs: Any = None, *, sampler: str = "ddim", shape: Any = None,
            schedule: Any = None, steps: int = 50, capture: Any = None, every: int = 1,
            seed: int | None = 0, prediction_type: str = "epsilon", eta: float = 0.0,
            conditioning: Any = None, **params: Any) -> Any:
        from topologies.inference_tasks.diffusion import DiffusionSchedule, sample_ddim, sample_ddpm

        if shape is None:
            raise ValueError("diffusion needs shape=(B, C, H, W)")
        schedule = schedule or DiffusionSchedule.create(1000)
        if isinstance(schedule, int):
            schedule = DiffusionSchedule.create(schedule)
        trajectory: dict[str, list[Any]] = {"step": [], "t": [], "x_t": [], "x0": []}

        def callback(index: int, t: int, x_t: torch.Tensor, x0: torch.Tensor, eps: torch.Tensor
                     ) -> None:
            if index % every == 0:
                trajectory["step"].append(index)
                trajectory["t"].append(int(t))
                trajectory["x_t"].append(x_t.detach().float().cpu())
                trajectory["x0"].append(x0.detach().float().cpu())
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        device = subject.device

        def drive(model: Any) -> Any:
            if sampler == "ddim":
                return sample_ddim(model, tuple(shape), schedule, steps=steps, eta=eta,
                                   prediction_type=prediction_type, device=device,
                                   generator=generator, conditioning=conditioning, callback=callback)
            if sampler == "ddpm":
                return sample_ddpm(model, tuple(shape), schedule, prediction_type=prediction_type,
                                   device=device, generator=generator, conditioning=conditioning,
                                   callback=callback)
            raise ValueError("sampler must be 'ddim' or 'ddpm' (use insp.process for others)")
        sites = capture if capture is not None else None
        session = subject.inspector.session(sites=sites, call_tree=False, keep="summary",
                                            occurrences="all")
        trace = session.run(drive)
        series = _site_series(trace)
        if every > 1:
            series = {s: v[::every] for s, v in series.items()}
        payload = DiffusionPayload(trajectory["step"], trajectory["t"],
                                   torch.stack(trajectory["x_t"]), torch.stack(trajectory["x0"]),
                                   trace.output.detach().cpu() if isinstance(trace.output, torch.Tensor)
                                   else trace.output, series)
        return make_result(self, payload, subject, warnings=trace.warnings, assumptions=[
            (f"{sampler} sampler, {steps} steps, every {every}; occurrence k of the denoiser = "
            "sampling step k"), "x_t and predicted x0 come from the sampler callback"],
            provenance={"shape": list(shape), "seed": seed})


# --------------------------------------------------------------------------- generic


@dataclass
class ProcessPayload(Payload):
    step_module: str
    steps: int
    site_series: dict[str, list[dict[str, Any]]]
    output: Any = None

    def to_records(self) -> list[dict[str, Any]]:
        return [{"site": s, "step": e["step"], **{k: v for k, v in e.items() if k != "step"}}
                for s, entries in self.site_series.items() for e in entries]

    def views(self, **options: Any) -> list[ViewSpec]:
        names, matrix = _norm_matrix(self.site_series)
        return finish([heatmap(f"Site L2 norm per step ({self.step_module} occurrences)", matrix,
                               rows=names, x_label="step", y_label="site")], **options)


@builtin
class Process(Technique):
    name = "process.loop"
    tier = "U"
    result_kind = "introspection.process"
    summary = "any user loop: steps = occurrences of a module"

    def run(self, subject: Any, inputs: Any = None, *, step_module: str | None = None,
            sites: Any = None, keep: str = "summary", **params: Any) -> Any:
        fn: Callable[[Any], Any] = inputs
        if not callable(fn):
            raise TypeError("process(fn, ...) needs a callable fn(model)")
        spec = subject.adapter.process() if subject.adapter is not None else None
        step = step_module if step_module is not None else (
            spec.step_module if spec is not None and spec.step_module is not None else ROOT)
        path = subject.index.resolve_modules(step)[0]
        selector = sites if sites is not None else (path or ROOT)
        trace = subject.inspector.trace(fn, sites=selector, keep=keep, call_tree=True, **params)
        steps = trace.calls.counts().get(path, 0)
        payload = ProcessPayload(path or ROOT, steps, _site_series(trace), trace.output)
        payload.trace = trace  # not a dataclass field: kept out of JSON
        return make_result(self, payload, subject, warnings=trace.warnings, assumptions=[
            f"steps = occurrences of {path or ROOT} ({steps})", *trace.assumptions])
