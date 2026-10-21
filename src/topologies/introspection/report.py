"""Automated, evidence-backed diagnostic reports.

Every finding carries the numbers it is based on, the thresholds used and a
``basis`` of ``measured`` or ``heuristic``. Correlational techniques never
produce causal wording; only interventions report effects, and they name the
intervention. A technique that fails becomes a ``critical`` ``technique_error``
finding unless ``strict=True``.
"""
from __future__ import annotations

import math
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from topologies.introspection.payloads import Payload, register_payload_serializer
from topologies.introspection.techniques import CapabilityRow, CapabilityTable
from topologies.introspection.views import finish, table, text
from topologies.visualization.views import ViewSpec

SEVERITY_ORDER = {"critical": 0, "warning": 1, "notice": 2, "info": 3}

PRESETS: dict[str, tuple[str, ...]] = {
    "quick": ("structure", "trace", "anomalies"),
    "standard": ("structure", "trace", "anomalies", "activations", "attention", "gradients",
                 "weights"),
    "deep": ("structure", "trace", "anomalies", "activations", "attention", "gradients", "weights",
             "attribution", "similarity", "geometry", "ablation_scan", "lens"),
}


@dataclass(frozen=True)
class Finding:
    id: str
    title: str
    severity: Literal["info", "notice", "warning", "critical"]
    basis: Literal["measured", "heuristic"]
    sites: tuple[str, ...]
    evidence: Mapping[str, Any]
    explanation: str
    suggestion: str | None = None
    views: tuple[ViewSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "severity": self.severity, "basis": self.basis,
                "sites": list(self.sites), "evidence": _plain(self.evidence),
                "explanation": self.explanation, "suggestion": self.suggestion}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@dataclass
class ReportSection:
    title: str
    technique: str | None = None
    text: str = ""
    findings: list[str] = field(default_factory=list)


@dataclass
class ReportPayload(Payload):
    subject: dict[str, Any]
    findings: list[dict[str, Any]]
    capabilities: dict[str, Any]
    sections: list[dict[str, Any]]

    def to_records(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in f.items() if k != "evidence"} for f in self.findings]

    def views(self, **options: Any) -> list[ViewSpec]:
        return finish([table("Findings", [{"severity": f["severity"], "basis": f["basis"],
                                           "title": f["title"]} for f in self.findings])], **options)


@dataclass
class Report:
    subject: Mapping[str, Any]
    sections: list[ReportSection]
    findings: list[Finding]
    results: dict[str, Any]
    capabilities: CapabilityTable
    preset: str = "quick"
    warnings: list[str] = field(default_factory=list)

    # -- text -------------------------------------------------------------------------

    def _sorted(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.id))

    def text(self) -> str:
        subject = self.subject
        lines = [f"Introspection report ({self.preset}) — {subject.get('model', '?')}",
                 (f"  parameters={subject.get('parameters')}, adapter="
                 f"{(subject.get('adapter') or {}).get('adapter')}, device={subject.get('device')}"),
                 "  tiers: " + ", ".join(f"{k}={'yes' if v else 'no'}"
                                         for k, v in self.capabilities.tiers.items())]
        ran = [r.name for r in self.capabilities.rows if r.available]
        skipped = [r for r in self.capabilities.rows if not r.available]
        lines.append(f"  ran: {', '.join(ran) if ran else '-'}")
        if self.findings:
            lines.append("Findings:")
            for f in self._sorted():
                lines.append(f"  [{f.severity}] ({f.basis}) {f.title}")
                lines.append(f"      {f.explanation}")
                evidence = ", ".join(f"{k}={_fmt(v)}" for k, v in list(f.evidence.items())[:5])
                if evidence:
                    lines.append(f"      evidence: {evidence}")
                if f.suggestion:
                    lines.append(f"      suggestion: {f.suggestion}")
        else:
            lines.append("No findings.")
        if skipped:
            lines.append("Skipped:")
            lines.extend(f"  {r.name}: {r.reason}" + (f" (fix: {r.fix})" if r.fix else "")
                         for r in skipped)
        lines.extend(f"warning: {w}" for w in self.warnings)
        return "\n".join(lines)

    def show(self) -> None:
        """Print the terminal summary (never plots)."""
        print(self.text())

    def to_markdown(self) -> str:
        lines = [f"# Introspection report ({self.preset})", "",
                 f"**Model:** `{self.subject.get('model')}`  ",
                 f"**Parameters:** {self.subject.get('parameters')}  ",
                 f"**Adapter:** {(self.subject.get('adapter') or {}).get('adapter')}", "",
                 "## Findings", ""]
        if self.findings:
            lines += ["| severity | basis | finding | sites | evidence |", "|---|---|---|---|---|"]
            for f in self._sorted():
                evidence = "; ".join(f"{k}={_fmt(v)}" for k, v in list(f.evidence.items())[:4])
                lines.append(f"| {f.severity} | {f.basis} | {f.title} | "
                             f"{', '.join(f.sites)[:60]} | {evidence} |")
            lines.append("")
            for f in self._sorted():
                lines += [f"### {f.title}", "", f.explanation, ""]
                if f.suggestion:
                    lines += [f"*Suggestion:* {f.suggestion}", ""]
        else:
            lines += ["No findings.", ""]
        lines += ["## Capabilities", "", "| technique | tier | ran | reason |", "|---|---|---|---|"]
        lines += [f"| {r.name} | {r.tier} | {'yes' if r.available else 'no'} | {r.reason} |"
                  for r in self.capabilities.rows]
        for section in self.sections:
            if section.text:
                lines += ["", f"## {section.title}", "", section.text]
        return "\n".join(lines) + "\n"

    def to_html(self, path: str | Path) -> Path:
        from topologies.introspection.html import report_html

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_html(self), encoding="utf-8")
        return out

    # -- results / views ------------------------------------------------------------------

    def payload(self) -> ReportPayload:
        return ReportPayload(dict(self.subject), [f.to_dict() for f in self._sorted()],
                             self.capabilities.to_dict(),
                             [{"title": s.title, "technique": s.technique, "text": s.text,
                               "findings": s.findings} for s in self.sections])

    def to_result(self) -> Any:
        from topologies.analysis.core import AnalysisResult

        register_payload_serializer()
        stages = [r for r in self.results.values() if hasattr(r, "payload")]
        return AnalysisResult(self.payload(), kind="introspection.report", name="report",
                              provenance={"subject": dict(self.subject), "preset": self.preset},
                              warnings=list(self.warnings),
                              assumptions=list(dict.fromkeys(a for r in stages for a in r.assumptions)),
                              stages=stages)

    def views(self, **options: Any) -> list[ViewSpec]:
        views = [text("Summary", self.text())]
        views.append(table("Findings", [{"severity": f.severity, "basis": f.basis, "title": f.title}
                                        for f in self._sorted()]))
        for f in self._sorted():
            views.extend(f.views)
        for result in self.results.values():
            producer = getattr(getattr(result, "payload", None), "views", None)
            if callable(producer):
                views.extend(producer())
        return finish(views, **options)

    def __visualization__(self, **options: Any) -> list[ViewSpec]:
        return self.views(**options)

    def visualize(self, **options: Any) -> Any:
        from topologies.visualization import visualize

        return visualize(self, **options)

    def __repr__(self) -> str:
        return f"Report(preset={self.preset!r}, findings={len(self.findings)})"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)) and len(value) > 6:
        return f"[{len(value)} items]"
    return str(value)


# --------------------------------------------------------------------------- builder


@dataclass
class _Step:
    name: str
    technique: str
    tier: str


_STEPS = {
    "structure": _Step("structure", "structure", "U"),
    "trace": _Step("trace", "structure.call_tree", "U"),
    "anomalies": _Step("anomalies", "anomalies", "U"),
    "activations": _Step("activations", "activations", "U"),
    "attention": _Step("attention", "attention", "F"),
    "gradients": _Step("gradients", "gradients", "U"),
    "weights": _Step("weights", "weights", "U"),
    "attribution": _Step("attribution", "attribution", "D"),
    "similarity": _Step("similarity", "similarity", "U"),
    "geometry": _Step("geometry", "geometry", "U"),
    "ablation_scan": _Step("ablation_scan", "interventions.ablation_scan", "U"),
    "lens": _Step("lens", "lens", "A"),
}


def build_report(inspector: Any, inputs: Any = None, *, data: Any = None, preset: str = "quick",
                 include: Any = None, exclude: Any = None, objective: Any = None,
                 strict: bool = False, max_batches: int | None = None, visualizer: Any = None,
                 **params: Any) -> Report:
    from topologies.introspection.techniques import DEFAULT_TECHNIQUES, Availability

    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    steps = list(PRESETS[preset])
    if include:
        steps += [s for s in ([include] if isinstance(include, str) else include) if s not in steps]
    if exclude:
        dropped = {exclude} if isinstance(exclude, str) else set(exclude)
        steps = [s for s in steps if s not in dropped]
    subject = inspector.subject
    if inputs is None and data is None:
        raise ValueError("report needs inputs=... (one invocation) and/or data=... (a dataset)")
    one = inputs if inputs is not None else _first(subject, data)
    findings: list[Finding] = []
    results: dict[str, Any] = {}
    sections: list[ReportSection] = []
    rows: list[CapabilityRow] = []
    warnings: list[str] = []
    group = _visualizer(visualizer)
    try:
        for step_name in steps:
            step = _STEPS.get(step_name)
            if step is None:
                raise ValueError(f"unknown report step {step_name!r}; known: {sorted(_STEPS)}")
            try:
                cls = DEFAULT_TECHNIQUES.get(step.technique)
            except KeyError:
                rows.append(CapabilityRow(step.name, step.tier, False, "technique not installed"))
                continue
            technique = cls()
            status = technique.check(subject)
            kwargs = _step_kwargs(step.name, objective=objective, max_batches=max_batches,
                                  data=data, params=params)
            if status and step.name == "gradients":
                status = _objective_available(subject, one, objective)
            if status and step.name == "attribution" and objective is None:
                status = Availability(True)
            if not status:
                rows.append(CapabilityRow(step.name, cls.tier, False, status.reason, status.fix))
                continue
            _emit(group, "technique_started", step.name)
            try:
                source = data if step.name in ("activations", "similarity", "geometry") and \
                    data is not None else one
                result = technique.run(subject, source, **kwargs)
            except Exception as exc:
                if strict:
                    raise
                tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
                findings.append(Finding("technique_error", f"{step.name} failed", "critical",
                                        "measured", (), {"technique": step.name, "error": tb},
                                        f"the {step.name} technique raised; other sections continue",
                                        "re-run with strict=True for the traceback"))
                rows.append(CapabilityRow(step.name, cls.tier, False, f"failed: {tb}"))
                continue
            rows.append(CapabilityRow(step.name, cls.tier, True))
            results[step.name] = result
            new = findings_for(step.name, result, params)
            findings.extend(new)
            for f in new:
                _emit(group, "finding", step.name, finding=f.title, severity=f.severity)
            sections.append(ReportSection(step.name, step.technique, _section_text(step.name, result),
                                          [f.id for f in new]))
            warnings.extend(getattr(result, "warnings", []))
            _emit(group, "technique_completed", step.name)
    finally:
        if group is not None:
            group.close()
    tiers = {t: any(r.available for r in rows if r.tier == t) for t in ("U", "F", "D", "A")}
    return Report(subject.describe(), sections, findings, results, CapabilityTable(rows, tiers),
                  preset, list(dict.fromkeys(warnings)))


def _visualizer(visualizer: Any) -> Any:
    if visualizer is None or visualizer is False:
        return None
    from topologies.visualization import create_visualizer

    group = create_visualizer(visualizer)
    group.prepare(source="introspection")
    return group


def _emit(group: Any, kind: str, technique: str, **data: Any) -> None:
    if group is None:
        return
    from topologies.visualization.core import VisualizationEvent

    group.handle(VisualizationEvent(source="introspection", kind=f"introspection.{kind}",
                                    data={"technique": technique, **data}))


def _first(subject: Any, data: Any) -> Any:
    from topologies.introspection.inputs import normalize_inputs

    return next(iter(normalize_inputs(subject, data, from_data=True)))


def _objective_available(subject: Any, inputs: Any, objective: Any) -> Any:
    from topologies.introspection.techniques import Availability

    if objective is not None:
        return Availability(True)
    try:
        from topologies.introspection.objectives import default_objective

        default_objective(subject, inputs)
    except Exception as exc:  # noqa: BLE001
        return Availability(False, f"no objective: {exc}".split("\n")[0],
                            "pass objective=Loss() with targets, or objective=Output()")
    return Availability(True)


def _step_kwargs(name: str, *, objective: Any, max_batches: int | None, data: Any,
                 params: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(params.get(name, {})) if isinstance(params.get(name), Mapping) else {}
    if name in ("gradients", "attribution") and objective is not None:
        kwargs.setdefault("objective", objective)
    if name == "attribution" and objective is None:
        kwargs.setdefault("target", "predicted")
    if name in ("activations",) and max_batches is not None:
        kwargs.setdefault("max_batches", max_batches)
    return kwargs


def _section_text(name: str, result: Any) -> str:
    try:
        rows = result.payload.to_records()
    except Exception:  # noqa: BLE001
        return ""
    return f"{len(rows)} row(s); see results['{name}'].to_markdown()" if rows else ""


# --------------------------------------------------------------------------- findings


def findings_for(step: str, result: Any, params: Mapping[str, Any] | None = None) -> list[Finding]:
    """Turn one technique result into findings (thresholds come from ``params``)."""
    params = params or {}
    payload = getattr(result, "payload", None)
    out: list[Finding] = []
    if step == "anomalies":
        for item in payload.items:
            out.append(Finding(item["id"], item["title"], item["severity"], item["basis"],
                               tuple(item["sites"]), item["evidence"], item["explanation"],
                               item.get("suggestion")))
    elif step == "trace":
        summary = payload.output_summary
        if summary:
            nonfinite = summary.get("nonfinite_frac")
            out.append(Finding("output_summary", "Model output summary",
                               "critical" if isinstance(nonfinite, float) and nonfinite > 0 else "info",
                               "measured", ("<root>",), dict(summary),
                               "statistics of the model's (first tensor) output for these inputs",
                               None))
    elif step == "activations":
        threshold = float(params.get("dead_unit_threshold", 0.25))
        for site, dead in payload.dead_units.items():
            if dead["fraction"] >= threshold and dead["dead"]:
                out.append(Finding("dead_units", f"{len(dead['dead'])}/{dead['units']} dead units at "
                                   f"{site}", "warning", dead["basis"], (site,),
                                   {"dead_fraction": dead["fraction"], "threshold": threshold,
                                    "examples": payload.examples, "units": dead["dead"][:20]},
                                   "these units produced zero (<= threshold) on every example "
                                   "seen; with few examples this is only a heuristic",
                                   "check initialisation / learning rate, or collect over more data"))
        saturation = float(params.get("saturation_threshold", 0.5))
        for site, fraction in payload.saturation.items():
            if fraction >= saturation:
                out.append(Finding("saturation", f"Saturated activations at {site}", "warning",
                                   "measured", (site,), {"fraction": fraction, "threshold": saturation},
                                   "most values sit at the activation's bounds, where gradients "
                                   "vanish", None))
        if payload.drift:
            worst = max(payload.drift.items(), key=lambda kv: kv[1].get("js_distance", 0) or 0)
            out.append(Finding("distribution_drift", f"Largest distribution shift at {worst[0]}",
                               "info", "measured", (worst[0],), dict(worst[1]),
                               "difference between the activation distributions of the two input "
                               "sets; it describes the inputs, not model quality", None))
    elif step == "weights":
        cutoff = float(params.get("alpha_threshold", 2.0))
        low = [r for r in payload.parameters if r.get("alpha") is not None and r["alpha"] < cutoff]
        if low:
            out.append(Finding("heavy_tailed_weights", f"{len(low)} weight matrices with alpha < {cutoff}",
                               "info", "heuristic", tuple(r["name"] for r in low),
                               {r["name"]: r["alpha"] for r in low[:10]},
                               "a heavy-tailed eigenvalue spectrum (power-law fit); in the "
                               "WeightWatcher literature this correlates with over-fitting, but it "
                               "is a heuristic", None))
        nonfinite = [r["name"] for r in payload.parameters if not math.isfinite(r["norm"])]
        if nonfinite:
            out.append(Finding("nonfinite_weights", "Non-finite weights", "critical", "measured",
                               tuple(nonfinite), {"parameters": nonfinite},
                               "these parameters contain NaN/Inf", "reload or re-initialise them"))
    else:
        hook = getattr(payload, "findings", None)
        if callable(hook):
            out.extend(hook(params))
    return out


__all__ = ["PRESETS", "Finding", "Report", "ReportPayload", "ReportSection", "build_report",
           "findings_for"]
