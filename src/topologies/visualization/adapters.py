"""Lazy adapters from existing static result objects to shared result views."""

from __future__ import annotations

from collections.abc import Mapping

from .core import VisualizationResult


def adapt_result(source, *, metric=None, view_options=None):
    """Adapt history, ResultSet, AnalysisResult, view producers, or analysis plot-spec input."""
    if isinstance(source, VisualizationResult):
        return source
    from topologies.analysis.core import AnalysisResult

    from .views import ViewSpec, is_view_list

    view_options = dict(view_options or {})
    if isinstance(source, ViewSpec):
        source = [source]
    if is_view_list(source):
        from .views import select_views

        return VisualizationResult("views", {"views": tuple(select_views(list(source), **{
            k: v for k, v in view_options.items() if k in ("kinds", "sites", "max_panels")})),
            "result": None})
    producer = getattr(source, "__visualization__", None)
    if callable(producer) and not isinstance(source, AnalysisResult):
        return VisualizationResult("views", {"views": tuple(producer(**view_options)),
                                             "result": source},
                                   tuple(getattr(source, "warnings", ()) or ()))
    views = getattr(source.payload, "views", None) if isinstance(source, AnalysisResult) else None
    if callable(views):
        return VisualizationResult("views", {"views": tuple(views(**view_options)), "result": source},
                                   tuple(source.warnings))

    if isinstance(source, AnalysisResult):
        payload = source.payload
        if isinstance(payload, Mapping) and (
            source.kind == "engineering_report"
            or "parameter_storage" in payload
            or "known_lower_bound_bytes" in payload
            or "module_parameters" in payload
            or "parameter_rows" in payload
        ):
            return VisualizationResult(
                "engineering", {"payload": payload, "result": source}, tuple(source.warnings)
            )
        if isinstance(payload, Mapping) and (
            "latency_seconds" in payload or "mean_latency_seconds" in payload
        ):
            return VisualizationResult(
                "profile", {"payload": payload, "result": source}, tuple(source.warnings)
            )
        if isinstance(payload, Mapping) and (
            "cpu" in payload or "cuda" in payload or "mps" in payload
        ):
            return VisualizationResult(
                "hardware", {"payload": payload, "result": source}, tuple(source.warnings)
            )
        from topologies.analysis.visualization import plot_spec

        try:
            spec = plot_spec(source)
        except TypeError:
            if not isinstance(payload, Mapping):
                raise
            # Report payloads (evaluation/optimization/...) without a plot spec: the
            # Matplotlib backend uses a registered ``result:<kind>`` view or the generic
            # numeric-leaves view.
            return VisualizationResult(str(source.kind or "mapping"), dict(payload),
                                       tuple(source.warnings))
        extra = {}
        # Preserve existing engineering/profile payload semantics without deriving estimates.
        if isinstance(payload, Mapping):
            for key in (
                "memory",
                "compute",
                "structure",
                "layers",
                "parameters",
                "flops",
                "activation_memory",
                "hardware",
                "kv_cache",
                "cost",
            ):
                if key in payload:
                    extra[key] = payload[key]
        return VisualizationResult(
            "analysis", {"spec": spec, "result": source, **extra}, tuple(source.warnings)
        )
    if isinstance(source, Mapping) and "kind" in source:
        return VisualizationResult(str(source["kind"]), dict(source))
    if isinstance(source, Mapping):
        rows = source.get("history", source)
        if isinstance(rows, (list, tuple)):
            return VisualizationResult(
                "history", tuple(dict(row) for row in rows if isinstance(row, Mapping))
            )
        return VisualizationResult("mapping", dict(rows))
    if isinstance(source, (list, tuple)) and all(isinstance(row, Mapping) for row in source):
        return VisualizationResult("history", tuple(dict(row) for row in source))
    result_set_type = None
    try:
        from topologies.experiments.results import ResultSet as result_set_type
    except ImportError:
        pass
    if result_set_type is not None and isinstance(source, result_set_type):
        name = metric or source.goal[0]
        summaries = source.summaries(name, include_failed=True, include_pruned=True)
        # Use the canonical ResultSet ordering/table APIs; retain raw honest missing values.
        table = source.table(metric=name, include_failed=True)
        rows = []
        for summary in summaries:
            trials = [
                {
                    "trial_id": t.trial_id,
                    "seed": t.seed,
                    "status": t.status,
                    "value": t.metric(name),
                    "history": tuple(t.series(name)),
                }
                for t in summary.trials
            ]
            rows.append(
                {
                    "labels": dict(summary.labels),
                    "metric": name,
                    "mode": summary.mode,
                    "mean": summary.mean,
                    "n": summary.n,
                    "statuses": dict(summary.statuses),
                    "curve": tuple(summary.curve(name)),
                    "best": summary in source.best(metric=name),
                    "values": tuple(summary.values),
                    "trials": tuple(trials),
                }
            )
        # This progression is over individual completed trials, not aggregate
        # configuration means. Prefer persisted completion timestamps; when
        # absent, preserve and label ResultSet source order honestly.
        ordered_trials = [r for r in source.records if r.status == "completed"
                          and r.metric(name) is not None]
        timestamped = all((r.raw.get("finished_at") or r.raw.get("completed_at"))
                          for r in ordered_trials)
        if timestamped:
            ordered_trials.sort(key=lambda r: (r.raw.get("finished_at") or
                                               r.raw.get("completed_at"),
                                               source.records.index(r)))
        best_so_far = []
        best_value = None
        for index, trial in enumerate(ordered_trials, start=1):
            value = trial.metric(name)
            if value is None:
                continue
            if value is not None and (
                best_value is None
                or (value < best_value if source.mode(name) == "min" else value > best_value)
            ):
                best_value = value
            best_so_far.append({"trial_id": trial.trial_id, "order": index,
                                "label": str(trial.labels), "value": value,
                                "best": best_value})
        return VisualizationResult(
            "experiments",
            {
                "goal": {"metric": name, "mode": source.mode(name)},
                "summaries": tuple(rows),
                "leaderboard": table,
                "best": tuple(source.best(metric=name)),
                "best_so_far": tuple(best_so_far),
                "best_so_far_order": "completion timestamp" if timestamped else
                                     "persisted source order (timestamps unavailable)",
                "resultset": source,
            },
        )
    raise TypeError(f"Unsupported visualization result: {type(source).__name__}")


def render_result(source, visualizer, *, metric=None, view_options=None):
    """Explicit static rendering; backend errors intentionally propagate."""
    result = (
        source if isinstance(source, VisualizationResult)
        else adapt_result(source, metric=metric, view_options=view_options)
    )
    return visualizer.render(result)


def _is_view_producer(source) -> bool:
    from topologies.analysis.core import AnalysisResult

    from .views import ViewSpec, is_view_list

    if isinstance(source, ViewSpec) or is_view_list(source):
        return True
    if isinstance(source, AnalysisResult):
        return callable(getattr(source.payload, "views", None))
    return callable(getattr(source, "__visualization__", None))


def visualize(source, *, visualizer="matplotlib", metric=None, views=None, view_options=None,
              **options):
    """Create, prepare, and explicitly render a static visualization.

    ``views`` means two things, depending on the source. For view producers
    (introspection results, ``ViewSpec`` lists, objects with ``__visualization__``)
    it selects view kinds such as ``"heatmap"``; ``view_options`` (``sites``,
    ``max_panels`` ...) goes to the producer. For training histories, experiments
    and analyses it names registered Matplotlib views (``register_view``).
    """
    from .core import create_visualizer

    view_options = dict(view_options or {})
    if views is not None:
        if _is_view_producer(source):
            view_options["kinds"] = views
        else:
            options["views"] = views
    backend = create_visualizer(visualizer, **options)
    try:
        backend.prepare(source="result")
        return render_result(source, backend, metric=metric, view_options=view_options)
    finally:
        backend.close()
