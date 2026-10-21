"""Logger, monitor, visualizer and reporter resolution for the Trainer."""

from __future__ import annotations

import warnings
from typing import Any

from topologies.training.errors import TrainingError
from topologies.training.logging import BaseLogger, build_logger
from topologies.training.monitoring import BaseMonitor

#: Sentinel: no visualizer selection was passed (use the legacy monitor dispatch).
_VISUALIZERS_UNSET = object()


class _ObservabilityMixin:
    """Trainer mixin (split out of engine.py; state lives on the Trainer)."""

    def _monitor_was_explicit(self) -> bool:
        if self._monitor_arg is not None:
            return True
        if self.config.monitoring is not None and self._monitoring_direct_supplied():
            return True
        explicit = set(self._config_input_dict or {}) | set(self._explicit_kw_keys or set())
        return bool(explicit & {"monitor", "monitoring"})

    def _monitoring_direct_supplied(self) -> bool:
        """True when monitoring was given (directly or through the config input)."""
        if "monitoring" in (self._config_input_dict or {}):
            return True
        return self.config.monitoring is not None

    def _resolve_logger_and_monitor(self, logger: Any | None, monitor: Any | None) -> None:
        from topologies.training.reporting.reporter import TrainingReporter

        lcfg = self.config.logging_cfg
        explicit = self._explicit_kw_keys or set()
        # -- log_every_steps / log_dir: explicit kwargs > logging_cfg > config.
        log_every = self.config.log_every_steps
        if ("log_every_steps" not in explicit and lcfg is not None
                and getattr(lcfg, "log_every_steps", None) is not None):
            log_every = int(lcfg.log_every_steps)
        log_every = max(1, int(log_every or 10))
        self._resolved_log_every = log_every
        log_dir = self.config.log_dir
        if "log_dir" not in explicit and lcfg is not None and getattr(lcfg, "directory", None):
            log_dir = lcfg.directory
        # -- logger.
        if isinstance(logger, BaseLogger):
            self.logger = logger
            self.logger.rank = self.dist.rank
        else:
            filename = getattr(lcfg, "filename", None) if lcfg is not None else None
            if isinstance(logger, str):
                kind = logger
            elif lcfg is not None:
                kind = str(getattr(lcfg, "format", "jsonl") or "jsonl")
            else:
                kind = self.config.logger
            try:
                self.logger = build_logger(kind, log_dir, rank=self.dist.rank,
                                           log_every_steps=log_every, filename=filename)
            except TrainingError:
                raise  # unknown logger / missing optional dependency: fail fast
            except Exception as exc:  # noqa: BLE001 - policy (b): a sink must not block training
                from topologies.training.logging import NullLogger

                warnings.warn(f"Logger {kind!r} could not be created ({exc}); logging disabled.",
                              RuntimeWarning, stacklevel=3)
                self.logger = NullLogger()
        # -- monitor options (the mode string or an explicit BaseMonitor).
        mcfg = self.config.monitoring
        opts: dict = {}
        mode: str | None = None
        if not isinstance(monitor, BaseMonitor):
            if isinstance(monitor, str):
                mode = monitor
            elif mcfg is not None:
                mode = str(getattr(mcfg, "mode", "standard") or "standard")
            else:
                mode = str(self.config.monitor or "standard")
            if mcfg is not None:
                opts = self._monitoring_to_opts(mcfg, log_every)
            opts.setdefault("log_every_steps", log_every)
        # -- presentation: one resolver maps monitor/monitoring/visualizer(s) to sinks.
        from topologies.visualization.training_views import resolve_training_visualizers

        selection_given = self._visualizer_selection is not _VISUALIZERS_UNSET
        try:
            facade, group = resolve_training_visualizers(
                monitor if isinstance(monitor, BaseMonitor) else mode,
                opts,
                None if not selection_given else self._visualizer_selection,
                request_stop=self.request_stop,
                log_every_steps=log_every,
                selection_given=selection_given,
            )
        except (TrainingError, TypeError, ValueError):
            raise  # unknown mode / bad options: fail fast
        except Exception as exc:  # noqa: BLE001 - policy (b): a sink must not block training
            from topologies.training.monitoring import NullMonitor
            from topologies.visualization import create_visualizer

            warnings.warn(f"Monitor {mode!r} could not be created ({exc}); monitoring "
                          "disabled.", RuntimeWarning, stacklevel=3)
            facade, group = NullMonitor(), create_visualizer(False)
        self.monitor: Any = facade
        visualizer = None
        if self.dist.rank == 0:
            visualizer = group
        else:
            try:
                group.close()
            except Exception:  # noqa: BLE001, S110 - closing an unused sink is best effort
                pass
        # -- reporter (additive; history still owned by Trainer). The monitor is
        # reached through the visualizer group, never directly.
        self.reporter = TrainingReporter(logger=self.logger, monitor=None, rank=self.dist.rank,
                                         log_every_steps=log_every, visualizer=visualizer)

    @staticmethod
    def _monitor_options(mode: str | None, opts: dict) -> dict:
        """The dashboard monitor accepts only presentation options."""
        if str(mode).lower() == "dashboard":
            return {key: opts[key] for key in ("charts", "refresh_rate", "tty", "width")
                    if key in opts}
        return dict(opts)

    @staticmethod
    def _monitoring_to_opts(mcfg: Any, log_every: int) -> dict:
        opts: dict = {}
        if getattr(mcfg, "columns", None) is not None:
            opts["columns"] = list(mcfg.columns)
        for k in ("show_eta", "show_mem", "show_lr", "precision"):
            v = getattr(mcfg, k, None)
            if v is not None:
                opts[k] = v
        opts["log_every_steps"] = int(log_every)
        if getattr(mcfg, "refresh_rate", None) is not None:
            opts["refresh_rate"] = float(mcfg.refresh_rate)
        if getattr(mcfg, "charts", None) is not None:
            opts["charts"] = list(mcfg.charts)
        for k in ("fullscreen", "panels", "view", "interactive", "controls", "capture_output",
                  "theme", "smoothing", "color"):
            v = getattr(mcfg, k, None)
            if v is not None:
                opts[k] = list(v) if k == "panels" else v
        return opts
