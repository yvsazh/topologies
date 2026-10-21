"""On-disk layout: one directory per experiment, one per trial, atomic JSON writes.

    <root>/                          # ./experiments or $TOPOLOGIES_EXPERIMENTS_DIR
      .cache/                        # shared cache (tokenizers, datasets)
      <experiment>/
        experiment.json              # spec snapshot + revisions
        trials.jsonl                 # append-only state log (coordinator only)
        trials/<trial_id>/{config.json, metrics.jsonl, result.json, ckpt/, model/}

Trials are content-addressed (``trial_id`` = hash of the resolved config),
so a name may be reused with a different spec; stale trials stay on disk.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topologies._util.fs import atomic_write_json
from topologies._util.jsonsafe import to_jsonable

ENV_ROOT = "TOPOLOGIES_EXPERIMENTS_DIR"
EXPERIMENT_FILE = "experiment.json"
TRIALS_LOG = "trials.jsonl"
RESULT_FILE = "result.json"
CONFIG_FILE = "config.json"


def default_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get(ENV_ROOT)
    return Path(env) if env else Path("experiments")


def slugify(name: str) -> str:
    """Experiment directory name: lowercase ``[a-z0-9._-]``."""
    s = re.sub(r"[^a-z0-9._-]+", "-", str(name).strip().lower()).strip("-.")
    if not s:
        raise ValueError(f"Experiment name {name!r} has no usable characters.")
    return s


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def jsonable(obj: Any) -> Any:
    """Recursively make ``obj`` JSON-safe (non-finite floats -> None, tuples -> lists)."""
    return to_jsonable(obj)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write via a temp file in the same directory, then ``os.replace``."""
    atomic_write_json(path, to_jsonable(obj), indent=1, allow_nan=False)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ExperimentStore:
    """Paths and I/O for one experiment directory."""

    def __init__(self, root: str | Path | None, name: str) -> None:
        self.root = default_root(root)
        self.name = name
        self.slug = slugify(name)
        self.dir = self.root / self.slug
        self.trials_dir = self.dir / "trials"
        self.spec_path = self.dir / EXPERIMENT_FILE
        self.log_path = self.dir / TRIALS_LOG

    # -- experiment spec -------------------------------------------------------
    def exists(self) -> bool:
        return self.spec_path.is_file()

    def read_spec(self) -> dict | None:
        try:
            return read_json(self.spec_path)
        except FileNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Ignoring unreadable {self.spec_path}: {e}", stacklevel=2)
            return None

    def write_spec(self, spec: dict, spec_hash: str) -> dict:
        """Write ``experiment.json``, appending a revision when the spec changed."""
        old = self.read_spec() or {}
        revisions = list(old.get("revisions") or [])
        if not revisions or revisions[-1].get("spec_hash") != spec_hash:
            revisions.append({"at": now_iso(), "spec_hash": spec_hash})
        doc = {"format": "topologies-experiment", "version": 1, "name": self.name,
               "created_at": old.get("created_at") or now_iso(), "updated_at": now_iso(),
               **{k: v for k, v in spec.items() if k not in ("format", "version", "name")},
               "revisions": revisions}
        write_json_atomic(self.spec_path, doc)
        return doc

    # -- trials ----------------------------------------------------------------
    def trial_dir(self, trial_id: str) -> Path:
        return self.trials_dir / trial_id

    def result_path(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id) / RESULT_FILE

    def write_config(self, trial_id: str, config: dict) -> None:
        write_json_atomic(self.trial_dir(trial_id) / CONFIG_FILE, config)

    def write_result(self, trial_id: str, result: dict) -> None:
        write_json_atomic(self.result_path(trial_id), result)

    def read_result(self, trial_id: str, *, warn: bool = True) -> dict | None:
        """``result.json`` or ``None``; corrupted files warn and count as missing."""
        path = self.result_path(trial_id)
        if not path.is_file():
            return None
        try:
            data = read_json(path)
            if not isinstance(data, dict) or "status" not in data:
                raise ValueError("not a trial result")
            return data
        except Exception as e:  # noqa: BLE001
            if warn:
                warnings.warn(f"Skipping corrupted trial result {path}: {e}. The trial will be "
                              "treated as not completed.", stacklevel=2)
            return None

    def clear_checkpoint(self, trial_id: str) -> None:
        """Remove ``ckpt/`` so the next attempt trains from step 0."""
        ckpt = self.trial_dir(trial_id) / "ckpt"
        if ckpt.is_dir():
            shutil.rmtree(ckpt, ignore_errors=True)

    def archive_result(self, trial_id: str) -> Path | None:
        """``result.json`` -> ``result.<timestamp>.json`` (before a forced re-run)."""
        path = self.result_path(trial_id)
        if not path.is_file():
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        target = path.with_name(f"result.{stamp}.json")
        os.replace(path, target)
        return target

    def trial_ids(self) -> list[str]:
        if not self.trials_dir.is_dir():
            return []
        return sorted(p.name for p in self.trials_dir.iterdir()
                      if p.is_dir() and not p.name.startswith("."))

    def log_event(self, trial_id: str, status: str, **extra: Any) -> None:
        """Append one state change to ``trials.jsonl`` (coordinator only)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        rec = {"at": now_iso(), "t": round(time.time(), 3), "trial_id": trial_id,
               "status": status, **jsonable(extra)}
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def events(self) -> list[dict]:
        out: list[dict] = []
        try:
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        except FileNotFoundError:
            pass
        return out


def list_experiments(root: str | Path | None = None) -> list[str]:
    """Experiment directory names under ``root`` that hold an ``experiment.json``."""
    base = default_root(root)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and not p.name.startswith(".") and (p / EXPERIMENT_FILE).is_file())


__all__ = [
    "ENV_ROOT",
    "ExperimentStore",
    "default_root",
    "jsonable",
    "list_experiments",
    "now_iso",
    "read_json",
    "slugify",
    "write_json_atomic",
]
