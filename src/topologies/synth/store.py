"""On-disk layout of synthetic datasets: specs, chunks, immutable versions, lineage.

    <root>/                              # $TOPOLOGIES_SYNTH_ROOT or synth/ next to experiments/
      .cache/verify.sqlite               # shared verifier verdict cache (all datasets)
      <name>/
        spec.json                        # canonical spec (no secrets) + spec_hash
        latest                           # id of the newest version
        events.jsonl                     # run events (start, chunk, max_time, errors)
        chunks/<chunk_id>.jsonl.gz       # processed chunk (accepted + rejected + lineage)
        chunks/<chunk_id>.done           # resumability marker (written last, atomically)
        blobs/ab/abcdef....png           # content-addressed binary payloads
        versions/<version_id>/
          manifest.json                  # counts, funnel, splits, schema, fingerprints, time
          records/<split>/part-00000.jsonl[.gz]
          rejected/part-00000.jsonl[.gz] # rejected records + verdicts
          lineage/part-00000.jsonl[.gz]  # per-record lineage (records stay lean)
          blobs/...                      # hard links of the blobs the version uses
          taxonomy.yaml  pools/  quality/  gold.jsonl

Versions are written to a temporary directory and renamed into place, so a
version either exists completely or not at all. Chunk files are written
atomically before their ``.done`` marker.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from topologies._util.fs import atomic_write_bytes
from topologies.experiments.store import jsonable, now_iso, read_json, slugify, write_json_atomic
from topologies.synth.errors import SynthError

ENV_ROOT = "TOPOLOGIES_SYNTH_ROOT"
MANIFEST = "manifest.json"
SHARD_SIZE = 50_000


def default_root(root: str | Path | None = None) -> Path:
    """``root`` > ``$TOPOLOGIES_SYNTH_ROOT`` > ``synth/`` next to the experiments root."""
    if root is not None:
        return Path(root)
    env = os.environ.get(ENV_ROOT)
    if env:
        return Path(env)
    from topologies.experiments.store import default_root as exp_root

    return exp_root().parent / "synth"


def dumps(obj: Any) -> str:
    """Compact, deterministic JSON line (insertion order kept, non-ASCII as is)."""
    return json.dumps(jsonable(obj), ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Lines of a ``.jsonl`` / ``.jsonl.gz`` file (blank lines skipped)."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_text_atomic(path: Path, text: str, *, compress: bool = False) -> None:
    data = text.encode("utf-8")
    if compress:
        buf = io.BytesIO()
        # mtime=0: identical content -> identical bytes (reproducible files).
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
            gz.write(data)
        data = buf.getvalue()
    atomic_write_bytes(path, data)


class ShardWriter:
    """Append records to ``part-00000.jsonl[.gz]``, ``part-00001...`` (``SHARD_SIZE`` each)."""

    def __init__(self, directory: Path, *, compress: bool = False,
                 shard_size: int = SHARD_SIZE) -> None:
        self.dir = directory
        self.compress = compress
        self.shard_size = int(shard_size)
        self.count = 0
        self._buf: list[str] = []
        self._part = 0
        self.files: list[str] = []

    def add(self, obj: Any) -> None:
        self._buf.append(dumps(obj))
        self.count += 1
        if len(self._buf) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        name = f"part-{self._part:05d}.jsonl" + (".gz" if self.compress else "")
        write_text_atomic(self.dir / name, "\n".join(self._buf) + "\n", compress=self.compress)
        self.files.append(name)
        self._buf = []
        self._part += 1

    def close(self) -> list[str]:
        self.flush()
        return list(self.files)


def shard_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.name.startswith("part-") and (p.suffix in (".jsonl", ".gz", ".parquet")))


def iter_shards(directory: Path) -> Iterator[dict]:
    for p in shard_files(directory):
        if p.suffix == ".parquet":
            yield from _iter_parquet(p)
        else:
            yield from iter_jsonl(p)


def _iter_parquet(path: Path) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - optional extra
        raise SynthError("Reading parquet shards needs pyarrow.", stage="store",
                         hint="pip install 'topologies[synth-parquet]'") from e
    table = pq.read_table(path)
    for row in table.to_pylist():
        yield {k: (json.loads(v) if isinstance(v, str) and k.startswith("_json_") else v)
               for k, v in row.items()}


class SynthStore:
    """Paths and I/O for one dataset directory (see module docstring)."""

    def __init__(self, root: str | Path | None, name: str) -> None:
        self.root = default_root(root)
        self.name = name
        try:
            self.slug = slugify(name)
        except ValueError as e:
            raise SynthError(str(e), stage="store",
                             hint="Use letters, digits, '.', '_' or '-'.") from e
        self.dir = self.root / self.slug
        self.spec_path = self.dir / "spec.json"
        self.chunks_dir = self.dir / "chunks"
        self.versions_dir = self.dir / "versions"
        self.blobs_dir = self.dir / "blobs"
        self.events_path = self.dir / "events.jsonl"
        self.latest_path = self.dir / "latest"

    # -- locking -----------------------------------------------------------
    @contextmanager
    def lock(self, timeout: float = 30.0):
        """Exclusive writer lock for this dataset (concurrent runs wait, then fail)."""
        from topologies.utils.locking import LockTimeout, file_lock

        try:
            with file_lock(self.dir / ".lock", timeout=timeout):
                yield
        except LockTimeout as e:
            raise SynthError(f"Dataset {self.name!r} is being written by another process.",
                             stage="store", hint="Wait for the other run to finish (or kill "
                                                 "it); a crashed run releases the lock.") from e

    # -- spec / events -------------------------------------------------------
    def exists(self) -> bool:
        return self.dir.is_dir()

    def write_spec(self, doc: dict) -> None:
        old = self.read_spec() or {}
        revisions = list(old.get("revisions") or [])
        if not revisions or revisions[-1].get("spec_hash") != doc.get("spec_hash"):
            revisions.append({"at": now_iso(), "spec_hash": doc.get("spec_hash")})
        write_json_atomic(self.spec_path, {"format": "topologies-synth-spec", "version": 1,
                                           "name": self.name,
                                           "created_at": old.get("created_at") or now_iso(),
                                           "updated_at": now_iso(), **doc,
                                           "revisions": revisions})

    def read_spec(self) -> dict | None:
        try:
            return read_json(self.spec_path)
        except (FileNotFoundError, ValueError):
            return None

    def log_event(self, kind: str, **extra: Any) -> None:
        """Append one event (never raises: observability does not break runs)."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            rec = {"at": now_iso(), "t": round(time.time(), 3), "event": kind, **jsonable(extra)}
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001, S110 - the event log is advisory; data writes are atomic
            pass

    def events(self) -> list[dict]:
        if not self.events_path.is_file():
            return []
        out = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    # -- chunks --------------------------------------------------------------------
    def chunk_path(self, chunk_id: str) -> Path:
        return self.chunks_dir / f"{chunk_id}.jsonl.gz"

    def chunk_done(self, chunk_id: str) -> bool:
        return (self.chunks_dir / f"{chunk_id}.done").is_file() and \
            self.chunk_path(chunk_id).is_file()

    def write_chunk(self, chunk_id: str, items: Iterable[Any], summary: dict) -> None:
        """Chunk payload first, then the ``.done`` marker (with the summary)."""
        text = "".join(dumps(x) + "\n" for x in items)
        write_text_atomic(self.chunk_path(chunk_id), text, compress=True)
        write_json_atomic(self.chunks_dir / f"{chunk_id}.done", summary)

    def read_chunk(self, chunk_id: str) -> Iterator[dict]:
        return iter_jsonl(self.chunk_path(chunk_id))

    def chunk_summary(self, chunk_id: str) -> dict | None:
        try:
            return read_json(self.chunks_dir / f"{chunk_id}.done")
        except (FileNotFoundError, ValueError):
            return None

    def clear_chunks(self, prefix: str | None = None) -> int:
        if not self.chunks_dir.is_dir():
            return 0
        n = 0
        for p in self.chunks_dir.iterdir():
            if prefix is None or p.name.startswith(prefix):
                p.unlink(missing_ok=True)
                n += 1
        return n

    # -- blobs -----------------------------------------------------------------------
    def put_blob(self, data: bytes, ext: str) -> dict[str, str]:
        """Store ``data`` content-addressed; returns the ``{"$blob", "type"}`` reference."""
        import hashlib

        sha = hashlib.sha256(data).hexdigest()
        ext = ext.lstrip(".") or "bin"
        path = self.blobs_dir / sha[:2] / f"{sha}.{ext}"
        if not path.is_file():
            atomic_write_bytes(path, data)
        return {"$blob": sha, "type": ext}

    def blob_path(self, ref: dict, version: str | None = None) -> Path:
        sha, ext = ref["$blob"], ref.get("type", "bin")
        if version is not None:
            vp = self.version_dir(version) / "blobs" / sha[:2] / f"{sha}.{ext}"
            if vp.is_file():
                return vp
        return self.blobs_dir / sha[:2] / f"{sha}.{ext}"

    # -- versions ---------------------------------------------------------------------
    def version_dir(self, version_id: str) -> Path:
        return self.versions_dir / version_id

    def has_version(self, version_id: str) -> bool:
        return (self.version_dir(version_id) / MANIFEST).is_file()

    def versions(self) -> list[dict]:
        """Manifests of every version, oldest first."""
        out = []
        if not self.versions_dir.is_dir():
            return out
        for p in self.versions_dir.iterdir():
            if p.name.startswith(".") or not (p / MANIFEST).is_file():
                continue
            try:
                m = read_json(p / MANIFEST)
            except ValueError:
                continue
            out.append(m)
        return sorted(out, key=lambda m: (m.get("created_at") or "", m.get("version_id", "")))

    def latest(self) -> str | None:
        try:
            v = self.latest_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return v if v and self.has_version(v) else None

    def set_latest(self, version_id: str) -> None:
        write_text_atomic(self.latest_path, version_id + "\n")

    def resolve_version(self, version: str | None = "latest") -> str:
        if version in (None, "latest"):
            v = self.latest()
            if v is None:
                raise SynthError(f"Dataset {self.name!r} has no versions under {self.dir}.",
                                 stage="store", hint="Generate it first: spec.run() / "
                                                     "`topologies synth run spec.yaml`.")
            return v
        if self.has_version(version):
            return version
        matches = [m["version_id"] for m in self.versions()
                   if m.get("version_id", "").startswith(str(version))
                   or m.get("label") == version]
        if len(matches) == 1:
            return matches[0]
        from topologies.synth.errors import unknown

        raise unknown("version", str(version), [m["version_id"] for m in self.versions()],
                      stage="store")

    def read_manifest(self, version_id: str) -> dict:
        return read_json(self.version_dir(version_id) / MANIFEST)

    @contextmanager
    def new_version(self, version_id: str):
        """Temporary directory that becomes ``versions/<version_id>`` on success."""
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{version_id}-", dir=self.versions_dir))
        try:
            yield tmp
            target = self.version_dir(version_id)
            if target.exists():
                shutil.rmtree(target)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def link_blob(self, ref: dict, version_tmp: Path) -> None:
        src = self.blob_path(ref)
        if not src.is_file():
            return
        dst = version_tmp / "blobs" / src.parent.name / src.name
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def disk_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.dir.rglob("*") if f.is_file()) \
            if self.dir.is_dir() else 0


def list_datasets(root: str | Path | None = None) -> list[str]:
    base = default_root(root)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and not p.name.startswith(".") and
                  ((p / "spec.json").is_file() or (p / "versions").is_dir()))


__all__ = ["ENV_ROOT", "MANIFEST", "ShardWriter", "SynthStore", "default_root", "dumps",
           "iter_jsonl", "iter_shards", "list_datasets", "shard_files", "write_text_atomic"]
