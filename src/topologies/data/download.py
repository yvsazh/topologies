"""Legacy dataset downloads (``topologies data download <name>``) — deprecated.

Use the declarative catalog instead: ``topologies.datasets`` /
``topologies datasets download <id>`` (checksums, versioning, offline cache).
This module will be removed in a future release (see ``docs/releases.md``).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from topologies.data.datasets import require
from topologies.data.text import DEFAULT_SEPARATOR

# Pinned to a dataset revision so the bytes (and the checksum) never change.
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_VALID_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    f"{TINYSTORIES_REVISION}/TinyStories-valid.txt"
)
TINYSTORIES_VALID_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
DOWNLOAD_TIMEOUT = 30


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str
    size: str
    extra: str | None
    fetch: Callable[[Path, bool], list[Path]]


def _torchvision(cls_name: str) -> Callable[[Path, bool], list[Path]]:
    def fetch(root: Path, force: bool) -> list[Path]:
        tv = require("torchvision", "vision")
        cls = getattr(tv.datasets, cls_name)
        if force and root.exists():
            shutil.rmtree(root)
        for train in (True, False):
            cls(root=str(root), train=train, download=True)
        return [root]

    return fetch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(name: str, done: int, total: int, elapsed: float, width: int) -> Any:
    from topologies.visualization.format import fmt_bytes, fmt_clock
    from topologies.visualization.terminal.text import line
    from topologies.visualization.terminal.widgets import fmt_percent, progress_bar

    frac = done / total if total else None
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if total and rate > 0 else None
    tail = line(f"  {fmt_bytes(done)}" + (f" / {fmt_bytes(total)}" if total else ""),
                ("  ·  ", "muted"), f"{fmt_bytes(rate)}/s",
                ("  ·  ", "muted"), fmt_clock(elapsed),
                (f" < {fmt_clock(eta)}" if eta is not None else "", "muted"))
    head = line((name, "title"), "  ")
    bar_w = max(0, min(30, width - head.width - tail.width - 5))
    bar = line(progress_bar(frac, bar_w), " ") if bar_w >= 8 else line()
    return line(head, bar, fmt_percent(frac), tail).clip(width)


def _download_file(url: str, dest: Path, force: bool, *, sha256: str | None = None,
                   stream: Any = None) -> Path:
    """Download ``url`` to ``dest`` atomically with a one-line progress on a terminal.

    Off a terminal only a start and a finish line are printed (log friendly).
    ``sha256`` is verified when given; the partial ``.part`` file is always
    removed on failure, and an existing ``dest`` whose checksum does not match
    is downloaded again.
    """
    import time

    from topologies.visualization.format import fmt_bytes, fmt_duration
    from topologies.visualization.terminal.progress import ProgressLine

    if dest.is_file() and not force and (sha256 is None or _sha256(dest) == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    progress = ProgressLine(stream if stream is not None else sys.stderr)
    start = time.monotonic()
    done = 0
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp, tmp.open("wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            if not progress.live:
                size = f" ({fmt_bytes(total)})" if total else ""
                progress.print(f"downloading {dest.name}{size} from {url}")
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                progress.update(_progress(dest.name, done, total, time.monotonic() - start,
                                          progress.width()))
        if total and done != total:
            raise OSError(f"incomplete download of {url}: {done} of {total} bytes")
        if sha256 is not None and (actual := _sha256(tmp)) != sha256:
            raise OSError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
        tmp.replace(dest)
    except BaseException:
        progress.close()
        raise
    finally:
        tmp.unlink(missing_ok=True)
    elapsed = fmt_duration(time.monotonic() - start)
    progress.close([f"downloaded {dest.name} · {fmt_bytes(done)} in {elapsed} → {dest}"])
    return dest


def make_text_subsets(
    source: Path,
    out_dir: Path,
    *,
    train: int = 1500,
    val: int = 200,
    prefix: str = "tinystories_mini",
    separator: str = DEFAULT_SEPARATOR,
) -> list[Path]:
    """Write the first ``train`` and next ``val`` documents of ``source`` as files."""
    stories = source.read_text(encoding="utf-8").split(separator)
    if len(stories) < train + val:
        raise ValueError(f"{source} has only {len(stories)} documents.")
    out = []
    for name, chunk in (("train", stories[:train]), ("val", stories[train : train + val])):
        p = out_dir / f"{prefix}_{name}.txt"
        p.write_text(separator.join(chunk) + separator, encoding="utf-8")
        out.append(p)
    return out


def _tinystories(root: Path, force: bool) -> list[Path]:
    valid = _download_file(TINYSTORIES_VALID_URL, root / "TinyStories-valid.txt", force,
                           sha256=TINYSTORIES_VALID_SHA256)
    return [valid, *make_text_subsets(valid, root)]


CATALOG: dict[str, DatasetInfo] = {
    "mnist": DatasetInfo("mnist", "MNIST digits, 1x28x28, 10 classes", "~55 MB", "vision",
                         _torchvision("MNIST")),
    "fashion-mnist": DatasetInfo("fashion-mnist", "Fashion-MNIST, 1x28x28, 10 classes",
                                 "~80 MB", "vision", _torchvision("FashionMNIST")),
    "cifar10": DatasetInfo("cifar10", "CIFAR-10, 3x32x32, 10 classes", "~340 MB", "vision",
                           _torchvision("CIFAR10")),
    "tinystories": DatasetInfo(
        "tinystories",
        "TinyStories validation split + 1500/200-story mini subsets",
        "~21 MB", None, _tinystories),
}


def list_datasets() -> list[DatasetInfo]:
    """Deprecated: the legacy download list (see ``topologies.datasets.list_datasets``)."""
    _deprecated("list_datasets")
    return list(CATALOG.values())


#: Catalog ids that replace the legacy names (``topologies.datasets``). TinyStories has
#: no catalog entry yet (catalog definitions are published through the artifact store).
REPLACEMENTS = {"mnist": "mnist", "fashion-mnist": "fashion-mnist", "cifar10": "cifar10",
                "tinystories": None}


def _deprecated(what: str, key: str | None = None) -> None:
    import warnings

    replacement = REPLACEMENTS.get(key or "")
    target = (f"topologies.datasets.get_dataset({replacement!r}) / `topologies datasets "
              f"download {replacement}`") if replacement else "topologies.datasets"
    warnings.warn(f"topologies.data.download.{what} is deprecated and will be removed; use "
                  f"{target}.", DeprecationWarning, stacklevel=3)


def download(name: str, root: str | Path | None = None, *, force: bool = False) -> list[Path]:
    """Deprecated: download dataset ``name`` into ``root`` (default ``data/<name>``)."""
    key = name.strip().lower()
    _deprecated("download", key)
    if key not in CATALOG:
        import difflib

        sug = difflib.get_close_matches(key, list(CATALOG), n=1)
        hint = f" Did you mean {sug[0]!r}?" if sug else ""
        raise ValueError(f"Unknown dataset {name!r}.{hint} Available: {', '.join(CATALOG)}.")
    target = Path(root) if root is not None else Path("data") / key
    return CATALOG[key].fetch(target, force)
