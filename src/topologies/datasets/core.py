"""Safe local dataset lifecycle with explicit download, prepare, and load steps."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import tarfile
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .api import DatasetDownloadError, DatasetOfflineError
from .schema import _option_type, _relative_path, validate_definition

#: Floor for the zip-bomb guard; the default limit is max(50 x archive size, this).
DEFAULT_MAX_EXTRACTED_BYTES = 20 * 1024**3


def _archive_member_parts(name: str, *, allow_root: bool = False) -> tuple[str, ...]:
    """Normalize an archive member after rejecting POSIX and Windows traversal forms."""
    if not isinstance(name, str) or "\x00" in name or "\\" in name:
        raise ValueError("archive member contains a NUL or backslash")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.drive or windows.root:
        raise ValueError("archive member is absolute")
    parts = []
    for part in name.split("/"):
        if part in ("", "."):
            continue
        if part == ".." or ":" in part:
            raise ValueError("archive member contains traversal or drive syntax")
        parts.append(part)
    if not parts and not allow_root:
        raise ValueError("archive member has no relative path")
    return tuple(parts)


class TopologiesDataset:
    """Base lifecycle for declarative and custom datasets; construction never downloads."""

    def __init__(
        self,
        definition: dict,
        version: str,
        split: str | None = None,
        root: str | Path | None = None,
        offline: bool = False,
        **options: Any,
    ):
        validate_definition(definition)
        if version not in definition["versions"]:
            raise ValueError(f"unknown version {version!r} for {definition['id']}")
        if split is not None and split not in definition["versions"][version]["splits"]:
            raise ValueError(f"unknown split {split!r} for {definition['id']}")
        declared = definition.get("options", {})
        unknown = set(options) - set(declared)
        if unknown:
            raise ValueError(f"unknown dataset option(s): {', '.join(sorted(unknown))}")
        self.definition = definition
        self.version = version
        self.split = split
        self.offline = offline
        self.options = {
            name: spec.get("default")
            for name, spec in declared.items()
            if isinstance(spec, dict) and "default" in spec
        }
        self.options.update(options)
        for name, value in self.options.items():
            if not _option_type(value, declared[name]["type"]):
                raise TypeError(f"dataset option {name!r} must be {declared[name]['type']}")
        base = (
            Path(root)
            if root is not None
            else Path(os.environ.get("TOPOLOGIES_DATA_HOME", Path.home() / ".cache" / "topologies"))
        )
        self.root = base / definition["id"] / version
        if (base / definition["id"]).is_symlink() or self.root.is_symlink():
            raise DatasetDownloadError(f"refusing symlink dataset/version cache path: {self.root}")
        self.version_info = definition["versions"][version]
        self._prepared: list[Path] | None = None

    def _resources(self) -> list[dict]:
        all_resources = self.version_info["resources"]
        refs = (
            set(self.version_info["splits"].get(self.split, []))
            if self.split
            else {r["name"] for r in all_resources}
        )
        return [resource for resource in all_resources if resource["name"] in refs]

    @staticmethod
    def _digest(path: Path, algorithm: str) -> str:
        digest = hashlib.new(algorithm)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _valid(cls, path: Path, resource: dict) -> bool:
        if not path.is_file():
            return False
        if "bytes" in resource and path.stat().st_size != resource["bytes"]:
            return False
        return all(
            not resource.get(key) or cls._digest(path, key).lower() == resource[key].lower()
            for key in ("sha256", "md5")
        )

    def download(self, force: bool = False) -> list[Path]:
        """Download selected provider resources atomically; downloads are not resumable."""
        if self.root.parent.is_symlink() or self.root.is_symlink():
            raise DatasetDownloadError(f"refusing symlink dataset/version cache path: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        results = []
        for resource in self._resources():
            dest = self.root / resource["filename"]
            if not dest.resolve(strict=False).is_relative_to(self.root.resolve()):
                raise DatasetDownloadError(
                    f"unsafe cache path for {self.definition['id']}: {resource['filename']}"
                )
            cursor = self.root
            for part in PurePosixPath(resource["filename"]).parts[:-1]:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise DatasetDownloadError(f"refusing symlink in cache path: {cursor}")
            if dest.is_symlink():
                raise DatasetDownloadError(f"refusing symlink resource path: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file() and self._valid(dest, resource):
                if force and self.offline:
                    raise DatasetOfflineError(
                        f"cannot force-redownload {self.definition['id']} from {resource['url']} while offline"
                    )
                if not force:
                    results.append(dest)
                    continue
            elif self.offline:
                raise DatasetOfflineError(
                    f"{self.definition['id']} {self.version}: missing or invalid cached resource {dest}; provider: {resource['url']}"
                )
            fd, temp_name = tempfile.mkstemp(prefix=".download-", dir=dest.parent)
            os.close(fd)
            try:
                try:
                    with (
                        urllib.request.urlopen(resource["url"], timeout=30) as response,
                        open(temp_name, "wb") as output,
                    ):
                        expected_length = response.headers.get("Content-Length")
                        shutil.copyfileobj(response, output, 1024 * 1024)
                    if expected_length is not None and Path(temp_name).stat().st_size != int(
                        expected_length
                    ):
                        raise DatasetDownloadError(
                            f"incomplete response for {self.definition['id']} from {resource['url']}: Content-Length mismatch"
                        )
                except DatasetDownloadError:
                    raise
                except Exception as exc:
                    raise DatasetDownloadError(
                        f"download failed for {self.definition['id']} from {resource['url']}: {exc}; consult the upstream provider"
                    ) from exc
                if not self._valid(Path(temp_name), resource):
                    raise DatasetDownloadError(
                        f"download verification failed for {self.definition['id']} from {resource['url']} (size/checksum mismatch)"
                    )
                os.replace(temp_name, dest)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            results.append(dest)
        return results

    def _target(self, resource: dict) -> Path:
        relative = resource.get("extract_to", f"extracted/{resource['name']}")
        target = self.root.joinpath(*PurePosixPath(relative).parts)
        self.root.mkdir(parents=True, exist_ok=True)
        resolved_root = self.root.resolve()
        if not target.resolve(strict=False).is_relative_to(resolved_root):
            raise DatasetDownloadError(f"unsafe extraction target for {resource['name']}")
        cursor = self.root
        for part in PurePosixPath(relative).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise DatasetDownloadError(f"refusing symlink in extraction path: {cursor}")
        return target

    def _fingerprint(self, resource: dict, source: Path) -> str:
        content_hash = self._digest(source, "sha256")
        fields = {"resource": resource, "content_sha256": content_hash, "options": self.options}
        return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _extraction_limit(resource: dict, source: Path) -> int:
        """Max total bytes an archive may expand to (zip-bomb guard)."""
        limit = resource.get("max_extracted_bytes")
        if limit is None:
            limit = max(50 * source.stat().st_size, DEFAULT_MAX_EXTRACTED_BYTES)
        return int(limit)

    def _extract_stage(self, resource: dict, source: Path, stage: Path) -> None:
        archive = resource["archive"]
        limit = self._extraction_limit(resource, source)

        def check_total(total: int) -> None:
            if total > limit:
                raise DatasetDownloadError(
                    f"archive {source} expands to more than {limit} bytes; refusing to extract "
                    "(set the resource's max_extracted_bytes to allow it)"
                )

        if archive == "zip":
            with zipfile.ZipFile(source) as archive_file:
                for info in archive_file.infolist():
                    mode = info.external_attr >> 16
                    kind = mode & 0o170000
                    try:
                        parts = _archive_member_parts(info.filename, allow_root=info.is_dir())
                    except ValueError as exc:
                        raise DatasetDownloadError(
                            f"unsafe zip member in {source}: {info.filename}"
                        ) from exc
                    if (
                        kind == 0o120000
                        or kind not in (0, 0o100000, 0o040000)
                        or ".topologies-manifest.json" in parts
                    ):
                        raise DatasetDownloadError(
                            f"unsafe zip member in {source}: {info.filename}"
                        )
                check_total(sum(info.file_size for info in archive_file.infolist()))
                archive_file.extractall(stage)
        elif archive in ("tar", "tar.gz"):
            with tarfile.open(source, "r:*") as archive_file:
                members = archive_file.getmembers()
                check_total(sum(member.size for member in members if member.isfile()))
                seen_members = set()
                for member in members:
                    try:
                        parts = _archive_member_parts(member.name, allow_root=member.isdir())
                    except ValueError as exc:
                        raise DatasetDownloadError(
                            f"unsafe tar member in {source}: {member.name}"
                        ) from exc
                    if ".topologies-manifest.json" in parts or not (
                        member.isfile() or member.isdir()
                    ):
                        raise DatasetDownloadError(f"unsafe tar member in {source}: {member.name}")
                    relative = "/".join(parts)
                    if relative in seen_members:
                        raise DatasetDownloadError(
                            f"duplicate tar member in {source}: {member.name}"
                        )
                    seen_members.add(relative)
                    if not parts:
                        continue  # Root `.` directory records carry no payload.
                    destination = stage.joinpath(*parts)
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    file_obj = archive_file.extractfile(member)
                    if file_obj is None:
                        raise DatasetDownloadError(
                            f"could not read tar member in {source}: {member.name}"
                        )
                    with file_obj, destination.open("xb") as output:
                        shutil.copyfileobj(file_obj, output)
        elif archive == "gzip":
            name = Path(resource["filename"]).name
            output_name = name[:-3] if name.endswith(".gz") else name + ".out"
            with (
                gzip.open(source, "rb") as input_file,
                (stage / output_name).open("wb") as output_file,
            ):
                total = 0
                for chunk in iter(lambda: input_file.read(1 << 20), b""):
                    total += len(chunk)
                    check_total(total)
                    output_file.write(chunk)

    def _manifest_valid(self, target: Path, fingerprint: str) -> bool:
        marker = target / ".topologies-manifest.json"
        try:
            if marker.is_symlink() or not marker.is_file():
                return False
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            records = manifest.get("files")
            if manifest.get("fingerprint") != fingerprint or not isinstance(records, list):
                return False
            declared = {}
            for record in records:
                if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
                    return False
                relative = record["path"]
                try:
                    _relative_path(relative, "manifest path")
                except (TypeError, ValueError):
                    return False
                if (
                    not isinstance(record["size"], int)
                    or isinstance(record["size"], bool)
                    or record["size"] < 0
                    or not isinstance(record["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                ):
                    return False
                if relative in declared:
                    return False
                declared[relative] = (record["size"], record["sha256"])

            actual = {}
            for item in target.rglob("*"):
                if item.is_symlink():
                    return False
                if item == marker:
                    continue
                if item.is_file():
                    relative = item.relative_to(target).as_posix()
                    actual[relative] = (item.stat().st_size, self._digest(item, "sha256"))
                elif not item.is_dir():
                    return False
            return actual == declared
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            return False

    def _manifest_records(self, root: Path) -> list[dict]:
        """Hash every regular extracted file; reject links and special files."""
        records = []
        for item in root.rglob("*"):
            if item.is_symlink():
                raise DatasetDownloadError(f"refusing symlink in prepared dataset: {item}")
            if item.is_dir():
                continue
            if not item.is_file():
                raise DatasetDownloadError(f"refusing special file in prepared dataset: {item}")
            relative = item.relative_to(root).as_posix()
            _relative_path(relative, "extracted file path")
            records.append(
                {
                    "path": relative,
                    "size": item.stat().st_size,
                    "sha256": self._digest(item, "sha256"),
                }
            )
        return sorted(records, key=lambda record: record["path"])

    def _publish(self, stage: Path, target: Path) -> None:
        backup = target.with_name(target.name + ".backup-" + uuid.uuid4().hex)
        had_target = target.exists()
        try:
            if had_target:
                os.replace(target, backup)
            os.replace(stage, target)
        except Exception:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if backup.exists():
                os.replace(backup, target)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)

    def prepare(self, force: bool = False) -> list[Path]:
        """Prepare resources; custom subclasses can transform paths via ``prepare_data``."""
        sources = self.download(force=force)
        outputs = []
        for resource, source in zip(self._resources(), sources):
            if resource["archive"] == "none":
                outputs.append(source)
                continue
            target = self._target(resource)
            fingerprint = self._fingerprint(resource, source)
            if not force and self._manifest_valid(target, fingerprint):
                outputs.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".extract-", dir=target.parent))
            try:
                self._extract_stage(resource, source, stage)
                files = self._manifest_records(stage)
                (stage / ".topologies-manifest.json").write_text(
                    json.dumps(
                        {"resource": resource["name"], "fingerprint": fingerprint, "files": files}
                    ),
                    encoding="utf-8",
                )
                self._publish(stage, target)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            outputs.append(target)
        self._prepared = list(self.prepare_data(outputs))
        return self._prepared

    def prepare_data(self, paths: list[Path]) -> list[Path]:
        """Subclass hook for post-extraction preparation; return paths used by load_data."""
        return paths

    def load(self) -> Any:
        """Prepare selected resources and delegate parsing to ``load_data``."""
        return self.load_data(self.prepare())

    def load_data(self, paths: list[Path]) -> Any:
        """Load generic or per-resource formats; subclasses may override this hook."""
        loaded = {}
        for resource, path in zip(self._resources(), paths):
            fmt = resource.get("format", self.version_info.get("format"))
            if fmt not in {"csv", "lines", "text", "jsonl", "idx"}:
                raise ValueError(
                    f"unsupported generic format {fmt!r} for {self.definition['id']}; provide an implementation subclass"
                )
            if path.is_dir():
                files = (
                    [path / item for item in resource.get("data_files", [])]
                    if resource.get("data_files")
                    else sorted(path.rglob("*"))
                )
                missing = (
                    [item for item in files if not item.is_file()]
                    if resource.get("data_files")
                    else []
                )
                if missing:
                    raise DatasetDownloadError(
                        f"declared data file missing for {resource['name']}: {missing[0]}"
                    )
            else:
                files = [path]
            files = [item for item in files if item.is_file() and not item.name.startswith(".")]
            options = {**resource.get("loader_options", {}), **self.options}
            loaded[resource["name"]] = self._load_files(fmt, files, options)
        return next(iter(loaded.values())) if len(loaded) == 1 else loaded

    @staticmethod
    def _load_files(fmt: str, files: list[Path], options: dict) -> Any:
        encoding = options.get("encoding", "utf-8")
        if fmt in ("lines", "text"):
            return [
                line.rstrip("\n")
                for path in files
                for line in path.read_text(encoding=encoding).splitlines()
            ]
        if fmt == "jsonl":
            return [
                json.loads(line)
                for path in files
                for line in path.read_text(encoding=encoding).splitlines()
                if line.strip()
            ]
        if fmt == "csv":
            if options.get("header") is False and not options.get("fieldnames"):
                raise ValueError("headerless CSV requires loader_options.fieldnames")
            rows = []
            for path in files:
                with path.open(encoding=encoding, newline="") as stream:
                    reader = csv.DictReader(
                        stream,
                        fieldnames=options.get("fieldnames"),
                        delimiter=options.get("delimiter", ","),
                    )
                    rows.extend(reader)
            return rows
        if fmt == "idx":
            try:
                import numpy as np
            except ImportError as exc:
                raise ImportError(
                    "IDX loading requires NumPy; install it with `pip install numpy`"
                ) from exc
            dtypes = {
                8: np.uint8,
                9: np.int8,
                11: np.dtype(">i2"),
                12: np.dtype(">i4"),
                13: np.dtype(">f4"),
                14: np.dtype(">f8"),
            }
            arrays = []
            for path in files:
                raw = path.read_bytes()
                if len(raw) < 4 or raw[:2] != b"\0\0" or raw[2] not in dtypes:
                    raise ValueError(f"invalid IDX file: {path}")
                dimensions = raw[3]
                offset = 4 + 4 * dimensions
                if dimensions < 1 or len(raw) < offset:
                    raise ValueError(f"invalid IDX dimensions: {path}")
                shape = struct.unpack(">" + "I" * dimensions, raw[4:offset])
                array = np.frombuffer(raw, dtype=dtypes[raw[2]], offset=offset)
                if array.size != int(np.prod(shape)):
                    raise ValueError(f"truncated IDX file: {path}")
                arrays.append(array.reshape(shape).copy())
            return arrays[0] if len(arrays) == 1 else arrays
        raise ValueError(f"unsupported generic format {fmt!r}")


class DeclarativeDataset(TopologiesDataset):
    """Default implementation; override ``prepare_data`` or ``load_data`` as needed."""
