"""Single HTTP integration point for the artifact store (httpx + retries + dry-run).

Nobody outside this module may talk HTTP to the store (enforced by review;
grep for ARTIFACT_STORE_URL outside this package).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .exceptions import (
    ArtifactAuthError,
    ArtifactDigestError,
    ArtifactNotFoundError,
    ArtifactStoreError,
    ArtifactUnavailableError,
)

logger = logging.getLogger("topologies.artifact_store")

# Default store endpoints per environment. Explicit ARTIFACT_STORE_URL always
# wins; otherwise TOPOLOGIES_ENV selects the matching deployment.
STORE_URLS_BY_ENV = {
    "local": "http://localhost:8008",
}
DEFAULT_STORE_URL = STORE_URLS_BY_ENV["local"]


def resolve_store_url(default: str = DEFAULT_STORE_URL) -> str:
    """Resolve the store URL. Explicit ARTIFACT_STORE_URL always wins, then
    the TOPOLOGIES_ENV mapping (local), then the given default
    (typically topologies.toml [artifact_store] url)."""
    explicit = os.environ.get("ARTIFACT_STORE_URL", "").strip()
    if explicit:
        return explicit
    env_name = os.environ.get("TOPOLOGIES_ENV", "").strip().lower()
    if env_name in STORE_URLS_BY_ENV:
        return STORE_URLS_BY_ENV[env_name]
    return default


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class ArtifactStoreConfig:
    url: str = "http://localhost:8008"
    read_token: str = ""
    write_token: str = ""
    timeout_seconds: float = 10.0
    max_retries: int = 2

    @classmethod
    def from_env(cls, toml_path: str | Path | None = None) -> ArtifactStoreConfig:
        """Env wins; falls back to topologies.toml [artifact_store], then defaults.

        URL precedence: explicit ARTIFACT_STORE_URL > TOPOLOGIES_ENV mapping
        (local/dev/prod, see resolve_store_url) > topologies.toml > default.
        """
        url = DEFAULT_STORE_URL
        timeout = 10.0
        retries = 2
        candidates: list[Path] = []
        if toml_path:
            candidates.append(Path(toml_path))
        else:
            candidates.append(Path("topologies.toml"))
            try:
                here = Path(__file__).resolve()
                for parent in [here.parent, *here.parents]:
                    cand = parent / "topologies.toml"
                    if cand.is_file() and cand not in candidates:
                        candidates.append(cand)
                        break
            except Exception:  # noqa: BLE001, S110 - best-effort config discovery
                pass
        for cand in candidates:
            if cand.is_file():
                try:
                    try:
                        import tomllib
                    except ImportError:
                        import tomli as tomllib  # type: ignore[no-redef]
                    data = tomllib.loads(cand.read_text()).get("artifact_store", {})
                    if data.get("url"):
                        url = str(data["url"])
                    if data.get("timeout_seconds"):
                        timeout = float(data["timeout_seconds"])
                    if data.get("max_retries") is not None:
                        retries = int(data["max_retries"])
                except Exception:  # noqa: BLE001, S110 - ignore bad config file
                    pass
                break
        return cls(
            url=resolve_store_url(default=url),
            read_token=os.environ.get("ARTIFACT_STORE_READ_TOKEN", ""),
            write_token=os.environ.get("ARTIFACT_STORE_WRITE_TOKEN", ""),
            timeout_seconds=float(os.environ.get("ARTIFACT_STORE_TIMEOUT_SECONDS", timeout)),
            max_retries=int(os.environ.get("ARTIFACT_STORE_MAX_RETRIES", retries)),
        )


def build_client_from_env(*, dry_run: bool = False,
                          transport: httpx.BaseTransport | None = None) -> ArtifactStoreClient:
    cfg = ArtifactStoreConfig.from_env()
    return ArtifactStoreClient(
        base_url=cfg.url, read_token=cfg.read_token, write_token=cfg.write_token,
        timeout=cfg.timeout_seconds, max_retries=cfg.max_retries,
        dry_run=dry_run, transport=transport)


class ArtifactStoreClient:
    """Thin wrapper over the v1 API with retries, timeouts, and dry-run simulation."""

    def __init__(
        self,
        base_url: str = "http://localhost:8008",
        *,
        read_token: str = "",
        write_token: str = "",
        timeout: float = 10.0,
        max_retries: int = 2,
        dry_run: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.read_token = read_token
        self.write_token = write_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = dry_run
        self._http: httpx.Client | None = None
        self._transport = transport
        self._dry_blobs: dict[str, bytes] = {}
        self._dry_manifests: dict[tuple[str, str, int], dict] = {}
        self._dry_versions: dict[tuple[str, str], int] = {}
        if not dry_run:
            self._http = httpx.Client(base_url=self.base_url,
                                      timeout=self.timeout, transport=transport)

    def _headers(self, *, write: bool = False) -> dict[str, str]:
        tok = self.write_token if write else (self.read_token or self.write_token)
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        assert self._http is not None, "dry-run client has no HTTP transport"
        last: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                resp = self._http.request(method, path, **kwargs)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
                last = e
                if attempt < self.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise ArtifactUnavailableError(f"store unreachable {method} {path}: {e}") from e
            if resp.status_code >= 500 and attempt < self.max_retries:
                time.sleep(0.2 * (attempt + 1))
                continue
            return resp
        raise ArtifactUnavailableError(f"store unreachable {method} {path}: {last}")

    def _raise_for(self, resp: httpx.Response, ctx: str) -> None:
        if resp.status_code in (401, 403):
            raise ArtifactAuthError(f"{ctx}: HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code == 404:
            raise ArtifactNotFoundError(f"{ctx}: not found")
        if resp.status_code >= 400:
            raise ArtifactStoreError(f"{ctx}: HTTP {resp.status_code}: {resp.text[:500]}")

    def health(self) -> dict:
        if self.dry_run:
            logger.info("artifact-store dry-run: GET /health (no network)")
            return {"status": "ok", "dry_run": True}
        resp = self._request("GET", "/health")
        if resp.status_code != 200:
            raise ArtifactStoreError(f"health failed: {resp.text[:200]}")
        return resp.json()

    def upload(
        self,
        namespace: str,
        name: str,
        files: dict[str, bytes] | list[tuple[str, bytes]],
        metadata: dict | None = None,
    ) -> dict:
        """Upload files where keys are in-package relative paths. Returns manifest."""
        if isinstance(files, list):
            items = list(files)
        else:
            items = sorted(files.items())
        entries = [{"path": p, "sha256": _sha256(b), "size": len(b)} for p, b in items]
        if self.dry_run:
            logger.info(
                "artifact-store dry-run: POST /v1/upload ns=%s name=%s files=%d (no network)",
                namespace, name, len(items))
            for _p, b in items:
                self._dry_blobs[_sha256(b)] = b
            key_ns = (namespace, name)
            ver = self._dry_versions.get(key_ns, 0) + 1
            self._dry_versions[key_ns] = ver
            manifest = {
                "namespace": namespace, "name": name, "version": ver,
                "status": "active", "files": sorted(entries, key=lambda e: e["path"]),
                "metadata": metadata or {}, "dry_run": True,
                "created_at": "dry-run", "promoted_from": None,
                "deprecated": False, "deprecate_reason": "",
            }
            self._dry_manifests[(namespace, name, ver)] = manifest
            return manifest
        import json as _json

        files_arg = [("files", (p, data)) for p, data in items]
        resp = self._request(
            "POST", "/v1/upload", files=files_arg,
            data={"namespace": namespace, "name": name,
                  "metadata": _json.dumps(metadata or {})},
            headers=self._headers(write=True))
        self._raise_for(resp, f"upload {namespace}/{name}")
        return resp.json()

    def list_artifacts(self, *, namespace: str | None = None, name: str | None = None,
                       status: str | None = None, limit: int = 50,
                       offset: int = 0) -> dict:
        if self.dry_run:
            logger.info("artifact-store dry-run: GET /v1/artifacts (no network)")
            items = [m for (ns, nm, _), m in self._dry_manifests.items()
                     if (namespace is None or ns == namespace)
                     and (name is None or nm == name)
                     and (status is None or m.get("status") == status)]
            return {"items": items[offset: offset + limit], "total": len(items),
                    "limit": limit, "offset": offset, "dry_run": True}
        params: dict[str, object] = {"limit": limit, "offset": offset}
        if namespace:
            params["namespace"] = namespace
        if name:
            params["name"] = name
        if status:
            params["status"] = status
        resp = self._request("GET", "/v1/artifacts", params=params,
                             headers=self._headers())
        self._raise_for(resp, "list artifacts")
        return resp.json()

    def get_manifest(self, namespace: str, name: str, version: int | str = "latest") -> dict:
        if self.dry_run:
            logger.info(
                "artifact-store dry-run: GET manifest %s/%s@%s (no network)",
                namespace, name, version)
            if version == "latest":
                vers = [v for (ns, nm, v) in self._dry_manifests if ns == namespace and nm == name]
                if not vers:
                    raise ArtifactNotFoundError(f"dry-run: no such artifact {namespace}/{name}")
                version = max(vers)
            key = (namespace, name, int(version))  # type: ignore[arg-type]
            if key not in self._dry_manifests:
                raise ArtifactNotFoundError(f"dry-run: no such artifact {namespace}/{name}@{version}")
            return self._dry_manifests[key]
        resp = self._request("GET", f"/v1/artifacts/{namespace}/{name}@{version}",
                             headers=self._headers())
        self._raise_for(resp, f"get manifest {namespace}/{name}@{version}")
        return resp.json()

    def list_versions(self, namespace: str, name: str) -> dict:
        if self.dry_run:
            logger.info(
                "artifact-store dry-run: GET versions %s/%s (no network)", namespace, name)
            manifests = [m for (ns, nm, _), m in sorted(self._dry_manifests.items())
                         if ns == namespace and nm == name]
            return {"namespace": namespace, "name": name,
                    "versions": manifests, "dry_run": True}
        resp = self._request("GET", f"/v1/artifacts/{namespace}/{name}",
                             headers=self._headers())
        self._raise_for(resp, f"list versions {namespace}/{name}")
        return resp.json()

    def download_blob(self, sha256: str) -> bytes:
        sha = sha256.strip().lower()
        if self.dry_run:
            logger.info("artifact-store dry-run: GET blob %s (no network)", sha[:12])
            if sha not in self._dry_blobs:
                raise ArtifactNotFoundError(f"dry-run: no such blob {sha[:12]}")
            return self._dry_blobs[sha]
        resp = self._request("GET", f"/v1/files/{sha}", headers=self._headers())
        self._raise_for(resp, f"download blob {sha[:12]}")
        data = resp.content
        if _sha256(data) != sha:
            raise ArtifactDigestError(f"digest mismatch for blob {sha[:12]}")
        return data

    def download_version(self, namespace: str, name: str, version: int | str = "latest") -> bytes:
        if self.dry_run:
            logger.info(
                "artifact-store dry-run: GET tarball %s/%s@%s (no network)",
                namespace, name, version)
            manifest = self.get_manifest(namespace, name, version)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for entry in manifest.get("files", []):
                    data = self._dry_blobs.get(entry["sha256"], b"")
                    ti = tarfile.TarInfo(name=entry["path"])
                    ti.size = len(data)
                    tar.addfile(ti, io.BytesIO(data))
            return buf.getvalue()
        resp = self._request("GET", f"/v1/download/{namespace}/{name}@{version}",
                             headers=self._headers())
        self._raise_for(resp, f"download {namespace}/{name}@{version}")
        return resp.content

    def download_version_files(
        self, namespace: str, name: str, version: int | str = "latest"
    ) -> dict[str, bytes]:
        """Download tarball and return {path: bytes}, verifying every sha against manifest."""
        manifest = self.get_manifest(namespace, name, version)
        tarball = self.download_version(namespace, name, version)
        out: dict[str, bytes] = {}
        buf = io.BytesIO(tarball)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is None:
                    continue
                out[member.name] = f.read()
        want = {e["path"]: e["sha256"] for e in manifest.get("files", [])}
        if set(out) != set(want):
            raise ArtifactDigestError(
                f"tarball contents mismatch for {namespace}/{name}@{version}: "
                f"got {sorted(out)}, want {sorted(want)}")
        for path, sha in want.items():
            if _sha256(out[path]) != sha:
                raise ArtifactDigestError(f"digest mismatch for {path} in {name}@{version}")
        return out

    def promote(self, from_namespace: str, to_namespace: str, name: str,
              version: int) -> dict:
        if self.dry_run:
            logger.info(
                "artifact-store dry-run: POST promote %s->%s %s@%s (no network)",
                from_namespace, to_namespace, name, version)
            src = self.get_manifest(from_namespace, name, version)
            key_ns = (to_namespace, name)
            ver = self._dry_versions.get(key_ns, 0) + 1
            self._dry_versions[key_ns] = ver
            manifest = {
                "namespace": to_namespace, "name": name, "version": ver,
                "status": "active", "files": list(src.get("files", [])),
                "metadata": dict(src.get("metadata", {})),
                "promoted_from": {"namespace": from_namespace, "name": name,
                                  "version": version},
                "dry_run": True, "created_at": "dry-run",
                "deprecated": False, "deprecate_reason": "",
            }
            self._dry_manifests[(to_namespace, name, ver)] = manifest
            return manifest
        resp = self._request(
            "POST", "/v1/promote",
            json={"from_namespace": from_namespace, "to_namespace": to_namespace,
                  "name": name, "version": version},
            headers=self._headers(write=True))
        self._raise_for(resp, f"promote {from_namespace}->{to_namespace} {name}@{version}")
        return resp.json()

    def deprecate(self, namespace: str, name: str, version: int, reason: str = "") -> dict:
        if self.dry_run:
            logger.info(
                "artifact-store dry-run: POST deprecate %s/%s@%s (no network)",
                namespace, name, version)
            m = self.get_manifest(namespace, name, version)
            m = dict(m)
            m["deprecated"] = True
            m["deprecate_reason"] = reason
            m["status"] = "deprecated"
            self._dry_manifests[(namespace, name, int(version))] = m  # type: ignore[arg-type]
            return m
        resp = self._request(
            "POST", "/v1/deprecate",
            json={"namespace": namespace, "name": name,
                  "version": version, "reason": reason},
            headers=self._headers(write=True))
        self._raise_for(resp, f"deprecate {namespace}/{name}@{version}")
        return resp.json()

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def __enter__(self) -> ArtifactStoreClient:  # noqa: PYI034 - runtime returns self
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
