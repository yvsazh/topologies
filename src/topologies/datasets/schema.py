"""Validation for the stable schema-v1 JSON dataset catalog."""

from __future__ import annotations

import re
from pathlib import PureWindowsPath
from urllib.parse import urlparse

ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ARCHIVES = {"none", "zip", "tar", "tar.gz", "gzip"}
MANIFEST_NAME = ".topologies-manifest.json"


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a safe relative POSIX path")
    windows = PureWindowsPath(value)
    if (
        value.startswith("/")
        or windows.drive
        or windows.root
        or any(part in ("", ".", "..") or ":" in part for part in value.split("/"))
    ):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    return value


def _http_url(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string URL")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{label} must be an HTTP(S) URL")


def validate_definition(definition: dict) -> dict:
    """Validate a schema-v1 definition and return it unchanged."""
    if not isinstance(definition, dict):
        raise TypeError("dataset definition must be an object")
    required = (
        "id",
        "name",
        "description",
        "domain",
        "tasks",
        "tags",
        "license",
        "source_url",
        "project_url",
        "citations",
        "default_version",
        "versions",
    )
    for key in required:
        if key not in definition:
            raise ValueError(f"dataset missing required field {key!r}")
    for key in ("id", "default_version"):
        value = definition[key]
        if not isinstance(value, str) or not ID.fullmatch(value) or value in (".", ".."):
            raise ValueError(f"invalid dataset {key}")
    for key in ("name", "description", "domain"):
        if not isinstance(definition[key], str) or not definition[key]:
            raise TypeError(f"{key} must be a nonempty string")
    for key in ("tasks", "tags", "citations"):
        if not isinstance(definition[key], list) or any(
            not isinstance(item, str) for item in definition[key]
        ):
            raise TypeError(f"{key} must be a list of strings")
    for key in ("source_url", "project_url"):
        _http_url(definition[key], key)
    license_info = definition["license"]
    if not isinstance(license_info, dict) or not isinstance(license_info.get("identifier"), str):
        raise TypeError("license requires a string identifier")
    license_url = license_info.get("url", "")
    if not isinstance(license_url, str):
        raise TypeError("license.url must be a string when provided")
    if license_url:
        _http_url(license_url, "license.url")
    if "notes" in license_info and not isinstance(license_info["notes"], str):
        raise TypeError("license.notes must be a string")
    if "access_mode" in definition and definition["access_mode"] not in {
        "direct", "manual", "metadata-only"
    }:
        raise ValueError("access_mode must be direct, manual, or metadata-only")
    if "access_notes" in definition and not isinstance(definition["access_notes"], str):
        raise TypeError("access_notes must be a string")

    versions = definition["versions"]
    if "catalog_only" in definition and not isinstance(definition["catalog_only"], bool):
        raise TypeError("catalog_only must be boolean")
    if (
        not isinstance(versions, dict)
        or not versions
        or definition["default_version"] not in versions
    ):
        raise ValueError("default_version must name a version")
    for version, version_info in versions.items():
        if not isinstance(version, str) or not ID.fullmatch(version) or version in (".", ".."):
            raise ValueError("invalid version")
        if not isinstance(version_info, dict):
            raise TypeError("version definition must be an object")
        if definition.get("catalog_only") and (
            definition.get("implementation") or version_info.get("resources")
            or version_info.get("splits")
        ):
            raise ValueError("catalog-only metadata cannot claim a loader, resources or splits")
        resources = version_info.get("resources")
        splits = version_info.get("splits")
        if not isinstance(resources, list) or not isinstance(splits, dict):
            raise TypeError("version requires resources list and splits object")
        names, filenames, targets = set(), set(), set()
        for resource in resources:
            if not isinstance(resource, dict):
                raise TypeError("resource must be an object")
            for key in ("name", "url", "filename", "archive", "format"):
                if key not in resource:
                    raise ValueError(f"resource missing {key!r}")
            name = resource["name"]
            if not isinstance(name, str) or not ID.fullmatch(name) or name in (".", ".."):
                raise ValueError("invalid resource name")
            if name in names:
                raise ValueError("duplicate resource name")
            names.add(name)
            _http_url(resource["url"], "resource url")
            filename = _relative_path(resource["filename"], "resource filename")
            if MANIFEST_NAME in filename.split("/") or any(
                part.startswith((".download-", ".extract-")) for part in filename.split("/")
            ):
                raise ValueError("resource filename uses a reserved cache name")
            if filename in filenames:
                raise ValueError("resource filenames must be unique")
            filenames.add(filename)
            if resource["archive"] not in ARCHIVES:
                raise ValueError("unsupported archive type")
            limit = resource.get("max_extracted_bytes")
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                                      or limit < 1):
                raise ValueError("max_extracted_bytes must be a positive integer")
            if not isinstance(resource["format"], str) or not resource["format"]:
                raise TypeError("resource format must be a nonempty string")
            target = _relative_path(resource.get("extract_to", f"extracted/{name}"), "extract_to")
            if MANIFEST_NAME in target.split("/") or any(
                part.startswith((".download-", ".extract-")) for part in target.split("/")
            ):
                raise ValueError("extraction target uses a reserved cache name")
            if target in targets:
                raise ValueError("resource extraction targets cannot collide")
            targets.add(target)
            if resource.get("data_files") is not None:
                if not isinstance(resource["data_files"], list) or not resource["data_files"]:
                    raise TypeError("data_files must be a nonempty list of relative paths")
                for data_file in resource["data_files"]:
                    _relative_path(data_file, "data_files entry")
            for algorithm, length in (("sha256", 64), ("md5", 32)):
                checksum = resource.get(algorithm)
                if checksum is not None and (
                    not isinstance(checksum, str)
                    or not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", checksum)
                ):
                    raise ValueError(f"invalid {algorithm}")
            if "bytes" in resource and (
                not isinstance(resource["bytes"], int)
                or isinstance(resource["bytes"], bool)
                or resource["bytes"] < 0
            ):
                raise ValueError("resource bytes must be a nonnegative integer")
            loader_options = resource.get("loader_options", {})
            if not isinstance(loader_options, dict):
                raise TypeError("loader_options must be an object")
            if "header" in loader_options and not isinstance(loader_options["header"], bool):
                raise TypeError("loader_options.header must be boolean")
            if "fieldnames" in loader_options and not (
                isinstance(loader_options["fieldnames"], list)
                and all(isinstance(x, str) for x in loader_options["fieldnames"])
            ):
                raise TypeError("loader_options.fieldnames must be a string list")
            if loader_options.get("header") is False and not loader_options.get("fieldnames"):
                raise ValueError("headerless CSV requires loader_options.fieldnames")
            if "delimiter" in loader_options and not isinstance(loader_options["delimiter"], str):
                raise TypeError("loader_options.delimiter must be a string")
            if "delimiter" in loader_options and len(loader_options["delimiter"]) != 1:
                raise ValueError("loader_options.delimiter must be exactly one character")
            if "encoding" in loader_options and not isinstance(loader_options["encoding"], str):
                raise TypeError("loader_options.encoding must be a string")
        for target in targets:
            if any(
                path == target or path.startswith(target + "/") or target.startswith(path + "/")
                for path in filenames
            ):
                raise ValueError("resource extraction targets cannot overlap resource files")
        target_paths = list(targets)
        if any(
            a.startswith(b + "/") or b.startswith(a + "/")
            for index, a in enumerate(target_paths)
            for b in target_paths[index + 1 :]
        ):
            raise ValueError("resource extraction targets cannot nest")
        paths = list(filenames)
        if any(
            a.startswith(b + "/") or b.startswith(a + "/")
            for index, a in enumerate(paths)
            for b in paths[index + 1 :]
        ):
            raise ValueError("resource filenames cannot overlap")
        for split, refs in splits.items():
            if (
                not isinstance(split, str)
                or not split
                or not isinstance(refs, list)
                or any(ref not in names for ref in refs)
            ):
                raise ValueError(f"split {split!r} references unknown resource")
        if "format" in version_info and (
            not isinstance(version_info["format"], str) or not version_info["format"]
        ):
            raise TypeError("version format must be a nonempty string")
    options = definition.get("options", {})
    if not isinstance(options, dict):
        raise TypeError("options must be an object mapping names to descriptors")
    for name, spec in options.items():
        if (
            not isinstance(name, str)
            or not isinstance(spec, dict)
            or spec.get("type") not in ("str", "int", "float", "bool")
        ):
            raise ValueError("options must map names to {type, ...} descriptors")
        if "default" in spec and not _option_type(spec["default"], spec["type"]):
            raise TypeError(f"invalid default for option {name!r}")
    dependencies = definition.get("dependencies", {"extras": [], "packages": []})
    if not isinstance(dependencies, dict) or set(dependencies) != {"extras", "packages"}:
        raise TypeError('dependencies must be {"extras": [...], "packages": [...]}')
    for key, values in dependencies.items():
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise TypeError(f"dependencies.{key} must be a list of nonempty strings")
    if not isinstance(license_info["identifier"], str) or not license_info["identifier"].strip():
        raise ValueError("license.identifier must be nonempty")
    implementation = definition.get("implementation")
    if implementation is not None:
        if not isinstance(implementation, str):
            raise TypeError("implementation must be a module:Class string")
        module, sep, attr = implementation.partition(":")
        if (
            not sep
            or not module
            or not attr
            or not all(part.isidentifier() for part in module.split("."))
            or not attr.isidentifier()
        ):
            raise ValueError("implementation must be module:Class")
    return definition


def _option_type(value: object, kind: str) -> bool:
    return {
        "str": lambda x: isinstance(x, str),
        "int": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "float": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "bool": lambda x: isinstance(x, bool),
    }[kind](value)


def validate_catalog(catalog: dict) -> dict:
    """Validate the packaged/authoring catalog and reject duplicate IDs."""
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != 1
        or not isinstance(catalog.get("datasets"), list)
    ):
        raise ValueError("catalog must have schema_version=1 and datasets list")
    seen = set()
    for definition in catalog["datasets"]:
        validate_definition(definition)
        if definition["id"] in seen:
            raise ValueError(f"duplicate dataset id {definition['id']!r}")
        seen.add(definition["id"])
    return catalog
