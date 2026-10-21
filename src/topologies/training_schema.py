"""Lightweight validation contract for executable training catalog metadata.

This module intentionally lives outside ``topologies.training`` so catalog
consumers can validate metadata without importing PyTorch training runtime.
"""

import ast
import io
import re
import tokenize

KINDS = {"loss": "losses", "optimizer": "optimizers"}
ENTRY_ID_RE = re.compile(r"\A[-a-zA-Z0-9_]+\Z")
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def validate_catalog(catalog, kind):
    if kind not in KINDS:
        raise ValueError(f"unknown training catalog kind: {kind}")
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != 1
        or not isinstance(catalog.get("entries"), list)
    ):
        raise ValueError("expected schema_version 1 and entries array")
    seen = set()
    for entry in catalog["entries"]:
        if not isinstance(entry, dict):
            raise TypeError("catalog entries must be objects")
        for key in ("id", "name", "description", "api", "usage"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"each entry requires non-empty {key}")
        if len(entry["name"]) > 300:
            raise ValueError("entry name must be at most 300 characters")
        parameters = entry.get("parameters", {})
        if not isinstance(parameters, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, (str, int, float, bool, list))
            or (
                isinstance(value, list)
                and not all(isinstance(item, (str, int, float, bool)) for item in value)
            )
            for key, value in parameters.items()
        ):
            raise ValueError("parameters must be a flat object of scalar values or scalar lists")
        if len(entry["id"]) > 200 or not ENTRY_ID_RE.fullmatch(entry["id"]):
            raise ValueError("entry id must be a URL slug of at most 200 characters")
        if entry["id"] in seen:
            raise ValueError(f"duplicate entry id: {entry['id']}")
        seen.add(entry["id"])
        examples = entry.get("examples")
        if not isinstance(examples, list) or len(examples) != 1:
            raise ValueError(f"{entry['id']}: exactly one executable example is required")
        example = examples[0]
        if (
            not isinstance(example, dict)
            or not isinstance(example.get("title"), str)
            or not example["title"].strip()
            or not isinstance(example.get("code"), str)
            or not example["code"].strip()
        ):
            raise ValueError(f"{entry['id']}: example requires non-empty title and code")
        code = example["code"]
        try:
            tree = ast.parse(code, filename=f"{entry['id']}-example.py")
            compile(tree, f"{entry['id']}-example.py", "exec")
        except SyntaxError as exc:
            raise ValueError(f"{entry['id']}: example does not compile: {exc}") from exc
        if any(
            token.type == tokenize.COMMENT
            for token in tokenize.generate_tokens(io.StringIO(code).readline)
        ):
            raise ValueError(f"{entry['id']}: example source must not contain comments")
        if any(
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(getattr(node.body[0].value, "value", None), str)
            for node in ast.walk(tree)
        ):
            raise ValueError(f"{entry['id']}: example source must not contain docstrings")
        impl = entry.get("implementation")
        if not isinstance(impl, dict):
            raise TypeError(f"{entry['id']}: executable implementation is required")
        path, symbol, digest = impl.get("path"), impl.get("symbol"), impl.get("sha256")
        if (
            not isinstance(path, str)
            or not path.startswith("implementations/")
            or path.endswith("/")
            or ".." in path.split("/")
            or not path.endswith(".py")
        ):
            raise ValueError(f"{entry['id']}: invalid implementation path")
        if not isinstance(symbol, str) or not symbol.isidentifier():
            raise ValueError(f"{entry['id']}: invalid implementation symbol")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"{entry['id']}: invalid implementation sha256")
    return catalog
