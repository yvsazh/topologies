"""Static source policy checks for bytes that are rendered as catalog code."""

from __future__ import annotations

import ast
import io
import tokenize


def comment_and_docstring_violations(source: str, filename: str = "<source>") -> list[str]:
    """Return comment-token and AST docstring findings without matching string data."""
    tree = ast.parse(source, filename=filename)
    violations = [f"comment at line {tok.start[0]}" for tok in
                  tokenize.generate_tokens(io.StringIO(source).readline)
                  if tok.type == tokenize.COMMENT]
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and ast.get_docstring(node, clean=False) is not None):
            violations.append(f"docstring at line {getattr(node, 'lineno', 1)}")
    return violations


__all__ = ["comment_and_docstring_violations"]
