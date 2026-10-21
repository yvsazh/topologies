"""Conservative detection of catalog facades that discard their algorithm."""

from __future__ import annotations

import ast


def discarded_algorithm_classes(source: str) -> list[str]:
    """Find forwards returning an input plus an explicitly zeroed computation.

    Deliberately narrow: identity branches in residual, dropout and masking
    algorithms are valid. This detects the unconditional terminal zero-term
    facade, without treating its unused numerical helpers as an implementation.
    """
    found = []
    for cls in ast.parse(source).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        forward = next((node for node in cls.body
                        if isinstance(node, ast.FunctionDef) and node.name == "forward"), None)
        if forward is None or not forward.body or not isinstance(forward.body[-1], ast.Return):
            continue
        value = forward.body[-1].value
        inputs = {arg.arg for arg in forward.args.args[1:]}

        def zeroed(node):
            return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
                    and any(isinstance(item, ast.Constant) and not isinstance(item.value, bool)
                            and item.value == 0 for item in (node.left, node.right)))

        if (isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub))
                and isinstance(value.left, ast.Name) and value.left.id in inputs
                and zeroed(value.right)):
            found.append(cls.name)
    return found
