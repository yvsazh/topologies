"""A minimal, safe text template engine (no Jinja, no ``eval``).

    t = Template('''Dear {{ name | title }},
    {% if items %}your order:
    {% for it in items %}- {{ it.qty }} x {{ it.product }}
    {% endfor %}{% endif %}Total: {{ total }} {{ currency | upper }}''')
    t.render(name="olena", items=[...], total=12.5, currency="uah")

Supported: ``{{ expr }}``, ``{{ expr | filter | filter(arg) }}``, ``{% if %}`` /
``{% elif %}`` / ``{% else %}`` / ``{% endif %}``, ``{% for x in expr %}`` /
``{% endfor %}`` (``loop.index``, ``loop.first``, ``loop.last``), ``{# comments #}``.
Expressions: dotted names (``order.items.0.product``, dict keys and
attributes), string / number literals, ``==``, ``!=``, ``in``, ``not in``,
``not``, ``and``, ``or``. Filters: ``join(sep)``, ``upper``, ``lower``,
``title``, ``default(value)``, ``json``, ``length``, ``trim``, ``truncate(n)``.
A missing name renders as empty (``default`` replaces it). The text and the
version are part of the dataset fingerprint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from topologies.synth import SynthError

_TOKEN = re.compile(r"({{.*?}}|{%.*?%}|{#.*?#})", re.DOTALL)
_MISSING = object()


class TemplateError(SynthError):
    """Template syntax or rendering error (with the offending tag)."""


# ---------------------------------------------------------------- expressions

_EXPR_TOKEN = re.compile(r"""\s*(?:
    (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*') |
    (?P<num>-?\d+(?:\.\d+)?) |
    (?P<op>==|!=|\(|\)|,|\|) |
    (?P<name>[A-Za-z_][\w.]*)
)""", re.VERBOSE)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    pos, out = 0, []
    expr = expr.strip()
    while pos < len(expr):
        m = _EXPR_TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            raise TemplateError(f"Cannot parse expression {expr!r} at {expr[pos:]!r}.",
                                stage="template")
        kind = m.lastgroup or "?"
        out.append((kind, m.group(kind)))
        pos = m.end()
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
    return out


def _lookup(ctx: Mapping[str, Any], dotted: str) -> Any:
    parts = dotted.split(".")
    if parts[0] in ("true", "True"):
        return True
    if parts[0] in ("false", "False"):
        return False
    if parts[0] in ("none", "None"):
        return None
    cur: Any = ctx.get(parts[0], _MISSING)
    for p in parts[1:]:
        if cur is _MISSING or cur is None:
            return _MISSING
        if isinstance(cur, Mapping):
            cur = cur.get(p, _MISSING)
        elif isinstance(cur, (list, tuple)) and p.lstrip("-").isdigit():
            i = int(p)
            cur = cur[i] if -len(cur) <= i < len(cur) else _MISSING
        else:
            cur = getattr(cur, p, _MISSING)
    return cur


def _to_str(v: Any) -> str:
    if v is _MISSING or v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


FILTERS: dict[str, Callable[..., Any]] = {
    "upper": lambda v: _to_str(v).upper(),
    "lower": lambda v: _to_str(v).lower(),
    "title": lambda v: _to_str(v).title(),
    "trim": lambda v: _to_str(v).strip(),
    "join": lambda v, sep=", ": sep.join(_to_str(x) for x in (v or [])) if v is not _MISSING
    else "",
    "default": lambda v, d="": d if v is _MISSING or v is None or v == "" else v,
    "json": lambda v: json.dumps(None if v is _MISSING else v, ensure_ascii=False),
    "length": lambda v: 0 if v is _MISSING or v is None else len(v),
    "truncate": lambda v, n=100: _to_str(v)[: int(n)],
}


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], ctx: Mapping[str, Any]) -> None:
        self.t, self.i, self.ctx = tokens, 0, ctx

    def peek(self) -> tuple[str, str] | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise TemplateError("Unexpected end of expression.", stage="template")
        self.i += 1
        return tok

    def expr(self) -> Any:
        v = self.and_()
        while self.peek() == ("name", "or"):
            self.take()
            rhs = self.and_()
            v = _truthy(v) or _truthy(rhs)
        return v

    def and_(self) -> Any:
        v = self.not_()
        while self.peek() == ("name", "and"):
            self.take()
            rhs = self.not_()
            v = _truthy(v) and _truthy(rhs)
        return v

    def not_(self) -> Any:
        if self.peek() == ("name", "not"):
            self.take()
            return not _truthy(self.not_())
        return self.cmp()

    def cmp(self) -> Any:
        left = self.filtered()
        tok = self.peek()
        if tok in (("op", "=="), ("op", "!=")):
            self.take()
            right = self.filtered()
            eq = _norm(left) == _norm(right)
            return eq if tok[1] == "==" else not eq
        if tok == ("name", "in") or tok == ("name", "not"):
            neg = False
            if tok == ("name", "not"):
                self.take()
                if self.take() != ("name", "in"):
                    raise TemplateError("Expected 'not in'.", stage="template")
                neg = True
            else:
                self.take()
            right = self.filtered()
            try:
                found = _norm(left) in (right if right is not _MISSING else [])
            except TypeError:
                found = False
            return not found if neg else found
        return left

    def filtered(self) -> Any:
        v = self.atom()
        while self.peek() == ("op", "|"):
            self.take()
            kind, name = self.take()
            fn = FILTERS.get(name)
            if kind != "name" or fn is None:
                raise TemplateError(f"Unknown filter {name!r}.", stage="template",
                                    available=sorted(FILTERS))
            args = []
            if self.peek() == ("op", "("):
                self.take()
                while self.peek() != ("op", ")"):
                    args.append(self.atom())
                    if self.peek() == ("op", ","):
                        self.take()
                self.take()
            v = fn(v, *args)
        return v

    def atom(self) -> Any:
        kind, val = self.take()
        if kind == "str":
            return val[1:-1].encode("utf-8").decode("unicode_escape") if "\\" in val else \
                val[1:-1]
        if kind == "num":
            return float(val) if "." in val else int(val)
        if kind == "name":
            return _lookup(self.ctx, val)
        if (kind, val) == ("op", "("):
            v = self.expr()
            if self.take() != ("op", ")"):
                raise TemplateError("Missing ')'.", stage="template")
            return v
        raise TemplateError(f"Unexpected {val!r}.", stage="template")


def _norm(v: Any) -> Any:
    return None if v is _MISSING else v


def _truthy(v: Any) -> bool:
    return False if v is _MISSING else bool(v)


def evaluate(expr: str, ctx: Mapping[str, Any]) -> Any:
    p = _Parser(_tokenize(expr), ctx)
    v = p.expr()
    if p.peek() is not None:
        raise TemplateError(f"Unexpected {p.peek()[1]!r} in {expr!r}.",  # type: ignore[index]
                            stage="template")
    return v


# ---------------------------------------------------------------- template tree


def _parse(text: str) -> list[Any]:
    """Nodes: ("text", s) | ("var", expr) | ("if", [(cond, body)...], else) |
    ("for", var, expr, body)."""
    stack: list[tuple[str, Any]] = [("root", [])]
    for piece in _TOKEN.split(text):
        if not piece:
            continue
        if piece.startswith("{#"):
            continue
        if piece.startswith("{{"):
            _append(stack, ("var", piece[2:-2].strip()))
            continue
        if piece.startswith("{%"):
            tag = piece[2:-2].strip()
            word = tag.split(None, 1)[0] if tag else ""
            rest = tag[len(word):].strip()
            if word == "if":
                stack.append(("if", {"branches": [(rest, [])], "else": None}))
            elif word in ("elif", "else"):
                if stack[-1][0] != "if":
                    raise TemplateError(f"{{% {word} %}} outside {{% if %}}.", stage="template")
                node = stack[-1][1]
                if word == "elif":
                    node["branches"].append((rest, []))
                else:
                    node["else"] = []
                    node["branches"].append((None, node["else"]))
            elif word == "endif":
                if stack[-1][0] != "if":
                    raise TemplateError("{% endif %} without {% if %}.", stage="template")
                _, node = stack.pop()
                branches = [(c, b) for c, b in node["branches"] if c is not None]
                _append(stack, ("if", branches, node["else"]))
            elif word == "for":
                m = re.match(r"(\w+)\s+in\s+(.+)$", rest)
                if not m:
                    raise TemplateError(f"Bad for tag {tag!r}; use {{% for x in items %}}.",
                                        stage="template")
                stack.append(("for", {"var": m.group(1), "expr": m.group(2), "body": []}))
            elif word == "endfor":
                if stack[-1][0] != "for":
                    raise TemplateError("{% endfor %} without {% for %}.", stage="template")
                _, node = stack.pop()
                _append(stack, ("for", node["var"], node["expr"], node["body"]))
            else:
                raise TemplateError(f"Unknown tag {{% {word} %}}.", stage="template",
                                    hint="Supported: if / elif / else / endif / for / endfor.")
            continue
        _append(stack, ("text", piece))
    if len(stack) != 1:
        raise TemplateError(f"Unclosed {{% {stack[-1][0]} %}}.", stage="template")
    return stack[0][1]


def _append(stack: list[tuple[str, Any]], node: Any) -> None:
    kind, payload = stack[-1]
    if kind == "root":
        payload.append(node)
    elif kind == "if":
        payload["branches"][-1][1].append(node)
    else:
        payload["body"].append(node)


def _render(nodes: list[Any], ctx: dict[str, Any], out: list[str]) -> None:
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(node[1])
        elif kind == "var":
            out.append(_to_str(evaluate(node[1], ctx)))
        elif kind == "if":
            for cond, body in node[1]:
                if _truthy(evaluate(cond, ctx)):
                    _render(body, ctx, out)
                    break
            else:
                if node[2] is not None:
                    _render(node[2], ctx, out)
        elif kind == "for":
            items = evaluate(node[2], ctx)
            if items is _MISSING or items is None:
                continue
            if isinstance(items, Mapping):
                items = list(items.items())
            items = list(items)
            for i, x in enumerate(items):
                inner = {**ctx, node[1]: x, "loop": {"index": i + 1, "index0": i,
                                                     "first": i == 0,
                                                     "last": i == len(items) - 1}}
                _render(node[3], inner, out)


class Template:
    """A versioned template (see module docstring)."""

    def __init__(self, template: str, *, version: str | None = None,
                 name: str | None = None) -> None:
        if not isinstance(template, str):
            raise SynthError("A template must be a string.", stage="template")
        self.template, self.version, self.name = template, version, name
        self._nodes = _parse(template)

    @classmethod
    def from_file(cls, path: str | Path, *, version: str | None = None) -> Template:
        p = Path(path)
        if not p.is_file():
            raise SynthError(f"Template file {p} does not exist.", stage="template")
        return cls(p.read_text(encoding="utf-8"), version=version, name=p.name)

    def render(self, **ctx: Any) -> str:
        out: list[str] = []
        _render(self._nodes, ctx, out)
        return _tidy("".join(out))

    def fingerprint(self) -> str:
        import hashlib

        return hashlib.sha256(json.dumps([self.template, self.version]).encode()).hexdigest()[:16]

    def to_config(self) -> dict[str, Any]:
        return {"template": self.template, "version": self.version}

    def __repr__(self) -> str:
        return f"Template({self.name or self.template[:30]!r}, version={self.version!r})"


def _tidy(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


__all__ = ["FILTERS", "Template", "TemplateError", "evaluate"]
