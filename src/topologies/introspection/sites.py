"""Addressable sites, the module index and the selector grammar.

A site is ``<module-path>[:<port>][@<occurrence>]``, an op site
``<module-path>/fn:<op>[#<i>][:<port>][@<occurrence>]`` or ``param:<name>``.
Selectors are comma-separated terms (union, left to right); a leading ``!``
excludes. See ``docs/introspection.md`` for the full grammar.
"""
from __future__ import annotations

import difflib
import importlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

ROOT = "<root>"
_WRAPPER_SEGMENTS = frozenset({"_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"})
_PORT_HEAD = re.compile(r"^(output|input|grad_output|grad_input|weights|q|k|v|mask)(\.|$)")
_OCC = re.compile(r"@(\d+|\*)$")
_DEPTH = re.compile(r"^depth\s*(<=|>=|<|>|==)\s*(\d+)$")
_PREFIXES = ("re:", "type:", "component:", "role:")


# --------------------------------------------------------------------------- keys


@dataclass(frozen=True, order=True)
class SiteKey:
    """Canonical identity of one captured value: path, port, occurrence (+ op)."""

    path: str
    port: str = "output"
    occurrence: int = 0
    op: str | None = None
    op_index: int | None = None

    @property
    def site(self) -> str:
        """The site string without occurrence (``output`` port omitted)."""
        base = self.path or ROOT
        if self.op is not None:
            base = f"{base}/fn:{self.op}#{self.op_index or 0}"
        return base if self.port == "output" else f"{base}:{self.port}"

    def __str__(self) -> str:
        return f"{self.site}@{self.occurrence}"


@dataclass(frozen=True)
class SiteSpec:
    """One parsed selector term."""

    module: str
    port: str | None = None
    occurrence: int | str | None = None
    op: str | None = None
    op_index: int | None = None
    param: str | None = None
    exclude: bool = False


def split_terms(selector: str) -> list[str]:
    """Split on top-level commas (commas inside ``[]``/``{}``/``()`` are kept)."""
    terms, depth, current = [], 0, []
    for char in selector:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            terms.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    terms.append("".join(current).strip())
    return [t for t in terms if t]


def parse_term(term: str) -> SiteSpec:
    """Parse one selector term into module selector, port, occurrence and op parts."""
    exclude = term.startswith("!")
    term = term[1:].strip() if exclude else term.strip()
    if term.startswith("param:"):
        return SiteSpec("", param=term[len("param:"):], exclude=exclude)
    occurrence: int | str | None = None
    match = _OCC.search(term)
    if match:
        occurrence = "*" if match.group(1) == "*" else int(match.group(1))
        term = term[: match.start()]
    op = op_index = None
    port: str | None = None
    if "/fn:" in term:
        term, op_part = term.split("/fn:", 1)
        op_name, _, op_port = op_part.partition(":")
        if "#" in op_name:
            op_name, index = op_name.split("#", 1)
            op_index = None if index == "*" else int(index)
        op = op_name
        port = op_port or None
    else:
        prefix = next((p for p in _PREFIXES if term.startswith(p)), None)
        head = prefix or ""
        rest = term[len(head):]
        parts = rest.split(":")
        # The selector value is everything up to the first token that starts a port.
        cut = next((i for i in range(1, len(parts)) if _PORT_HEAD.match(parts[i])), len(parts))
        term = head + ":".join(parts[:cut])
        port = ":".join(parts[cut:]) or None
    return SiteSpec(term, port, occurrence, op, op_index, None, exclude)


# --------------------------------------------------------------------------- index


@dataclass
class ModuleInfo:
    path: str
    type_name: str
    type_path: str
    component_id: str | None
    params: int
    depth: int
    leaf: bool
    aliases: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path or ROOT, "type": self.type_path, "component_id": self.component_id,
                "params": self.params, "depth": self.depth, "leaf": self.leaf,
                "aliases": list(self.aliases), "roles": list(self.roles)}


def clean_path(name: str) -> str:
    """Strip FSDP / checkpoint / compile wrapper segments from a module name."""
    if not name:
        return name
    return ".".join(seg for seg in name.split(".") if seg not in _WRAPPER_SEGMENTS)


class SiteIndex:
    """One walk over ``named_modules(remove_duplicate=False)``: paths, types, aliases, roles."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.modules: dict[str, ModuleInfo] = {}
        self._objects: dict[str, nn.Module] = {}
        self.by_id: dict[int, str] = {}
        aliases: dict[int, list[str]] = {}
        for raw, module in model.named_modules(remove_duplicate=False):
            path = clean_path(raw)
            key = id(module)
            if key in self.by_id:
                if path != self.by_id[key]:
                    aliases.setdefault(key, []).append(path)
                continue
            if path in self._objects:  # a stripped wrapper collapsed onto its child
                continue
            self.by_id[key] = path
            self._objects[path] = module
        for path, module in self._objects.items():
            cls = type(module)
            self.modules[path] = ModuleInfo(
                path=path, type_name=cls.__name__, type_path=f"{cls.__module__}.{cls.__qualname__}",
                component_id=getattr(cls, "component_id", None) if isinstance(
                    getattr(cls, "component_id", None), str) else None,
                params=sum(p.numel() for p in module.parameters()),
                depth=0 if not path else path.count(".") + 1,
                leaf=not any(True for _ in module.children()),
                aliases=tuple(dict.fromkeys(aliases.get(id(module), ()))))
        self.roles: dict[str, list[str]] = {}

    @classmethod
    def build(cls, model: nn.Module) -> SiteIndex:
        return cls(model)

    def __contains__(self, path: str) -> bool:
        return self.canonical(path) in self._objects

    def __len__(self) -> int:
        return len(self._objects)

    def paths(self) -> list[str]:
        return list(self._objects)

    def canonical(self, path: str) -> str:
        path = "" if path in (ROOT, "") else clean_path(path)
        if path in self._objects:
            return path
        for info in self.modules.values():
            if path in info.aliases:
                return info.path
        return path

    def module(self, path: str) -> nn.Module:
        canonical = self.canonical(path)
        try:
            return self._objects[canonical]
        except KeyError:
            raise KeyError(self.unknown_message(path)) from None

    def path_of(self, module: nn.Module) -> str | None:
        return self.by_id.get(id(module))

    def set_roles(self, roles: Mapping[str, Iterable[str]]) -> None:
        """Install role -> selectors (adapter roles merged with user declarations)."""
        self.roles = {name: list(selectors) for name, selectors in roles.items()}
        for info in self.modules.values():
            info.roles = ()
        for name in self.roles:
            try:
                paths = self.resolve_modules(f"role:{name}", allow_empty=True)
            except ValueError:
                continue
            for path in paths:
                info = self.modules[path]
                info.roles = (*info.roles, name)

    def unknown_message(self, name: str) -> str:
        names = [p or ROOT for p in self._objects]
        close = difflib.get_close_matches(name, names, n=5, cutoff=0.4)
        hint = f" Closest paths: {', '.join(close)}." if close else ""
        return f"no module matches {name!r}.{hint} List paths with insp.sites()."

    # -- selection ------------------------------------------------------------

    def resolve_modules(self, selector: Any, *, allow_empty: bool = False,
                        _seen: frozenset[str] = frozenset()) -> list[str]:
        """Resolve a module selector (str, ``Select``, callable or list) to canonical paths."""
        paths = list(self._objects)
        if isinstance(selector, Select):
            chosen = [p for p in paths if selector.predicate(p, self._objects[p], self)]
        elif callable(selector) and not isinstance(selector, str):
            chosen = [p for p in paths if selector(p or ROOT, self._objects[p])]
        elif isinstance(selector, (list, tuple, set, frozenset)):
            chosen_set: dict[str, None] = {}
            for item in selector:
                for p in self.resolve_modules(item, allow_empty=True, _seen=_seen):
                    chosen_set[p] = None
            chosen = [p for p in paths if p in chosen_set]
        elif isinstance(selector, str):
            terms = [parse_term(t) for t in split_terms(selector)]
            selected: dict[str, None] = {}
            if terms and terms[0].exclude:
                selected = dict.fromkeys(paths)
            for term in terms:
                if term.param is not None:
                    continue
                matched = self._match_term(term.module, _seen)
                if term.exclude:
                    for p in matched:
                        selected.pop(p, None)
                else:
                    for p in matched:
                        selected[p] = None
            chosen = [p for p in paths if p in selected]
        else:
            raise TypeError("selector must be a string, Select, callable(path, module) or a list")
        if not chosen and not allow_empty:
            raise ValueError(self.unknown_message(str(selector)))
        return chosen

    def _match_term(self, term: str, seen: frozenset[str]) -> list[str]:
        paths = list(self._objects)
        if term in ("*", "**"):
            return paths
        if term == ROOT:
            return [""]
        if term == "leaf":
            return [p for p in paths if self.modules[p].leaf]
        depth = _DEPTH.match(term)
        if depth:
            op, bound = depth.group(1), int(depth.group(2))
            compare = {"<=": int.__le__, ">=": int.__ge__, "<": int.__lt__, ">": int.__gt__,
                       "==": int.__eq__}[op]
            return [p for p in paths if compare(self.modules[p].depth, bound)]
        if term.startswith("re:"):
            pattern = re.compile(term[3:])
            return [p for p in paths if pattern.search(p or ROOT)]
        if term.startswith("type:"):
            matcher = _type_matcher(term[5:])
            return [p for p in paths if matcher(self._objects[p])]
        if term.startswith("component:"):
            wanted = term[len("component:"):]
            return [p for p in paths if self.modules[p].component_id == wanted]
        if term.startswith("role:"):
            role = term[5:]
            if role in seen:
                raise ValueError(f"role {role!r} is defined in terms of itself")
            out: dict[str, None] = {}
            for selector in self.roles.get(role, ()):
                for p in self.resolve_modules(selector, allow_empty=True, _seen=seen | {role}):
                    out[p] = None
            return [p for p in paths if p in out]
        pattern = glob_regex(term)
        matched = [p for p in paths if pattern.fullmatch(p)]
        if not matched:  # aliases (e.g. tied ``head`` vs ``core.lm_head``) map to canonical paths
            matched = [info.path for info in self.modules.values()
                       if any(pattern.fullmatch(a) for a in info.aliases)]
        return matched


def glob_regex(pattern: str) -> re.Pattern[str]:
    """``*`` = one path segment (or part of one), ``**`` = any depth, ``[...]`` = char class."""
    if pattern in ("", ROOT):
        return re.compile("")
    parts = pattern.split(".")
    regex = ""
    for i, part in enumerate(parts):
        if part == "**":
            if len(parts) == 1:
                regex += ".*"
            elif i == 0:
                regex += r"(?:.*\.)?"
            elif i == len(parts) - 1:
                regex += r"(?:\..*)?"
            else:
                regex += r"\.(?:.*\.)?"
            continue
        sep = "" if i == 0 or parts[i - 1] == "**" else r"\."
        regex += sep + _segment_regex(part)
    return re.compile(regex)


def _segment_regex(part: str) -> str:
    piece, i = [], 0
    while i < len(part):
        char = part[i]
        if char == "*":
            piece.append(r"[^.]*")
        elif char == "?":
            piece.append(r"[^.]")
        elif char == "[" and part.find("]", i) != -1:
            end = part.find("]", i)
            piece.append(part[i:end + 1])
            i = end
        else:
            piece.append(re.escape(char))
        i += 1
    return "".join(piece)


def _type_matcher(name: str) -> Callable[[nn.Module], bool]:
    target: type | None = None
    if "." in name:
        module_name, _, attr = name.rpartition(".")
        try:
            target = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
            target = None
    else:
        candidate = getattr(nn, name, None)
        target = candidate if isinstance(candidate, type) else None
    if isinstance(target, type):
        resolved = target
        return lambda module: isinstance(module, resolved)

    def by_name(module: nn.Module) -> bool:
        for cls in type(module).__mro__:
            if name in (cls.__name__, f"{cls.__module__}.{cls.__qualname__}"):
                return True
        return False
    return by_name


# --------------------------------------------------------------------------- Select


@dataclass(frozen=True)
class Select:
    """Composable programmatic selector: ``Select.glob("blocks.*") & Select.type(nn.Linear)``."""

    predicate: Callable[[str, nn.Module, SiteIndex], bool]
    label: str = field(default="select", compare=False)

    def __and__(self, other: Any) -> Select:
        other = Select.of(other)
        return Select(lambda p, m, i: self.predicate(p, m, i) and other.predicate(p, m, i),
                      f"({self.label} & {other.label})")

    def __or__(self, other: Any) -> Select:
        other = Select.of(other)
        return Select(lambda p, m, i: self.predicate(p, m, i) or other.predicate(p, m, i),
                      f"({self.label} | {other.label})")

    def __invert__(self) -> Select:
        return Select(lambda p, m, i: not self.predicate(p, m, i), f"~{self.label}")

    def __str__(self) -> str:
        return self.label

    @staticmethod
    def of(value: Any) -> Select:
        if isinstance(value, Select):
            return value
        if isinstance(value, str):
            return Select(lambda p, m, i, s=value: p in i.resolve_modules(s, allow_empty=True), value)
        if callable(value):
            return Select(lambda p, m, i: bool(value(p or ROOT, m)), getattr(value, "__name__", "fn"))
        raise TypeError("cannot build a Select from " + type(value).__name__)

    @staticmethod
    def glob(pattern: str) -> Select:
        regex = glob_regex(pattern)
        return Select(lambda p, m, i: bool(regex.fullmatch(p)), f"glob:{pattern}")

    @staticmethod
    def regex(pattern: str) -> Select:
        compiled = re.compile(pattern)
        return Select(lambda p, m, i: bool(compiled.search(p or ROOT)), f"re:{pattern}")

    @staticmethod
    def type(*types: type | str) -> Select:
        matchers = [(lambda m, t=t: isinstance(m, t)) if isinstance(t, type) else _type_matcher(t)
                    for t in types]
        return Select(lambda p, m, i: any(f(m) for f in matchers),
                      "type:" + ",".join(getattr(t, "__name__", str(t)) for t in types))

    @staticmethod
    def component(component_id: str) -> Select:
        return Select(lambda p, m, i: getattr(m, "component_id", None) == component_id,
                      f"component:{component_id}")

    @staticmethod
    def role(role: str) -> Select:
        return Select(lambda p, m, i: p in i.resolve_modules(f"role:{role}", allow_empty=True),
                      f"role:{role}")

    @staticmethod
    def leaf() -> Select:
        return Select(lambda p, m, i: not any(True for _ in m.children()), "leaf")

    @staticmethod
    def depth(maximum: int) -> Select:
        return Select(lambda p, m, i: (0 if not p else p.count(".") + 1) <= maximum,
                      f"depth<={maximum}")

    @staticmethod
    def where(fn: Callable[[str, nn.Module], bool]) -> Select:
        return Select(lambda p, m, i: bool(fn(p or ROOT, m)), getattr(fn, "__name__", "where"))


# --------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class SiteRequest:
    """A resolved capture request: one module path + port (+ op) + occurrence filter."""

    path: str
    port: str = "output"
    occurrence: int | str = "*"
    op: str | None = None
    op_index: int | None = None


def resolve_sites(selector: Any, index: SiteIndex, *, default_ports: tuple[str, ...] = ("output",),
                  allow_empty: bool = False) -> tuple[list[SiteRequest], list[str]]:
    """Resolve a full site selector into capture requests and parameter names."""
    if selector is None:
        return [], []
    if not isinstance(selector, str):
        paths = index.resolve_modules(selector, allow_empty=allow_empty)
        return [SiteRequest(p, port) for p in paths for port in default_ports], []
    requests: dict[SiteRequest, None] = {}
    params: list[str] = []
    terms = [parse_term(t) for t in split_terms(selector)]
    if terms and terms[0].exclude:
        for p in index.paths():
            for port in default_ports:
                requests[SiteRequest(p, port)] = None
    for term in terms:
        if term.param is not None:
            names = [n for n, _ in index.model.named_parameters()]
            regex = glob_regex(term.param)
            matched = [n for n in names if regex.fullmatch(n)]
            if not matched and not allow_empty:
                close = difflib.get_close_matches(term.param, names, n=5, cutoff=0.4)
                raise ValueError(f"no parameter matches {term.param!r}; closest: {close}")
            params.extend(n for n in matched if n not in params)
            continue
        paths = index._match_term(term.module, frozenset()) if term.module else [""]
        if term.exclude:
            for req in list(requests):
                if req.path in paths:
                    del requests[req]
            continue
        ports = (term.port,) if term.port else (("output",) if term.op else default_ports)
        occurrence = "*" if term.occurrence is None else term.occurrence
        for p in paths:
            for port in ports:
                requests[SiteRequest(p, port, occurrence, term.op, term.op_index)] = None
    if not requests and not params and not allow_empty:
        raise ValueError(index.unknown_message(selector))
    return list(requests), params


__all__ = ["ROOT", "ModuleInfo", "Select", "SiteIndex", "SiteKey", "SiteRequest", "SiteSpec",
           "clean_path", "glob_regex", "parse_term", "resolve_sites", "split_terms"]
