"""Verifiers: executable checks of generated records (evidence kind ``verifier``).

    synth.ops.verify(my_fn)                                  # fn(record, ctx) -> bool | Verdict
    synth.ops.verify.python_exec("code", tests_field="tests", timeout=5)
    synth.ops.verify.math_answer("problem_expr", "answer")
    synth.ops.verify.json_schema(Invoice, field="output")
    synth.ops.verify.regex(r"^\\d{5}$", field="zip")
    synth.ops.verify.round_trip(parse_invoice, source="output", text="input",
                                tolerance="fields>=0.95")   # a parser recovers the latent
    synth.ops.verify.span_alignment()
    synth.ops.verify.label_rule(lambda r: ("refund" in r["text"]) == (r["label"] == "refund"))

Results are cached by ``(verifier fingerprint, record content hash)`` in
``<root>/.cache/verify.sqlite``, so re-runs and ablations never re-execute code
or re-run parsers for the same record.

``python_exec`` runs code in a subprocess with a temp working directory, an
empty environment (no proxies, no keys), ``-I`` isolated mode and POSIX
resource limits (CPU seconds, address space). **It is not a security
boundary**: run untrusted generated code in a container or VM.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from topologies.experiments.cache import stable_hash
from topologies.synth.errors import SynthError
from topologies.synth.fingerprint import callable_fingerprint, fp_canonical
from topologies.synth.ops.base import FilterOp
from topologies.synth.records import strip_meta
from topologies.synth.verdict import Verdict, as_verdict

# ---------------------------------------------------------------- verdict cache


class VerdictCache:
    """sqlite map ``key -> verdict`` (shared, WAL; failures never break a run)."""

    _instances: dict[str, VerdictCache] = {}

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def for_root(cls, root: Any = None) -> VerdictCache:
        from topologies.synth.store import default_root

        p = default_root(root) / ".cache" / "verify.sqlite"
        key = str(p.resolve()) if p.parent.exists() else str(p)
        if key not in cls._instances:
            cls._instances[key] = cls(p)
        return cls._instances[key]

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False,
                                isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, v TEXT)")
            self._conn = c
        return self._conn

    def get(self, key: str) -> Verdict | None:
        try:
            with self._lock:
                row = self._db().execute("SELECT v FROM verdicts WHERE key = ?",
                                         (key,)).fetchone()
        except sqlite3.Error:
            return None
        return Verdict.from_dict(json.loads(row[0])) if row else None

    def put(self, key: str, v: Verdict) -> None:
        try:
            with self._lock:
                self._db().execute("INSERT OR REPLACE INTO verdicts VALUES (?, ?)",
                                   (key, json.dumps(v.as_dict(), default=str)))
        except sqlite3.Error:
            pass


class Verifier(FilterOp):
    """Base verifier: implement ``run(record, ctx) -> Verdict``."""

    evidence = "verifier"
    name = "verify"
    cache = True

    def setup(self, runtime: Any) -> None:
        root = getattr(getattr(runtime, "spec", None), "root", None)
        self._cache = VerdictCache.for_root(root) if self.cache else None

    def _key(self, record: Mapping[str, Any]) -> str:
        return stable_hash({"v": self.fingerprint(), "r": strip_meta(record)}, n=32)

    def run(self, record: dict, ctx: Any) -> Any:
        raise NotImplementedError

    def check(self, record: dict, ctx: Any) -> Verdict:
        cache = getattr(self, "_cache", None)
        key = self._key(record) if cache is not None else None
        if key is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        v = as_verdict(self.run(record, ctx))
        if key is not None:
            cache.put(key, v)
        return v

    def check_sync(self, record: dict, ctx: Any) -> Verdict:
        return as_verdict(self.run(record, ctx))


class FnVerifier(Verifier):
    def __init__(self, fn: Callable[..., Any], *, name: str | None = None) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "verify")
        if self.name == "<lambda>":
            self.name = "verify"
        import inspect

        try:
            self._two = len([p for p in inspect.signature(fn).parameters.values()
                             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 2
        except (TypeError, ValueError):
            self._two = True

    def run(self, record: dict, ctx: Any) -> Any:
        return self.fn(record, ctx) if self._two else self.fn(record)

    def config(self) -> dict[str, Any]:
        return {"op": "verify", "fn": callable_fingerprint(self.fn), "name": self.name}


def verify(fn: Callable[..., Any], *, name: str | None = None) -> FnVerifier:
    """``fn(record[, ctx]) -> bool | Verdict`` as a verifier (evidence ``verifier``)."""
    return FnVerifier(fn, name=name)


custom = verify


def label_rule(fn: Callable[..., Any], *, name: str = "verify.label_rule") -> FnVerifier:
    """A programmatic rule the label must satisfy (``fn(record) -> bool``)."""
    return FnVerifier(fn, name=name)


# ---------------------------------------------------------------- python_exec


_RUNNER = r"""
import json, sys, traceback
ns = {"__name__": "__synth__"}
src = open(sys.argv[1], encoding="utf-8").read()
try:
    exec(compile(src, "<generated>", "exec"), ns)
except BaseException as e:
    print(json.dumps({"ok": False, "reason": "".join(traceback.format_exception_only(type(e), e)).strip()[-500:]}))
    sys.exit(1)
print(json.dumps({"ok": True}))
"""


def _limits(cpu_s: float, mem_mb: int) -> Callable[[], None] | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None

    def apply() -> None:
        sec = max(1, int(cpu_s) + 1)
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CPU, (sec, sec))
        if mem_mb and sys.platform != "darwin":  # RLIMIT_AS is unreliable on macOS
            b = int(mem_mb) * 1024 * 1024
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
        os.setsid()

    return apply


def run_python(code: str, *, timeout: float = 5.0, memory_mb: int = 512) -> Verdict:
    """Execute ``code`` (code + asserts) in a restricted subprocess; ok = exit status 0.

    **This is not a sandbox.** The child runs with a minimal environment, a
    throwaway working directory and CPU (plus, except on macOS, address-space)
    rlimits, but it can still read and write any file the user can and open
    network connections. Use it only for code from generators you trust.
    """
    with tempfile.TemporaryDirectory(prefix="synth-exec-") as d:
        src = Path(d) / "main.py"
        src.write_text(code, encoding="utf-8")
        runner = Path(d) / "runner.py"
        runner.write_text(_RUNNER, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "HOME": d,
               "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", str(runner), str(src)], cwd=d, env=env,
                capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limits(timeout, memory_mb),
                check=False)
        except subprocess.TimeoutExpired:
            return Verdict(False, reason=f"timeout after {timeout:g}s")
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        try:
            res = json.loads(lines[-1]) if lines else {}
        except ValueError:
            res = {}
        if proc.returncode == 0 and res.get("ok"):
            return Verdict(True)
        reason = res.get("reason") or (proc.stderr.strip().splitlines() or
                                       [f"exit status {proc.returncode}"])[-1]
        return Verdict(False, reason=reason[:500])


class PythonExec(Verifier):
    """Code + tests must run (``code_field`` + ``tests_field``) without errors.

    Runs the code through ``run_python``, which is **not a sandbox** (no
    filesystem or network isolation); only for trusted generator output.
    """

    name = "verify.python_exec"

    def __init__(self, code_field: str = "code", *, tests_field: str | None = "tests",
                 timeout: float = 5.0, memory_mb: int = 512, prelude: str = "") -> None:
        self.code_field, self.tests_field = code_field, tests_field
        self.timeout, self.memory_mb, self.prelude = float(timeout), int(memory_mb), prelude

    def run(self, record: dict, ctx: Any) -> Verdict:
        code = str(record.get(self.code_field) or "")
        if not code.strip():
            return Verdict(False, reason=f"no {self.code_field!r}")
        tests = record.get(self.tests_field) if self.tests_field else ""
        if isinstance(tests, list):
            tests = "\n".join(map(str, tests))
        program = "\n\n".join(x for x in (self.prelude, _strip_fence(code),
                                          _strip_fence(str(tests or ""))) if x)
        return run_python(program, timeout=self.timeout, memory_mb=self.memory_mb)

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "code": self.code_field, "tests": self.tests_field,
                "timeout": self.timeout, "memory_mb": self.memory_mb, "prelude": self.prelude}


def _strip_fence(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def python_exec(code_field: str = "code", **kw: Any) -> PythonExec:
    return PythonExec(code_field, **kw)


# ---------------------------------------------------------------- math / schema / regex


class MathAnswer(Verifier):
    """``answer_field`` must equal the exact value of ``expr_field`` (or ``solve(record)``)."""

    name = "verify.math_answer"

    def __init__(self, expr_field: str | None = "expr", answer_field: str = "answer", *,
                 solve: Callable[[dict], Any] | None = None, env_field: str | None = None,
                 rel_tol: float = 1e-6) -> None:
        if expr_field is None and solve is None:
            raise SynthError("math_answer needs expr_field or solve=fn(record).", stage="verify")
        self.expr_field, self.answer_field, self.solve = expr_field, answer_field, solve
        self.env_field, self.rel_tol = env_field, float(rel_tol)

    def run(self, record: dict, ctx: Any) -> Verdict:
        from topologies.synth.lib.math.expr import MathError, evaluate, fmt, same_answer

        try:
            if self.solve is not None:
                truth = self.solve(record)
            else:
                env = record.get(self.env_field) if self.env_field else None
                truth = evaluate(str(record[self.expr_field]), env if isinstance(env, dict)
                                 else None)
        except (MathError, KeyError, TypeError) as e:
            return Verdict(False, reason=f"cannot re-derive the answer: {e}")
        given = record.get(self.answer_field)
        expected = fmt(truth) if hasattr(truth, "numerator") else str(truth)
        ok = same_answer(given, expected, rel_tol=self.rel_tol)
        return Verdict(ok, reason="" if ok else f"answer {given!r} != {expected}",
                       evidence={"expected": expected})

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "expr": self.expr_field, "answer": self.answer_field,
                "solve": callable_fingerprint(self.solve), "env": self.env_field,
                "rel_tol": self.rel_tol}


def math_answer(expr_field: str | None = "expr", answer_field: str = "answer",
                **kw: Any) -> MathAnswer:
    return MathAnswer(expr_field, answer_field, **kw)


class JsonSchemaCheck(Verifier):
    name = "verify.json_schema"

    def __init__(self, model: Any, *, field: str | None = None) -> None:
        from topologies.synth.schema import Json

        self.model, self.field = model, field
        self._json = Json(model) if not isinstance(model, Json) else model

    def run(self, record: dict, ctx: Any) -> Verdict:
        value = record.get(self.field) if self.field else strip_meta(record)
        value = self._json.coerce(value)
        errs = self._json.validate(value)
        return Verdict(not errs, reason="; ".join(errs[:3]))

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "schema": self._json.to_config(), "field": self.field}


def json_schema(model: Any, *, field: str | None = None) -> JsonSchemaCheck:
    return JsonSchemaCheck(model, field=field)


class RegexCheck(Verifier):
    name = "verify.regex"
    cache = False

    def __init__(self, pattern: str, *, field: str = "text", must_match: bool = True,
                 flags: int = 0) -> None:
        self.pattern, self.field, self.must_match, self.flags = pattern, field, must_match, flags
        self._re = re.compile(pattern, flags)

    def run(self, record: dict, ctx: Any) -> Verdict:
        hit = self._re.search(str(record.get(self.field, ""))) is not None
        ok = hit == self.must_match
        return Verdict(ok, reason="" if ok else (f"{self.field} does not match {self.pattern!r}"
                                                 if self.must_match else
                                                 f"{self.field} matches {self.pattern!r}"))

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "pattern": self.pattern, "field": self.field,
                "must_match": self.must_match, "flags": self.flags}


def regex(pattern: str, **kw: Any) -> RegexCheck:
    return RegexCheck(pattern, **kw)


class SpanAlignment(Verifier):
    name = "verify.span_alignment"
    cache = False

    def __init__(self, *, text_field: str = "text", spans_field: str = "spans",
                 labels: Sequence[str] | None = None) -> None:
        from topologies.synth.schema import Spans

        self.text_field, self.spans_field = text_field, spans_field
        self.labels = list(labels) if labels else None
        self._f = Spans(text_field=text_field, labels=self.labels, must_align=True)

    def run(self, record: dict, ctx: Any) -> Verdict:
        spans = record.get(self.spans_field)
        if spans is None:
            return Verdict(False, reason=f"no {self.spans_field!r}")
        errs = self._f.validate(spans, record)
        if errs:
            return Verdict(False, reason=errs[0])
        text = str(record.get(self.text_field, ""))
        for s in spans:  # without explicit span text, the span must be non-blank text
            if not text[s["start"]:s["end"]].strip():
                return Verdict(False, reason=f"blank span {s}")
        return Verdict(True)

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "text": self.text_field, "spans": self.spans_field,
                "labels": self.labels}


def span_alignment(**kw: Any) -> SpanAlignment:
    return SpanAlignment(**kw)


# ---------------------------------------------------------------- round trip


_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y",
                 "%d %b %Y", "%b %d, %Y", "%Y/%m/%d")


def _norm_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    for fmt in _DATE_FORMATS:
        try:
            return _dt.datetime.strptime(s, fmt).date().isoformat()  # noqa: DTZ007
        except ValueError:
            continue
    num = re.fullmatch(r"[-+]?[$€£]?\s*[\d,]*\.?\d+\s*(?:%|eur|usd|gbp|uah)?", s.lower())
    if num:
        try:
            from topologies.synth.lib.math.expr import parse_number

            return float(parse_number(re.sub(r"(?i)\s*(eur|usd|gbp|uah)$", "", s)))
        except ValueError:
            pass
    return re.sub(r"\s+", " ", s.casefold()).strip(" .")


def _flatten(v: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(v, Mapping):
        out: dict[str, Any] = {}
        for k, x in v.items():
            out.update(_flatten(x, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(v, list):
        out = {}
        for i, x in enumerate(v):
            out.update(_flatten(x, f"{prefix}[{i}]"))
        return out or {prefix: []}
    return {prefix: v}


def compare_fields(expected: Any, got: Any, *, rel_tol: float = 0.01) -> tuple[float, list[str]]:
    """Share of leaf fields of ``expected`` matched in ``got`` (normalized) + mismatches."""
    exp, act = _flatten(expected), _flatten(got if got is not None else {})
    if not exp:
        return 1.0, []
    bad = []
    for k, ev in exp.items():
        a, b = _norm_scalar(ev), _norm_scalar(act.get(k))
        if isinstance(a, float) and isinstance(b, float):
            ok = abs(a - b) <= rel_tol * max(1.0, abs(a))
        else:
            ok = a == b
        if not ok:
            bad.append(k)
    return 1 - len(bad) / len(exp), bad


def parse_tolerance(tol: str | float) -> float:
    if isinstance(tol, (int, float)):
        return float(tol)
    m = re.fullmatch(r"\s*fields\s*>=\s*([0-9.]+)\s*", str(tol))
    if not m:
        raise SynthError(f"Invalid tolerance {tol!r}.", stage="verify",
                         hint="Use tolerance='fields>=0.95' or a number in [0, 1].")
    return float(m.group(1))


class RoundTrip(Verifier):
    """A parser must recover the latent from the rendered value, field by field.

    ``parser(text) -> structured value``; the result is compared with the record's
    ``source`` field (leaf by leaf, numbers / dates / casing normalized).
    """

    name = "verify.round_trip"

    def __init__(self, parser: Callable[[Any], Any], *, source: str = "output",
                 text: str = "input", tolerance: str | float = "fields>=0.95",
                 rel_tol: float = 0.01, compare: Callable[[Any, Any], float] | None = None
                 ) -> None:
        if not callable(parser):
            raise SynthError("round_trip needs a parser: fn(rendered) -> structured value.",
                             stage="verify", hint="synth.ops.verify.round_trip(json.loads, "
                                                  "source='output', text='input')")
        self.parser, self.source, self.text = parser, source, text
        self.tolerance = tolerance
        self._min = parse_tolerance(tolerance)
        self.rel_tol, self.compare = float(rel_tol), compare

    def run(self, record: dict, ctx: Any) -> Verdict:
        try:
            got = self.parser(record.get(self.text, ""))
        except (ValueError, KeyError, TypeError, SynthError) as e:
            return Verdict(False, reason=f"parsing failed: {e}")
        expected = record.get(self.source)
        if self.compare is not None:
            score, bad = float(self.compare(expected, got)), []
        else:
            score, bad = compare_fields(expected, got, rel_tol=self.rel_tol)
        ok = score >= self._min
        return Verdict(ok, score=round(score, 4),
                       reason="" if ok else f"{len(bad)} field(s) differ: {', '.join(bad[:5])}",
                       evidence={"mismatched": bad[:20]})

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "parser": callable_fingerprint(self.parser),
                "source": self.source, "text": self.text, "tolerance": self.tolerance,
                "rel_tol": self.rel_tol, "compare": callable_fingerprint(self.compare)}


def round_trip(parser: Callable[[Any], Any], **kw: Any) -> RoundTrip:
    """``parser(rendered) -> value`` must recover ``record[source]`` (see ``RoundTrip``)."""
    return RoundTrip(parser, **kw)


_ = fp_canonical

__all__ = ["FnVerifier", "JsonSchemaCheck", "MathAnswer", "PythonExec", "RegexCheck",
           "RoundTrip", "SpanAlignment", "VerdictCache", "Verifier", "compare_fields",
           "custom", "json_schema", "label_rule", "math_answer", "python_exec", "regex",
           "round_trip", "run_python", "span_alignment", "verify"]


from topologies.synth.ops._callable import make_callable

make_callable(__name__, "verify")
