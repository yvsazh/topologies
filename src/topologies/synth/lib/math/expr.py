"""Safe arithmetic with exact rationals (``synth.math``): an ``ast`` walker (no ``eval``) over ``fractions``.

    evaluate("3/4 + 2*(5 - 1)")          # Fraction(35, 4)
    evaluate("x**2 - 1", {"x": 3})        # Fraction(8, 1)
    parse_answer("The answer is $1,200.50")   # Fraction(240100, 200)
    same_answer("0.75", "3/4")            # True

Allowed: numbers, ``+ - * / // % **`` (integer exponents, |exp| <= 64), unary
signs, parentheses, names bound in ``env``, and ``abs``, ``min``, ``max``,
``round``, ``sqrt`` (exact for perfect squares, float otherwise), ``gcd``,
``lcm``, ``floor``, ``ceil``. Anything else raises ``MathError``.
"""

from __future__ import annotations

import ast
import math
import re
from fractions import Fraction
from typing import Any

Number = Fraction | float


class MathError(ValueError):
    """Unsupported or invalid expression."""


def _sqrt(x: Number) -> Number:
    if isinstance(x, Fraction) and x >= 0:
        n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
        if n * n == x.numerator and d * d == x.denominator:
            return Fraction(n, d)
    if x < 0:
        raise MathError("sqrt of a negative number")
    return math.sqrt(float(x))


def _int_arg(x: Number) -> int:
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    raise MathError("integer argument expected")


FUNCS = {
    "abs": lambda x: abs(x),
    "min": lambda *a: min(a),
    "max": lambda *a: max(a),
    "round": lambda x, n=Fraction(0): Fraction(round(Fraction(x), _int_arg(n))),
    "sqrt": _sqrt,
    "gcd": lambda a, b: Fraction(math.gcd(_int_arg(a), _int_arg(b))),
    "lcm": lambda a, b: Fraction(abs(_int_arg(a) * _int_arg(b)) //
                                 math.gcd(_int_arg(a), _int_arg(b))),
    "floor": lambda x: Fraction(math.floor(x)),
    "ceil": lambda x: Fraction(math.ceil(x)),
}


def _num(v: Any) -> Number:
    if isinstance(v, bool):
        raise MathError("booleans are not numbers")
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(v).limit_denominator(10**9) if math.isfinite(v) else v
    if isinstance(v, (Fraction, )):
        return v
    if isinstance(v, str):
        return parse_number(v)
    raise MathError(f"not a number: {v!r}")


def _eval(node: ast.AST, env: dict[str, Any]) -> Number:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return _num(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return _num(env[node.id])
        if node.id in ("pi", "e"):
            return math.pi if node.id == "pi" else math.e
        raise MathError(f"unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand, env)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        op = node.op
        try:
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return a / b
            if isinstance(op, ast.FloorDiv):
                return Fraction(a // b)
            if isinstance(op, ast.Mod):
                return a % b
            if isinstance(op, ast.Pow):
                e = _int_arg(b)
                if abs(e) > 64:
                    raise MathError("exponent too large")
                return a ** e
        except ZeroDivisionError as err:
            raise MathError("division by zero") from err
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
            node.func.id in FUNCS and not node.keywords:
        return FUNCS[node.func.id](*[_eval(a, env) for a in node.args])
    raise MathError(f"unsupported syntax: {type(node).__name__}")


def evaluate(expr: str, env: dict[str, Any] | None = None) -> Number:
    """Exact value of ``expr`` (see module docstring)."""
    text = str(expr).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    if len(text) > 2000:
        raise MathError("expression too long")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise MathError(f"cannot parse {expr!r}") from e
    return _eval(tree, dict(env or {}))


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?|-?\.\d+")


def parse_number(text: str) -> Fraction:
    """A number written as ``1,200.50``, ``3/4``, ``-0.5``, ``75%`` (= 3/4) or ``$12``."""
    t = str(text).strip().replace("$", "").replace("€", "").replace("£", "").strip()
    pct = t.endswith("%")
    t = t.rstrip("%").strip()
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            v = Fraction(a.replace(",", "").strip()) / Fraction(b.replace(",", "").strip())
        else:
            v = Fraction(t.replace(",", ""))
    except (ValueError, ZeroDivisionError) as e:
        raise MathError(f"not a number: {text!r}") from e
    return v / 100 if pct else v


def parse_answer(text: Any) -> Number:
    """The final number in a free-text answer (``"#### 42"``, ``"x = 5"``, ``"\\boxed{3/4}"``)."""
    if isinstance(text, (int, float, Fraction)) and not isinstance(text, bool):
        return _num(text)
    s = str(text).strip()
    if s.endswith("%"):
        try:
            return parse_number(s)
        except MathError:
            pass
    for pat in (r"\\boxed\{([^}]*)\}", r"####\s*(.+)$", r"answer\s*(?:is|:|=)\s*(.+)$"):
        m = re.search(pat, s, re.IGNORECASE | re.MULTILINE)
        if m:
            s = m.group(1)
            break
    try:
        return evaluate(s.strip().rstrip(".").replace(",", "").replace("$", ""))
    except MathError:
        pass
    found = _NUM.findall(s)
    if not found:
        raise MathError(f"no number in {text!r}")
    return parse_number(found[-1])


def same_answer(a: Any, b: Any, *, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    try:
        x, y = parse_answer(a), parse_answer(b)
    except MathError:
        return str(a).strip().casefold() == str(b).strip().casefold()
    if isinstance(x, Fraction) and isinstance(y, Fraction):
        return x == y or math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=abs_tol)
    return math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=abs_tol)


def fmt(v: Number) -> str:
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    return f"{v:g}"


__all__ = ["FUNCS", "MathError", "evaluate", "fmt", "parse_answer", "parse_number",
           "same_answer"]
