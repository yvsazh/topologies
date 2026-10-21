"""``synth.math``: exact arithmetic, random expressions, step-by-step solutions, word problems.

    synth.math.evaluate("3/4 + 2*(5 - 1)")                  # Fraction(35, 4), no eval()
    synth.math.expression(depth=3, ops="+-*/", low=1, high=20)
    # -> {"expr": "(7 - 3) * (12 / 4)", "answer": "12", "steps": ["7 - 3 = 4", ...]}
    synth.math.word_problem(kinds=["percent", "rate"], language="uk")
    # -> {"question": ..., "expr": ..., "answer": ..., "solution": ..., "facts": {...}}

Answers are computed exactly (rationals), so they are correct by construction;
templates per language only write the surface form. ``same_answer`` /
``parse_answer`` check model outputs (``ops.verify.math_answer``).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from topologies import synth
from topologies.synth.lib.math.expr import (
    FUNCS,
    MathError,
    evaluate,
    fmt,
    parse_answer,
    parse_number,
    same_answer,
)

_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
        "/": lambda a, b: a / b}


def _tree(r: Any, depth: int, ops: str, low: int, high: int, allow_neg: bool) -> Any:
    if depth <= 0 or r.random() < 0.25:
        return Fraction(r.randint(low, high))
    op = r.choice(ops)
    left = _tree(r, depth - 1, ops, low, high, allow_neg)
    right = _tree(r, depth - 1, ops, low, high, allow_neg)
    if op == "/" and _value(right) == 0:
        right = Fraction(r.randint(max(1, low), max(1, high)))
    if op == "-" and not allow_neg and _value(left) < _value(right):
        left, right = right, left
    return (op, left, right)


def _value(t: Any) -> Fraction:
    if isinstance(t, Fraction):
        return t
    op, a, b = t
    return _OPS[op](_value(a), _value(b))


def _text(t: Any, top: bool = True) -> str:
    if isinstance(t, Fraction):
        return fmt(t)
    op, a, b = t
    s = f"{_text(a, False)} {op} {_text(b, False)}"
    return s if top else f"({s})"


def _steps(t: Any, out: list[str]) -> Fraction:
    if isinstance(t, Fraction):
        return t
    op, a, b = t
    va, vb = _steps(a, out), _steps(b, out)
    v = _OPS[op](va, vb)
    out.append(f"{fmt(va)} {op} {fmt(vb)} = {fmt(v)}")
    return v


@synth.node(tags=["math", "reasoning"], example={"depth": 2})
def expression(r, *, depth: int = 2, ops: str = "+-*", low: int = 1, high: int = 20,
               allow_negative: bool = False, integer: bool = False, max_tries: int = 50):
    """A random arithmetic expression tree with its exact answer and solution steps
    (``integer=True`` retries until the answer is an integer)."""
    for _ in range(max_tries):
        t = _tree(r, int(depth), ops, int(low), int(high), bool(allow_negative))
        if isinstance(t, Fraction):
            t = ("+", t, Fraction(r.randint(low, high)))
        try:
            steps: list[str] = []
            v = _steps(t, steps)
        except ZeroDivisionError:
            continue
        if integer and v.denominator != 1:
            continue
        return {"expr": _text(t), "answer": fmt(v), "steps": steps}
    raise synth.SynthError("expression(): no integer answer found.", stage="math",
                           hint="Drop '/' from ops or set integer=False.")


KINDS = ("arithmetic", "percent", "rate", "linear", "sharing")

_TEMPLATES = {
    "en": {
        "arithmetic": "{who} has {a} boxes and buys {b} more; each holds {c} items. "
                      "How many items are there in total?",
        "percent": "An item costs {price}; {who} gets {pct}% off. How much does {who} pay?",
        "rate": "{who} drives {hours} h at {speed} km/h, then {extra} km more. "
                "How far does {who} drive?",
        "linear": "{who} thinks of a number, multiplies it by {a} and adds {b}, getting {c}. "
                  "What is the number?",
        "sharing": "{who} shares {total} items equally among {k} friends. How many does each "
                   "friend get?",
        "answer": "Answer",
    },
    "uk": {
        "arithmetic": "У {who} є {a} коробок, і {who} купує ще {b}; у кожній по {c} речей. "
                      "Скільки всього речей?",
        "percent": "Товар коштує {price}; {who} отримує знижку {pct}%. Скільки заплатить {who}?",
        "rate": "{who} їде {hours} год зі швидкістю {speed} км/год, потім ще {extra} км. "
                "Яку відстань подолає {who}?",
        "linear": "{who} задумує число, множить його на {a} і додає {b}, отримує {c}. "
                  "Яке число задумано?",
        "sharing": "{who} ділить {total} речей порівну між {k} друзями. Скільки отримає "
                   "кожен?",
        "answer": "Відповідь",
    },
}


@synth.node(tags=["math", "reasoning", "text"], example={"kinds": ["percent"], "language": "en"})
def word_problem(r, *, kinds: Any = KINDS, difficulty: str = "medium", language: str = "en"):
    """A word problem with an exact answer: question text (``en`` / ``uk`` templates),
    expression, answer, step-by-step solution and the facts that produced it."""
    from topologies.synth.lib.text.faker import Faker

    if language not in _TEMPLATES:
        raise synth.unknown("math language", language, _TEMPLATES, stage="math")
    tpl = _TEMPLATES[language]
    scale = {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2)
    kind = r.choice(list(kinds)) if not isinstance(kinds, str) else kinds
    who = Faker(language, r).first_name(gender="male" if language == "uk" else None)
    if language == "uk":
        from topologies.synth.lib.text.packs import pack

        # The templates use "У {who}" and therefore require the genitive case.
        who = pack("uk").paradigms.get(who, {}).get("gen", who)
    if kind == "arithmetic":
        a, b, c = r.randint(2, 20 * scale), r.randint(2, 20 * scale), r.randint(2, 9 * scale)
        expr, facts = f"({a} + {b}) * {c}", {"a": a, "b": b, "c": c}
        steps = [f"{a} + {b} = {a + b}", f"{a + b} * {c} = {(a + b) * c}"]
    elif kind == "percent":
        price, pct = r.randint(10, 200 * scale), r.choice([5, 10, 15, 20, 25, 30, 40, 50])
        expr, facts = f"{price} * (100 - {pct}) / 100", {"price": price, "pct": pct}
        steps = [f"{price} * {pct} / 100 = {fmt(evaluate(f'{price}*{pct}/100'))}",
                 f"{price} - {fmt(evaluate(f'{price}*{pct}/100'))} = {fmt(evaluate(expr))}"]
    elif kind == "rate":
        speed, hours, extra = r.randint(20, 90), r.randint(1, 3 * scale), r.randint(5, 30)
        expr = f"{speed} * {hours} + {extra}"
        facts = {"speed": speed, "hours": hours, "extra": extra}
        steps = [f"{speed} * {hours} = {speed * hours}",
                 f"{speed * hours} + {extra} = {speed * hours + extra}"]
    elif kind == "linear":
        a, x, b = r.randint(2, 9), r.randint(1, 12 * scale), r.randint(1, 30)
        c = a * x + b
        expr, facts = f"({c} - {b}) / {a}", {"a": a, "b": b, "c": c}
        steps = [f"{a}x + {b} = {c}", f"{a}x = {c - b}", f"x = {c - b} / {a} = {x}"]
    elif kind == "sharing":
        k = r.randint(2, 6)
        total = k * r.randint(3, 15 * scale)
        expr, facts = f"{total} / {k}", {"total": total, "k": k}
        steps = [f"{total} / {k} = {total // k}"]
    else:
        raise synth.unknown("word problem kind", kind, KINDS, stage="math")
    answer = fmt(evaluate(expr))
    question = tpl[kind].format(who=who, **facts)
    return {"kind": kind, "question": question, "expr": expr, "answer": answer,
            "solution": "\n".join(steps) + f"\n{tpl['answer']}: {answer}",
            "facts": {"person": who, **facts}}


__all__ = ["FUNCS", "KINDS", "MathError", "evaluate", "expression", "fmt", "parse_answer",
           "parse_number", "same_answer", "word_problem"]
