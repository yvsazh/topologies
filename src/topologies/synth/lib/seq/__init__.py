"""``synth.seq``: algorithmic probe tasks with exact labels (architecture research).

Every task emits ``{"tokens": [...], "labels": [...]}`` where ``labels`` is ``-100``
everywhere except the answer positions, aligned so that ``labels[t]`` is the token
to predict *at* position ``t`` - exactly what the ``causal-lm`` recipe expects
(it shifts internally, so ``labels`` here equals the next tokens at answer slots).
Accuracy over answer positions is the probe score (``val_acc`` /
``val_acc_answer``).

    synth.seq.mqar(n_pairs=8, n_queries=4, vocab=64)          # a generator node
    synth.seq.Probe("parity", length=16)                      # the same, by task name
    synth.seq.sample("parity", rng, length=16)                # one example
    synth.seq.vocab_size("arithmetic", digits=3)
    synth.seq.batch("mqar", n=100_000, seed=0, n_pairs=8)     # vectorized torch

Tasks: associative_recall, mqar, induction_heads, selective_copy, copy, reverse,
sort, arithmetic, parity, majority, dyck, modular_arithmetic, graph_reachability,
in_context_regression, needle_in_haystack. Knobs set the difficulty (length,
number of keys, vocabulary, depth); ``length`` is recorded so
``synth.ops.split(by_length={"field": "tokens", "train_max": L})`` builds
length-generalization splits.
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Callable
from typing import Any

import torch

from topologies import synth

SynthError, unknown = synth.SynthError, synth.unknown

IGNORE = -100
# Shared special ids at the top of every task vocabulary.
PAD, SEP, QUERY, YES, NO = 0, 1, 2, 3, 4
N_SPECIAL = 5

Example = dict[str, list[int]]


def _ex(tokens: list[int], answer_at: dict[int, int]) -> Example:
    """``answer_at[t] = token``: the token to predict *after* position ``t``.

    ``causal-lm`` shifts labels by one (logits[t] predict labels[t+1]), so the
    answer for the slot after position ``t`` lives at ``labels[t + 1]``.
    """
    labels = [IGNORE] * len(tokens)
    for t, tok in answer_at.items():
        labels[t + 1] = tok
    return {"tokens": tokens, "labels": labels}


def _seq_with_answer(context: list[int], answer: list[int]) -> Example:
    """``context + answer``; every answer token is a target."""
    tokens = context + answer
    return _ex(tokens, {len(context) - 1 + i: a for i, a in enumerate(answer)})


# ---------------------------------------------------------------- tasks


def associative_recall(rng: random.Random, *, n_pairs: int = 8, vocab: int = 64) -> Example:
    keys_n = (vocab - N_SPECIAL) // 2
    if n_pairs > keys_n:
        raise SynthError(f"associative_recall: n_pairs={n_pairs} needs vocab >= "
                         f"{2 * n_pairs + N_SPECIAL}.", stage="algorithmic")
    keys = rng.sample(range(N_SPECIAL, N_SPECIAL + keys_n), n_pairs)
    vals = [rng.randrange(N_SPECIAL + keys_n, vocab) for _ in keys]
    q = rng.randrange(n_pairs)
    context = [t for kv in zip(keys, vals) for t in kv] + [QUERY, keys[q]]
    return _seq_with_answer(context, [vals[q]])


def mqar(rng: random.Random, *, n_pairs: int = 8, n_queries: int | None = None,
         vocab: int = 64, seq_len: int | None = None) -> Example:
    """Multi-query associative recall (Zoology): pairs, then queries scattered later."""
    keys_n = (vocab - N_SPECIAL) // 2
    nq = n_queries or n_pairs
    if n_pairs > keys_n:
        raise SynthError(f"mqar: n_pairs={n_pairs} needs vocab >= {2 * n_pairs + N_SPECIAL}.",
                         stage="algorithmic")
    keys = rng.sample(range(N_SPECIAL, N_SPECIAL + keys_n), n_pairs)
    vals = [rng.randrange(N_SPECIAL + keys_n, vocab) for _ in keys]
    tokens = [t for kv in zip(keys, vals) for t in kv]
    total = seq_len or (2 * n_pairs + 2 * nq + 2)
    gap = max(0, total - len(tokens) - 2 * nq)
    tokens += [PAD] * gap
    answers: dict[int, int] = {}
    for _ in range(nq):
        q = rng.randrange(n_pairs)
        tokens.append(keys[q])
        answers[len(tokens) - 1] = vals[q]
        tokens.append(vals[q])
    return _ex(tokens, answers)


def induction_heads(rng: random.Random, *, seq_len: int = 64, vocab: int = 32) -> Example:
    """A random sequence where token ``A`` appears twice; after the second ``A``, predict the
    token that followed the first."""
    toks = [rng.randrange(N_SPECIAL, vocab) for _ in range(seq_len - 1)]
    a = rng.randrange(N_SPECIAL, vocab)
    i = rng.randrange(1, seq_len // 2)
    span = vocab - N_SPECIAL
    toks = [t if t != a else N_SPECIAL + (t - N_SPECIAL + 1) % span for t in toks]
    toks[i] = a
    b = toks[i + 1]
    toks[-1] = a
    return _ex(toks + [b], {len(toks) - 1: b})


def selective_copy(rng: random.Random, *, seq_len: int = 32, n_tokens: int = 4,
                   vocab: int = 16) -> Example:
    noise = PAD
    data = [rng.randrange(N_SPECIAL, vocab) for _ in range(n_tokens)]
    pos = sorted(rng.sample(range(seq_len), n_tokens))
    seq = [noise] * seq_len
    for p, d in zip(pos, data):
        seq[p] = d
    return _seq_with_answer(seq + [SEP], data)


def copy(rng: random.Random, *, length: int = 16, vocab: int = 16) -> Example:
    x = [rng.randrange(N_SPECIAL, vocab) for _ in range(length)]
    return _seq_with_answer(x + [SEP], x)


def reverse(rng: random.Random, *, length: int = 16, vocab: int = 16) -> Example:
    x = [rng.randrange(N_SPECIAL, vocab) for _ in range(length)]
    return _seq_with_answer(x + [SEP], x[::-1])


def sort(rng: random.Random, *, length: int = 16, vocab: int = 32) -> Example:
    x = [rng.randrange(N_SPECIAL, vocab) for _ in range(length)]
    return _seq_with_answer(x + [SEP], sorted(x))


_DIGIT0 = N_SPECIAL  # digits 0-9
_OPS = {"+": N_SPECIAL + 10, "-": N_SPECIAL + 11, "*": N_SPECIAL + 12}
_EQ, _NEG = N_SPECIAL + 13, N_SPECIAL + 14


def _digits(n: int) -> list[int]:
    neg = n < 0
    out = [_DIGIT0 + int(c) for c in str(abs(n))]
    return ([_NEG] if neg else []) + out


def arithmetic(rng: random.Random, *, digits: int = 3, ops: str | tuple = ("+", "-"),
               reverse_answer: bool = False) -> Example:
    """``a op b = c`` on digit tokens (``reverse_answer`` writes ``c`` least-significant
    digit first, which is easier for causal models)."""
    op = rng.choice(list(ops))
    if op not in _OPS:
        raise unknown("arithmetic op", op, _OPS, stage="algorithmic")
    a, b = rng.randrange(10 ** digits), rng.randrange(10 ** digits)
    c = a + b if op == "+" else a - b if op == "-" else a * b
    ans = _digits(c)
    if reverse_answer:
        ans = ans[::-1]
    return _seq_with_answer(_digits(a) + [_OPS[op]] + _digits(b) + [_EQ], ans + [SEP])


def parity(rng: random.Random, *, length: int = 16) -> Example:
    bits = [rng.randrange(2) for _ in range(length)]
    return _seq_with_answer([N_SPECIAL + b for b in bits] + [SEP],
                            [YES if sum(bits) % 2 else NO])


def majority(rng: random.Random, *, length: int = 15, vocab: int = 8) -> Example:
    x = [rng.randrange(N_SPECIAL, vocab) for _ in range(length)]
    counts = {t: x.count(t) for t in set(x)}
    top = max(counts.values())
    winner = min(t for t, c in counts.items() if c == top)
    return _seq_with_answer(x + [SEP], [winner])


def dyck(rng: random.Random, *, k: int = 2, depth: int = 4, length: int = 16) -> Example:
    """Is a bracket sequence over ``k`` bracket types balanced (max nesting ``depth``)?"""
    opens = [N_SPECIAL + 2 * i for i in range(k)]
    closes = [N_SPECIAL + 2 * i + 1 for i in range(k)]
    seq, stack = [], []
    n = length - (length % 2)
    while len(seq) < n:
        remaining = n - len(seq)
        if stack and (len(stack) >= depth or remaining <= len(stack) or rng.random() < 0.45):
            seq.append(closes[stack.pop()])
        else:
            t = rng.randrange(k)
            stack.append(t)
            seq.append(opens[t])
    valid = rng.random() < 0.5
    if not valid:
        i = rng.randrange(len(seq))
        seq[i] = rng.choice([x for x in opens + closes if x != seq[i]])
        valid = _balanced(seq, opens, closes)
    return _seq_with_answer(seq + [SEP], [YES if valid else NO])


def _balanced(seq: list[int], opens: list[int], closes: list[int]) -> bool:
    stack = []
    for t in seq:
        if t in opens:
            stack.append(opens.index(t))
        elif not stack or closes.index(t) != stack.pop():
            return False
    return not stack


def modular_arithmetic(rng: random.Random, *, p: int = 97, ops: tuple = ("+",)) -> Example:
    op = rng.choice(list(ops))
    a, b = rng.randrange(p), rng.randrange(p)
    c = (a + b) % p if op == "+" else (a - b) % p if op == "-" else (a * b) % p
    base = N_SPECIAL + 3
    return _seq_with_answer([base + a, N_SPECIAL + ("+-*".index(op)), base + b, SEP],
                            [base + c])


def graph_reachability(rng: random.Random, *, n_nodes: int = 8, n_edges: int = 10) -> Example:
    nodes = list(range(N_SPECIAL, N_SPECIAL + n_nodes))
    edges = set()
    while len(edges) < min(n_edges, n_nodes * (n_nodes - 1)):
        a, b = rng.sample(nodes, 2)
        edges.add((a, b))
    s, t = rng.sample(nodes, 2)
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    seen, frontier = {s}, [s]
    while frontier:
        nxt = []
        for x in frontier:
            for y in adj.get(x, []):
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        frontier = nxt
    flat = [x for e in sorted(edges) for x in e]
    return _seq_with_answer(flat + [QUERY, s, t], [YES if t in seen else NO])


def in_context_regression(rng: random.Random, *, n_points: int = 8, modulus: int = 17
                          ) -> Example:
    """``y = (a*x + b) mod m`` shown on examples; predict ``y`` for a query ``x``."""
    a, b = rng.randrange(1, modulus), rng.randrange(modulus)
    base = N_SPECIAL
    xs = [rng.randrange(modulus) for _ in range(n_points + 1)]
    toks = [t for x in xs[:-1] for t in (base + x, base + (a * x + b) % modulus)]
    q = xs[-1]
    return _seq_with_answer(toks + [QUERY, base + q], [base + (a * q + b) % modulus])


def needle_in_haystack(rng: random.Random, *, length: int = 128, vocab: int = 64) -> Example:
    half = N_SPECIAL + (vocab - N_SPECIAL) // 2
    hay = [rng.randrange(N_SPECIAL, half) for _ in range(length)]
    key, val = rng.randrange(half, vocab), rng.randrange(N_SPECIAL, half)
    pos = rng.randrange(length - 1)
    hay[pos], hay[pos + 1] = key, val
    return _seq_with_answer(hay + [QUERY, key], [val])


TASKS: dict[str, Callable[..., Example]] = {
    "associative_recall": associative_recall, "mqar": mqar, "induction_heads": induction_heads,
    "selective_copy": selective_copy, "copy": copy, "reverse": reverse, "sort": sort,
    "arithmetic": arithmetic, "parity": parity, "majority": majority, "dyck": dyck,
    "modular_arithmetic": modular_arithmetic, "graph_reachability": graph_reachability,
    "in_context_regression": in_context_regression, "needle_in_haystack": needle_in_haystack,
}


def vocab_size(task: str, **knobs: Any) -> int:
    """Vocabulary a model needs for ``task`` with these knobs."""
    fn = _task(task)
    defaults = {k: p.default for k, p in inspect.signature(fn).parameters.items()
                if p.default is not p.empty}
    k = {**defaults, **knobs}
    if task == "arithmetic":
        return N_SPECIAL + 15
    if task == "modular_arithmetic":
        return N_SPECIAL + 3 + int(k["p"])
    if task in ("parity",):
        return N_SPECIAL + 2
    if task == "dyck":
        return N_SPECIAL + 2 * int(k["k"])
    if task == "graph_reachability":
        return N_SPECIAL + int(k["n_nodes"])
    if task == "in_context_regression":
        return N_SPECIAL + int(k["modulus"])
    return int(k.get("vocab", 64))


def _task(name: str) -> Callable[..., Example]:
    fn = TASKS.get(name)
    if fn is None:
        raise unknown("algorithmic task", name, TASKS, stage="algorithmic")
    return fn


def sample(task: str, rng: random.Random | None = None, **knobs: Any) -> Example:
    return _task(task)(rng or random.Random(0), **knobs)


def check(example: Example) -> None:
    """Invariants every probe example satisfies (used by tests)."""
    t, lab = example["tokens"], example["labels"]
    if len(t) != len(lab):
        raise AssertionError("tokens/labels length mismatch")
    for i, y in enumerate(lab):
        if y != IGNORE and t[i] != y:
            raise AssertionError(f"label at {i} ({y}) is not the token there ({t[i]})")


def batch(task: str, n: int, *, seed: int = 0, **knobs: Any) -> dict[str, torch.Tensor]:
    """``n`` examples as padded tensors (``mqar`` / ``copy`` / ``parity`` are fully vectorized;
    others loop in Python)."""
    g = torch.Generator().manual_seed(seed)
    if task == "parity":
        length = int(knobs.get("length", 16))
        bits = torch.randint(0, 2, (n, length), generator=g)
        ans = torch.where(bits.sum(1) % 2 == 1, YES, NO)
        tokens = torch.cat([bits + N_SPECIAL, torch.full((n, 1), SEP), ans[:, None]], 1)
        labels = torch.full_like(tokens, IGNORE)
        labels[:, -1] = ans
        return {"tokens": tokens, "labels": labels}
    if task == "copy":
        length, vocab = int(knobs.get("length", 16)), int(knobs.get("vocab", 16))
        x = torch.randint(N_SPECIAL, vocab, (n, length), generator=g)
        tokens = torch.cat([x, torch.full((n, 1), SEP), x], 1)
        labels = torch.full_like(tokens, IGNORE)
        labels[:, length + 1:] = x
        return {"tokens": tokens, "labels": labels}
    if task == "mqar" and knobs.get("seq_len") is None:
        n_pairs = int(knobs.get("n_pairs", 8))
        nq = int(knobs.get("n_queries") or n_pairs)
        vocab = int(knobs.get("vocab", 64))
        keys_n = (vocab - N_SPECIAL) // 2
        keys = torch.argsort(torch.rand(n, keys_n, generator=g), 1)[:, :n_pairs] + N_SPECIAL
        vals = torch.randint(N_SPECIAL + keys_n, vocab, (n, n_pairs), generator=g)
        kv = torch.stack([keys, vals], 2).reshape(n, 2 * n_pairs)
        q = torch.randint(0, n_pairs, (n, nq), generator=g)
        qk, qv = keys.gather(1, q), vals.gather(1, q)
        queries = torch.stack([qk, qv], 2).reshape(n, 2 * nq)
        tokens = torch.cat([kv, torch.full((n, 2), PAD), queries], 1)
        labels = torch.full_like(tokens, IGNORE)
        labels[:, 2 * n_pairs + 3::2] = qv
        return {"tokens": tokens, "labels": labels}
    rng = random.Random(seed)
    exs = [sample(task, rng, **knobs) for _ in range(n)]
    T = max(len(e["tokens"]) for e in exs)
    tokens = torch.full((n, T), PAD, dtype=torch.long)
    labels = torch.full((n, T), IGNORE, dtype=torch.long)
    for i, e in enumerate(exs):
        tokens[i, : len(e["tokens"])] = torch.tensor(e["tokens"])
        labels[i, : len(e["labels"])] = torch.tensor(e["labels"])
    return {"tokens": tokens, "labels": labels}


# ---------------------------------------------------------------- generators


class Probe(synth.Generator):
    """A probe task as a generator node; knobs fixed here or read from attributes with the
    same names (``attributes={"n_pairs": [4, 8, 16]}``)."""

    evidence = "construction"
    tags = ("seq", "probe", "lm")
    example = {"task": "parity", "length": 8}

    def __init__(self, task: str, **knobs: Any) -> None:
        self.fn = _task(task)
        self.task, self.name, self.knobs = task, task, dict(knobs)
        self._params = set(inspect.signature(self.fn).parameters) - {"rng"}
        bad = [k for k in knobs if k not in self._params]
        if bad:
            raise unknown(f"{task} knob", bad[0], sorted(self._params), stage="algorithmic")
        self.produces = synth.Schema(tokens=synth.Tokens(vocab=vocab_size(task, **knobs)),
                                     labels=synth.Tokens())

    def params(self) -> dict[str, Any]:
        return {}

    def generate(self, ctx: Any, attrs: dict[str, Any]) -> Any:
        kw = {**self.knobs, **{k: v for k, v in attrs.items() if k in self._params}}
        ex = self.fn(ctx.rng, **kw)
        return synth.Raw(ex, task=self.task, length=len(ex["tokens"]))

    def config(self) -> dict[str, Any]:
        return {"type": "probe", "task": self.task, "knobs": self.knobs}

    def describe(self) -> str:
        return f"probe({self.task})"


def _dual(name: str) -> Callable[..., Any]:
    """``mqar(rng, **knobs)`` -> one example; ``mqar(**knobs)`` -> a Probe node."""
    fn = TASKS[name]

    def task(*args: Any, **knobs: Any) -> Any:
        if args:
            return fn(*args, **knobs)
        return Probe(name, **knobs)

    task.__name__ = name
    task.__qualname__ = name
    task.__doc__ = fn.__doc__
    return task


for _name in list(TASKS):
    globals()[_name] = _dual(_name)

__all__ = ["IGNORE", "N_SPECIAL", "TASKS", "Probe", "arithmetic", "associative_recall", "batch",
           "check", "copy", "dyck", "graph_reachability", "in_context_regression",
           "induction_heads", "majority", "modular_arithmetic", "mqar", "needle_in_haystack",
           "parity", "reverse", "sample", "selective_copy", "sort", "vocab_size"]
