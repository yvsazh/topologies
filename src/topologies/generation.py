"""Autoregressive generation for any token model + any tokenizer.

Works with every model whose ``forward(tokens)`` returns logits
``[B, T, vocab]`` (tuple/dict outputs are unwrapped). The context window
comes from the model contract (``context_length`` / ``max_seq`` / ...) or
the ``context_length`` argument; prompts longer than it raise a clear
error (or are truncated with ``truncate_prompt=True``), and generation
beyond it slides the window.

    text = generate(model, "Once upon a time", tokenizer=tok,
                    max_new_tokens=200, temperature=0.8, top_k=50)
    for piece in stream_generate(model, "Hi", tokenizer=tok):
        print(piece, end="", flush=True)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
import torch.nn as nn

from topologies._execution import model_execution, primary_output
from topologies.build.contract import model_context_length
from topologies.inference import model_device


class GenerationError(ValueError):
    """Invalid generation request (prompt too long, no tokenizer, ...)."""


#: Tokens to hold back while the decoded tail is an incomplete character; past this the
#: bytes are invalid, not incomplete, and are emitted as replacement characters.
_MAX_PENDING = 8


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick next-token ids from ``[B, V]`` logits (greedy / temperature / top-k / top-p)."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    logits = logits.float()
    if greedy or temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / float(temperature)
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum - sorted_logits.softmax(dim=-1) >= top_p  # keep the token crossing p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(dim=-1)
    # Sample where the logits live when the generator can (CUDA); otherwise on the CPU
    # (MPS generators are not supported for multinomial).
    on_device = probs.device.type == "cuda" and (
        generator is None or generator.device.type == "cuda")
    source = probs if on_device else probs.cpu()
    return torch.multinomial(source, 1, generator=generator).squeeze(-1).to(logits.device)


def _can_use_cache(model: nn.Module, prompt_len: int, context: int | None,
                   max_new_tokens: int) -> bool:
    """Prefill plus one-token steps, when the whole generation fits in context."""
    if not callable(getattr(model, "init_cache", None)):
        return False
    if not callable(getattr(model, "step", None)):
        return False
    return context is None or prompt_len + max_new_tokens <= context


def _prompt_ids(prompt: Any, tokenizer: Any) -> list[int]:
    if isinstance(prompt, str):
        if tokenizer is None:
            raise GenerationError("A text prompt needs tokenizer=...; or pass token ids.")
        ids = list(tokenizer.encode(prompt))
        bos = getattr(tokenizer, "bos_id", None)
        if bos is not None:
            ids = [bos, *ids]
    elif isinstance(prompt, torch.Tensor):
        ids = [int(i) for i in prompt.reshape(-1).tolist()]
    else:
        ids = [int(i) for i in prompt]
    if not ids:
        start = None
        if tokenizer is not None:
            start = getattr(tokenizer, "bos_id", None)
            if start is None:
                start = getattr(tokenizer, "eos_id", None)
        if start is None:
            raise GenerationError("Empty prompt and the tokenizer has no bos/eos token to start.")
        ids = [int(start)]
    return ids


def stream_generate(
    model: nn.Module,
    prompt: str | Sequence[int] | torch.Tensor,
    *,
    tokenizer: Any = None,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    stop: str | Sequence[str] | None = None,
    stop_at_eos: bool = True,
    context_length: int | None = None,
    truncate_prompt: bool = False,
    seed: int | None = None,
    as_text: bool | None = None,
    callback: Callable[[int, list[int], torch.Tensor], None] | None = None,
) -> Iterator[Any]:
    """Yield the continuation incrementally: text pieces (with a tokenizer) or token ids.

    ``callback(step, ids, logits)`` (optional) sees every decoding step: ``ids`` ends with the
    sampled token and ``logits`` are the next-token logits ``[1, vocab]``.
    """
    if max_new_tokens < 0:
        raise GenerationError("max_new_tokens must be >= 0.")
    as_text = (tokenizer is not None) if as_text is None else as_text
    if as_text and tokenizer is None:
        raise GenerationError("as_text=True needs tokenizer=...")
    ids = _prompt_ids(prompt, tokenizer)
    ctx = context_length or model_context_length(model)
    if ctx is not None and len(ids) > ctx:
        if not truncate_prompt:
            raise GenerationError(
                f"Prompt has {len(ids)} tokens but the model context is {ctx}. "
                "Shorten the prompt or pass truncate_prompt=True (keeps the last tokens)."
            )
        ids = ids[-ctx:]
    stops = [stop] if isinstance(stop, str) else list(stop or [])
    eos = getattr(tokenizer, "eos_id", None) if (stop_at_eos and tokenizer is not None) else None
    device = model_device(model)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
        gen.manual_seed(int(seed))
    new: list[int] = []
    emitted = ""  # text already yielded
    # Incremental detokenization: decode only a short window of recent ids, so
    # the cost per token is O(1) instead of re-decoding everything (O(n^2)).
    prefix_offset = read_offset = 0
    prefix_text = ""
    longest_stop = max((len(s) for s in stops), default=0)
    # Modes are restored per sub-module when the generator finishes or is closed.
    # Grad mode is scoped to each forward (never held across a yield), and the
    # global RNG is not restored: unseeded sampling must keep advancing it.
    with model_execution(model, training=False, grad_mode="unchanged", preserve_rng=False):
        # Models with ``init_cache``/``step`` decode incrementally (KV cache) when
        # the whole generation fits in the context; others recompute the window.
        cache = None
        logits: torch.Tensor | None = None
        if _can_use_cache(model, len(ids), ctx, max_new_tokens):
            try:
                dtype = next(model.parameters()).dtype
            except StopIteration:
                dtype = torch.float32
            with torch.inference_mode():
                cache = model.init_cache(1, device=device, dtype=dtype)
                prompt_tokens = torch.tensor([ids], dtype=torch.long, device=device)
                logits = primary_output(model.step(prompt_tokens, cache)[0], floating=True)[:, -1, :]
        for step_i in range(max_new_tokens):
            if cache is None:
                window = ids[-ctx:] if ctx is not None else ids
                x = torch.tensor([window], dtype=torch.long, device=device)
                with torch.inference_mode():
                    logits = primary_output(model(x), floating=True)[:, -1, :]
            nxt = int(sample_next_token(logits, temperature=temperature, top_k=top_k,
                                        top_p=top_p, greedy=greedy, generator=gen)[0])
            if callback is not None:
                callback(step_i, [*ids, nxt], logits)
            if eos is not None and nxt == eos:
                break
            ids.append(nxt)
            new.append(nxt)
            if cache is not None and step_i + 1 < max_new_tokens:
                token = torch.tensor([[nxt]], dtype=torch.long, device=device)
                with torch.inference_mode():
                    logits = primary_output(model.step(token, cache)[0], floating=True)[:, -1, :]
            if not as_text:
                yield nxt
                continue
            window_text = tokenizer.decode(new[prefix_offset:])
            if window_text.endswith("\ufffd") and len(new) - read_offset < _MAX_PENDING:
                continue  # incomplete multi-byte character: wait for the rest (bounded)
            piece = window_text[len(prefix_text):]
            prefix_offset, read_offset = read_offset, len(new)
            prefix_text = tokenizer.decode(new[prefix_offset:read_offset])
            if longest_stop:
                buffer = emitted + piece
                start = max(0, len(emitted) - longest_stop + 1)  # stops may span pieces
                hits = [i for i in (buffer.find(s, start) for s in stops if s) if i >= 0]
                if hits:
                    cut = min(hits)
                    if cut > len(emitted):
                        yield buffer[len(emitted):cut]
                    return
            if piece:
                yield piece
                emitted += piece
        if as_text and read_offset < len(new):
            window_text = tokenizer.decode(new[prefix_offset:])
            piece = window_text[len(prefix_text):]
            if longest_stop:
                buffer = emitted + piece
                start = max(0, len(emitted) - longest_stop + 1)
                hits = [i for i in (buffer.find(s, start) for s in stops if s) if i >= 0]
                if hits:
                    piece = buffer[len(emitted):min(hits)] if min(hits) > len(emitted) else ""
            if piece:
                yield piece


def generate(
    model: nn.Module,
    prompt: str | Sequence[int] | torch.Tensor,
    *,
    tokenizer: Any = None,
    include_prompt: bool = True,
    **kwargs: Any,
) -> Any:
    """Generate a continuation; returns text with a tokenizer, else a list of ids.

    Keyword arguments are those of ``stream_generate`` (``max_new_tokens``,
    ``temperature``, ``top_k``, ``top_p``, ``greedy``, ``stop``, ``seed``, ...).
    """
    pieces = list(stream_generate(model, prompt, tokenizer=tokenizer, **kwargs))
    as_text = kwargs.get("as_text")
    as_text = (tokenizer is not None) if as_text is None else as_text
    if as_text:
        out = "".join(pieces)
        if include_prompt:
            head = prompt if isinstance(prompt, str) else tokenizer.decode(_prompt_ids(prompt,
                                                                                      tokenizer))
            return head + out
        return out
    if include_prompt:
        return _prompt_ids(prompt, tokenizer) + pieces
    return pieces
