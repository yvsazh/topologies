"""Reproducible autoregressive symbolic motif sampling helpers."""
import random

import torch


def make_seeded_motifs(count, *, seed=0, steps=8):
    """Generate a reproducible local motif corpus as event dicts (durations in ticks)."""
    if count < 1 or steps < 1: raise ValueError("count and steps must be positive")
    rng = random.Random(seed)
    scale = (0, 2, 4, 5, 7, 9, 11)
    motifs = []
    for _ in range(count):
        root = rng.randrange(48, 73)
        motifs.append([{"pitch": min(127, max(0, root + scale[rng.randrange(len(scale))])),
                        "duration_ticks": rng.choice((3, 6, 12)), "velocity": rng.randrange(48, 113)}
                       for _ in range(steps)])
    return motifs


def sample_tokens(model, prefix, count, *, seed=0, top_k=8, temperature=1.0, pad_id=2):
    if count < 0 or top_k < 1 or temperature <= 0: raise ValueError("invalid sampling settings")
    generator = torch.Generator(device=prefix.device).manual_seed(seed)
    result = prefix.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(count):
            logits = model(result[:, -model.context_length:])[:, -1] / temperature
            logits[:, pad_id] = -torch.inf
            values, indices = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
            probs = torch.softmax(values, dim=-1)
            choice = torch.multinomial(probs, 1, generator=generator)
            result = torch.cat((result, indices.gather(-1, choice)), dim=1)
    return result
