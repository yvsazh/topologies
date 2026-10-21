"""Synthetic loaders matching each Recipe's expected schema."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def _loader(tensors: dict[str, torch.Tensor], batch_size: int = 4,
            generator: torch.Generator | None = None) -> DataLoader:
    keys = list(tensors.keys())
    ds = TensorDataset(*[tensors[k] for k in keys])

    def _collate(batch: list[tuple]) -> dict:
        return {k: torch.stack([b[i] for b in batch]) for i, k in enumerate(keys)}

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, generator=generator)


def make_synthetic_loader(
    recipe_name: str, *, batch_size: int = 4, steps: int = 4, seed: int = 0, **kwargs: Any
) -> DataLoader:
    """Build a small CPU loader whose batches match ``recipe_name``'s schema.

    Single source of truth for tests / smoke / examples: batch keys come from
    each Recipe's ``expected_batch_spec()`` / ``dummy_batch()`` shapes.
    """
    for parameter, value in (("batch_size", batch_size), ("steps", steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{parameter} must be a positive integer")

    from topologies.training.recipe_base import get_recipe

    g = torch.Generator().manual_seed(seed)
    name = str(recipe_name)
    r = get_recipe(name)
    if hasattr(r, "synthetic_batch"):
        if name == "canine-c":
            if "context_length" not in kwargs:
                raise ValueError("canine-c synthetic data requires model context_length")
            batch = r.synthetic_batch(context_length=kwargs["context_length"],
                                      batch_size=batch_size, seed=seed, config=r.ConfigSchema())
            tensors = {key: value.repeat(max(steps, 1), *([1] * (value.ndim - 1)))
                       for key, value in batch.items()}
            return _loader(tensors, batch_size=batch_size, generator=g)
        if "vocab_size" not in kwargs or "context_length" not in kwargs:
            message = ("albert-pretraining synthetic data requires model vocab_size and context_length"
                       if name == "albert-pretraining" else
                       f"{name} synthetic data requires model vocab_size and context_length")
            raise ValueError(message)
        config_type = type(r).ConfigSchema
        config_fields = getattr(config_type, "model_fields", {})
        config_values = {key: value for key, value in kwargs.items() if key in config_fields}
        if name == "distilbert-pretraining":
            from topologies.training.recipes.lm import distilbert_special_token_ids

            controls = distilbert_special_token_ids(kwargs["vocab_size"])
            for key, value in zip(("pad_token_id", "unknown_token_id", "cls_token_id",
                                   "sep_token_id", "mask_token_id"), controls):
                config_values.setdefault(key, value)
            config_values.setdefault("special_token_ids", controls)
        if name == "electra-pretraining":
            from topologies.training.recipes.lm import electra_pretraining_special_token_ids

            controls = electra_pretraining_special_token_ids(kwargs["vocab_size"])
            for key, value in zip(("pad_token_id", "unknown_token_id", "cls_token_id",
                                   "sep_token_id", "mask_token_id"), controls):
                config_values.setdefault(key, value)
            config_values.setdefault("special_token_ids", controls)
        config = config_type(**config_values)
        batch = r.synthetic_batch(
            vocab_size=kwargs["vocab_size"], context_length=kwargs["context_length"],
            batch_size=batch_size, seed=seed, config=config)
        if name == "distilbert-pretraining":
            class DynamicRecipeLoader:
                def __init__(self):
                    self.batch_size = batch_size
                    self.steps = max(steps, 1)

                def __len__(self):
                    return self.steps

                def __iter__(self):
                        for index in range(self.steps):
                            yield r.synthetic_batch(vocab_size=kwargs["vocab_size"],
                                context_length=kwargs["context_length"], batch_size=batch_size,
                                seed=seed, mask_seed=seed + index, config=config)

            return DynamicRecipeLoader()
        tensors = {key: value.repeat(max(steps, 1), *([1] * (value.ndim - 1)))
                   for key, value in batch.items()}
        return _loader(tensors, batch_size=batch_size, generator=g)
    # Prefer the recipe's own dummy_batch as the canonical definition.
    try:
        r = get_recipe(name)
        dummy = r.dummy_batch(device="cpu")
        tensors: dict[str, torch.Tensor] = {}
        for k, v in dummy.items():
            if isinstance(v, torch.Tensor):
                reps = [max(steps, 1)] + [1] * (v.dim() - 1) if v.dim() > 0 else [1]
                big = v.repeat(*reps) if v.dim() > 0 else v.unsqueeze(0).repeat(max(steps, 1))
                # Re-randomize beyond the first copy for variety (keep dtypes).
                if big.shape[0] > v.shape[0]:
                    extra = big[v.shape[0]:].clone()
                    if extra.is_floating_point():
                        extra = torch.randn_like(extra, generator=g)
                    else:
                        original = extra
                        non_negative = v[v >= 0]
                        high = max(2, int(non_negative.max().item()) + 1) if non_negative.numel() else 2
                        if name in ("causal-lm", "masked-lm", "albert-pretraining", "electra", "seq2seq-lm") and k in (
                                "tokens", "src", "tgt", "labels"):
                            high = kwargs.get("vocab_size", 64)
                        if k.endswith("lengths"):
                            extra = original
                        else:
                            extra = torch.randint(0, high, original.shape, dtype=original.dtype, generator=g)
                            extra = torch.where(original < 0, original, extra)
                    big = torch.cat([big[: v.shape[0]], extra])
                tensors[k] = big
            else:
                tensors[k] = v
        vocab_size = kwargs.get("vocab_size")
        if (name in ("causal-lm", "masked-lm", "albert-pretraining") and isinstance(vocab_size, int)
                and not isinstance(vocab_size, bool) and vocab_size > 0):
            if "tokens" in tensors:
                tensors["tokens"] = tensors["tokens"].remainder(vocab_size)
            if "labels" in tensors:
                labels = tensors["labels"]
                ignore_index = int(kwargs.get("ignore_index", -100))
                tensors["labels"] = torch.where(
                    labels == ignore_index, labels, labels.remainder(vocab_size))
        if tensors and all(isinstance(v, torch.Tensor) for v in tensors.values()):
            return _loader(tensors, batch_size=batch_size, generator=g)
    except Exception:  # noqa: BLE001, S110 - fall through to explicit builders
        pass

    def _rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    if name in ("causal-lm", "masked-lm", "electra"):
        n = batch_size * steps
        vocab_size = kwargs.get("vocab_size", 64)
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        return _loader(
            {
                "tokens": torch.randint(0, vocab_size, (n, 8), generator=g),
                "labels": (
                    torch.randint(0, vocab_size, (n, 8), generator=g)
                    if name != "electra"
                    else torch.randint(0, 2, (n, 8), generator=g).float()
                ),
            },
            batch_size=batch_size,
        )
    if name == "seq2seq-lm":
        n = batch_size * steps
        return _loader(
            {
                "src": torch.randint(0, 64, (n, 8), generator=g),
                "tgt": torch.randint(0, 64, (n, 8), generator=g),
                "labels": torch.randint(0, 64, (n, 8), generator=g),
            },
            batch_size=batch_size,
        )
    if name == "vision-classify":
        n = batch_size * steps
        return _loader(
            {"pixels": _rand(n, 3, 16, 16), "labels": torch.randint(0, 10, (n,), generator=g)},
            batch_size=batch_size,
        )
    if name in ("autoencoder", "vae", "vqvae"):
        n = batch_size * steps
        return _loader({"x": torch.rand(n, 32, generator=g)}, batch_size=batch_size)
    if name == "diffusion":
        n = batch_size * steps
        return _loader({"x": _rand(n, 16, 8, 8)}, batch_size=batch_size)
    if name == "recommender-bce":
        n = batch_size * steps
        return _loader(
            {
                "user": torch.randint(0, 32, (n,), generator=g),
                "item": torch.randint(0, 64, (n,), generator=g),
                "label": torch.randint(0, 2, (n,), generator=g).float(),
            },
            batch_size=batch_size,
        )
    if name == "tts":
        n = batch_size * steps
        return _loader(
            {
                "tokens": torch.randint(0, 64, (n, 16), generator=g),
                "mel": _rand(n, 32, 16),
                "durations": torch.randint(1, 4, (n, 16), generator=g).float(),
            },
            batch_size=batch_size,
        )
    # Generic fallback (gan/distillation/ppo use custom paths; keep loader usable).
    n = batch_size * steps
    return _loader({"x": _rand(n, 32)}, batch_size=batch_size)
