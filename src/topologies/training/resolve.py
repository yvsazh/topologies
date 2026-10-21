"""Recipe resolution with confidence + explanation (never class-name matching)."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch.nn as nn

from topologies.training.adapters import ModelSpec, model_spec_for
from topologies.training.errors import TrainingError

TASK_TO_RECIPE: dict[str, str] = {
    "language-modeling": "causal-lm",
    "causal-language-modeling": "causal-lm",
    "masked-language-modeling": "masked-lm",
    "permutation-language-modeling": "permutation-lm",
    "seq2seq": "seq2seq-lm",
    "summarization": "seq2seq-lm",
    "translation": "seq2seq-lm",
    "image-classification": "vision-classify",
    "image-classification-distilled": "vision-classify",
    "autoencoding": "autoencoder",
    "reconstruction": "autoencoder",
    "vae": "vae",
    "vqvae": "vqvae",
    "vector-quantization": "vqvae",
    "diffusion": "diffusion",
    "denoising": "diffusion",
    "recommendation": "recommender-bce",
    "collaborative-filtering": "recommender-bce",
    "replaced-token-detection": "electra-pretraining",
    "electra": "electra-pretraining",
    "text-to-speech": "tts",
    "tts": "tts",
    "gan": "gan",
    "distillation": "distillation",
    "reinforcement-learning": "ppo",
    "ppo": "ppo",
}


@dataclass
class ResolvedRecipe:
    recipe: Any  # Recipe instance
    confidence: float
    explanation: str


def _registry_recipe_for_model(model: nn.Module) -> tuple[str | None, str]:
    """Look up registry training.recipe via architecture_id (metadata only)."""
    arch_id = getattr(model, "architecture_id", "") or getattr(
        type(model), "architecture_id", ""
    )
    if not arch_id:
        return None, ""
    packaged = _packaged_model_spec(str(arch_id))
    if packaged:
        if packaged.get("recipe"):
            return str(packaged["recipe"]), f"packaged ModelSpec recipe={packaged['recipe']!r} for {arch_id!r}"
        for task in packaged.get("tasks", []):
            if str(task).lower() in TASK_TO_RECIPE:
                return TASK_TO_RECIPE[str(task).lower()], f"packaged ModelSpec task {task!r}"
    try:
        from topologies.registry.api import Registry

        reg = Registry.load(None)
        rec = None
        try:
            rec = reg.get(str(arch_id))
        except Exception:  # noqa: BLE001
            rec = None
        if rec is None:
            return None, ""
        training = getattr(rec, "training", None)
        if training is None:
            # Also support extra-field dict form.
            try:
                dump = rec.model_dump(mode="python")
                training = dump.get("training")
            except Exception:  # noqa: BLE001
                training = None
        recipe_name = None
        if training is not None:
            if isinstance(training, dict):
                recipe_name = training.get("recipe")
            else:
                recipe_name = getattr(training, "recipe", None)
        if recipe_name:
            return str(recipe_name), f"registry training.recipe={recipe_name!r} for {arch_id!r}"
        # Task/modality mapping from canonical metadata (no class names).
        tasks = [str(t).lower() for t in (getattr(rec, "tasks", None) or [])]
        for t in tasks:
            if t in TASK_TO_RECIPE:
                return TASK_TO_RECIPE[t], f"registry tasks {tasks} -> {TASK_TO_RECIPE[t]!r}"
        modalities = [str(m).lower() for m in (getattr(rec, "modalities", None) or [])]
        if "vision" in modalities and "text" not in modalities:
            return "vision-classify", f"registry modalities {modalities} -> 'vision-classify'"
        return None, ""
    except Exception:  # noqa: BLE001
        return None, ""


@lru_cache(maxsize=1)
def _packaged_model_specs() -> dict:
    """Metadata projection shipped in the wheel; independent of checkout YAML."""
    path = Path(__file__).with_name("model_recipes.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("models", {}) if data.get("schema_version") == 1 else {}
    except (OSError, ValueError):
        return {}


def _packaged_model_spec(architecture_id: str) -> dict | None:
    return _packaged_model_specs().get(architecture_id)


def _infer_from_batch(
    model: nn.Module, batch: dict | None
) -> tuple[str | None, float, str]:
    if not batch:
        return None, 0.0, "no batch available for inference"
    keys = set(batch.keys())
    sig_params: set[str] = set()
    try:
        sig = inspect.signature(model.forward)
        sig_params = {
            p.name
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
    except (ValueError, TypeError):
        sig_params = set()
    # High-confidence exact-schema matches.
    if keys == {"tokens", "labels"}:
        return "causal-lm", 0.85, "batch keys {tokens, labels} match causal-lm schema"
    if keys in ({"tokens"}, {"tokens", "mask"}):
        return "causal-lm", 0.75, (
            f"batch keys {sorted(keys)} match causal-lm (labels derived from tokens)"
        )
    if keys == {"tokens", "labels", "mask"} or keys == {"tokens", "labels", "attention_mask"}:
        # Padded causal batches and masked-LM batches look identical: never guess.
        return None, 0.5, (
            f"batch keys {sorted(keys)} fit both causal-lm (padded) and masked-lm"
        )
    if keys >= {"src", "tgt", "labels"}:
        return "seq2seq-lm", 0.85, "batch keys {src, tgt, labels} match seq2seq-lm schema"
    if keys >= {"pixels", "labels"} or keys >= {"images", "labels"}:
        return "vision-classify", 0.85, "batch keys {pixels/images, labels} match vision schema"
    if keys >= {"user", "item", "label"}:
        return "recommender-bce", 0.85, "batch keys {user, item, label} match recommender schema"
    if keys >= {"x"} and len(keys) == 1 and "tokens" not in sig_params:
        return "autoencoder", 0.6, "batch keys {x} are ambiguous (autoencoder/vae/vqvae)"
    # Forward-signature hints raise confidence only for distinctive arities.
    if sig_params == {"src", "tgt"}:
        return "seq2seq-lm", 0.75, "forward(src, tgt) matches seq2seq-lm"
    if sig_params == {"user", "item"}:
        return "recommender-bce", 0.75, "forward(user, item) matches recommender-bce"
    if "tokens" in keys and "labels" in keys and not ("mask" in keys or "attention_mask" in keys):
        return "causal-lm", 0.7, "batch has tokens+labels"
    return None, 0.4, (
        f"ambiguous batch keys={sorted(keys)} forward={sorted(sig_params)}; "
        "candidates: causal-lm, masked-lm, seq2seq-lm"
    )


def resolve_recipe(
    *,
    recipe: Any = None,
    model: nn.Module | None = None,
    model_spec: ModelSpec | dict | None = None,
    train_batch: dict | None = None,
    allow_auto: bool = True,
) -> ResolvedRecipe:
    """Resolve a Recipe instance. Priority: instance > string > ModelSpec > registry > mapping > infer."""
    from topologies.training.recipe_base import get_recipe

    if recipe is not None and not isinstance(recipe, str):
        return ResolvedRecipe(
            recipe=recipe, confidence=1.0, explanation="explicit Recipe instance"
        )
    if isinstance(recipe, str):
        return ResolvedRecipe(
            recipe=get_recipe(recipe),
            confidence=1.0,
            explanation=f"explicit recipe={recipe!r}",
        )
    spec = model_spec_for(model, explicit=model_spec) if model is not None else None
    if spec is not None and spec.recipe:
        return ResolvedRecipe(
            recipe=get_recipe(spec.recipe),
            confidence=0.95,
            explanation=f"ModelSpec recipe={spec.recipe!r}",
        )
    if model is not None:
        reg_name, reg_why = _registry_recipe_for_model(model)
        if reg_name:
            return ResolvedRecipe(
                recipe=get_recipe(reg_name), confidence=0.9, explanation=reg_why
            )
    if model is not None and allow_auto:
        name, conf, why = _infer_from_batch(model, train_batch)
        if name is not None and conf >= 0.7:
            return ResolvedRecipe(
                recipe=get_recipe(name),
                confidence=conf,
                explanation=f"inferred {name!r}: {why}",
            )
        # Low confidence -> actionable error, never silent guess.
        fwd = ""
        try:
            fwd = str(inspect.signature(model.forward))
        except (ValueError, TypeError):
            fwd = "(unknown signature)"
        raise TrainingError(
            "Could not confidently determine a training recipe.\n\n"
            f"Model:\n    {type(model).__name__ if model is not None else '?'}\n\n"
            f"Detected forward:\n    forward{fwd}\n\n"
            "Possible recipes:\n"
            "    causal-lm\n    masked-lm\n    seq2seq-lm\n\n"
            "Please specify one explicitly:\n\n"
            '    Trainer(..., recipe="causal-lm")',
            stage="resolve",
            received=f"batch keys={sorted((train_batch or {}).keys())} forward{fwd}",
            hint="Pass recipe explicitly or attach a ModelSpec / registry training.recipe.",
            available=["causal-lm", "masked-lm", "seq2seq-lm", "vision-classify",
                        "autoencoder", "vae", "vqvae", "diffusion", "recommender-bce",
                        "electra", "tts", "gan", "distillation", "ppo"],
        )
    fwd = ""
    try:
        fwd = str(inspect.signature(model.forward)) if model is not None else ""
    except (ValueError, TypeError):
        fwd = ""
    raise TrainingError(
        "No training recipe specified and automatic inference is disabled.",
        stage="resolve",
        received=f"forward{fwd}",
        hint='Pass recipe explicitly, e.g. Trainer(..., recipe="causal-lm").',
        available=["causal-lm", "masked-lm", "seq2seq-lm", "vision-classify",
                    "autoencoder", "vae", "vqvae", "diffusion", "recommender-bce",
                    "electra", "tts", "gan", "distillation", "ppo"],
    )
