"""Paired image/text symmetric InfoNCE objective and retrieval evaluation."""
import torch
import torch.nn.functional as F

from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import Recipe, register_recipe


def symmetric_info_nce(image_embeddings, text_embeddings, temperature=0.07, pair_ids=None):
    if image_embeddings.ndim != 2 or text_embeddings.shape != image_embeddings.shape:
        raise ValueError("embeddings must have matching [B,D] shapes")
    if image_embeddings.shape[0] < 2: raise ValueError("InfoNCE requires at least two pairs")
    if temperature <= 0: raise ValueError("temperature must be positive")
    i, t = F.normalize(image_embeddings, dim=-1), F.normalize(text_embeddings, dim=-1)
    logits = i @ t.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    if pair_ids is not None and (len(pair_ids) != len(labels) or len(set(pair_ids)) != len(pair_ids)):
        raise ValueError("pair_ids must be unique within a contrastive batch")
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def retrieval_recall(image_embeddings, text_embeddings, pair_ids, ks=(1, 5)):
    """Bidirectional recall@K; IDs must identify held-out paired examples."""
    if len(pair_ids) != image_embeddings.shape[0] or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("retrieval pair identities must be unique and match batch")
    i, t = F.normalize(image_embeddings, dim=-1), F.normalize(text_embeddings, dim=-1)
    sims = i @ t.T
    ids = torch.arange(len(pair_ids), device=sims.device)
    result = {}
    for k in ks:
        if k < 1: raise ValueError("K must be positive")
        kk = min(k, len(pair_ids))
        result[f"image_to_text_recall@{k}"] = float((sims.topk(kk, dim=1).indices == ids[:, None]).any(1).float().mean())
        result[f"text_to_image_recall@{k}"] = float((sims.T.topk(kk, dim=1).indices == ids[:, None]).any(1).float().mean())
    return result


class ContrastiveConfig(LossOverrideConfig):
    temperature: float = 0.07


@register_recipe("contrastive-pairs", help="Symmetric paired image/text InfoNCE.")
@register_recipe("image-text-contrastive", help="Symmetric paired image/text InfoNCE.")
class ContrastiveRecipe(Recipe):
    ConfigSchema = ContrastiveConfig
    ComponentsSchema = None
    level = "loss"

    def expected_batch_spec(self):
        return {"images": "FloatTensor[B,C,H,W]", "tokens": "LongTensor[B,T]", "mask": "BoolTensor[B,T] optional", "pair_ids": "unique held-out-safe identities"}

    def optional_batch_keys(self): return {"mask", "pair_ids"}

    def dummy_batch(self, device="cpu"):
        return {"images": torch.randn(2, 3, 32, 32, device=device), "tokens": torch.randint(1, 32, (2, 8), device=device)}

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        image, text, _ = model(batch["images"], batch["tokens"], batch.get("mask"))
        if cfg.loss is None:
            loss = symmetric_info_nce(image, text, temperature=cfg.temperature, pair_ids=batch.get("pair_ids"))
        else:
            loss = compute_loss(cfg, image, text, temperature=cfg.temperature, normalize=True)
        return loss, {"loss": float(loss.detach())}


@register_recipe("sentence-contrastive", help="In-batch InfoNCE for paired sentence embeddings.")
class SentenceContrastiveRecipe(Recipe):
    ConfigSchema = ContrastiveConfig
    ComponentsSchema = None
    level = "loss"

    def expected_batch_spec(self):
        return {"left": "LongTensor[B,T]", "right": "LongTensor[B,T]"}

    def dummy_batch(self, device="cpu"):
        return {
            "left": torch.randint(1, 32, (4, 8), device=device),
            "right": torch.randint(1, 32, (4, 8), device=device),
        }

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        loss = symmetric_info_nce(model(batch["left"]), model(batch["right"]), temperature=cfg.temperature)
        return loss, {"loss": float(loss.detach())}
