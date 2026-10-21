import torch

from topologies.training.recipe_base import Recipe, RecipeConfig, register_recipe


class NegativeSamplingConfig(RecipeConfig):
    pass


@register_recipe("negative-sampling", help="Skip-gram negative-sampling loss for static embeddings.")
class NegativeSamplingRecipe(Recipe):
    ConfigSchema = NegativeSamplingConfig
    ComponentsSchema = None
    level = "loss"

    def expected_batch_spec(self):
        return {"target": "LongTensor[B]", "context": "LongTensor[B]", "negatives": "LongTensor[B,K]"}

    def dummy_batch(self, device="cpu"):
        return {
            "target": torch.randint(1, 32, (4,), device=device),
            "context": torch.randint(1, 32, (4,), device=device),
            "negatives": torch.randint(1, 32, (4, 2), device=device),
        }

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        loss = model(batch["target"], batch["context"], batch["negatives"])
        return loss, {"loss": float(loss.detach())}
