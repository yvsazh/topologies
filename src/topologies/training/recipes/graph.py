"""Supervised node/graph classification using explicit masks and batch IDs."""
from torch.nn import functional as F

from topologies.training.recipe_base import Recipe, RecipeConfig, register_recipe


class GraphConfig(RecipeConfig):
    level: str = "node"


@register_recipe("graph-supervised", help="Masked node or graph classification.")
class GraphSupervisedRecipe(Recipe):
    level = "loss"
    ConfigSchema = GraphConfig
    ComponentsSchema = None

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        required = {"x", "edge_index", "y", "train_mask"}
        if not required <= batch.keys():
            raise ValueError(f"graph batch requires {sorted(required)}")
        graph_level = cfg.level == "graph"
        logits = model(batch["x"], batch["edge_index"], edge_weight=batch.get("edge_weight"), batch=batch.get("batch"), node_level=not graph_level)
        mask = batch["train_mask"].bool()
        if mask.shape != (logits.shape[0],):
            raise ValueError("train_mask must select exactly the prediction rows")
        if not mask.any():
            raise ValueError("train_mask must contain at least one training example")
        loss = F.cross_entropy(logits[mask], batch["y"][mask].long())
        return loss, {"loss": float(loss.detach())}
