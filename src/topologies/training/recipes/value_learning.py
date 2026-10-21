"""One-step TD target utilities; target network ownership stays with caller."""
import torch
from torch.nn import functional as F

from topologies.training.recipe_base import Recipe, RecipeComponents, RecipeConfig, register_recipe


class TDValueConfig(RecipeConfig):
    gamma: float = 0.99
    loss: str = "huber"


class TDValueComponents(RecipeComponents):
    target_model: object


@register_recipe("td-value", help="One-step Q-learning with explicit terminal/truncation flags.")
class TDValueRecipe(Recipe):
    level = "loss"
    ConfigSchema = TDValueConfig
    ComponentsSchema = TDValueComponents

    @staticmethod
    def td_target(rewards, next_q_values, terminated, gamma, truncated=None):
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        # Truncation is not termination: bootstrap unless the caller defines it as terminal.
        return rewards + gamma * next_q_values.detach().max(dim=-1).values * (~terminated.bool()).to(rewards)

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        if components is None or components.target_model is None:
            raise ValueError("td-value requires an externally managed target_model component")
        states, actions = batch["state"], batch["action"].long()
        with torch.no_grad():
            next_q = components.target_model(batch["next_state"])
            target = self.td_target(batch["reward"], next_q, batch["terminated"], cfg.gamma, batch.get("truncated"))
        predicted = model(states).gather(-1, actions.reshape(-1, 1)).squeeze(-1)
        loss = F.smooth_l1_loss(predicted, target) if cfg.loss == "huber" else F.mse_loss(predicted, target)
        return loss, {"loss": float(loss.detach()), "td_target": float(target.mean())}
