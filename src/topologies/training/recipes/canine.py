import torch

from topologies.training.recipe_base import Recipe, RecipeConfig, register_recipe


class CanineCConfig(RecipeConfig):
    pass


@register_recipe("canine-c", help="CANINE-C shuffled autoregressive hashed character-span prediction.")
class CanineCRecipe(Recipe):
    level = "loss"
    ConfigSchema = CanineCConfig
    ComponentsSchema = None

    def expected_batch_spec(self):
        return {"codepoints": "LongTensor[B,T], begins with CLS codepoint", "targets": "LongTensor[B,T] original Unicode values", "prediction_mask": "BoolTensor[B,T] selected character positions", "input_mask": "BoolTensor[B,T] valid character positions"}

    def dummy_batch(self, device="cpu"):
        device = torch.device(device)
        codepoints = torch.tensor([[2, 97, 98, 99], [2, 100, 101, 0]], device=device)
        targets = codepoints.clone()
        prediction_mask = torch.tensor([[False, True, True, False], [False, True, False, False]], device=device)
        codepoints[prediction_mask] = 0xF0000
        return {"codepoints": codepoints, "targets": targets, "prediction_mask": prediction_mask, "input_mask": codepoints.ne(0)}

    def synthetic_batch(self, *, context_length, batch_size, seed, config):
        if context_length < 4:
            raise ValueError("CANINE-C synthetic context_length must be at least four")
        generator = torch.Generator().manual_seed(seed)
        codepoints = torch.randint(97, 123, (batch_size, context_length), generator=generator)
        codepoints[:, 0] = 2
        valid = torch.ones_like(codepoints, dtype=torch.bool)
        selected = torch.zeros_like(valid)
        selected[:, 1::4] = True
        targets = codepoints.clone()
        codepoints[selected] = 0xF0000
        return {"codepoints": codepoints, "targets": targets,
                "prediction_mask": selected, "input_mask": valid}

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        loss = model.mlm_loss(batch["codepoints"], batch["targets"], batch["prediction_mask"], batch["input_mask"], batch.get("segments"), batch.get("permutation"))
        return loss, {"loss": float(loss.detach())}

    def validation_step(self, model, batch, cfg, components, adapter=None):
        _loss, metrics = self.batch_to_loss(model, batch, cfg, components, adapter)
        return metrics
