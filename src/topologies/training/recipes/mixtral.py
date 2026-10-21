from __future__ import annotations

import math

from pydantic import Field

from topologies.evaluation import EvaluationRecord
from topologies.training.recipe_base import EvaluationStepResult, register_recipe
from topologies.training.recipes.lm import (
    CausalLMConfig,
    CausalLMRecipe,
    _call,
    _token_inputs,
    _token_loss,
)


class MixtralConfig(CausalLMConfig):
    router_aux_loss_coef: float = Field(default=0.02, ge=0.0)


@register_recipe("mixtral-causal-lm", help="Causal LM with Mixtral router load-balancing loss.")
class MixtralCausalLMRecipe(CausalLMRecipe):
    ConfigSchema = MixtralConfig

    def _logits_aux_targets(self, model, batch, cfg, adapter):
        self.validate_batch(batch, strict=True)
        inputs, keys = _token_inputs(batch)
        aux_model = getattr(model, "module", model)
        aux_forward = getattr(aux_model, "forward_with_aux", None)
        if aux_forward is not None and (adapter is None or getattr(adapter, "forward_map", None) is None):
            adapted = adapter.adapted_batch(inputs) if adapter is not None else inputs
            logits, aux = aux_forward(**adapted)
        else:
            output = _call(model, inputs, keys, adapter)
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise ValueError("mixtral-causal-lm requires forward_with_aux -> (logits, aux_loss)")
            logits, aux = output
        labels = self._targets(batch, cfg)
        if cfg.shift_labels:
            logits, labels = logits[:, :-1], labels[:, 1:]
        return logits, aux, labels

    def batch_to_loss(self, model, batch, cfg: MixtralConfig, components, adapter=None):
        logits, aux, labels = self._logits_aux_targets(model, batch, cfg, adapter)
        ce = _token_loss(cfg, logits, labels)
        loss = ce + cfg.router_aux_loss_coef * aux
        return loss, {"loss": float(loss.detach()), "cross_entropy": float(ce.detach()), "router_aux_loss": float(aux.detach()), "total_loss": float(loss.detach())}

    def validation_step(self, model, batch, cfg, components, adapter=None):
        logits, aux, labels = self._logits_aux_targets(model, batch, cfg, adapter)
        ce = _token_loss(cfg, logits, labels)
        total = ce + cfg.router_aux_loss_coef * aux
        return {"loss": float(total.detach()), "cross_entropy": float(ce.detach()), "router_aux_loss": float(aux.detach()), "total_loss": float(total.detach())}

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits, aux, labels = self._logits_aux_targets(model, batch, cfg, adapter)
        ce = _token_loss(cfg, logits, labels)
        total = ce + cfg.router_aux_loss_coef * aux
        metrics = {"loss": float(total.detach()), "cross_entropy": float(ce.detach()), "router_aux_loss": float(aux.detach()), "total_loss": float(total.detach())}
        return EvaluationStepResult(metrics, EvaluationRecord(logits, labels, mask=labels != cfg.ignore_index, sample_ids=batch.get("sample_ids"), metadata=batch))

    def summarize_metrics(self, metrics):
        if "cross_entropy" in metrics:
            metrics["ppl"] = math.exp(min(metrics["cross_entropy"], 50.0))
        return metrics
