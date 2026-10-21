"""Logit lens and tuned lens (A): decode intermediate residual states through the unembedding.

Requires the ``residual`` role and an adapter ``unembed`` decoder (final norm + head).
The tuned lens trains one affine translator per layer on your data and is a trained
probe: it shows what is decodable with that translator, not what the model computes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection.errors import SemanticsRequired
from topologies.introspection.inputs import is_collection, normalize_inputs
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.techniques import Availability, Technique, builtin
from topologies.introspection.views import finish, heatmap, table
from topologies.visualization.views import ViewSpec


@dataclass
class LensPayload(Payload):
    kind: str
    layers: list[str]
    tokens: list[str] | None
    top_tokens: list[list[list[str]]]    # [L][T] -> top-k token texts
    final_probability: torch.Tensor      # [L, T] p(final prediction) under each layer's lens
    kl_to_final: torch.Tensor            # [L, T]
    translators: dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for i, layer in enumerate(self.layers):
            for t in range(self.kl_to_final.shape[1]):
                rows.append({"layer": layer, "position": t,
                             "token": self.tokens[t] if self.tokens and t < len(self.tokens) else None,
                             "top": " | ".join(self.top_tokens[i][t]),
                             "p_final": float(self.final_probability[i, t]),
                             "kl_to_final": float(self.kl_to_final[i, t])})
        return rows

    def views(self, **options: Any) -> list[ViewSpec]:
        caption = f"{self.kind} lens" + ("; trained translators (a trained probe)"
                                         if self.kind == "tuned" else "")
        return finish([heatmap(f"{self.kind.capitalize()} lens: KL to final distribution",
                               self.kl_to_final, rows=self.layers, cols=self.tokens, caption=caption,
                               x_label="position", y_label="layer"),
                       heatmap("Probability of the final prediction", self.final_probability,
                               rows=self.layers, cols=self.tokens),
                       table("Top tokens per layer (last position)",
                             [{"layer": layer, "top": " | ".join(self.top_tokens[i][-1])}
                              for i, layer in enumerate(self.layers)])], **options)


def _decoder(subject: Any) -> Any:
    decoders = subject.adapter.decoders(subject.model) if subject.adapter is not None else {}
    if "unembed" not in decoders:
        raise SemanticsRequired("the lens needs an adapter 'unembed' decoder (final norm + head); "
                                "use a catalog/HF model or an adapter whose decoders() returns it")
    return decoders["unembed"]


def _states(subject: Any, inputs: Any, layers: list[str]) -> tuple[dict[str, torch.Tensor], Any, Any]:
    trace = subject.inspector.trace(inputs, sites=",".join(layers), keep="tensors",
                                    call_tree=False, max_bytes=None)
    states = {}
    for layer in layers:
        record = max((r for r in trace.records.values() if r.key.path == layer and r.key.op is None),
                     key=lambda r: r.occurrence)
        states[layer] = record.value.float()
    return states, trace.output, trace.metadata.get("tokens")


def _logits(output: Any) -> torch.Tensor:
    from topologies.introspection.objectives import primary

    return primary(output).float()


@builtin
class Lens(Technique):
    name = "lens"
    tier = "A"
    requires = frozenset({"role:residual"})
    result_kind = "introspection.lens"
    summary = "logit lens / tuned lens over the residual stream"

    def check(self, subject: Any) -> Availability:
        base = super().check(subject)
        if not base:
            return base
        try:
            _decoder(subject)
        except SemanticsRequired as exc:
            return Availability(False, str(exc), "an adapter with an unembed decoder")
        return base

    def run(self, subject: Any, inputs: Any = None, *, kind: str = "logit", top_k: int = 5,
            train_data: Any = None, epochs: int = 50, lr: float = 1e-3, example: int = 0,
            translators: dict[str, Any] | None = None, **params: Any) -> Any:
        if kind not in ("logit", "tuned"):
            raise ValueError("kind must be 'logit' or 'tuned'")
        decoder = _decoder(subject)
        layers = subject.index.resolve_modules("role:residual")
        states, output, tokens = _states(subject, inputs, layers)
        final = _logits(output)[example]
        assumptions = [("lens = unembed decoder (adapter: final norm + head) applied to each "
                       "residual state; KL(final || lens)")]
        if kind == "tuned":
            translators = translators or train_tuned_lens(subject, train_data or inputs, layers,
                                                          decoder, epochs=epochs, lr=lr)
            assumptions.append(f"tuned lens: per-layer affine translators trained for {epochs} "
                               "epochs (AdamW) on the given data — a trained probe")
        log_final = final.log_softmax(-1)
        prediction = final.argmax(-1)
        tops, p_final, kls = [], [], []
        with torch.no_grad():
            for layer in layers:
                h = states[layer][example]
                if kind == "tuned":
                    h = translators[layer](h)
                logits = decoder(h).float()
                logp = logits.log_softmax(-1)
                kls.append((log_final.exp() * (log_final - logp)).sum(-1))
                p_final.append(logp.gather(-1, prediction[..., None]).squeeze(-1).exp())
                top = logits.topk(top_k, dim=-1).indices
                tops.append([[_text(subject.tokenizer, int(i)) for i in row] for row in top])
        payload = LensPayload(kind, layers, tokens[example] if tokens else None, tops,
                              torch.stack(p_final), torch.stack(kls),
                              {k: v.state_dict() for k, v in (translators or {}).items()}
                              if kind == "tuned" else {})
        return make_result(self, payload, subject, assumptions=assumptions,
                           provenance={"inputs": input_fingerprint(inputs), "kind": kind})


def _text(tokenizer: Any, token: int) -> str:
    if tokenizer is None:
        return str(token)
    try:
        return tokenizer.decode([token])
    except Exception:  # noqa: BLE001
        return str(token)


def train_tuned_lens(subject: Any, data: Any, layers: list[str], decoder: Any, *, epochs: int = 50,
                     lr: float = 1e-3) -> dict[str, nn.Module]:
    """One affine translator per layer (identity init) minimising KL(final || lens(A h + b))."""
    source = data if is_collection(data) else [data]
    batches = list(normalize_inputs(subject, source, from_data=True))
    cached = [_states(subject, b, layers)[:2] for b in batches]
    width = next(iter(cached[0][0].values())).shape[-1]
    translators = {}
    for layer in layers:
        translator = nn.Linear(width, width)
        with torch.no_grad():
            translator.weight.copy_(torch.eye(width))
            translator.bias.zero_()
        optimizer = torch.optim.AdamW(translator.parameters(), lr=lr, weight_decay=0.0)
        with torch.enable_grad():
            for _ in range(epochs):
                for states, output in cached:
                    target = _logits(output).log_softmax(-1).detach()
                    logp = decoder(translator(states[layer])).float().log_softmax(-1)
                    loss = (target.exp() * (target - logp)).sum(-1).mean()
                    # autograd.grad: the model's own parameters never accumulate .grad
                    params = list(translator.parameters())
                    for p, g in zip(params, torch.autograd.grad(loss, params)):
                        p.grad = g
                    optimizer.step()
        translators[layer] = translator.eval()
    return translators
