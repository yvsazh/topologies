"""Sparse dictionary learning (U, experimental): a sparse autoencoder over site activations.

ReLU encoder, an L1 or TopK sparsity penalty, unit-norm decoder rows and resampling of
dead features. Reported: the dictionary, reconstruction error, fraction of variance
explained, the effect on the model output when the reconstruction is spliced in
(through ``Replace``), and the top examples for each feature. Feature activations are
correlational.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection.errors import SemanticsRequired
from topologies.introspection.interventions import Replace
from topologies.introspection.objectives import primary
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.reducers import batch_positions_features, feature_last
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.techniques._features import batches
from topologies.introspection.views import bars, finish, heatmap, table
from topologies.visualization.views import ViewSpec


class SparseAutoencoder(nn.Module):
    def __init__(self, width: int, size: int, *, sparsity: str = "l1", k: int = 8):
        super().__init__()
        self.encoder = nn.Linear(width, size)
        self.decoder = nn.Linear(size, width)
        self.sparsity, self.k = sparsity, k
        with torch.no_grad():
            self.decoder.weight.copy_(nn.functional.normalize(torch.randn(width, size), dim=0))
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        codes = torch.relu(self.encoder(x - self.decoder.bias))
        if self.sparsity == "topk":
            top = codes.topk(min(self.k, codes.shape[-1]), dim=-1)
            codes = torch.zeros_like(codes).scatter(-1, top.indices, top.values)
        return codes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.encode(x)
        return self.decoder(codes), codes

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.copy_(nn.functional.normalize(self.decoder.weight, dim=0))


def train_sae(rows: torch.Tensor, size: int, *, sparsity: str = "l1", k: int = 8,
              l1: float = 1e-3, steps: int = 500, batch_size: int = 256, lr: float = 1e-3,
              resample_every: int = 200, seed: int = 0) -> tuple[SparseAutoencoder, dict[str, Any]]:
    torch.manual_seed(seed)
    rows = rows.float()
    sae = SparseAutoencoder(rows.shape[1], size, sparsity=sparsity, k=k)
    with torch.no_grad():
        sae.decoder.bias.copy_(rows.mean(0))
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    fired = torch.zeros(size)
    resampled = 0
    with torch.enable_grad():
        for step in range(1, steps + 1):
            batch = rows[torch.randint(0, rows.shape[0], (min(batch_size, rows.shape[0]),),
                                       generator=g)]
            recon, codes = sae(batch)
            loss = (recon - batch).pow(2).sum(-1).mean()
            if sparsity == "l1":
                loss = loss + l1 * codes.abs().sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            fired += (codes > 0).float().sum(0).detach()
            if resample_every and step % resample_every == 0 and step < steps:
                dead = (fired == 0).nonzero().flatten()
                if dead.numel():
                    resampled += int(dead.numel())
                    with torch.no_grad():
                        err = (sae(rows)[0] - rows).pow(2).sum(-1)
                        picks = torch.multinomial(err / err.sum(), dead.numel(), replacement=True,
                                                  generator=g)
                        directions = nn.functional.normalize(rows[picks] - sae.decoder.bias, dim=-1)
                        sae.decoder.weight[:, dead] = directions.T
                        sae.encoder.weight[dead] = directions * 0.2
                        sae.encoder.bias[dead] = 0.0
                fired.zero_()
    with torch.no_grad():
        recon, codes = sae(rows)
        mse = float((recon - rows).pow(2).sum(-1).mean())
        total = float((rows - rows.mean(0)).pow(2).sum(-1).mean())
        stats = {"mse": mse, "relative_error": mse / max(total, 1e-12),
                 "fraction_variance_explained": 1 - mse / max(total, 1e-12),
                 "l0": float((codes > 0).float().sum(-1).mean()),
                 "dead_features": int(((codes > 0).sum(0) == 0).sum()), "resampled": resampled}
    return sae.eval(), stats


@dataclass
class DictionaryPayload(Payload):
    site: str
    size: int
    stats: dict[str, Any]
    decoder: torch.Tensor                     # [size, width]
    feature_frequency: torch.Tensor           # [size]
    top_examples: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    splice: dict[str, float] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        return [{"site": self.site, "size": self.size, **self.stats, **{f"splice_{k}": v for k, v in
                                                                        self.splice.items()}}]

    def views(self, **options: Any) -> list[ViewSpec]:
        order = self.feature_frequency.argsort(descending=True)[:30].tolist()
        rows = [{"feature": f, "rank": i, **e} for f, entries in list(self.top_examples.items())[:8]
                for i, e in enumerate(entries)]
        return finish([bars("Feature firing frequency", [str(i) for i in order],
                            [float(self.feature_frequency[i]) for i in order],
                            caption="experimental sparse autoencoder; features are correlational"),
                       heatmap("Dictionary (decoder rows)", self.decoder[:64], x_label="unit",
                               y_label="feature"),
                       table("Top examples per feature", rows[:60])], **options)


@builtin
class Dictionary(Technique):
    name = "dictionary"
    tier = "U"
    result_kind = "introspection.dictionary"
    summary = "sparse autoencoder over site activations (experimental)"

    def run(self, subject: Any, inputs: Any = None, *, site: str = "", size: int | None = None,
            sparsity: str = "l1", k: int = 8, l1: float = 1e-3, steps: int = 500,
            batch_size: int = 256, lr: float = 1e-3, max_batches: int | None = None,
            max_rows: int = 200_000, top_k: int = 5, seed: int = 0, **params: Any) -> Any:
        if sparsity not in ("l1", "topk"):
            raise ValueError("sparsity must be 'l1' or 'topk'")
        path = subject.index.resolve_modules(site)[0]
        rows, owners, first = [], [], None
        example = 0
        for batch in batches(subject, inputs, max_batches):
            trace = subject.inspector.trace(batch, sites=path or "<root>", keep="tensors",
                                            call_tree=False, max_bytes=None)
            record = max((r for r in trace.records.values() if r.key.op is None),
                         key=lambda r: r.occurrence)
            if record.axes is None or record.axes.feature_axis is None:
                raise SemanticsRequired(f"dictionary learning at {site!r} needs a feature axis; "
                                        f"insp.declare({site!r}, axes='b t d')")
            if record.axes.batch_axis is not None:
                bpf = batch_positions_features(record.value.float(), record.axes, site=path)
                value = bpf.reshape(-1, bpf.shape[-1])
                owners.extend((example + b, p) for b in range(bpf.shape[0])
                              for p in range(bpf.shape[1]))
                example += bpf.shape[0]
            else:
                value = feature_last(record.value.float(), record.axes, site=path)
                owners.extend((example + r, 0) for r in range(value.shape[0]))
                example += value.shape[0]
            rows.append(value)
            first = first or batch
            if sum(len(x) for x in rows) >= max_rows:
                break
        data = torch.cat(rows)[:max_rows]
        size = size or 4 * data.shape[1]
        sae, stats = train_sae(data, size, sparsity=sparsity, k=k, l1=l1, steps=steps,
                               batch_size=batch_size, lr=lr, seed=seed)
        with torch.no_grad():
            codes = sae.encode(data)
        frequency = (codes > 0).float().mean(0)
        tops: dict[int, list[dict[str, Any]]] = {}
        for feature in frequency.argsort(descending=True)[:16].tolist():
            values, idx = codes[:, feature].topk(min(top_k, codes.shape[0]))
            tops[feature] = [{"value": float(v), "example": owners[i][0], "position": owners[i][1]}
                             for v, i in zip(values.tolist(), idx.tolist())]
        splice = _splice_effect(subject, first, path, sae, record.axes) if first is not None else {}
        payload = DictionaryPayload(path or "<root>", size, stats, sae.decoder.weight.T.detach(),
                                    frequency, tops, splice)
        return make_result(self, payload, subject, assumptions=[
            ("experimental: sparse autoencoder (ReLU encoder, "
            f"{'L1 penalty ' + str(l1) if sparsity == 'l1' else 'TopK k=' + str(k)}, unit-norm "
            "decoder, dead-feature resampling)"),
            "splice effect: model output with the site replaced by the SAE reconstruction",
            "features are correlational descriptions of activations"],
            provenance={"inputs": input_fingerprint(inputs), "rows": int(data.shape[0])})


def _splice_effect(subject: Any, batch: Any, path: str, sae: SparseAutoencoder, axes: Any
                   ) -> dict[str, float]:
    def reconstruct(value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        flat = value.movedim(axes.feature_axis % value.ndim, -1)
        recon = sae(flat.float())[0].to(value.dtype)
        return recon.movedim(-1, axes.feature_axis % value.ndim)
    clean = primary(subject.inspector.run_with(batch).output).float()
    spliced = primary(subject.inspector.run_with(batch, Replace(path or "<root>", reconstruct))
                      .output).float()
    out = {"relative_l2": float((spliced - clean).norm() / clean.norm().clamp_min(1e-12))}
    if clean.ndim >= 2 and clean.shape[-1] > 1:
        p, q = clean.log_softmax(-1), spliced.log_softmax(-1)
        out["kl"] = float((p.exp() * (p - q)).sum(-1).mean())
    return out
