"""Persistent magnitude pruning and deliberately narrow structural surgery."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


def magnitude_prune(
    model: Any,
    amount: float,
    *,
    modules: tuple[str, ...] = (),
    global_unstructured: bool = False,
    per_scope: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply persistent magnitude masks to Linear/Conv weights."""
    if not 0 <= amount < 1:
        raise ValueError("amount must be in [0, 1)")
    import torch.nn as nn
    import torch.nn.utils.prune as prune

    selected = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        and (not modules or n in modules)
    ]
    if not selected:
        raise ValueError("No selected Linear/Conv modules to prune")
    scoped = dict(per_scope or {})
    unknown = set(scoped) - {n for n, _ in selected}
    if unknown:
        raise ValueError(f"per_scope contains unselected module paths: {sorted(unknown)}")
    global_modules = [(n, m) for n, m in selected if n not in scoped]
    if global_unstructured and global_modules:
        prune.global_unstructured(
            [(m, "weight") for _, m in global_modules],
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    else:
        for _, module in global_modules:
            prune.l1_unstructured(module, name="weight", amount=amount)
    for name, fraction in scoped.items():
        if not 0 <= fraction < 1:
            raise ValueError(f"per_scope[{name!r}] must be in [0,1)")
        module = dict(selected)[name]
        prune.l1_unstructured(module, name="weight", amount=fraction)
    return {
        name: float((module.weight == 0).sum().item() / module.weight.numel())
        for name, module in selected
    }


class MaskController:
    """Enforce existing pruning masks after optimizer steps; serializable independent state."""

    def __init__(
        self,
        model: Any,
        schedule: list[tuple[int, float]] | None = None,
        *,
        global_unstructured: bool = False,
        per_scope: dict[str, float] | None = None,
    ):
        self.model = model
        self.schedule = tuple(sorted(schedule or ()))
        self.global_unstructured = bool(global_unstructured)
        self.per_scope = dict(per_scope or {})
        self.step = 0
        self._attached = {}
        if any(int(s) < 1 or not 0 <= float(f) < 1 for s, f in self.schedule):
            raise ValueError("schedule milestones must be >=1 and fractions in [0,1)")
        if len({int(s) for s, _ in self.schedule}) != len(self.schedule):
            raise ValueError("schedule milestones must be unique")
        self._ensure_masks()

    def _ensure_masks(self, require: bool = False):
        import torch.nn.utils.prune as prune

        for name, module in self.model.named_modules():
            if name in self.per_scope and not hasattr(module, "weight_orig"):
                prune.identity(module, "weight")
        self.entries = [
            (n, m)
            for n, m in self.model.named_modules()
            if hasattr(m, "weight_orig") and hasattr(m, "weight_mask")
        ]
        if self.per_scope and not set(self.per_scope).issubset(dict(self.entries)):
            raise ValueError("per-scope mask policy paths must refer to masked modules")
        if require and not self.entries:
            raise ValueError("model has no persistent pruning masks; apply magnitude_prune first")

    def _apply_fraction(self, fraction: float):
        groups = []
        if self.per_scope:
            groups = [([entry], self.per_scope.get(entry[0], fraction)) for entry in self.entries]
        elif self.global_unstructured:
            groups = [(self.entries, fraction)]
        else:
            groups = [([entry], fraction) for entry in self.entries]
        for entries, amount in groups:
            amount = float(amount)
            if not 0 <= amount < 1:
                raise ValueError("scheduled sparsity must be in [0,1)")
            tensors = [m.weight_orig.detach().abs().flatten() for _, m in entries]
            if not tensors:
                continue
            scores = torch.cat(tensors)
            prune_n = int(amount * scores.numel())
            mask = torch.ones_like(scores, dtype=torch.bool)
            if prune_n:
                mask[scores.argsort()[:prune_n]] = False
            cursor = 0
            for _, module in entries:
                count = module.weight_orig.numel()
                module.weight_mask.copy_(mask[cursor : cursor + count].view_as(module.weight_mask))
                cursor += count
        self.apply()

    def apply(self) -> None:
        for _, module in self.entries:
            module.weight_orig.data.mul_(module.weight_mask)

    def attach(self, trainer: Any) -> MaskController:
        """Bind enforcement to Trainer's actual optimizer-update boundary."""
        self._ensure_masks(require=True)

        def after_step(_trainer, _name):
            self.step += 1
            due = [fraction for milestone, fraction in self.schedule if milestone == self.step]
            if due:
                self._apply_fraction(due[-1])
            self.apply()

        identity = id(trainer)
        if identity not in self._attached:
            self._attached[identity] = trainer.add_hook("after_optimizer_step", after_step)
        self.apply()
        return self

    def detach(self, trainer: Any) -> None:
        handle = self._attached.pop(id(trainer), None)
        if handle is not None:
            handle.remove()

    def optimizer_step(self, optimizer: Any) -> None:
        optimizer.step()
        self.step += 1
        self.apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "schedule": self.schedule,
            "global_unstructured": self.global_unstructured,
            "per_scope": self.per_scope,
            "masks": {n: m.weight_mask.detach().clone() for n, m in self.entries},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schedule": tuple(tuple(x) for x in self.schedule),
            "global_unstructured": self.global_unstructured,
            "per_scope": self.per_scope,
        }
        found = {k: state.get(k) for k in expected}
        if found["schedule"] is not None:
            found["schedule"] = tuple(tuple(x) for x in found["schedule"])
        if found != expected:
            raise ValueError(
                f"pruning schedule/policy mismatch on resume: expected {expected}, got {found}"
            )
        current = dict(self.entries)
        if set(current) != set(state["masks"]):
            # Recreate persistent parametrizations on a freshly constructed dense topology.
            import torch.nn.utils.prune as prune

            named = dict(self.model.named_modules())
            for name, mask in state["masks"].items():
                if name not in named or named[name].weight.shape != mask.shape:
                    raise ValueError("pruning mask topology mismatch on resume")
                if not hasattr(named[name], "weight_orig"):
                    prune.custom_from_mask(named[name], "weight", mask.to(named[name].weight))
            self._ensure_masks()
            current = dict(self.entries)
        if set(current) != set(state["masks"]):
            raise ValueError("pruning mask topology mismatch on resume")
        for name, mask in state["masks"].items():
            if current[name].weight_mask.shape != mask.shape:
                raise ValueError(f"pruning mask shape mismatch for {name}")
            current[name].weight_mask.copy_(mask.to(current[name].weight_mask))
        self.step = int(state["step"])
        self.apply()


@dataclass(frozen=True)
class StructuralPruningSpec:
    retained: dict[int, tuple[int, ...]]
    handler: str = "linear"

    def reconstruct(self, factory: Callable[[], Any], state_dict: dict | None = None) -> Any:
        model = structural_prune_sequential(
            factory(), {i: torch.tensor(v) for i, v in self.retained.items()}, factory=factory
        )
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model


def structural_prune_sequential(
    model: Any, keep: dict[int, torch.Tensor], *, factory: Callable[[], Any] | None = None
) -> Any:
    """Physically remove hidden Linear channels in Sequential Linear/ReLU chains.

    Requires every Linear output to feed the next Linear directly (only ReLU/Identity may
    intervene); output dimension is fixed. Returns rebuilt model; source stays untouched.
    """
    import torch.nn as nn

    if not isinstance(model, nn.Sequential):
        raise TypeError("structural handler currently requires nn.Sequential")
    layers = list(model.children())
    convs = [(i, m) for i, m in enumerate(layers) if isinstance(m, nn.Conv2d)]
    if convs:
        if len(convs) < 2 or any(isinstance(m, (nn.Conv1d, nn.Conv3d, nn.Linear)) for m in layers):
            raise ValueError(
                "Conv structural handler requires at least two Conv2d layers only; provide a dependency handler"
            )
        if any(m.groups != 1 for _, m in convs):
            raise ValueError(
                "grouped/depthwise Conv2d structural pruning is unsupported; groups must equal one"
            )
        for (ia, first), (ib, second) in itertools.pairwise(convs):
            chain = layers[ia + 1 : ib]
            if (
                len(chain) != 2
                or not isinstance(chain[0], nn.BatchNorm2d)
                or not isinstance(chain[1], nn.ReLU)
            ):
                raise ValueError(
                    "supported Conv dependency must be Conv2d-BatchNorm2d-ReLU-Conv2d; residual/flatten/functional paths require a custom dependency handler"
                )
            if (
                first.out_channels != chain[0].num_features
                or first.out_channels != second.in_channels
            ):
                raise ValueError(
                    "Conv-BatchNorm-Conv channel dimensions disagree; inspect graph or supply custom dependency handler"
                )
        if convs[0][0] != 0 or convs[-1][0] != len(layers) - 1:
            raise ValueError(
                "Conv handler requires a direct image/channel Sequential chain with no leading/trailing flatten, residual, or output adapters; provide a dependency handler"
            )
        selected = {}
        for idx, layer in convs[:-1]:
            if idx not in keep:
                continue
            k = torch.as_tensor(keep[idx], dtype=torch.long, device=layer.weight.device)
            if (
                k.ndim != 1
                or not k.numel()
                or k.unique().numel() != k.numel()
                or k.min() < 0
                or k.max() >= layer.out_channels
            ):
                raise ValueError(f"invalid retained output channel indices for Conv2d layer {idx}")
            selected[idx] = k
        if not selected:
            raise ValueError("no hidden Conv2d layers selected; final output channels are fixed")
        rebuilt = factory() if factory else __import__("copy").deepcopy(model)
        out_layers = list(rebuilt.children())
        for pos, (idx, old) in enumerate(convs):
            inidx = (
                selected.get(
                    convs[pos - 1][0], torch.arange(old.in_channels, device=old.weight.device)
                )
                if pos
                else torch.arange(old.in_channels, device=old.weight.device)
            )
            outidx = selected.get(idx, torch.arange(old.out_channels, device=old.weight.device))
            newconv = nn.Conv2d(
                len(inidx),
                len(outidx),
                old.kernel_size,
                old.stride,
                old.padding,
                old.dilation,
                groups=1,
                bias=old.bias is not None,
                padding_mode=old.padding_mode,
                device=old.weight.device,
                dtype=old.weight.dtype,
            )
            with torch.no_grad():
                newconv.weight.copy_(old.weight[outidx][:, inidx])
                if old.bias is not None:
                    newconv.bias.copy_(old.bias[outidx])
            out_layers[idx] = newconv
            if pos + 1 < len(convs):
                bnidx = idx + 1
                norm = layers[bnidx]
                bn = nn.BatchNorm2d(
                    len(outidx),
                    eps=norm.eps,
                    momentum=norm.momentum,
                    affine=norm.affine,
                    track_running_stats=norm.track_running_stats,
                    device=norm.weight.device if norm.affine else None,
                    dtype=norm.weight.dtype if norm.affine else None,
                )
                with torch.no_grad():
                    if norm.affine:
                        bn.weight.copy_(norm.weight[outidx])
                        bn.bias.copy_(norm.bias[outidx])
                    if norm.track_running_stats:
                        bn.running_mean.copy_(norm.running_mean[outidx])
                        bn.running_var.copy_(norm.running_var[outidx])
                        bn.num_batches_tracked.copy_(norm.num_batches_tracked)
                bn.train(norm.training)
                out_layers[bnidx] = bn
        result = nn.Sequential(*out_layers)
        result._topologies_pruning_provenance = {
            "handler": "sequential-conv2d-bn2d-relu-v1",
            "retained": {str(i): v.cpu().tolist() for i, v in selected.items()},
        }
        result._topologies_pruning_spec = StructuralPruningSpec(
            {i: tuple(map(int, v.cpu().tolist())) for i, v in selected.items()}, "conv2d"
        )
        return result
    linears = [(i, m) for i, m in enumerate(layers) if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        raise ValueError("structural pruning requires at least two Linear layers")
    for (ia, _), (ib, _) in itertools.pairwise(linears):
        if any(
            not isinstance(x, (nn.ReLU, nn.Identity, nn.BatchNorm1d)) for x in layers[ia + 1 : ib]
        ):
            raise ValueError(
                "unsupported dependency between Linear layers; supported chain is Linear-[BatchNorm1d]-ReLU-Linear; provide a dependency handler"
            )
    indices = {}
    for pos, (idx, layer) in enumerate(linears[:-1]):
        k = keep.get(idx)
        if k is None:
            continue
        k = torch.as_tensor(k, dtype=torch.long, device=layer.weight.device)
        if (
            k.ndim != 1
            or k.numel() == 0
            or k.unique().numel() != k.numel()
            or k.min() < 0
            or k.max() >= layer.out_features
        ):
            raise ValueError(f"invalid retained output indices for Sequential layer {idx}")
        indices[idx] = k
    if not indices:
        raise ValueError("no hidden Linear layers selected")
    new = factory() if factory else __import__("copy").deepcopy(model)
    new_layers = list(new.children())
    for pos, (idx, old) in enumerate(linears):
        outidx = indices.get(idx, torch.arange(old.out_features, device=old.weight.device))
        inidx = (
            indices.get(
                linears[pos - 1][0], torch.arange(old.in_features, device=old.weight.device)
            )
            if pos
            else torch.arange(old.in_features, device=old.weight.device)
        )
        replacement = nn.Linear(
            len(inidx),
            len(outidx),
            bias=old.bias is not None,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(old.weight[outidx][:, inidx])
            if old.bias is not None:
                replacement.bias.copy_(old.bias[outidx])
        new_layers[idx] = replacement
        for ni in range(idx + 1, linears[pos + 1][0] if pos + 1 < len(linears) else len(layers)):
            norm = layers[ni]
            if isinstance(norm, nn.BatchNorm1d):
                bn = nn.BatchNorm1d(
                    len(outidx),
                    eps=norm.eps,
                    momentum=norm.momentum,
                    affine=norm.affine,
                    track_running_stats=norm.track_running_stats,
                    device=norm.weight.device if norm.affine else None,
                    dtype=norm.weight.dtype if norm.affine else None,
                )
                with torch.no_grad():
                    if norm.affine:
                        bn.weight.copy_(norm.weight[outidx])
                        bn.bias.copy_(norm.bias[outidx])
                    if norm.track_running_stats:
                        bn.running_mean.copy_(norm.running_mean[outidx])
                        bn.running_var.copy_(norm.running_var[outidx])
                        bn.num_batches_tracked.copy_(norm.num_batches_tracked)
                bn.train(norm.training)
                new_layers[ni] = bn
    result = nn.Sequential(*new_layers)
    result._topologies_pruning_provenance = {
        "handler": "sequential-linear-norm-relu-v1",
        "retained": {str(k): v.cpu().tolist() for k, v in indices.items()},
    }
    result._topologies_pruning_spec = StructuralPruningSpec(
        {k: tuple(int(x) for x in v.cpu().tolist()) for k, v in indices.items()}, "linear"
    )
    return result
