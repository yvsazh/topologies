"""Task-agnostic teacher/student lifecycle with explicit execution adapters."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .contracts import ModelCall


class TeacherCache:
    """Bounded cache using an explicit backend (memory by default)."""

    def __init__(
        self, max_items: int = 1024, *, max_bytes: int = 256 * 1024 * 1024, backend: Any = None
    ):
        if max_items < 1 or max_bytes < 1:
            raise ValueError("cache bounds must be positive")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.backend = backend or _MemoryCacheBackend()

    @staticmethod
    def fingerprint(model: Any) -> str:
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            if torch.is_tensor(tensor):
                x = tensor.detach().cpu().contiguous()
                h.update(name.encode())
                h.update(str((tuple(x.shape), x.dtype)).encode())
                h.update(x.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    @classmethod
    def verified_key(
        cls,
        *,
        teacher_model: Any,
        preprocessing: str,
        sample: str,
        output: str,
        alignment: str,
        augmentation: str,
        stochastic: bool = False,
    ):
        return cls.key(
            teacher=cls.fingerprint(teacher_model),
            preprocessing=preprocessing,
            sample=sample,
            output=output,
            alignment=alignment,
            augmentation=augmentation,
            stochastic=stochastic,
        )

    @staticmethod
    def key(
        *,
        teacher: str,
        preprocessing: str,
        sample: str,
        output: str,
        alignment: str,
        augmentation: str,
        stochastic: bool = False,
    ):
        if stochastic:
            raise ValueError("teacher output caching refused for stochastic augmentation")
        if not all((teacher, preprocessing, sample, output, alignment, augmentation)):
            raise ValueError(
                "cache identity requires teacher/preprocessing/sample/output/alignment/augmentation IDs"
            )
        return hashlib.sha256(
            f"{teacher}\0{preprocessing}\0{sample}\0{output}\0{alignment}\0{augmentation}".encode()
        ).hexdigest()

    def get(self, key):
        return self.backend.get(key)

    def put(self, key, value):
        if self.backend.count >= self.max_items and not self.backend.contains(key):
            raise RuntimeError("teacher cache item bound exceeded")
        self.backend.put(key, value, max_bytes=self.max_bytes)


class _MemoryCacheBackend:
    def __init__(self):
        self.items = {}

    @property
    def count(self):
        return len(self.items)

    def contains(self, key):
        return key in self.items

    def get(self, key):
        return self.items.get(key)

    def put(self, key, value, max_bytes):
        val = value.detach().cpu() if torch.is_tensor(value) else value
        size = sum(x.numel() * x.element_size() for x in _tensors(val))
        used = sum(
            sum(x.numel() * x.element_size() for x in _tensors(v))
            for v in self.items.values()
            if v is not None
        )
        prior = self.items.get(key)
        used -= sum(x.numel() * x.element_size() for x in _tensors(prior))
        if used + size > max_bytes:
            raise RuntimeError("teacher cache byte bound exceeded")
        self.items[key] = val


def _tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _tensors(v)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    return value


class DiskTeacherCacheBackend:
    """Atomic tensor-tree file cache, bounded by total on-disk bytes."""

    def __init__(self, path: str | Path, *, max_disk_bytes: int = 2 * 1024 * 1024 * 1024):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_disk_bytes = max_disk_bytes
        if max_disk_bytes < 1:
            raise ValueError("max_disk_bytes must be positive")

    def _file(self, key):
        return self.path / (hashlib.sha256(str(key).encode()).hexdigest() + ".pt")

    @property
    def count(self):
        return sum(1 for p in self.path.glob("*.pt") if p.is_file())

    def contains(self, key):
        return self._file(key).is_file()

    def get(self, key):
        path = self._file(key)
        return torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None

    def put(self, key, value, max_bytes):
        tensors = list(_tensors(value))
        if not tensors:
            raise TypeError("disk cache currently stores tensor outputs/trees only")
        prior = self._file(key)
        used = sum(p.stat().st_size for p in self.path.glob("*.pt") if p.is_file()) - (
            prior.stat().st_size if prior.exists() else 0
        )
        with tempfile.NamedTemporaryFile(
            dir=self.path, prefix=".cache-", suffix=".tmp", delete=False
        ) as f:
            temp = Path(f.name)
        try:
            torch.save(value, temp)
            size = temp.stat().st_size
            if size > max_bytes or used + size > self.max_disk_bytes:
                raise RuntimeError("teacher cache disk-byte bound exceeded")
            os.replace(temp, prior)
        finally:
            if temp.exists():
                temp.unlink()


class FeatureCapture:
    """Scoped bounded latest-value forward hooks; context exit always removes hooks."""

    def __init__(self, model: Any, modules: tuple[str, ...], *, max_bytes: int = 64 * 1024 * 1024):
        self.model = model
        self.modules = modules
        self.max_bytes = max_bytes
        self.values = {}
        self.handles = []

    def __enter__(self):
        named = dict(self.model.named_modules())
        missing = set(self.modules) - set(named)
        if missing:
            raise ValueError(f"unknown feature module(s): {sorted(missing)}")

        def hook(name):
            def capture(_module, _inputs, output):
                self.values.pop(name, None)
                tensors = []

                def collect(v):
                    if torch.is_tensor(v):
                        tensors.append(v)
                    elif isinstance(v, (tuple, list)):
                        [collect(x) for x in v]
                    elif isinstance(v, dict):
                        [collect(x) for x in v.values()]

                collect(output)
                size = sum(t.numel() * t.element_size() for t in tensors)
                if size > self.max_bytes:
                    raise RuntimeError(f"feature {name!r} exceeds capture bound")
                self.values[name] = output

            return capture

        self.handles = [named[n].register_forward_hook(hook(n)) for n in self.modules]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.values.clear()


def _shared(student: Any, teacher: Any) -> bool:
    return bool(
        {id(x) for x in student.modules()} & {id(x) for x in teacher.modules()}
        or {id(x) for x in student.parameters()} & {id(x) for x in teacher.parameters()}
    )


class DistillationRun:
    def __init__(
        self,
        student: Any,
        teacher: Any,
        train_data: Any,
        *,
        objective: Callable[..., Any],
        optimizer: Any = None,
        adapter: Callable[[Any], Any] | None = None,
        student_call: Callable[[Any], ModelCall] | None = None,
        teacher_call: Callable[[Any], ModelCall] | None = None,
        target: Callable[[Any], Any] | None = None,
        mask: Callable[[Any], Any] | None = None,
        metadata: Callable[[Any], Any] | None = None,
        student_device: Any = None,
        teacher_device: Any = None,
        aligners: Any = None,
        cache: TeacherCache | None = None,
        cache_identity: Callable[[Any], dict[str, str]] | None = None,
        student_feature_modules: tuple[str, ...] = (),
        teacher_feature_modules: tuple[str, ...] = (),
        feature_max_bytes: int = 64 * 1024 * 1024,
        teacher_reference: str | None = None,
        resume_contract: dict[str, Any] | None = None,
    ):
        if _shared(student, teacher):
            raise ValueError(
                "teacher and student share modules/parameters; separate ownership is required"
            )
        self.student, self.teacher, self.train_data, self.objective = (
            student,
            teacher,
            train_data,
            objective,
        )
        self.optimizer, self.adapter = optimizer, adapter
        self.student_call = student_call or (
            lambda b: ModelCall((b[0],) if isinstance(b, (tuple, list)) else (b,))
        )
        self.teacher_call = teacher_call or self.student_call
        self.target_fn, self.mask_fn, self.metadata_fn = target, mask, metadata
        self.student_device, self.teacher_device = student_device, teacher_device
        self.aligners = aligners
        self.cache = cache
        self.cache_identity = cache_identity
        self.student_feature_modules = student_feature_modules
        self.teacher_feature_modules = teacher_feature_modules
        self.feature_max_bytes = feature_max_bytes
        self.teacher_reference = (
            teacher_reference or f"{type(teacher).__module__}.{type(teacher).__qualname__}"
        )
        self.resume_contract = dict(resume_contract or {})
        self.prepared = False
        self.epoch = 0
        self.trainer = None

    def prepare(self):
        owned = (
            {id(p) for g in self.optimizer.param_groups for p in g["params"]}
            if self.optimizer is not None
            else set()
        )
        if any(id(p) in owned for p in self.teacher.parameters()):
            raise ValueError("student optimizer contains teacher parameters")
        from topologies.training.engine import Trainer

        recipe = _DistillationRecipe(self)
        initial_device = next(self.student.parameters(), torch.empty(0)).device
        self.trainer = Trainer(
            self.student,
            self.train_data,
            recipe=recipe,
            components={"teacher": self.teacher, "aligners": self.aligners},
            optimizer=self.optimizer or "auto",
            config={"max_epochs": 1, "device": str(self.student_device or initial_device)},
        )
        self.optimizer = self.trainer.optimizers.get("default")
        self.student.__topologies_optimization_session__ = self
        self.prepared = True
        return self

    def fit(self, epochs=1):
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if not self.prepared:
            self.prepare()
        teacher_modes = [(m, m.training) for m in self.teacher.modules()]
        teacher_flags = [(p, p.requires_grad) for p in self.teacher.parameters()]
        student_device = next(self.student.parameters(), torch.empty(0)).device
        teacher_device = next(self.teacher.parameters(), torch.empty(0)).device
        try:
            if self.student_device is not None:
                self.student.to(self.student_device)
            self.teacher.to(self.teacher_device or self.trainer.device)
            if self.aligners is not None:
                self.aligners.to(self.trainer.device)
            self.teacher.eval()
            for p, _ in teacher_flags:
                p.requires_grad_(False)
            self.trainer.config.max_epochs = int(epochs)
            self.trainer.fit()
            self.epoch += epochs
        finally:
            for p, flag in teacher_flags:
                p.requires_grad_(flag)
            for mod, flag in teacher_modes:
                mod.training = flag
            if self.student_device is None:
                self.student.to(student_device)
            if self.teacher_device is None:
                self.teacher.to(teacher_device)
            if self.aligners is not None and self.student_device is None:
                self.aligners.to(student_device)
        return self

    @staticmethod
    def _callable_id(fn):
        return (
            None
            if fn is None
            else f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', type(fn).__qualname__)}"
        )

    def get_transformation_state(self):
        callables = (
            self.objective,
            self.adapter,
            self.student_call,
            self.teacher_call,
            self.target_fn,
            self.mask_fn,
            self.metadata_fn,
            self.cache_identity,
        )
        if not self.resume_contract and any(
            fn is not None
            and ("<lambda>" in self._callable_id(fn) or "<locals>" in self._callable_id(fn))
            for fn in callables
        ):
            raise ValueError(
                "checkpointing local/lambda distillation callables requires an explicit resume_contract"
            )
        return {
            "kind": "generic-distillation-v1",
            "teacher_reference": self.teacher_reference,
            "teacher_fingerprint": TeacherCache.fingerprint(self.teacher),
            "objective": self._callable_id(self.objective),
            "adapter": self._callable_id(self.adapter),
            "student_call": self._callable_id(self.student_call),
            "teacher_call": self._callable_id(self.teacher_call),
            "target": self._callable_id(self.target_fn),
            "mask": self._callable_id(self.mask_fn),
            "metadata": self._callable_id(self.metadata_fn),
            "cache_identity": self._callable_id(self.cache_identity),
            "student_features": self.student_feature_modules,
            "teacher_features": self.teacher_feature_modules,
            "feature_max_bytes": self.feature_max_bytes,
            "student_device": str(self.student_device),
            "teacher_device": str(self.teacher_device),
            "aligner_type": None
            if self.aligners is None
            else f"{type(self.aligners).__module__}.{type(self.aligners).__qualname__}",
            "cache_backend": None
            if self.cache is None
            else f"{type(self.cache.backend).__module__}.{type(self.cache.backend).__qualname__}",
            "cache_max_items": None if self.cache is None else self.cache.max_items,
            "cache_max_bytes": None if self.cache is None else self.cache.max_bytes,
            "resume_contract": self.resume_contract,
            "aligners": self.aligners.state_dict() if self.aligners is not None else None,
            "epoch": self.epoch,
        }

    def validate_transformation_state(self, state):
        expected = self.get_transformation_state()
        for key, value in expected.items():
            if key in ("aligners", "epoch"):
                continue
            if state.get(key) != value:
                raise ValueError(f"distillation resume contract mismatch for {key!r}")

    def load_transformation_state(self, state):
        if not self.prepared:
            self.prepare()
        self.validate_transformation_state(state)
        if self.aligners is not None:
            if state.get("aligners") is None:
                raise ValueError("checkpoint missing alignment projection state")
            self.aligners.load_state_dict(state["aligners"], strict=True)
        elif state.get("aligners") is not None:
            raise ValueError("checkpoint contains alignment state but run has no aligners")
        self.epoch = int(state.get("epoch", 0))
        return self

    def state_dict(self):
        return self.get_transformation_state()

    def load_state_dict(self, state):
        return self.load_transformation_state(state)

    def evaluate(self, data, adapter=None):
        adapter = adapter or self.student_call
        was = [(m, m.training) for m in self.student.modules()]
        self.student.eval()
        try:
            with torch.no_grad():
                return [self.student(*c.args, **c.kwargs) for c in (adapter(b) for b in data)]
        finally:
            for m, flag in was:
                m.training = flag

    def resume(self, path, *, map_location="cpu", trust_checkpoint=False):
        if not self.prepared:
            self.prepare()
        from topologies.training.checkpoint import read_checkpoint

        payload = read_checkpoint(path, map_location="cpu", trust_checkpoint=trust_checkpoint)
        transformation = payload.get("transformation_state")
        if transformation is None:
            raise ValueError("distillation checkpoint is missing transformation-state contract")
        self.validate_transformation_state(transformation)
        self.trainer._do_resume(path, trust_checkpoint=trust_checkpoint)
        return self

    def save(self, path):
        if not self.prepared:
            self.prepare()
        return self.trainer.save(str(path))

    def save_student(self, path, *, model_factory, metadata=None):
        from types import SimpleNamespace

        from .artifacts import save_artifact

        result = SimpleNamespace(
            model=self.student,
            backend="distillation-student",
            config={"kind": "distilled-student"},
            validation="not-evaluated",
        )
        meta = {
            "role": "student-deployment",
            "teacher_reference": self.teacher_reference,
            "teacher_fingerprint": TeacherCache.fingerprint(self.teacher),
            **(metadata or {}),
        }
        return save_artifact(result, path, model_factory=model_factory, metadata=meta)


from topologies.training.recipe_base import Recipe


class _DistillConfig(
    __import__("topologies.training.recipe_base", fromlist=["RecipeConfig"]).RecipeConfig
):
    pass


class _DistillationRecipe(Recipe):
    """Trainer loss recipe: owns objective calculation, never the training loop."""

    name = "generic-distillation"
    level = "loss"
    ConfigSchema = _DistillConfig
    ComponentsSchema = None

    def __init__(self, run):
        self.run = run

    def default_config(self):
        return {}

    def expected_batch_spec(self):
        return {"_distill_batch": "application-defined batch"}

    def input_keys(self):
        return ["_distill_batch"]

    def target_keys(self):
        return set()

    def default_training_policy(self):
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.0, "scheduler": None}

    def optimizer_params(self, optimizer_name, model, components):
        return list(model.parameters()) + (
            list(self.run.aligners.parameters()) if self.run.aligners is not None else []
        )

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        r = self.run
        batch = batch.get("_distill_batch", batch)
        if r.adapter:
            sc, tc, y, m, meta = r.adapter(batch)
        else:
            sc, tc = r.student_call(batch), r.teacher_call(batch)
            y = r.target_fn(batch) if r.target_fn else None
            m = r.mask_fn(batch) if r.mask_fn else None
            meta = r.metadata_fn(batch) if r.metadata_fn else None
        if not isinstance(sc, ModelCall):
            sc = ModelCall(*sc)
        if not isinstance(tc, ModelCall):
            tc = ModelCall(*tc)
        scapture = (
            FeatureCapture(model, r.student_feature_modules, max_bytes=r.feature_max_bytes)
            if r.student_feature_modules
            else None
        )
        tcapture = (
            FeatureCapture(r.teacher, r.teacher_feature_modules, max_bytes=r.feature_max_bytes)
            if r.teacher_feature_modules
            else None
        )
        if scapture:
            scapture.__enter__()
        try:
            pred = model(*sc.args, **sc.kwargs)
            cache_key = None
            ref = None
            if r.cache is not None:
                if r.cache_identity is None:
                    raise ValueError(
                        "cache requires cache_identity(batch) with preprocessing/sample/output/alignment/augmentation IDs"
                    )
                ident = r.cache_identity(batch)
                cache_key = TeacherCache.verified_key(
                    teacher_model=r.teacher,
                    stochastic=bool(ident.get("stochastic", False)),
                    **{
                        k: ident[k]
                        for k in ("preprocessing", "sample", "output", "alignment", "augmentation")
                    },
                )
                ref = r.cache.get(cache_key)
                try:
                    teacher_device = next(r.teacher.parameters()).device
                except StopIteration:
                    teacher_device = torch.device("cpu")
                ref = _to_device(ref, teacher_device) if ref is not None else None
            if ref is None:
                if tcapture:
                    tcapture.__enter__()
                try:
                    with torch.no_grad():
                        ref = r.teacher(*tc.args, **tc.kwargs)
                    if cache_key is not None:
                        r.cache.put(cache_key, ref)
                finally:
                    if tcapture:
                        tcapture.__exit__(None, None, None)
            if scapture or tcapture:
                meta = dict(meta or {})
                if scapture:
                    meta["student_features"] = dict(scapture.values)
                if tcapture:
                    meta["teacher_features"] = dict(tcapture.values)
        finally:
            if scapture:
                scapture.__exit__(None, None, None)
        loss = r.objective(pred, ref, y, m, meta)
        if not torch.is_tensor(loss) or loss.ndim:
            raise ValueError("objective must return a scalar tensor")
        return loss, {"distillation_loss": loss.detach()}


def distill(
    student: Any,
    teacher: Any,
    train_data: Any,
    *,
    objective: Callable[..., Any],
    optimizer: Any = None,
    adapter: Callable[[Any], Any] | None = None,
    epochs: int = 1,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper returning the trained student; objective(pred, ref, targets, mask, metadata)."""
    run = DistillationRun(
        student,
        teacher,
        train_data,
        objective=objective,
        optimizer=optimizer,
        adapter=adapter,
        **kwargs,
    )
    run.fit(epochs)
    return student
