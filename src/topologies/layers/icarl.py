import torch
from torch import nn
from torch.nn import functional as F


class ICaRL(nn.Module):
    component_id = "icarl"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("ICaRL does not accept constructor arguments")

    @staticmethod
    def _floating_tensor(value, name, dimensions):
        tensor = torch.as_tensor(value)
        if tensor.ndim != dimensions:
            raise ValueError(f"{name} must have {dimensions} dimensions")
        if any(size == 0 for size in tensor.shape):
            raise ValueError(f"{name} dimensions must be nonempty")
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must be a floating point tensor")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")
        if (torch.linalg.vector_norm(tensor, dim=-1) == 0).any().item():
            raise ValueError(f"{name} rows must have nonzero norms")
        return tensor

    @staticmethod
    def _integer_tensor(value, name, dimensions):
        tensor = torch.as_tensor(value)
        if tensor.ndim != dimensions:
            raise ValueError(f"{name} must have {dimensions} dimensions")
        if any(size == 0 for size in tensor.shape):
            raise ValueError(f"{name} dimensions must be nonempty")
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if tensor.dtype not in integer_types:
            raise ValueError(f"{name} must contain integer values")
        return tensor.to(dtype=torch.long)

    def forward(self, query_features, exemplar_features=None, exemplar_labels=None):
        query_features = self._floating_tensor(query_features, "query_features", 2)
        if exemplar_features is None and exemplar_labels is None:
            return F.normalize(query_features, p=2, dim=-1)
        if exemplar_features is None or exemplar_labels is None:
            raise ValueError("exemplar_features and exemplar_labels must be supplied together")

        exemplar_features = self._floating_tensor(exemplar_features, "exemplar_features", 2)
        exemplar_labels = self._integer_tensor(exemplar_labels, "exemplar_labels", 1)

        if query_features.shape[1] != exemplar_features.shape[1]:
            raise ValueError("query and exemplar feature dimensions must match")
        if exemplar_features.shape[0] != exemplar_labels.shape[0]:
            raise ValueError("exemplar_labels must match the exemplar count")
        if query_features.device != exemplar_features.device:
            raise ValueError("query and exemplar features must be on the same device")
        if query_features.dtype != exemplar_features.dtype:
            raise ValueError("query and exemplar features must have the same dtype")

        exemplar_labels = exemplar_labels.to(device=exemplar_features.device)
        class_ids = torch.unique(exemplar_labels, sorted=True)
        expected_ids = torch.arange(class_ids.numel(), device=class_ids.device)
        if not torch.equal(class_ids, expected_ids):
            raise ValueError("exemplar class IDs must be contiguous and start at zero")

        query = F.normalize(query_features, p=2, dim=-1)
        exemplars = F.normalize(exemplar_features, p=2, dim=-1)
        assignments = F.one_hot(exemplar_labels, num_classes=class_ids.numel()).to(
            dtype=exemplars.dtype
        )
        counts = assignments.sum(dim=0).unsqueeze(-1)
        means = assignments.transpose(0, 1) @ exemplars / counts
        if (torch.linalg.vector_norm(means, dim=-1) == 0).any().item():
            raise ValueError("each exemplar class must have a nonzero mean feature")
        prototypes = F.normalize(means, p=2, dim=-1)
        distances = torch.linalg.vector_norm(query.unsqueeze(1) - prototypes.unsqueeze(0), dim=-1)
        return -distances

    @staticmethod
    def herding(features, m):
        features = ICaRL._floating_tensor(features, "features", 2)
        if not isinstance(m, int) or isinstance(m, bool) or m < 0:
            raise ValueError("m must be a nonnegative integer")
        if m > features.shape[0]:
            raise ValueError("m cannot exceed the number of input features")
        normalized = F.normalize(features, p=2, dim=-1)
        mean = normalized.mean(dim=0)
        if torch.linalg.vector_norm(mean).item() == 0:
            raise ValueError("features must have a nonzero normalized mean")
        mean = F.normalize(mean, p=2, dim=0)
        if m == 0:
            return torch.empty((0,), dtype=torch.long, device=features.device)

        selected = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        selected_sum = torch.zeros_like(mean)
        indices = []
        for k in range(1, m + 1):
            candidate_means = F.normalize((normalized + selected_sum) / k, p=2, dim=-1)
            distances = torch.linalg.vector_norm(candidate_means - mean, dim=-1)
            index = distances.masked_fill(selected, torch.inf).argmin()
            indices.append(index)
            selected[index] = True
            selected_sum = selected_sum + normalized[index]
        return torch.stack(indices)

    @staticmethod
    def reduce_exemplar_set(exemplars, m):
        exemplars = torch.as_tensor(exemplars)
        if exemplars.ndim == 0 or exemplars.shape[0] == 0:
            raise ValueError("exemplars must have a nonempty leading dimension")
        if not isinstance(m, int) or isinstance(m, bool) or m < 0:
            raise ValueError("m must be a nonnegative integer")
        if m > exemplars.shape[0]:
            raise ValueError("m cannot exceed the number of exemplars")
        return exemplars[:m]

    @staticmethod
    def representation_loss(logits, labels, old_targets):
        logits = torch.as_tensor(logits)
        if logits.ndim != 2 or any(size == 0 for size in logits.shape):
            raise ValueError("logits must have nonempty shape [B, C]")
        if not torch.is_floating_point(logits):
            raise ValueError("logits must be a floating point tensor")
        if not torch.isfinite(logits).all().item():
            raise ValueError("logits must contain only finite values")

        labels = ICaRL._integer_tensor(labels, "labels", 1)
        old_targets = torch.as_tensor(old_targets)
        if old_targets.ndim != 2 or old_targets.shape[0] != logits.shape[0]:
            raise ValueError("old_targets must have shape [B, S] matching logits")
        old_count = old_targets.shape[1]
        if old_count > logits.shape[1]:
            raise ValueError("old_targets class count cannot exceed the logits class count")
        if not torch.is_floating_point(old_targets):
            raise ValueError("old_targets must contain floating point probabilities")
        if not torch.isfinite(old_targets).all().item():
            raise ValueError("old_targets must contain only finite probabilities")
        if ((old_targets < 0) | (old_targets > 1)).any().item():
            raise ValueError("old_targets probabilities must be between zero and one")
        if labels.shape[0] != logits.shape[0]:
            raise ValueError("labels must match the logits batch size")
        labels = labels.to(device=logits.device)
        if (labels < 0).any().item() or (labels >= logits.shape[1]).any().item():
            raise ValueError("labels must refer to a class represented by logits")

        old_targets = old_targets.to(device=logits.device, dtype=logits.dtype).detach()
        old_loss = F.binary_cross_entropy_with_logits(
            logits[:, :old_count], old_targets, reduction="sum"
        ) if old_count else logits.new_zeros(())
        new_logits = logits[:, old_count:]
        new_targets = torch.zeros_like(new_logits)
        new_rows = labels >= old_count
        if new_rows.any().item():
            rows = torch.arange(logits.shape[0], device=logits.device)[new_rows]
            columns = labels[new_rows] - old_count
            new_targets[rows, columns] = 1
        new_loss = F.binary_cross_entropy_with_logits(
            new_logits, new_targets, reduction="sum"
        ) if new_logits.shape[1] else logits.new_zeros(())
        return old_loss + new_loss
