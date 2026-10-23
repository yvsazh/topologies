import torch
from torch import nn
from torch.nn import functional as F


class PrototypicalNetworks(nn.Module):
    component_id = "prototypical-networks"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("PrototypicalNetworks does not accept constructor arguments")

    def _prepare_episode(self, support_embeddings, support_labels, query_embeddings):
        support_embeddings = torch.as_tensor(support_embeddings)
        support_labels = torch.as_tensor(support_labels)
        query_embeddings = torch.as_tensor(query_embeddings)

        if support_embeddings.ndim != 3 or query_embeddings.ndim != 3:
            raise ValueError("support_embeddings and query_embeddings must have shape [B, N, D]")
        if support_labels.ndim != 2:
            raise ValueError("support_labels must have shape [B, N_s]")
        if support_embeddings.shape[0] != query_embeddings.shape[0] or (
            support_embeddings.shape[0] != support_labels.shape[0]
        ):
            raise ValueError("episode batch dimensions must match")
        if support_embeddings.shape[1] != support_labels.shape[1]:
            raise ValueError("support_embeddings and support_labels must have matching support counts")
        if support_embeddings.shape[2] != query_embeddings.shape[2]:
            raise ValueError("support and query embedding dimensions must match")
        if min(support_embeddings.shape) == 0 or min(query_embeddings.shape) == 0:
            raise ValueError("episode batch, support, query, and embedding dimensions must be nonempty")
        if not torch.is_floating_point(support_embeddings) or not torch.is_floating_point(
            query_embeddings
        ):
            raise ValueError("support and query embeddings must be floating point tensors")
        if support_embeddings.device != query_embeddings.device:
            raise ValueError("support and query embeddings must be on the same device")
        if support_embeddings.dtype != query_embeddings.dtype:
            raise ValueError("support and query embeddings must have the same dtype")

        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if support_labels.dtype not in integer_types:
            raise ValueError("support_labels must contain integer class indices")
        support_labels = support_labels.to(device=support_embeddings.device, dtype=torch.long)

        class_count = None
        for episode_labels in support_labels:
            classes = torch.unique(episode_labels, sorted=True)
            if classes.numel() == 0:
                raise ValueError("each episode must contain at least one support class")
            if classes[0].item() != 0:
                raise ValueError("support class indices must start at zero")
            expected = torch.arange(classes.numel(), device=classes.device)
            if not torch.equal(classes, expected):
                raise ValueError("support labels must include every class index without gaps")
            if class_count is None:
                class_count = classes.numel()
            elif class_count != classes.numel():
                raise ValueError("all episodes in a batch must contain the same number of classes")

        return support_embeddings, support_labels, query_embeddings, class_count

    def forward(self, support_embeddings, support_labels, query_embeddings):
        support_embeddings, support_labels, query_embeddings, class_count = self._prepare_episode(
            support_embeddings, support_labels, query_embeddings
        )
        assignments = F.one_hot(support_labels, num_classes=class_count).to(
            dtype=support_embeddings.dtype
        )
        counts = assignments.sum(dim=1).unsqueeze(-1)
        prototypes = torch.bmm(assignments.transpose(1, 2), support_embeddings) / counts
        differences = query_embeddings.unsqueeze(2) - prototypes.unsqueeze(1)
        return -differences.square().sum(dim=-1)

    def episode_loss(self, support_embeddings, support_labels, query_embeddings, query_labels):
        scores = self.forward(support_embeddings, support_labels, query_embeddings)
        query_labels = torch.as_tensor(query_labels)
        if query_labels.ndim != 2 or tuple(query_labels.shape) != tuple(scores.shape[:2]):
            raise ValueError("query_labels must have shape [B, N_q] matching the episode queries")
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if query_labels.dtype not in integer_types:
            raise ValueError("query_labels must contain integer class indices")
        query_labels = query_labels.to(device=scores.device, dtype=torch.long)
        if (query_labels < 0).any().item() or (query_labels >= scores.shape[-1]).any().item():
            raise ValueError("query_labels must refer to a support class in each episode")
        return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), query_labels.reshape(-1))
