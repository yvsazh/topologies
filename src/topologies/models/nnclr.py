import math

import torch
from torch import nn
from torch.nn import functional as F


class NNCLR(nn.Module):
    architecture_id = "nnclr"
    PRESETS = {
        "tiny": {
            "input_dim": 32,
            "projection_hidden_dim": 64,
            "embedding_dim": 16,
            "predictor_hidden_dim": 64,
            "queue_size": 128,
            "temperature": 0.1,
        },
        "small": {
            "input_dim": 2048,
            "projection_hidden_dim": 2048,
            "embedding_dim": 256,
            "predictor_hidden_dim": 4096,
            "queue_size": 98304,
            "temperature": 0.1,
        },
    }

    def __init__(
        self,
        input_dim=2048,
        projection_hidden_dim=2048,
        embedding_dim=256,
        predictor_hidden_dim=4096,
        queue_size=98304,
        temperature=0.1,
    ):
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "projection_hidden_dim": projection_hidden_dim,
            "embedding_dim": embedding_dim,
            "predictor_hidden_dim": predictor_hidden_dim,
            "queue_size": queue_size,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive number")
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.temperature = float(temperature)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, predictor_hidden_dim),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dim, embedding_dim),
        )
        self.register_buffer("support_queue", torch.randn(queue_size, embedding_dim))

    def nearest_neighbor(self, query):
        if not torch.is_tensor(query) or query.ndim != 2:
            raise ValueError("query must have shape (batch, embedding_dim)")
        if query.shape[0] == 0 or query.shape[1] != self.embedding_dim:
            raise ValueError("query batch and embedding dimensions must be positive and match")
        if query.device != self.support_queue.device or query.dtype != self.support_queue.dtype:
            raise ValueError("query and support queue must share device and dtype")
        normalized_query = F.normalize(query, dim=1)
        normalized_queue = F.normalize(self.support_queue, dim=1)
        indices = normalized_query @ normalized_queue.transpose(0, 1)
        return normalized_queue[indices.argmax(dim=1)]

    def directional_loss(self, neighbors, predictions):
        if (
            not torch.is_tensor(neighbors)
            or not torch.is_tensor(predictions)
            or neighbors.ndim != 2
            or predictions.ndim != 2
            or neighbors.shape != predictions.shape
            or neighbors.shape[0] == 0
            or neighbors.shape[1] != self.embedding_dim
        ):
            raise ValueError("neighbors and predictions must have matching (batch, embedding_dim) shapes")
        if neighbors.device != predictions.device or neighbors.dtype != predictions.dtype:
            raise ValueError("neighbors and predictions must share device and dtype")
        logits = F.normalize(neighbors, dim=1) @ F.normalize(predictions, dim=1).transpose(0, 1)
        logits = logits / self.temperature
        targets = torch.arange(neighbors.shape[0], device=neighbors.device)
        return F.cross_entropy(logits, targets)

    def symmetric_loss(self, neighbors1, neighbors2, predictions1, predictions2):
        loss12 = self.directional_loss(neighbors1, predictions2)
        loss21 = self.directional_loss(neighbors2, predictions1)
        return (loss12 + loss21) * 0.5

    def update_queue(self, embeddings):
        if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (batch, embedding_dim)")
        if embeddings.shape[0] == 0 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError("embeddings batch and feature dimensions must be positive and match")
        if embeddings.device != self.support_queue.device or embeddings.dtype != self.support_queue.dtype:
            raise ValueError("embeddings and support queue must share device and dtype")
        with torch.no_grad():
            combined = torch.cat((self.support_queue, embeddings.detach()), dim=0)
            self.support_queue.copy_(combined[-self.queue_size :])

    def forward(self, view1, view2):
        if not torch.is_tensor(view1) or not torch.is_tensor(view2):
            raise ValueError("views must be tensors with matching (batch, input_dim) shapes")
        if (
            view1.ndim != 2
            or view2.ndim != 2
            or view1.shape != view2.shape
            or view1.shape[0] == 0
            or view1.shape[1] != self.input_dim
        ):
            raise ValueError("views must have matching nonempty (batch, input_dim) shapes")
        if view1.device != view2.device or view1.dtype != view2.dtype:
            raise ValueError("views must share device and dtype")
        if not view1.is_floating_point() or not view2.is_floating_point():
            raise ValueError("views must have floating point dtype")
        if view1.device != self.support_queue.device or view1.dtype != self.support_queue.dtype:
            raise ValueError("views and model must share device and dtype")
        z1 = self.projector(view1)
        z2 = self.projector(view2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        neighbor1 = self.nearest_neighbor(z1)
        neighbor2 = self.nearest_neighbor(z2)
        loss = self.symmetric_loss(neighbor1, neighbor2, p1, p2)
        if self.training:
            self.update_queue(z1)
        return loss
