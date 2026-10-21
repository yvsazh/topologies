import torch
from torch import nn
from torch.nn import functional as F


class DINOv2(nn.Module):
    component_id = "dinov2"
    architecture_id = "dinov2"

    def __init__(
        self,
        embedding_dim=8,
        num_prototypes=4,
        teacher_temperature=0.04,
        student_temperature=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        if embedding_dim <= 0 or num_prototypes <= 1:
            raise ValueError("embedding_dim must be positive and num_prototypes must exceed one")
        if teacher_temperature <= 0 or student_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if not 0.0 <= center_momentum < 1.0:
            raise ValueError("center_momentum must be in [0, 1)")
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embedding_dim))
        nn.init.normal_(self.prototypes, std=embedding_dim ** -0.5)
        self.register_buffer("teacher_center", torch.zeros(1, num_prototypes))

    def forward(self, student_embedding, teacher_embedding, prototype_matrix=None):
        if student_embedding.ndim != 2 or teacher_embedding.ndim != 2:
            raise ValueError("student and teacher embeddings must have shape [batch, embedding_dim]")
        if student_embedding.shape != teacher_embedding.shape or student_embedding.shape[1] != self.embedding_dim:
            raise ValueError("student and teacher embeddings must have matching configured dimensions")
        prototypes = self.prototypes if prototype_matrix is None else prototype_matrix
        if prototypes.shape != (self.num_prototypes, self.embedding_dim):
            raise ValueError("prototype matrix has an incompatible shape")
        student_scores = student_embedding @ prototypes.t()
        with torch.no_grad():
            teacher_scores = teacher_embedding @ prototypes.detach().t()
            centered_teacher_scores = teacher_scores - self.teacher_center
            teacher_targets = F.softmax(centered_teacher_scores / self.teacher_temperature, dim=-1)
            if self.training:
                batch_center = teacher_scores.mean(dim=0, keepdim=True)
                self.teacher_center.mul_(self.center_momentum).add_(
                    batch_center, alpha=1.0 - self.center_momentum
                )
        student_log_probabilities = F.log_softmax(student_scores / self.student_temperature, dim=-1)
        return -(teacher_targets * student_log_probabilities).sum(dim=-1).mean()
