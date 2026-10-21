import torch
from torch import nn


class SimKD(nn.Module):
    component_id = "simkd"

    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        if student_dim <= 0 or teacher_dim <= 0:
            raise ValueError("student_dim and teacher_dim must be positive")
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.projector = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_features):
        if student_features.ndim < 1 or student_features.shape[-1] != self.student_dim:
            raise ValueError("student_features final dimension must equal student_dim")
        return self.projector(student_features)

    def alignment_loss(self, student_features, teacher_features):
        projected = self(student_features)
        if teacher_features.shape != projected.shape:
            raise ValueError("teacher_features must match projected student feature shape")
        return (projected - teacher_features).square().sum()

    def classify(self, student_features, teacher_classifier):
        return teacher_classifier(self(student_features))
