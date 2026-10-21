import torch
from torch import nn
import torch.nn.functional as F


def dino_loss(student, teacher, center, student_temp=1.0, teacher_temp=1.0):
    student_log = F.log_softmax(student / student_temp, dim=-1)
    teacher_prob = F.softmax((teacher - center) / teacher_temp, dim=-1)
    return -(teacher_prob * student_log).sum(dim=-1).mean()


def koleo(values):
    normalized = F.normalize(values, dim=-1)
    distances = torch.cdist(normalized, normalized)
    distances.fill_diagonal_(1e9)
    nearest = distances.min(dim=-1).values
    return -torch.log(nearest).mean()


def ibot_loss(student, teacher, mask):
    student_log = F.log_softmax(student, dim=-1)
    teacher_prob = F.softmax(teacher, dim=-1)
    token = -(teacher_prob * student_log).sum(dim=-1)
    return (token * mask).sum() / mask.sum()


def beit_loss(logits, tokens, mask):
    per_token = F.cross_entropy(logits.transpose(1, 2), tokens, reduction="none")
    return (per_token * mask).sum() / mask.sum()


def beit_v2_loss(token_loss, student_feature, teacher_feature):
    student_feature = F.normalize(student_feature, dim=-1)
    teacher_feature = F.normalize(teacher_feature, dim=-1)
    cosine = (student_feature * teacher_feature).sum(dim=-1).mean()
    return token_loss + (1 - cosine)


def data2vec_loss(student, teacher, mask):
    per_token = (student - teacher).pow(2).mean(dim=-1)
    return (per_token * mask).sum() / mask.sum()


def msn_loss(student, teacher):
    return (student - teacher).pow(2).mean()


class VisionSSL(nn.Module):
    architecture_id = "vision-ssl"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "dino"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "beit"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="dino", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"dino", "dinov2", "ibot", "beit", "beit-v2", "data2vec", "msn"}:
            raise ValueError("unknown vision SSL mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid vision SSL dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.patch = nn.Conv2d(in_channels, width, 16, stride=16, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)

    def objective(self, mode, student, teacher, mask=None, tokens=None, center=None, token_loss=None):
        if mode == "dino":
            return dino_loss(student, teacher, center)
        if mode == "dinov2":
            return dino_loss(student, teacher, center) + koleo(student)
        if mode == "ibot":
            return ibot_loss(student, teacher, mask)
        if mode == "beit":
            return beit_loss(student, tokens, mask)
        if mode == "beit-v2":
            return beit_v2_loss(token_loss, student, teacher)
        if mode == "data2vec":
            return data2vec_loss(student, teacher, mask)
        return msn_loss(student, teacher)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        tokens = self.patch(pixels).flatten(2).transpose(1, 2)
        return self.fc(self.proj(tokens).mean(dim=1))
