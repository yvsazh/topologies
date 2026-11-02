import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class CoTTA(nn.Module):
    component_id = "cotta"

    @staticmethod
    def _validated_real(name, value):
        try:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError
            value = float(value)
        except (OverflowError, TypeError) as error:
            raise ValueError(f"{name} must be a finite number") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return value

    @staticmethod
    def _validated_bool(name, value):
        try:
            if not isinstance(value, bool):
                raise TypeError
        except TypeError as error:
            raise ValueError(f"{name} must be a boolean") from error
        return value

    def __init__(
        self,
        source_model=None,
        num_classes=3,
        width=4,
        ema_decay=0.99,
        restore_prob=0.01,
        confidence_threshold=0.9,
        augmentation_count=32,
        steps=1,
        learning_rate=1e-3,
        adapt_on_forward=True,
    ):
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        if source_model is not None and not isinstance(source_model, nn.Module):
            raise ValueError("source_model must be a torch.nn.Module")
        ema_decay = self._validated_real("ema_decay", ema_decay)
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be a number in [0, 1)")
        restore_prob = self._validated_real("restore_prob", restore_prob)
        if not 0.0 <= restore_prob <= 1.0:
            raise ValueError("restore_prob must be a number in [0, 1]")
        confidence_threshold = self._validated_real(
            "confidence_threshold", confidence_threshold
        )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be a number in [0, 1]")
        if (
            not isinstance(augmentation_count, int)
            or isinstance(augmentation_count, bool)
            or augmentation_count <= 0
        ):
            raise ValueError("augmentation_count must be a positive integer")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        adapt_on_forward = self._validated_bool("adapt_on_forward", adapt_on_forward)
        learning_rate = self._validated_real("learning_rate", learning_rate)
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite number")

        if source_model is None:
            source_model = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(width, num_classes),
            )
        parameters = list(source_model.parameters())
        if not parameters:
            raise ValueError("source_model must have at least one parameter")
        for module in source_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.num_batches_tracked = None

        self.student = source_model
        self.student.requires_grad_(True)
        self.anchor = copy.deepcopy(self.student)
        self.teacher = copy.deepcopy(self.student)
        self.anchor.requires_grad_(False)
        self.teacher.requires_grad_(False)
        self.anchor.eval()
        self.teacher.eval()
        self.num_classes = num_classes
        self.ema_decay = float(ema_decay)
        self.restore_prob = float(restore_prob)
        self.confidence_threshold = float(confidence_threshold)
        self.augmentation_count = augmentation_count
        self.steps = steps
        self.adapt_on_forward = adapt_on_forward
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=float(learning_rate))

    @staticmethod
    def _augmentation_average(predictions):
        if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
            raise ValueError("predictions must have shape [augmentations, batch, classes]")
        if predictions.shape[0] == 0:
            raise ValueError("predictions must contain at least one augmentation")
        return predictions.mean(dim=0)

    def _select_target(self, direct_probabilities, augmentation_probabilities, confidence):
        if direct_probabilities.shape != augmentation_probabilities.shape:
            raise ValueError("direct and augmentation probabilities must have matching shapes")
        if confidence.shape != direct_probabilities.shape[:-1]:
            raise ValueError("confidence must have one value per batch item")
        use_direct = confidence >= self.confidence_threshold
        return torch.where(
            use_direct.unsqueeze(-1), direct_probabilities, augmentation_probabilities
        )

    @staticmethod
    def _soft_target_loss(student_logits, target):
        if student_logits.shape != target.shape or student_logits.ndim != 2:
            raise ValueError("student_logits and target must have matching [batch, classes] shapes")
        return -(target * F.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()

    @torch.no_grad()
    def _update_teacher(self):
        for teacher_parameter, student_parameter in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_parameter.mul_(self.ema_decay).add_(
                student_parameter, alpha=1.0 - self.ema_decay
            )
        for teacher_buffer, student_buffer in zip(self.teacher.buffers(), self.student.buffers()):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def _restore_student(self):
        for student_parameter, anchor_parameter in zip(
            self.student.parameters(), self.anchor.parameters()
        ):
            if self.restore_prob == 1.0:
                student_parameter.copy_(anchor_parameter)
            elif self.restore_prob > 0.0:
                mask = torch.rand_like(student_parameter) < self.restore_prob
                student_parameter.copy_(torch.where(mask, anchor_parameter, student_parameter))

    @staticmethod
    def _uniform_like(images, low, high):
        shape = (images.shape[0], 1, 1, 1)
        return torch.empty(shape, device=images.device, dtype=images.dtype).uniform_(low, high)

    def _augment(self, images):
        augmented = images.clamp(0.0, 1.0)
        augmented = augmented * self._uniform_like(augmented, 0.6, 1.4)
        contrast = self._uniform_like(augmented, 0.7, 1.3)
        mean = augmented.mean(dim=(1, 2, 3), keepdim=True)
        augmented = (augmented - mean) * contrast + mean
        saturation = self._uniform_like(augmented, 0.5, 1.5)
        gray = augmented.mean(dim=1, keepdim=True)
        augmented = gray + (augmented - gray) * saturation
        flip = torch.rand((images.shape[0], 1, 1, 1), device=images.device) < 0.5
        augmented = torch.where(flip, torch.flip(augmented, dims=(-1,)), augmented)
        return augmented.clamp(0.0, 1.0)

    def _checked_logits(self, model, images):
        logits = model(images)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("source_model must return logits with shape [batch, classes]")
        if logits.shape[0] != images.shape[0] or logits.shape[1] != self.num_classes:
            raise ValueError("source_model logits must match the input batch and num_classes")
        return logits

    def forward(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if images.shape[1] != 3 or images.shape[2] <= 0 or images.shape[3] <= 0:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if not images.is_floating_point():
            raise ValueError("images must have a floating point dtype")
        if images.shape[0] <= 0:
            raise ValueError("images must contain at least one batch item")
        if not self.adapt_on_forward:
            return self._checked_logits(self.student, images)

        self.student.train()
        self.anchor.eval()
        self.teacher.eval()
        with torch.enable_grad():
            with torch.no_grad():
                direct_logits = self._checked_logits(self.teacher, images)
                direct_probabilities = direct_logits.softmax(dim=-1)
                source_probabilities = self._checked_logits(self.anchor, images).softmax(dim=-1)
                confidence = source_probabilities.max(dim=-1).values
                if bool((confidence < self.confidence_threshold).any()):
                    augmentation_probabilities = self._augmentation_average(
                        torch.stack(
                            [
                                self._checked_logits(self.teacher, self._augment(images)).softmax(
                                    dim=-1
                                )
                                for _ in range(self.augmentation_count)
                            ]
                        )
                    )
                else:
                    augmentation_probabilities = direct_probabilities
                target = self._select_target(
                    direct_probabilities, augmentation_probabilities, confidence
                ).detach()

            for _ in range(self.steps):
                self.optimizer.zero_grad(set_to_none=True)
                student_logits = self._checked_logits(self.student, images)
                loss = self._soft_target_loss(student_logits, target)
                loss.backward()
                self.optimizer.step()
                self._update_teacher()
                self._restore_student()

            with torch.no_grad():
                return self._checked_logits(self.teacher, images)
