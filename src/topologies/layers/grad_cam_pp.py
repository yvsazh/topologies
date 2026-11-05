import torch
from torch import nn


class GradCAMPP(nn.Module):
    component_id = "grad-cam-pp"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("GradCAMPP does not accept constructor arguments")
        super().__init__()

    def forward(self, activations, class_score, score_gradients=None):
        if not torch.is_tensor(activations):
            raise ValueError("activations must be a tensor")
        if activations.ndim not in (4, 5):
            raise ValueError("activations must have rank 4 or 5")
        if any(size < 1 for size in activations.shape):
            raise ValueError("activations must have nonempty dimensions")
        if not activations.is_floating_point():
            raise ValueError("activations must have a floating-point dtype")
        if not torch.is_tensor(class_score):
            if score_gradients is None or type(class_score) not in (int, float):
                raise ValueError("class_score must be a scalar tensor or one scalar per sample")
            class_score = activations.new_tensor(class_score)
        if class_score.ndim != 0 and (
            class_score.ndim != 1 or class_score.shape[0] != activations.shape[0]
        ):
            raise ValueError("class_score must be a scalar tensor or one scalar per sample")
        if not class_score.is_floating_point():
            raise ValueError("class_score must have a floating-point dtype")
        if class_score.device != activations.device or class_score.dtype != activations.dtype:
            raise ValueError("class_score must share the device and dtype of activations")

        if score_gradients is None:
            if not activations.requires_grad or not class_score.requires_grad:
                raise ValueError("activations and class_score must be connected by autograd")
            try:
                with torch.enable_grad():
                    score_gradients = torch.autograd.grad(
                        class_score.sum(),
                        activations,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
            except RuntimeError as error:
                raise ValueError("class_score must depend on activations") from error
        elif not torch.is_tensor(score_gradients):
            raise ValueError("score_gradients must be a tensor")

        if score_gradients.shape != activations.shape:
            raise ValueError("score_gradients must have the same shape as activations")
        if not score_gradients.is_floating_point():
            raise ValueError("score_gradients must have a floating-point dtype")
        if score_gradients.device != activations.device or score_gradients.dtype != activations.dtype:
            raise ValueError("score_gradients must share the device and dtype of activations")

        spatial_dims = tuple(range(2, activations.ndim))
        activation_sum = activations.sum(dim=spatial_dims, keepdim=True)
        positive_gradient = score_gradients > 0
        denominator = 2 * score_gradients.square() + activation_sum * score_gradients.pow(3)
        safe_denominator = torch.where(positive_gradient, denominator, torch.ones_like(denominator))
        alpha = torch.where(
            positive_gradient,
            score_gradients.square() / safe_denominator,
            torch.zeros_like(score_gradients),
        )
        score_scale = torch.exp(class_score)
        if class_score.ndim == 1:
            score_scale = score_scale.reshape(
                (activations.shape[0],) + (1,) * (activations.ndim - 1)
            )
        class_gradient = score_scale * score_gradients.clamp_min(0)
        channel_weights = (alpha * class_gradient).sum(dim=spatial_dims, keepdim=True)
        return torch.relu((channel_weights * activations).sum(dim=1))
