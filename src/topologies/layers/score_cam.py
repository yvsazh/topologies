import torch
from torch import nn


class ScoreCAM(nn.Module):
    component_id = "score-cam"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("ScoreCAM does not accept constructor arguments")

    def forward(self, model, inputs, activations, baseline, target):
        if not callable(model) and not torch.is_tensor(model):
            raise ValueError("model must be callable or a one-dimensional tensor of constant logits")
        if not torch.is_tensor(inputs) or inputs.ndim != 4:
            raise ValueError("inputs must be a tensor with shape [B, C_in, H, W]")
        if not torch.is_tensor(activations) or activations.ndim != 4:
            raise ValueError("activations must be a tensor with shape [B, C_l, h_l, w_l]")
        if not torch.is_tensor(baseline) or baseline.ndim != 4:
            raise ValueError("baseline must be a tensor with shape [B, C_in, H, W]")
        if any(size < 1 for size in inputs.shape):
            raise ValueError("inputs must have nonempty batch, channel, and spatial dimensions")
        if any(size < 1 for size in activations.shape):
            raise ValueError("activations must have nonempty batch, channel, and spatial dimensions")
        if not inputs.is_floating_point() or not activations.is_floating_point() or not baseline.is_floating_point():
            raise ValueError("inputs, activations, and baseline must have floating-point dtypes")
        if activations.shape[0] != inputs.shape[0]:
            raise ValueError("inputs and activations must have matching batch dimensions")
        if baseline.shape != inputs.shape:
            raise ValueError("baseline must have the same shape as inputs")
        if inputs.device != activations.device or inputs.device != baseline.device:
            raise ValueError("inputs, activations, and baseline must be on the same device")
        if inputs.dtype != baseline.dtype:
            raise ValueError("inputs and baseline must have the same dtype")
        if type(target) is int:
            target_indices = torch.full((inputs.shape[0],), target, dtype=torch.long, device=inputs.device)
        elif torch.is_tensor(target):
            if target.ndim == 0:
                target_indices = target.reshape(1).expand(inputs.shape[0])
            elif target.ndim == 1 and target.shape[0] == inputs.shape[0]:
                target_indices = target
            else:
                raise ValueError("target must be a scalar or one index per sample")
            if target_indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("target must be an integer tensor")
            target_indices = target_indices.to(device=inputs.device, dtype=torch.long)
        else:
            raise ValueError("target must be an integer or integer tensor")

        batch_size = inputs.shape[0]

        def run_model(values):
            if torch.is_tensor(model):
                if model.ndim == 1 and model.shape[0] > 0:
                    sample_logits = model.unsqueeze(0).expand(batch_size, -1)
                elif model.ndim == 2 and model.shape[1] > 0 and model.shape[0] in (1, batch_size):
                    sample_logits = model.expand(batch_size, -1)
                else:
                    raise ValueError("constant model logits must have shape [K] or [B, K]")
                if values.shape[0] == batch_size:
                    logits = sample_logits
                elif values.shape[0] % batch_size == 0:
                    logits = sample_logits.repeat_interleave(values.shape[0] // batch_size, dim=0)
                else:
                    raise ValueError("constant model logits cannot match the requested batch size")
            else:
                logits = model(values)
            if not torch.is_tensor(logits) or logits.ndim != 2 or logits.shape[0] != values.shape[0] or logits.shape[1] < 1:
                raise ValueError("model output must have shape [N, K] with at least one class")
            if not logits.is_floating_point():
                raise ValueError("model output must have a floating-point dtype")
            if logits.device != values.device:
                raise ValueError("model output must be on the same device as its inputs")
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("model output must contain only finite values")
            return logits

        with torch.no_grad():
            baseline_logits = run_model(baseline)
        if target_indices.numel() and (bool((target_indices < 0).any()) or bool((target_indices >= baseline_logits.shape[1]).any())):
            raise ValueError("target indices must be in range for the model output classes")
        baseline_probabilities = torch.softmax(baseline_logits, dim=1).gather(
            1, target_indices[:, None]
        ).squeeze(1)

        height, width = inputs.shape[-2:]
        upsampled = nn.functional.interpolate(
            activations,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        minimum = upsampled.amin(dim=(-2, -1), keepdim=True)
        maximum = upsampled.amax(dim=(-2, -1), keepdim=True)
        valid_channels = (maximum > minimum).reshape(inputs.shape[0], activations.shape[1])
        denominator = torch.where(maximum > minimum, maximum - minimum, torch.ones_like(maximum))
        masks = (upsampled - minimum) / denominator
        masks = masks * valid_channels[:, :, None, None].to(dtype=masks.dtype)

        channel_count = activations.shape[1]
        probability_parts = []
        for start in range(0, channel_count, 32):
            stop = min(start + 32, channel_count)
            chunk_masks = masks[:, start:stop].to(dtype=inputs.dtype)
            masked_inputs = (inputs[:, None] * chunk_masks[:, :, None]).reshape(
                batch_size * (stop - start), inputs.shape[1], height, width
            )
            with torch.no_grad():
                masked_logits = run_model(masked_inputs)
            if masked_logits.shape[1] != baseline_logits.shape[1]:
                raise ValueError("model output class dimensions must match for baseline and masked inputs")
            repeated_targets = target_indices.repeat_interleave(stop - start)
            probabilities = torch.softmax(masked_logits, dim=1).gather(
                1, repeated_targets[:, None]
            ).reshape(batch_size, stop - start)
            probability_parts.append(probabilities)

        masked_probabilities = torch.cat(probability_parts, dim=1)
        scores = masked_probabilities - baseline_probabilities[:, None]
        safe_scores = torch.where(valid_channels, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=1) * valid_channels.to(dtype=safe_scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        return torch.relu((activations * weights.to(dtype=activations.dtype)[:, :, None, None]).sum(dim=1))
