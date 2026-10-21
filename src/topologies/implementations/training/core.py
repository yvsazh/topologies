import torch
import torch.nn.functional as F


def cross_entropy(logits, targets, *, ignore_index=-100, reduction="mean", weight=None, label_smoothing=0.0):
    return F.cross_entropy(logits, targets, weight=weight, ignore_index=ignore_index,
                           reduction=reduction, label_smoothing=label_smoothing)


def binary_cross_entropy_logits(logits, targets, *, pos_weight=None, reduction="mean"):
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    return F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), pos_weight=pos_weight, reduction=reduction)


def mean_squared_error(prediction, targets, *, reduction="mean"):
    if prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    return F.mse_loss(prediction, targets.to(prediction.dtype), reduction=reduction)


def mean_absolute_error(prediction, targets, *, reduction="mean"):
    if prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    return F.l1_loss(prediction, targets.to(prediction.dtype), reduction=reduction)


def huber(prediction, targets, *, delta=1.0, reduction="mean"):
    if prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if delta <= 0:
        raise ValueError("delta must be positive")
    return F.huber_loss(prediction, targets.to(prediction.dtype), delta=delta, reduction=reduction)


def dice(logits, targets, *, mode="binary", reduction="mean", smooth=1.0, mask=None, include_background=True):
    if smooth <= 0:
        raise ValueError("smooth must be positive")
    if mode == "binary":
        if logits.shape != targets.shape:
            raise ValueError("binary logits and targets must have identical shapes")
        probs = logits.sigmoid().reshape(logits.shape[0], -1)
        truth = targets.to(logits.dtype).reshape(logits.shape[0], -1)
    elif mode == "multiclass":
        if logits.ndim < 3 or targets.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError("multiclass expects logits [N,C,...] and integer targets [N,...]")
        probs = logits.softmax(dim=1)
        truth = F.one_hot(targets.long(), logits.shape[1]).movedim(-1, 1).to(logits.dtype)
        if not include_background and logits.shape[1] > 1:
            probs, truth = probs[:, 1:], truth[:, 1:]
        probs = probs.reshape(probs.shape[0], probs.shape[1], -1)
        truth = truth.reshape(truth.shape[0], truth.shape[1], -1)
    else:
        raise ValueError("mode must be 'binary' or 'multiclass'")
    if mask is not None:
        valid = mask.to(device=logits.device, dtype=torch.bool)
        if mode == "binary":
            if valid.shape != logits.shape:
                raise ValueError("binary mask must match logits shape")
            valid = valid.reshape(logits.shape[0], -1)
            probs, truth = probs * valid, truth * valid
        else:
            if valid.shape != targets.shape:
                raise ValueError("multiclass mask must match target shape")
            valid = valid.reshape(logits.shape[0], 1, -1)
            probs, truth = probs * valid, truth * valid
    dims = -1
    score = (2 * (probs * truth).sum(dims) + smooth) / (probs.sum(dims) + truth.sum(dims) + smooth)
    losses = 1 - score
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be none, mean, or sum")


def gaussian_kl(mean, logvar, *, reduction="mean"):
    if mean.shape != logvar.shape or mean.ndim < 2:
        raise ValueError("mean/logvar must have matching [batch,...] shapes")
    per_example = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).reshape(mean.shape[0], -1).sum(1)
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    if reduction == "mean":
        return per_example.mean()
    raise ValueError("reduction must be none, mean, or sum")


def smooth_l1(prediction: torch.Tensor, targets: torch.Tensor, *, beta: float = 1.0,
              reduction: str = "mean") -> torch.Tensor:
    if prediction.shape != targets.shape or beta < 0:
        raise ValueError("matching shapes and nonnegative beta are required")
    return F.smooth_l1_loss(prediction, targets.to(prediction.dtype), beta=beta, reduction=reduction)


def sigmoid_focal(logits: torch.Tensor, targets: torch.Tensor, *, alpha: float = 0.25,
                  gamma: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    if logits.shape != targets.shape or not 0 <= alpha <= 1 or gamma < 0:
        raise ValueError("matching shapes, alpha in [0,1], and gamma >= 0 required")
    targets = targets.to(logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = logits.sigmoid() * targets + (1 - logits.sigmoid()) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return _reduce(alpha_t * (1 - p_t).pow(gamma) * ce, reduction)


def softmax_focal(logits: torch.Tensor, targets: torch.Tensor, *, alpha=None,
                  gamma: float = 2.0, ignore_index: int = -100,
                  reduction: str = "mean") -> torch.Tensor:
    if logits.ndim < 2 or targets.shape != (logits.shape[0], *logits.shape[2:]) or gamma < 0:
        raise ValueError("expected logits [N,C,...], matching class-index targets, gamma >= 0")
    logp = logits.log_softmax(dim=1)
    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0).long()
    selected = logp.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    p_t = selected.exp()
    losses = -(1 - p_t).pow(gamma) * selected
    if alpha is not None:
        weights = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device)
        if weights.ndim == 0:
            losses = losses * weights
        elif weights.shape == (logits.shape[1],):
            losses = losses * weights[safe_targets]
        else:
            raise ValueError("alpha must be scalar or have one value per class")
    losses = losses * valid
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.sum() / valid.sum().clamp_min(1)
    raise ValueError("reduction must be none, mean, or sum")


def generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    if boxes1.shape != boxes2.shape or boxes1.ndim < 2 or boxes1.shape[-1] != 4:
        raise ValueError("boxes must have matching [...,4] xyxy shapes")
    if bool(((boxes1[..., 2:] < boxes1[..., :2]) | (boxes2[..., 2:] < boxes2[..., :2])).any()):
        raise ValueError("xyxy boxes must have x2>=x1 and y2>=y1")
    lo = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    hi = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter = (hi - lo).clamp_min(0).prod(-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(-1)
    union = area1 + area2 - inter
    iou = inter / union.clamp_min(torch.finfo(boxes1.dtype).eps)
    enc_lo = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enc_hi = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing = (enc_hi - enc_lo).clamp_min(0).prod(-1)
    giou = iou - (enclosing - union).clamp_min(0) / enclosing.clamp_min(torch.finfo(boxes1.dtype).eps)
    return _reduce(1 - giou, reduction)


def gaussian_nll(mean: torch.Tensor, targets: torch.Tensor, variance: torch.Tensor, *,
                 full: bool = False, eps: float = 1e-6, reduction: str = "mean") -> torch.Tensor:
    if mean.shape != targets.shape or eps <= 0:
        raise ValueError("mean/targets must match and eps must be positive")
    variance = torch.as_tensor(variance, dtype=mean.dtype, device=mean.device).clamp_min(eps)
    try:
        if torch.broadcast_shapes(mean.shape, variance.shape) != mean.shape:
            raise ValueError("variance must broadcast to mean shape")
    except RuntimeError as exc:
        raise ValueError("variance must broadcast to mean shape") from exc
    loss = 0.5 * (variance.log() + (mean - targets.to(mean.dtype)).square() / variance)
    if full:
        loss = loss + 0.5 * 1.8378770664093453
    return _reduce(loss, reduction)


def pinball(prediction: torch.Tensor, targets: torch.Tensor, *, quantiles,
            reduction: str = "mean") -> torch.Tensor:
    q = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device)
    if prediction.shape[:-1] != targets.shape or prediction.shape[-1] != q.numel():
        raise ValueError("prediction [...,Q], targets [...], and Q quantiles required")
    if q.ndim != 1 or q.numel() == 0 or bool(((q <= 0) | (q >= 1)).any()) or bool((q[1:] < q[:-1]).any()):
        raise ValueError("quantiles must be a nonempty sorted 1-D sequence in (0,1)")
    error = targets.to(prediction.dtype).unsqueeze(-1) - prediction
    return _reduce(torch.maximum(q * error, (q - 1) * error), reduction)


def ctc(logits: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor,
        target_lengths: torch.Tensor, *, blank: int = 0, reduction: str = "mean",
        zero_infinity: bool = False, log_probs: bool = False) -> torch.Tensor:
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if (logits.ndim != 3 or input_lengths.ndim != 1 or target_lengths.ndim != 1
            or input_lengths.shape != target_lengths.shape or input_lengths.numel() != logits.shape[1]
            or input_lengths.dtype not in integer_dtypes or target_lengths.dtype not in integer_dtypes
            or targets.dtype not in integer_dtypes):
        raise ValueError("logits [T,N,C] and one input/target length per batch item required")
    if (not isinstance(blank, int) or isinstance(blank, bool) or not 0 <= blank < logits.shape[2]
            or bool((input_lengths < 0).any()) or bool((input_lengths > logits.shape[0]).any())
            or bool((target_lengths < 0).any())):
        raise ValueError("invalid blank or input lengths")
    target_capacity = targets.shape[1] if targets.ndim == 2 else targets.numel() if targets.ndim == 1 else -1
    if bool((target_lengths > target_capacity).any()):
        raise ValueError("target lengths exceed padded or concatenated target capacity")
    lp = logits if log_probs else logits.log_softmax(-1)
    return F.ctc_loss(lp, targets.long(), input_lengths.long(), target_lengths.long(),
                      blank=blank, reduction=reduction, zero_infinity=zero_infinity)


def multi_resolution_stft(prediction: torch.Tensor, targets: torch.Tensor, *,
                          resolutions=((1024, 256), (512, 128), (256, 64)),
                          center: bool = True, eps: float = 1e-7,
                          reduction: str = "mean") -> torch.Tensor:
    if prediction.shape != targets.shape or prediction.ndim != 2 or eps <= 0:
        raise ValueError("waveforms must have matching [batch,samples] shapes and eps > 0")
    terms = []
    for n_fft, hop in resolutions:
        if n_fft <= 0 or hop <= 0 or hop > n_fft:
            raise ValueError("each resolution requires positive n_fft and 0 < hop <= n_fft")
        window = torch.hann_window(n_fft, dtype=prediction.dtype, device=prediction.device)
        pred_spec = torch.stft(prediction, n_fft, hop, window=window, center=center, return_complex=True).abs()
        target_spec = torch.stft(targets, n_fft, hop, window=window, center=center, return_complex=True).abs()
        convergence = (pred_spec - target_spec).flatten(1).norm(dim=1) / target_spec.flatten(1).norm(dim=1).clamp_min(eps)
        logmag = (torch.log(pred_spec + eps) - torch.log(target_spec + eps)).abs().flatten(1).mean(1)
        terms.append(convergence + logmag)
    if not terms:
        raise ValueError("at least one STFT resolution is required")
    per_example = torch.stack(terms).mean(0)
    return _reduce(per_example, reduction)


def info_nce(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, *,
             temperature: float = 0.07, mask: torch.Tensor = None,
             normalize: bool = True, reduction: str = "mean") -> torch.Tensor:
    if image_embeddings.ndim != 2 or image_embeddings.shape != text_embeddings.shape or temperature <= 0:
        raise ValueError("paired embeddings must have matching [N,D] shapes and positive temperature")
    if image_embeddings.shape[0] == 0:
        raise ValueError("batch must be nonempty")
    a, b = image_embeddings, text_embeddings
    if normalize:
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    scores = a @ b.transpose(0, 1) / temperature
    if mask is not None:
        valid = mask.to(device=scores.device, dtype=torch.bool)
        if valid.shape != scores.shape or bool((~valid.diagonal()).any()):
            raise ValueError("mask must be [N,N] and retain every diagonal positive")
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    target = torch.arange(scores.shape[0], device=scores.device)
    losses = (F.cross_entropy(scores, target, reduction="none") + F.cross_entropy(scores.T, target, reduction="none")) / 2
    return _reduce(losses, reduction)


def bpr(positive_scores: torch.Tensor, negative_scores: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative score shapes must match")
    return _reduce(F.softplus(negative_scores - positive_scores), reduction)


def triplet_margin(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, *,
                   margin: float = 1.0, p: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    if anchor.shape != positive.shape or anchor.shape != negative.shape or margin < 0 or p <= 0:
        raise ValueError("matching embedding shapes, margin >= 0, and p > 0 required")
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=p, reduction=reduction)


def adversarial_hinge_discriminator(real_scores: torch.Tensor, fake_scores: torch.Tensor, *,
                                    reduction: str = "mean") -> torch.Tensor:
    if real_scores.numel() == 0 or fake_scores.numel() == 0:
        raise ValueError("real and fake scores must be nonempty")
    return _reduce(F.relu(1 - real_scores), reduction) + _reduce(F.relu(1 + fake_scores), reduction)


def adversarial_hinge_generator(fake_scores: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    if fake_scores.numel() == 0:
        raise ValueError("fake scores must be nonempty")
    return _reduce(-fake_scores, reduction)


def density_nll(log_prob: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    return _reduce(-log_prob, reduction)


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8,
           reduction: str = "mean") -> torch.Tensor:
    if estimate.shape != target.shape or estimate.ndim < 2 or eps <= 0:
        raise ValueError("matching [batch,...] waveforms and positive eps required")
    e, t = estimate.reshape(estimate.shape[0], -1), target.reshape(target.shape[0], -1)
    e, t = e - e.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
    projection = (e * t).sum(-1, keepdim=True) / t.square().sum(-1, keepdim=True).clamp_min(eps) * t
    noise = e - projection
    score = 10 * torch.log10(projection.square().sum(-1).clamp_min(eps) / noise.square().sum(-1).clamp_min(eps))
    return _reduce(-score, reduction)


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be none, mean, or sum")
