import torch
from torch import nn


def _raise_value_error(message):
    raise ValueError(message)


class _SpeculativeDecodeResult(torch.Tensor):
    @classmethod
    def create(cls, tokens, lengths, final_probs):
        result = final_probs.as_subclass(cls)
        result.tokens = tokens
        result.lengths = lengths
        return result

    def __iter__(self):
        return iter((self.tokens, self.lengths, self.as_subclass(torch.Tensor)))


class SpeculativeDecode(nn.Module):
    component_id = "speculative-decode"

    def __init__(self, gamma=None, pad_value=-1, generator=None):
        super().__init__()
        if gamma is not None and (not isinstance(gamma, int) or isinstance(gamma, bool) or gamma < 1):
            raise ValueError("gamma must be a positive integer or None")
        if not isinstance(pad_value, int) or isinstance(pad_value, bool):
            _raise_value_error("pad_value must be an integer")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator or None")
        self.gamma = gamma
        self.pad_value = pad_value
        self.generator = generator

    def _validate_probabilities(self, probabilities, name):
        if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, positions, vocabulary]")
        if not torch.is_floating_point(probabilities):
            raise ValueError(f"{name} must use a floating point dtype")
        if probabilities.shape[0] < 1 or probabilities.shape[1] < 1 or probabilities.shape[2] < 1:
            raise ValueError(f"{name} dimensions must all be positive")
        if not torch.isfinite(probabilities).all():
            raise ValueError(f"{name} must contain only finite probabilities")
        if (probabilities < 0).any():
            raise ValueError(f"{name} probabilities must be nonnegative")
        tolerance = max(1e-6, 4 * torch.finfo(probabilities.dtype).eps)
        sums = probabilities.float().sum(dim=-1)
        expected = torch.ones_like(sums)
        if not torch.allclose(sums, expected, atol=tolerance, rtol=tolerance):
            raise ValueError(f"each {name} probability row must sum to 1")

    def forward(self, draft_probs, draft_tokens, target_probs):
        self._validate_probabilities(draft_probs, "draft_probs")
        self._validate_probabilities(target_probs, "target_probs")
        if not isinstance(draft_tokens, torch.Tensor) or draft_tokens.ndim != 2:
            raise ValueError("draft_tokens must have shape [batch, gamma]")
        if draft_tokens.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("draft_tokens must use an integer dtype")
        batch, gamma, vocabulary = draft_probs.shape
        if gamma < 1:
            raise ValueError("gamma must be positive")
        if self.gamma is not None and gamma != self.gamma:
            raise ValueError(f"input gamma {gamma} does not match configured gamma {self.gamma}")
        if draft_tokens.shape != (batch, gamma):
            raise ValueError("draft_tokens shape must match the draft batch and gamma dimensions")
        if target_probs.shape != (batch, gamma + 1, vocabulary):
            raise ValueError("target_probs must match the batch, gamma plus one, and vocabulary dimensions")
        if draft_probs.device != target_probs.device or draft_probs.device != draft_tokens.device:
            raise ValueError("draft_probs, draft_tokens, and target_probs must be on the same device")
        if draft_probs.dtype != target_probs.dtype:
            raise ValueError("draft_probs and target_probs must use the same floating point dtype")
        if self.generator is not None and self.generator.device != draft_probs.device:
            raise ValueError("generator device must match the probability tensors")
        tokens = draft_tokens.to(dtype=torch.long)
        if (tokens < 0).any() or (tokens >= vocabulary).any():
            raise ValueError("draft_tokens must be within the vocabulary range")
        draft_token_probs = draft_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        if (draft_token_probs <= 0).any():
            raise ValueError("each draft token must have positive draft probability")
        uniforms = torch.rand(
            (batch, gamma),
            dtype=draft_probs.dtype,
            device=draft_probs.device,
            generator=self.generator,
        )
        target_token_probs = target_probs[:, :gamma].gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        ratios = target_token_probs / draft_token_probs
        tokens_out = torch.full(
            (batch, gamma + 1),
            self.pad_value,
            dtype=torch.long,
            device=draft_probs.device,
        )
        lengths = torch.empty(batch, dtype=torch.long, device=draft_probs.device)
        final_distributions = []
        for batch_index in range(batch):
            rejected = torch.nonzero(uniforms[batch_index] >= ratios[batch_index], as_tuple=False)
            if rejected.numel():
                position = int(rejected[0, 0].item())
                if position:
                    tokens_out[batch_index, :position] = tokens[batch_index, :position]
                residual = (target_probs[batch_index, position] - draft_probs[batch_index, position]).clamp_min(0)
                final_probs = residual / residual.sum()
                sampled = torch.multinomial(final_probs, 1, generator=self.generator)
                tokens_out[batch_index, position] = sampled[0]
                lengths[batch_index] = position + 1
            else:
                tokens_out[batch_index, :gamma] = tokens[batch_index]
                final_probs = target_probs[batch_index, gamma]
                sampled = torch.multinomial(final_probs, 1, generator=self.generator)
                tokens_out[batch_index, gamma] = sampled[0]
                lengths[batch_index] = gamma + 1
            final_distributions.append(final_probs)
        return _SpeculativeDecodeResult.create(tokens_out, lengths, torch.stack(final_distributions))
