import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


class ORPO(nn.Module):
    component_id = "orpo"

    def __init__(self, lambda_=1.0, disable_prompt_loss=False, reduction="mean"):
        super().__init__()
        if isinstance(lambda_, bool) or not isinstance(lambda_, numbers.Real):
            error = ValueError("lambda_ must be a finite non-negative number")
            raise error
        weight = float(lambda_)
        if not math.isfinite(weight) or weight < 0:
            error = ValueError("lambda_ must be a finite non-negative number")
            raise error
        if not isinstance(disable_prompt_loss, bool):
            error = ValueError("disable_prompt_loss must be a boolean")
            raise error
        if reduction not in ("mean", "none"):
            error = ValueError("reduction must be 'mean' or 'none'")
            raise error
        self.lambda_ = weight
        self.disable_prompt_loss = disable_prompt_loss
        self.reduction = reduction

    def forward(
        self,
        chosen_logits,
        rejected_logits,
        chosen_input_ids,
        rejected_input_ids,
        chosen_attention_mask,
        rejected_attention_mask,
        prompt_attention_mask,
    ):
        tensors = {
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_attention_mask": rejected_attention_mask,
            "prompt_attention_mask": prompt_attention_mask,
        }
        if any(not torch.is_tensor(value) for value in tensors.values()):
            raise ValueError("all logits, token IDs, and masks must be tensors")
        if chosen_logits.ndim != 3:
            raise ValueError("chosen_logits must have shape [batch, sequence, vocabulary]")
        if rejected_logits.ndim != 3:
            raise ValueError("rejected_logits must have shape [batch, sequence, vocabulary]")
        if not chosen_logits.is_floating_point() or not rejected_logits.is_floating_point():
            raise ValueError("chosen_logits and rejected_logits must be floating-point tensors")
        if chosen_logits.shape != rejected_logits.shape:
            raise ValueError("chosen_logits and rejected_logits must have the same shape")
        batch_size, sequence_length, vocab_size = chosen_logits.shape
        if batch_size < 1 or sequence_length < 2 or vocab_size < 2:
            raise ValueError("logits require non-empty batches, at least two sequence positions, and two vocabulary items")
        expected_sequence_shape = (batch_size, sequence_length)
        for name in ("chosen_input_ids", "rejected_input_ids"):
            value = tensors[name]
            if value.shape != expected_sequence_shape:
                raise ValueError(f"{name} must have shape [batch, sequence] matching logits")
            if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError(f"{name} must contain integer token IDs")
            if torch.any(value < 0) or torch.any(value >= vocab_size):
                raise ValueError(f"{name} values must be within the logits vocabulary")
        for name in (
            "chosen_attention_mask",
            "rejected_attention_mask",
            "prompt_attention_mask",
        ):
            value = tensors[name]
            if value.shape != expected_sequence_shape:
                raise ValueError(f"{name} must have shape [batch, sequence] matching logits")
            if torch.any((value != 0) & (value != 1)):
                raise ValueError(f"{name} must be binary")
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("logits, token IDs, and masks must be on the same device")
        chosen_attention = chosen_attention_mask.bool()
        rejected_attention = rejected_attention_mask.bool()
        prompt_mask = prompt_attention_mask.bool()
        if torch.any(prompt_mask & ~chosen_attention) or torch.any(prompt_mask & ~rejected_attention):
            raise ValueError("prompt_attention_mask must only mark attended tokens")
        chosen_response_mask = chosen_attention[:, 1:] & ~prompt_mask[:, 1:]
        rejected_response_mask = rejected_attention[:, 1:] & ~prompt_mask[:, 1:]
        if torch.any(chosen_response_mask.sum(dim=1) == 0) or torch.any(
            rejected_response_mask.sum(dim=1) == 0
        ):
            raise ValueError("each pair must have at least one valid response token")
        chosen_log_probs = F.log_softmax(chosen_logits[:, :-1, :], dim=-1).gather(
            -1, chosen_input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        rejected_log_probs = F.log_softmax(rejected_logits[:, :-1, :], dim=-1).gather(
            -1, rejected_input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        chosen_response_logp = torch.where(
            chosen_response_mask, chosen_log_probs, torch.zeros_like(chosen_log_probs)
        ).sum(dim=1) / chosen_response_mask.sum(dim=1)
        rejected_response_logp = torch.where(
            rejected_response_mask, rejected_log_probs, torch.zeros_like(rejected_log_probs)
        ).sum(dim=1) / rejected_response_mask.sum(dim=1)
        chosen_log_odds = chosen_response_logp - torch.log(-torch.expm1(chosen_response_logp))
        rejected_log_odds = rejected_response_logp - torch.log(
            -torch.expm1(rejected_response_logp)
        )
        odds_loss = F.softplus(-(chosen_log_odds - rejected_log_odds))
        chosen_nll_mask = chosen_attention[:, 1:]
        if self.disable_prompt_loss:
            chosen_nll_mask = chosen_nll_mask & ~prompt_mask[:, 1:]
        chosen_token_nll = -torch.where(
            chosen_nll_mask, chosen_log_probs, torch.zeros_like(chosen_log_probs)
        )
        if self.reduction == "none":
            chosen_nll = chosen_token_nll.sum(dim=1) / chosen_nll_mask.sum(dim=1)
            return chosen_nll + self.lambda_ * odds_loss
        chosen_nll = chosen_token_nll.sum() / chosen_nll_mask.sum()
        return chosen_nll + self.lambda_ * odds_loss.mean()
