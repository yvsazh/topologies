import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveSearch(nn.Module):
    component_id = "contrastive-search"

    def __init__(self, alpha=0.6):
        super().__init__()
        if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)

    def forward(self, logits, candidate_ids, candidate_hidden, context_hidden):
        if logits.ndim < 1 or logits.shape[-1] == 0:
            raise ValueError("logits must have a non-empty vocabulary dimension")
        if candidate_ids.ndim < 1 or candidate_hidden.ndim < 2 or context_hidden.ndim < 2:
            raise ValueError("candidate IDs and hidden states must include their sequence dimensions")
        if candidate_ids.shape[-1] == 0 or candidate_hidden.shape[-2] == 0:
            raise ValueError("at least one candidate is required")
        if context_hidden.shape[-2] == 0:
            raise ValueError("at least one prior context state is required")
        if candidate_ids.shape[:-1] != logits.shape[:-1]:
            raise ValueError("candidate_ids batch dimensions must match logits batch dimensions")
        if candidate_hidden.shape[:-1] != candidate_ids.shape:
            raise ValueError("candidate_hidden must have one hidden state per candidate ID")
        if context_hidden.shape[:-2] != logits.shape[:-1]:
            raise ValueError("context_hidden batch dimensions must match logits batch dimensions")
        if candidate_hidden.shape[-1] != context_hidden.shape[-1]:
            raise ValueError("candidate and context hidden dimensions must match")
        if candidate_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("candidate_ids must contain integer token IDs")

        candidate_ids = candidate_ids.to(device=logits.device, dtype=torch.long)
        if torch.any(candidate_ids < 0) or torch.any(candidate_ids >= logits.shape[-1]):
            raise ValueError("candidate_ids must be within the logits vocabulary dimension")

        full_vocab_probabilities = F.softmax(logits, dim=-1)
        confidence = full_vocab_probabilities.gather(dim=-1, index=candidate_ids)
        candidates = F.normalize(candidate_hidden, p=2, dim=-1)
        context = F.normalize(context_hidden, p=2, dim=-1)
        similarities = torch.matmul(candidates, context.transpose(-1, -2))
        penalty = similarities.amax(dim=-1)
        return (1 - self.alpha) * confidence - self.alpha * penalty
