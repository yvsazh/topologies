import torch
from torch import nn


class LinearChainCRF(nn.Module):
    component_id = "linear-chain-crf"

    def __init__(self, num_labels=4, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if num_labels <= 1:
            raise ValueError("num_labels must be at least 2")
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.zeros(num_labels, num_labels))

    def _gold(self, emissions, tags):
        score = emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for index in range(1, emissions.shape[1]):
            score = score + self.transitions[tags[:, index - 1], tags[:, index]]
            score = score + emissions[:, index].gather(1, tags[:, index:index + 1]).squeeze(1)
        return score

    def log_partition(self, emissions):
        alpha = emissions[:, 0]
        for index in range(1, emissions.shape[1]):
            scores = alpha[:, :, None] + self.transitions[None, :, :]
            alpha = emissions[:, index] + torch.logsumexp(scores, dim=1)
        return torch.logsumexp(alpha, dim=-1)

    def nll(self, emissions, tags):
        return self.log_partition(emissions) - self._gold(emissions, tags)

    def viterbi(self, emissions):
        trellis = emissions[:, 0]
        pointers = []
        for index in range(1, emissions.shape[1]):
            scores = trellis[:, :, None] + self.transitions[None, :, :]
            best, choice = scores.max(dim=1)
            trellis = emissions[:, index] + best
            pointers.append(choice)
        last = trellis.argmax(dim=-1)
        path = [last]
        for choice in reversed(pointers):
            last = choice.gather(1, last[:, None]).squeeze(1)
            path.append(last)
        return torch.stack(list(reversed(path)), dim=1)

    def forward(self, emissions, tags=None):
        emissions = torch.as_tensor(emissions)
        if emissions.ndim != 3 or emissions.shape[-1] != self.num_labels or emissions.shape[1] < 1:
            raise ValueError("emissions must have shape (batch, length, num_labels)")
        if tags is None:
            return self.log_partition(emissions)
        tags = torch.as_tensor(tags)
        if tags.shape[:2] != emissions.shape[:2]:
            raise ValueError("tags must have shape (batch, length)")
        return self.nll(emissions, tags)
