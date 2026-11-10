import torch
import torch.nn.functional as F
from torch import nn


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class EndToEndMemory(nn.Module):
    component_id = "memory-networks"
    PRESETS = {
        "tiny": {"vocab_size": 8, "embedding_dim": 4, "hops": 1, "max_memory_slots": 2, "padding_idx": 0},
        "small": {"vocab_size": 32, "embedding_dim": 20, "hops": 2, "max_memory_slots": 16, "padding_idx": 0},
        "base": {"vocab_size": 177, "embedding_dim": 50, "hops": 3, "max_memory_slots": 50, "padding_idx": 0},
    }

    def __init__(self, vocab_size=177, embedding_dim=50, hops=3, max_memory_slots=50,
                 padding_idx=0):
        super().__init__()
        _positive_integer("vocab_size", vocab_size)
        _positive_integer("embedding_dim", embedding_dim)
        _positive_integer("hops", hops)
        _positive_integer("max_memory_slots", max_memory_slots)
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if padding_idx is not None and (
            isinstance(padding_idx, bool)
            or not isinstance(padding_idx, int)
            or not 0 <= padding_idx < vocab_size
        ):
            raise ValueError("padding_idx must be None or an integer within the vocabulary")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hops = hops
        self.max_memory_slots = max_memory_slots
        self.padding_idx = padding_idx
        self.embedding_tables = nn.ModuleList(
            nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
            for _ in range(hops + 1)
        )
        with torch.no_grad():
            for embedding in self.embedding_tables:
                nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
                if padding_idx is not None:
                    embedding.weight[padding_idx].zero_()

    @property
    def query_embedding(self):
        return self.embedding_tables[0]

    @property
    def key_embeddings(self):
        return tuple(self.embedding_tables[:-1])

    @property
    def value_embeddings(self):
        return tuple(self.embedding_tables[1:])

    @property
    def answer_embedding(self):
        return self.embedding_tables[-1]

    def _validate_inputs(self, question, memory):
        if not isinstance(question, torch.Tensor) or question.ndim != 2:
            raise ValueError("question must be a rank-2 int64 tensor shaped [B, Q]")
        if not isinstance(memory, torch.Tensor) or memory.ndim != 3:
            raise ValueError("memory must be a rank-3 int64 tensor shaped [B, M, S]")
        if question.dtype != torch.int64 or memory.dtype != torch.int64:
            raise ValueError("question and memory token IDs must use torch.int64")
        if question.shape[0] == 0 or question.shape[1] == 0:
            raise ValueError("question batch and token axes must be nonempty")
        if memory.shape[0] == 0 or memory.shape[1] == 0 or memory.shape[2] == 0:
            raise ValueError("memory batch, slot, and token axes must be nonempty")
        if question.shape[0] != memory.shape[0]:
            raise ValueError("question and memory batch sizes must match")
        if memory.shape[1] > self.max_memory_slots:
            raise ValueError("memory slot count exceeds max_memory_slots")
        if question.device != memory.device:
            raise ValueError("question and memory tensors must be on the same device")
        if bool((question < 0).any()) or bool((question >= self.vocab_size).any()):
            raise ValueError("question token IDs must be within the vocabulary")
        if bool((memory < 0).any()) or bool((memory >= self.vocab_size).any()):
            raise ValueError("memory token IDs must be within the vocabulary")

    def forward_with_details(self, question, memory):
        self._validate_inputs(question, memory)
        state = self.query_embedding(question).sum(dim=1)
        attentions = []
        responses = []
        for hop in range(self.hops):
            keys = self.key_embeddings[hop](memory).sum(dim=2)
            values = self.value_embeddings[hop](memory).sum(dim=2)
            scores = (state.unsqueeze(1) * keys).sum(dim=-1)
            attention = torch.softmax(scores, dim=1)
            response = (attention.unsqueeze(-1) * values).sum(dim=1)
            state = state + response
            attentions.append(attention)
            responses.append(response)
        logits = F.linear(state, self.answer_embedding.weight)
        return logits, tuple(attentions), tuple(responses)

    def forward(self, question, memory):
        logits, _, _ = self.forward_with_details(question, memory)
        return logits

    def loss(self, logits, targets):
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("logits must be a rank-2 tensor shaped [B, vocab_size]")
        if logits.shape[0] == 0 or logits.shape[1] != self.vocab_size:
            raise ValueError("logits must have a nonempty batch and vocab_size classes")
        if not logits.is_floating_point():
            raise ValueError("logits must use a floating-point dtype")
        if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
            raise ValueError("targets must be a rank-1 int64 tensor shaped [B]")
        if targets.dtype != torch.int64:
            raise ValueError("targets must use torch.int64")
        if targets.shape[0] != logits.shape[0]:
            raise ValueError("target count must match the logits batch size")
        if targets.device != logits.device:
            raise ValueError("targets and logits must be on the same device")
        if bool((targets < 0).any()) or bool((targets >= self.vocab_size).any()):
            raise ValueError("target IDs must be within the vocabulary")
        return F.cross_entropy(logits, targets, reduction="sum")
