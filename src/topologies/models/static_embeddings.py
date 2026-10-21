import torch
from torch import nn
import torch.nn.functional as F


class StaticEmbeddings(nn.Module):
    architecture_id = "static-embeddings"
    PRESETS = {
        "tiny": {"vocab_size": 64, "d_model": 16, "ngram_buckets": 8, "mode": "skipgram"},
        "base": {"vocab_size": 256, "d_model": 32, "ngram_buckets": 16, "mode": "skipgram"},
    }

    def __init__(self, vocab_size=128, d_model=32, ngram_buckets=16, mode="skipgram", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"skipgram", "cbow", "glove", "fasttext"}:
            raise ValueError("mode must be skipgram, cbow, glove, or fasttext")
        if min(vocab_size, d_model, ngram_buckets) <= 0 or vocab_size <= 1:
            raise ValueError("invalid embedding dimensions")
        self.vocab_size = vocab_size
        self.context_length = 8
        self.d_model = d_model
        self.ngram_buckets = ngram_buckets
        self.mode = mode
        self.inputs = nn.Embedding(vocab_size, d_model)
        self.outputs = nn.Embedding(vocab_size, d_model)
        self.ngrams = nn.Embedding(ngram_buckets, d_model)
        self.input_bias = nn.Parameter(torch.zeros(vocab_size))
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def _center(self, target):
        if self.mode == "fasttext":
            return self.ngrams(target).sum(dim=1)
        return self.inputs(target)

    def _positive(self, target, context):
        if self.mode == "cbow":
            center = self.inputs(context).mean(dim=1)
            return (center * self.outputs(target)).sum(dim=-1)
        return (self._center(target) * self.outputs(context)).sum(dim=-1)

    def forward(self, target, context=None, negatives=None, counts=None):
        target = torch.as_tensor(target)
        if context is None:
            if self.mode == "fasttext" and target.ndim == 2:
                return self._center(target)
            if target.ndim == 1:
                return self._center(target)
            raise ValueError("target ids must have shape (batch,) when context is omitted")
        context = torch.as_tensor(context)
        if self.mode == "glove":
            if counts is None:
                raise ValueError("glove mode requires counts")
            counts = torch.as_tensor(counts)
            prediction = self._positive(target, context) + self.input_bias[target] + self.output_bias[context]
            weight = torch.clamp(counts / 100.0, max=1.0).pow(0.75)
            return (weight * (prediction - torch.log(counts.clamp_min(1.0))).pow(2)).mean()
        positive = F.logsigmoid(self._positive(target, context))
        if negatives is None:
            return -positive.mean()
        negatives = torch.as_tensor(negatives)
        if self.mode == "cbow":
            center = self.inputs(context).mean(dim=1)
            negative_score = (center[:, None, :] * self.outputs(negatives)).sum(dim=-1)
        else:
            negative_score = (self._center(target)[:, None, :] * self.outputs(negatives)).sum(dim=-1)
        return -(positive + F.logsigmoid(-negative_score).sum(dim=-1)).mean()
