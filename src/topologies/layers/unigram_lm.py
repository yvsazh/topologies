import math
from collections.abc import Sequence

import torch
from torch import nn


class UnigramLM(nn.Module):
    component_id = "unigram-lm"

    @staticmethod
    def _invalid_argument(message):
        return ValueError(message)

    def __init__(self, vocab, initial_probs):
        super().__init__()
        if isinstance(vocab, (str, bytes)) or not isinstance(vocab, Sequence):
            raise self._invalid_argument("vocab must be a sequence of non-empty strings")
        pieces = tuple(vocab)
        if not pieces or any(not isinstance(piece, str) or not piece for piece in pieces):
            raise ValueError("vocab must contain non-empty strings")
        if len(set(pieces)) != len(pieces):
            raise ValueError("vocab pieces must be unique")
        try:
            probabilities = torch.as_tensor(initial_probs, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("initial_probs must be a one-dimensional sequence of probabilities") from error
        if probabilities.ndim != 1 or probabilities.numel() != len(pieces):
            raise ValueError("initial_probs must have one probability per vocabulary piece")
        if not torch.isfinite(probabilities).all() or not (probabilities > 0).all():
            raise ValueError("initial_probs must be finite and positive")
        if not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0), rtol=1e-5, atol=1e-6):
            raise ValueError("initial_probs must sum to 1")
        self.vocab = pieces
        self._piece_to_index = {piece: index for index, piece in enumerate(pieces)}
        self.piece_logits = nn.Parameter(probabilities.log())

    @property
    def log_probs(self):
        return torch.log_softmax(self.piece_logits, dim=0)

    def _edges(self, text):
        edges = [[] for _ in range(len(text) + 1)]
        for start in range(len(text)):
            for index, piece in enumerate(self.vocab):
                end = start + len(piece)
                if end <= len(text) and text.startswith(piece, start):
                    edges[start].append((end, index))
        return edges

    def _forward_backward(self, text, log_probs):
        edges = self._edges(text)
        size = len(text)
        zero = log_probs.new_zeros(())
        negative_infinity = log_probs.new_full((), -torch.inf)
        alpha = [negative_infinity for _ in range(size + 1)]
        alpha[0] = zero
        for start, outgoing in enumerate(edges[:-1]):
            for end, index in outgoing:
                alpha[end] = torch.logaddexp(alpha[end], alpha[start] + log_probs[index])
        if not torch.isfinite(alpha[-1]):
            return None
        beta = [negative_infinity for _ in range(size + 1)]
        beta[-1] = zero
        for start in range(size - 1, -1, -1):
            scores = [log_probs[index] + beta[end] for end, index in edges[start] if torch.isfinite(beta[end])]
            if scores:
                beta[start] = torch.logsumexp(torch.stack(scores), dim=0)
        return edges, alpha, beta

    def forward(self, texts, batch_indices=None):
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise self._invalid_argument("texts must be a sequence of strings")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must contain only strings")
        if batch_indices is not None:
            if not torch.is_tensor(batch_indices) or batch_indices.ndim not in (1, 2):
                raise ValueError("batch_indices must be a one- or two-dimensional tensor")
            if batch_indices.ndim == 2:
                if batch_indices.shape[1] == 0:
                    raise ValueError("batch_indices rows must contain at least one index")
                batch_indices = batch_indices[:, 0]
            if batch_indices.is_floating_point() or batch_indices.is_complex():
                raise ValueError("batch_indices must have an integer dtype")
            indices = batch_indices.detach().tolist()
            if any(index < 0 or index >= len(texts) for index in indices):
                raise ValueError("batch_indices must refer to texts")
            texts = [texts[index] for index in indices]
        log_probs = self.log_probs
        scores = []
        for text in texts:
            result = self._forward_backward(text, log_probs)
            if result is None:
                scores.append(log_probs.new_full((), -torch.inf))
            else:
                scores.append(result[1][-1])
        if not scores:
            return log_probs.new_empty((0,))
        return torch.stack(scores)

    def encode(self, text):
        if not isinstance(text, str):
            raise self._invalid_argument("text must be a string")
        size = len(text)
        log_probs = self.log_probs.detach()
        edges = self._edges(text)
        best_scores = [float("-inf")] * (size + 1)
        best_scores[0] = 0.0
        best_previous = [None] * (size + 1)
        for start in range(size):
            if not math.isfinite(best_scores[start]):
                continue
            for end, index in edges[start]:
                score = best_scores[start] + float(log_probs[index])
                if score > best_scores[end]:
                    best_scores[end] = score
                    best_previous[end] = (start, index)
        if not math.isfinite(best_scores[-1]):
            raise ValueError("text cannot be segmented with vocab")
        result = []
        cursor = size
        while cursor:
            previous = best_previous[cursor]
            result.append(self.vocab[previous[1]])
            cursor = previous[0]
        result.reverse()
        return result

    @staticmethod
    def _corpus(corpus):
        if isinstance(corpus, (str, bytes)):
            raise UnigramLM._invalid_argument("corpus must be a sequence of strings")
        try:
            texts = list(corpus)
        except TypeError as error:
            raise UnigramLM._invalid_argument("corpus must be a sequence of strings") from error
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("corpus must contain only strings")
        if not texts:
            raise ValueError("corpus must contain at least one string")
        return texts

    @staticmethod
    def _iteration_count(em_iterations):
        if isinstance(em_iterations, bool) or not isinstance(em_iterations, int) or em_iterations < 1:
            raise ValueError("em_iterations must be a positive integer")
        return em_iterations

    def fit_em(self, corpus, em_iterations):
        texts = self._corpus(corpus)
        iterations = self._iteration_count(em_iterations)
        with torch.no_grad():
            for _ in range(iterations):
                log_probs = self.log_probs
                expected_counts = torch.zeros_like(log_probs)
                for text in texts:
                    result = self._forward_backward(text, log_probs)
                    if result is None:
                        raise ValueError("every corpus string must be segmentable with vocab")
                    edges, alpha, beta = result
                    likelihood = alpha[-1]
                    for start, outgoing in enumerate(edges[:-1]):
                        if not torch.isfinite(alpha[start]):
                            continue
                        for end, index in outgoing:
                            if torch.isfinite(beta[end]):
                                posterior = torch.exp(alpha[start] + log_probs[index] + beta[end] - likelihood)
                                expected_counts[index] += posterior
                total = expected_counts.sum()
                if not torch.isfinite(total) or total <= 0:
                    raise ValueError("corpus must contain at least one segmentable non-empty string")
                probabilities = expected_counts / total
                self.piece_logits.copy_(probabilities.log())
        return self

    def prune(self, corpus, target_vocab_size, em_iterations, keep_fraction=0.8):
        texts = self._corpus(corpus)
        iterations = self._iteration_count(em_iterations)
        if isinstance(target_vocab_size, bool) or not isinstance(target_vocab_size, int):
            raise self._invalid_argument("target_vocab_size must be an integer")
        if target_vocab_size < 1 or target_vocab_size > len(self.vocab):
            raise ValueError("target_vocab_size must be between 1 and the current vocabulary size")
        if isinstance(keep_fraction, bool) or not isinstance(keep_fraction, (int, float)):
            raise self._invalid_argument("keep_fraction must be between 0 and 1")
        if not math.isfinite(keep_fraction) or not 0 < keep_fraction < 1:
            raise ValueError("keep_fraction must be between 0 and 1")
        required = {piece for piece in self.vocab if len(piece) == 1}
        if target_vocab_size < len(required):
            raise ValueError("target_vocab_size cannot remove single-character pieces")
        self.fit_em(texts, iterations)
        while len(self.vocab) > target_vocab_size:
            current_likelihood = float(self(texts).sum().detach())
            probabilities = self.log_probs.detach().exp()
            losses = {}
            for index, piece in enumerate(self.vocab):
                if piece in required:
                    continue
                kept_indices = [other for other in range(len(self.vocab)) if other != index]
                candidate_vocab = [self.vocab[other] for other in kept_indices]
                candidate_probs = probabilities[kept_indices].clamp_min(torch.finfo(probabilities.dtype).tiny)
                candidate_probs = candidate_probs / candidate_probs.sum()
                candidate = UnigramLM(candidate_vocab, candidate_probs)
                try:
                    candidate.fit_em(texts, iterations)
                except ValueError as error:
                    if "segmentable" not in str(error):
                        raise
                    losses[index] = -math.inf
                else:
                    candidate_likelihood = float(candidate(texts).sum().detach())
                    losses[index] = current_likelihood - candidate_likelihood
            keep_count = max(target_vocab_size, len(required), math.floor(len(self.vocab) * keep_fraction))
            keep_count = min(len(self.vocab) - 1, keep_count)
            retain_noncharacters = keep_count - len(required)
            ranked = sorted(losses, key=lambda index: (-losses[index], index))
            retained = required | {self.vocab[index] for index in ranked[:retain_noncharacters]}
            kept_indices = [index for index, piece in enumerate(self.vocab) if piece in retained]
            if len(kept_indices) >= len(self.vocab):
                raise ValueError("target_vocab_size cannot be reached while preserving single-character pieces")
            kept_probs = probabilities[kept_indices].clamp_min(torch.finfo(probabilities.dtype).tiny)
            kept_probs = kept_probs / kept_probs.sum()
            self.vocab = tuple(self.vocab[index] for index in kept_indices)
            self._piece_to_index = {piece: index for index, piece in enumerate(self.vocab)}
            self.piece_logits = nn.Parameter(kept_probs.log())
            self.fit_em(texts, iterations)
        return self
