import math

import torch
from torch import nn


class BM25(nn.Module):
    component_id = "bm25"

    def __init__(self, k1=1.2, b=0.75):
        super().__init__()
        try:
            k1_value = float(k1)
            k1_valid = math.isfinite(k1) and k1 > 0
        except (TypeError, ValueError, OverflowError):
            k1_value = 0.0
            k1_valid = False
        try:
            b_value = float(b)
            b_valid = math.isfinite(b) and 0 <= b <= 1
        except (TypeError, ValueError, OverflowError):
            b_value = 0.0
            b_valid = False
        if not k1_valid:
            raise ValueError("k1 must be finite and greater than zero")
        if not b_valid:
            raise ValueError("b must be finite and between zero and one")
        self.k1 = k1_value
        self.b = b_value

    def forward(self, query_tf, document_tf):
        if not isinstance(query_tf, torch.Tensor) or not isinstance(document_tf, torch.Tensor):
            raise TypeError("query_tf and document_tf must be tensors")
        if query_tf.ndim != 2 or document_tf.ndim != 3:
            raise ValueError("query_tf must have shape [B,V] and document_tf [B,D,V]")
        if query_tf.shape[0] != document_tf.shape[0] or query_tf.shape[1] != document_tf.shape[2]:
            raise ValueError("query_tf and document_tf batch and vocabulary shapes must match")
        if any(size == 0 for size in (*query_tf.shape, *document_tf.shape)):
            raise ValueError("batch, document, and vocabulary dimensions must be nonempty")
        if query_tf.device != document_tf.device:
            raise ValueError("query_tf and document_tf must be on the same device")
        if query_tf.is_complex() or document_tf.is_complex():
            raise ValueError("term counts must be real numeric tensors")
        if not torch.isfinite(query_tf).all() or not torch.isfinite(document_tf).all():
            raise ValueError("term counts must be finite")
        if (query_tf < 0).any() or (document_tf < 0).any():
            raise ValueError("term counts must be nonnegative")

        score_dtype = torch.promote_types(query_tf.dtype, document_tf.dtype)
        if not score_dtype.is_floating_point:
            score_dtype = torch.get_default_dtype()
        query_values = query_tf.to(dtype=score_dtype)
        document_values = document_tf.to(dtype=score_dtype)
        document_lengths = document_values.sum(dim=-1)
        average_length = document_lengths.mean(dim=1, keepdim=True)
        if self.b > 0 and (average_length == 0).any():
            raise ValueError("average document length must be positive when b is greater than zero")
        safe_average_length = torch.where(average_length > 0, average_length, torch.ones_like(average_length))
        length_normalization = (1 - self.b) + self.b * document_lengths / safe_average_length
        document_frequency = (document_values > 0).sum(dim=1).to(dtype=score_dtype)
        document_count = document_tf.shape[1]
        inverse_document_frequency = torch.log(
            (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        term_frequency_factor = document_values / (
            self.k1 * length_normalization.unsqueeze(-1) + document_values
        )
        query_weights = query_values * inverse_document_frequency
        return (query_weights.unsqueeze(1) * term_frequency_factor).sum(dim=-1)
