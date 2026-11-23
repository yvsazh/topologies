import math

import torch
from torch import nn


class TalkingHeads(nn.Module):
    component_id = "talking-heads"

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_dim=None,
        output_dim=None,
        key_dim=None,
        value_dim=None,
        logit_heads=None,
        value_heads=None,
        projection_bias=False,
        fixed_logit_scale=None,
    ):
        super().__init__()
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (embed_dim, num_heads)):
            raise ValueError("embed_dim and num_heads must be positive integers")
        context_dim = embed_dim if context_dim is None else context_dim
        output_dim = embed_dim if output_dim is None else output_dim
        logit_heads = num_heads if logit_heads is None else logit_heads
        value_heads = num_heads if value_heads is None else value_heads
        if key_dim is None:
            if embed_dim % num_heads:
                raise ValueError("embed_dim must be divisible by num_heads when key_dim is omitted")
            key_dim = embed_dim // num_heads
        value_dim = key_dim if value_dim is None else value_dim
        dimensions = (context_dim, output_dim, key_dim, value_dim, logit_heads, value_heads)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in dimensions):
            raise ValueError("all attention dimensions must be positive integers")
        if projection_bias is not True and projection_bias is not False:
            raise ValueError("projection_bias must be a bool")
        if fixed_logit_scale is not None:
            if isinstance(fixed_logit_scale, bool) or not isinstance(fixed_logit_scale, (int, float)):
                raise ValueError("fixed_logit_scale must be a finite number or None")
            try:
                if not math.isfinite(float(fixed_logit_scale)):
                    raise ValueError("fixed_logit_scale must be a finite number or None")
            except OverflowError:
                raise ValueError("fixed_logit_scale must be a finite number or None") from None
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.logit_heads = logit_heads
        self.value_heads = value_heads
        self.fixed_logit_scale = fixed_logit_scale
        self.query_projection = nn.Linear(embed_dim, num_heads * key_dim, bias=projection_bias)
        self.key_projection = nn.Linear(context_dim, num_heads * key_dim, bias=projection_bias)
        self.value_projection = nn.Linear(context_dim, value_heads * value_dim, bias=projection_bias)
        self.logit_mix = nn.Parameter(torch.empty(num_heads, logit_heads))
        self.weight_mix = nn.Parameter(torch.empty(logit_heads, value_heads))
        self.output_projection = nn.Linear(value_dim * value_heads, output_dim, bias=projection_bias)
        nn.init.xavier_uniform_(self.logit_mix)
        nn.init.xavier_uniform_(self.weight_mix)

    def forward(self, x, context=None):
        context = x if context is None else context
        if not torch.is_tensor(x) or not torch.is_tensor(context):
            raise ValueError("x and context must be tensors with shapes [B,N,embed_dim] and [B,M,context_dim]")
        if x.ndim != 3 or context.ndim != 3:
            raise ValueError("x and context must have shapes [B,N,embed_dim] and [B,M,context_dim]")
        if x.shape[0] != context.shape[0]:
            raise ValueError("x and context must have matching batch sizes")
        if x.shape[1] == 0 or context.shape[1] == 0:
            raise ValueError("x and context must contain at least one sequence position")
        if x.shape[-1] != self.embed_dim or context.shape[-1] != self.context_dim:
            raise ValueError("x or context feature width does not match the configured dimension")
        if not torch.is_floating_point(x) or not torch.is_floating_point(context):
            raise ValueError("x and context must have floating point values")
        if x.dtype != context.dtype or x.device != context.device:
            raise ValueError("x and context must have matching dtype and device")
        batch_size, query_length, _ = x.shape
        memory_length = context.shape[1]
        query = self.query_projection(x).reshape(
            batch_size, query_length, self.num_heads, self.key_dim
        ).transpose(-1, -2)
        key = self.key_projection(context).reshape(
            batch_size, memory_length, self.num_heads, self.key_dim
        ).transpose(-1, -2)
        value = self.value_projection(context).reshape(
            batch_size, memory_length, self.value_heads, self.value_dim
        ).transpose(-1, -2)
        logits = torch.einsum("bnkh,bmkh->bnmh", query, key)
        if self.fixed_logit_scale is not None:
            logits = logits * self.fixed_logit_scale
        logits = torch.einsum("bnmh,hg->bnmg", logits, self.logit_mix)
        weights = torch.softmax(logits, dim=2)
        mixed_weights = torch.einsum("bnmg,gv->bnmv", weights, self.weight_mix)
        combined = torch.einsum("bnmv,bmdv->bndv", mixed_weights, value)
        return self.output_projection(combined.reshape(batch_size, query_length, -1))
