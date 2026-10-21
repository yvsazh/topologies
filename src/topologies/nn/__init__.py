"""Maintained toolbox of reusable components and generic model stacks.

``topologies.models`` / ``topologies.layers`` are the research catalog
(registry entries, one per paper). ``topologies.nn`` is the construction
kit: building blocks and size-configurable stacks that follow the
constructor rules (explicit geometry, fail-fast validation, dropout,
mask-aware token models, presets + parameter budgets via
``topologies.build``).
"""

from __future__ import annotations

from topologies.nn.aux import collect_aux_losses, normalize_aux_loss
from topologies.nn.causal import CausalLM
from topologies.nn.layers import (
    FeedForward,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    PatchEmbedding,
    ResidualConvBlock,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
    make_norm,
    padding_attention_mask,
    resize_pos_embedding,
)
from topologies.nn.mixers import Attention, build_mixer, register_mixer
from topologies.nn.models import (
    MLP,
    ConvNet,
    TabularMLP,
    TemporalConvNet,
    TransformerClassifier,
    TransformerForecaster,
    TransformerLM,
    TransformerTagger,
    VisionTransformer,
)
from topologies.nn.slots import Slot, register_attention, register_ffn, slot

__all__ = [
    "MLP",
    "Attention",
    "CausalLM",
    "ConvNet",
    "FeedForward",
    "LearnedPositionalEmbedding",
    "MultiHeadAttention",
    "PatchEmbedding",
    "RMSNorm",
    "ResidualConvBlock",
    "RotaryEmbedding",
    "Slot",
    "TabularMLP",
    "TemporalConvNet",
    "TransformerBlock",
    "TransformerClassifier",
    "TransformerForecaster",
    "TransformerLM",
    "TransformerTagger",
    "VisionTransformer",
    "build_mixer",
    "collect_aux_losses",
    "make_norm",
    "normalize_aux_loss",
    "padding_attention_mask",
    "register_attention",
    "register_ffn",
    "register_mixer",
    "resize_pos_embedding",
    "slot",
]
