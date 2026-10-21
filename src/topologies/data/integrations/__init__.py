"""Explicit local/integration data bridges with lazy optional dependencies."""

from .karate import karate_club
from .local import (
    CartPoleAdapter,
    ImageFolder,
    collate_wavs,
    encode_text_documents,
    fit_csv_pipeline,
    image_folder,
    load_pcm_wav,
    midi_to_event_tokens,
    paired_caption_manifest,
    paired_retrieval_batch,
    read_csv_rows,
    text_windows_by_split,
)

__all__ = [
    "CartPoleAdapter", "ImageFolder", "collate_wavs", "encode_text_documents",
    "fit_csv_pipeline", "image_folder", "karate_club", "load_pcm_wav",
    "midi_to_event_tokens", "paired_caption_manifest", "paired_retrieval_batch",
    "read_csv_rows", "text_windows_by_split",
]
