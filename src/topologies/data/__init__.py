"""Data utilities: adapt real datasets to recipe batch contracts.

Every recipe declares its batch contract (``expected_batch_spec()``); the
helpers here produce exactly that, so any dataset/loader works with
``Trainer`` without hand-written wrapper classes.

Core (torch only): adapters, collation, splits, text blocks.
``image_folder`` needs ``pip install 'topologies[vision]'``. Named datasets
are only *downloaded* here (``download``); loading is always generic.
"""

from __future__ import annotations

from topologies.data.adapters import (
    DictDataset,
    MapDataset,
    TensorDictDataset,
    as_dict_dataset,
    dict_collate,
    for_recipe,
    recipe_keys,
    tuple_to_dict,
)
from topologies.data.collate import PadCollator, pad_collate
from topologies.data.corpus import TextCorpus
from topologies.data.datasets import image_folder
from topologies.data.download import download, list_datasets
from topologies.data.loaders import dataset_collate_fn, make_loader
from topologies.data.samplers import ShardedEvalSampler
from topologies.data.splits import dataset_labels, take_subset, train_val_split
from topologies.data.stream import TokenStream, write_token_stream
from topologies.data.text import (
    SequenceDataset,
    TokenBlockDataset,
    lm_dataset,
    pack_sequences,
    read_documents,
    read_text,
    sliding_blocks,
    split_documents,
)
from topologies.data.windows import TokenWindows


def from_synth(name_or_ds, split="train", recipe=None, **kw):
    """Torch dataset from a synthetic dataset split (``topologies.synth``)."""
    from topologies.synth.tasks import from_synth as _fs

    return _fs(name_or_ds, split, recipe, **kw)


__all__ = [
    "DictDataset",
    "MapDataset",
    "PadCollator",
    "SequenceDataset",
    "ShardedEvalSampler",
    "TensorDictDataset",
    "TextCorpus",
    "TokenBlockDataset",
    "TokenStream",
    "TokenWindows",
    "as_dict_dataset",
    "dataset_collate_fn",
    "dataset_labels",
    "dict_collate",
    "download",
    "for_recipe",
    "from_synth",
    "image_folder",
    "list_datasets",
    "lm_dataset",
    "make_loader",
    "pack_sequences",
    "pad_collate",
    "read_documents",
    "read_text",
    "recipe_keys",
    "sliding_blocks",
    "split_documents",
    "take_subset",
    "train_val_split",
    "tuple_to_dict",
    "write_token_stream",
]
