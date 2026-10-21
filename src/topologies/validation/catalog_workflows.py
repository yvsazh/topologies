"""Small, deterministic synthetic training workflows for foundational catalog models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topologies import synth


def _tabular(ctx: Any) -> dict[str, Any]:
    import torch

    features = torch.randn(6, generator=ctx.rng.torch)
    label = int(features[0] > 0)
    return {"features": features, "labels": label}


def _image(ctx: Any) -> dict[str, Any]:
    import torch

    image = torch.rand((1, 32, 32), generator=ctx.rng.torch)
    label = int(image[:, :, :16].mean() > image[:, :, 16:].mean())
    return {"image": synth.Image(image), "label": label}


def _reconstruction(ctx: Any) -> dict[str, Any]:
    import torch

    latent = torch.randn(2, generator=ctx.rng.torch)
    x = torch.stack((latent[0], latent[1], latent[0] + latent[1], latent[0] - latent[1],
                     latent[0] * 0.5, latent[1] * 0.5, latent.sum(), latent.prod()))
    return {"x": x}


def _tokens(ctx: Any) -> dict[str, Any]:
    import torch

    start = int(torch.randint(0, 16, (), generator=ctx.rng.torch))
    values = [(start + i) % 16 for i in range(9)]
    return {"tokens": values}


def _transformer_xl_tokens(ctx: Any) -> dict[str, Any]:
    import torch

    start = int(torch.randint(0, 16, (), generator=ctx.rng.torch))
    return {"tokens": [(start + i) % 16 for i in range(13)]}


def _masked_lm(ctx: Any) -> dict[str, Any]:
    import torch

    start = 1 + int(ctx.index) % 7
    tokens = [1 + ((start + i) % 15) for i in range(9)]
    position = int(torch.randint(1, len(tokens) - 1, (), generator=ctx.rng.torch))
    labels = [-100] * len(tokens)
    labels[position] = tokens[position]
    tokens[position] = 1
    return {"tokens": tokens, "labels": labels}


def _modernbert_tokens(ctx: Any) -> dict[str, Any]:
    import torch

    tokens = [2, *(5 + (int(ctx.index) + offset) % 23 for offset in range(6)), 3]
    position = int(torch.randint(1, len(tokens) - 1, (), generator=ctx.rng.torch))
    labels = [-100] * len(tokens)
    labels[position] = tokens[position]
    tokens[position] = 4
    return {"tokens": tokens, "labels": labels}


def _distilbert_masked_sample(ctx: Any) -> dict[str, Any]:
    import torch

    from topologies.training.recipes.lm import distilbert_special_token_ids

    constructor = _ENTRY_SPECS["distilbert-encoder"][1]
    controls = distilbert_special_token_ids(constructor["vocab_size"])
    excluded = set(controls)
    ordinary_ids = [token for token in range(constructor["vocab_size"]) if token not in excluded]
    body_length = min(constructor["context_length"] - 2, len(ordinary_ids))
    ordinary = [ordinary_ids[(int(ctx.index) + offset) % len(ordinary_ids)]
                for offset in range(body_length)]
    tokens = [controls[2], *ordinary, controls[3]]
    position = 1 + int(torch.randint(0, len(ordinary), (), generator=ctx.rng.torch))
    labels = [-100] * len(tokens)
    labels[position] = tokens[position]
    tokens[position] = controls[4]
    return {"tokens": tokens, "labels": labels}


def _roberta_mlm(ctx: Any) -> dict[str, Any]:
    import torch

    tokens = [0, *(4 + (int(ctx.index) + i) % 11 for i in range(7)), 2]
    position = int(torch.randint(1, 8, (), generator=ctx.rng.torch))
    labels = [-100] * len(tokens)
    labels[position] = tokens[position]
    tokens[position] = 15
    return {"tokens": tokens, "labels": labels}


def _electra_tokens(ctx: Any) -> dict[str, Any]:
    base = 5 + int(ctx.index) % 6
    return {"tokens": [2, base, base + 1, base + 2, base + 3, base + 4, base + 5, 3],
             "mask": [True] * 8, "token_type_ids": [0] * 8}


def _deberta_masked_sample(ctx: Any) -> dict[str, Any]:
    import torch

    length = 8
    vocab = 32
    ordinary = [5 + (int(ctx.index) + offset) % (vocab - 5) for offset in range(length - 2)]
    tokens = [2, *ordinary, 3]
    position = 1 + int(torch.randint(0, len(ordinary), (), generator=ctx.rng.torch))
    labels = [-100] * length
    labels[position] = tokens[position]
    tokens[position] = 4
    return {"tokens": tokens, "labels": labels}


def _permutation_tokens(ctx: Any) -> dict[str, Any]:
    start = int(ctx.index) % 24
    return {"tokens": [1 + (start + i) % 31 for i in range(8)], "mask": [True] * 8,
            "token_type_ids": [0] * 4 + [1] * 4}


def _spanbert_tokens(ctx: Any) -> dict[str, Any]:
    import torch

    body = [5 + (int(ctx.index) * 3 + offset) % 27 for offset in range(6)]
    return {"tokens": [2, *body, 3], "word_ids": [-1, 0, 0, 1, 2, 2, 3, -1]}


class _SpanBERTPretrainingDataset:
    def __init__(self, dataset, *, seed: int, vocab_size: int):
        self.dataset, self.seed, self.vocab_size = dataset, seed, vocab_size

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        if recipe != "spanbert-pretraining":
            raise ValueError(f"SpanBERT workflow requires spanbert-pretraining, got {recipe!r}")
        from topologies.models.spanbert import SpanMaskingCollator

        view = self.dataset.view(split)
        records = view.to_records(meta=False)
        batches = [records[index:index + 4] for index in range(0, len(records), 4)]
        seed, vocab_size = self.seed, self.vocab_size
        class SpanLoader:
            batch_size = 4

            def __iter__(self):
                if split == "train":
                    current_epoch = getattr(self, "_epoch", 0)
                    self._epoch = current_epoch + 1
                else:
                    current_epoch = 0
                for batch_index, rows in enumerate(batches):
                    collator = SpanMaskingCollator(vocab_size=vocab_size, mask_token_id=4,
                        pad_token_id=0, special_token_ids=(0, 1, 2, 3, 4), masking_ratio=.15,
                        geometric_p=.2, span_lower=1, span_upper=6, max_span_length=200,
                        seed=seed + current_epoch * 100000 + batch_index if split == "train"
                        else seed + 10000 + batch_index)
                    tokens = [torch.tensor(row["tokens"], dtype=torch.long) for row in rows]
                    words = [torch.tensor(row["word_ids"], dtype=torch.long) for row in rows]
                    width = max(t.numel() for t in tokens)
                    padded = torch.zeros((len(tokens), width), dtype=torch.long)
                    word_ids = torch.full_like(padded, -1)
                    for index, (token, word) in enumerate(zip(tokens, words)):
                        padded[index, :token.numel()] = token
                        word_ids[index, :word.numel()] = word
                    yield collator(padded, word_ids)

            def __len__(self):
                return len(batches)
        import torch
        return SpanLoader()


def _albert_documents(ctx: Any) -> dict[str, Any]:
    base = 5 + (int(getattr(ctx, "index", ctx)) % 5)
    return {"document": [[base, base + 1], [base + 2, base + 3], [base + 4, base + 5]]}


class _AlbertPretrainingDataset:
    def __init__(self, *, seed: int, size: int):
        self.seed, self.size = seed, size

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        if recipe != "albert-pretraining":
            raise ValueError(f"ALBERT workflow requires albert-pretraining, got {recipe!r}")
        from topologies.models.bert_encoder import ALBERTPretrainingCollator

        start = 0 if split == "train" else self.size // 2
        docs = [_albert_documents(i + self.seed)["document"]
                for i in range(start, start + max(4, self.size // 2))]

        class DocumentLoader:
            batch_size = 2

            def __iter__(self):
                for offset in range(0, len(docs), self.batch_size):
                    yield docs[offset:offset + self.batch_size]

            def __len__(self):
                return (len(docs) + self.batch_size - 1) // self.batch_size

        collator = ALBERTPretrainingCollator(vocab_size=16, mask_token_id=4,
            pad_token_id=0, cls_token_id=2, sep_token_id=3, special_token_ids=(0, 1, 2, 3, 4),
            seed=self.seed + (0 if split == "train" else 1))
        return collator.wrap(DocumentLoader())


class _DistilBertPretrainingDataset:
    def __init__(self, *, seed: int, size: int, vocab_size: int,
                 context_length: int, pad_token_id: int = 0):
        self.seed, self.size = seed, size
        self.vocab_size, self.context_length = vocab_size, context_length
        self.pad_token_id = pad_token_id

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        if recipe != "distilbert-pretraining":
            raise ValueError(f"DistilBERT workflow requires distilbert-pretraining, got {recipe!r}")
        from topologies.training import make_synthetic_loader
        from topologies.training.recipes.lm import distilbert_special_token_ids

        controls = distilbert_special_token_ids(self.vocab_size)
        return make_synthetic_loader(recipe, batch_size=4, steps=max(1, self.size // 4),
            seed=self.seed + (0 if split == "train" else self.size), vocab_size=self.vocab_size,
            context_length=self.context_length, pad_token_id=self.pad_token_id,
            unknown_token_id=controls[1], cls_token_id=controls[2],
            sep_token_id=controls[3], mask_token_id=controls[4], special_token_ids=controls)


class _ElectraPretrainingDataset:
    def __init__(self, *, seed: int, size: int, vocab_size: int, context_length: int):
        self.seed, self.size = seed, size
        self.vocab_size, self.context_length = vocab_size, context_length

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        if recipe != "electra-pretraining":
            raise ValueError(f"ELECTRA workflow requires electra-pretraining, got {recipe!r}")
        from topologies.training import make_synthetic_loader

        return make_synthetic_loader(recipe, batch_size=4, steps=max(1, self.size // 4),
            seed=self.seed + (0 if split == "train" else self.size),
            vocab_size=self.vocab_size, context_length=self.context_length,
            pad_token_id=0, unknown_token_id=1, cls_token_id=2,
            sep_token_id=3, mask_token_id=4, special_token_ids=(0, 1, 2, 3, 4))


def _tabular_mixed(ctx: Any) -> dict[str, Any]:
    import torch

    return {"numeric_a": float(torch.randn((), generator=ctx.rng.torch)),
            "numeric_b": float(torch.randn((), generator=ctx.rng.torch)),
            "numeric_c": float(torch.randn((), generator=ctx.rng.torch)),
            "labels": int(torch.randint(0, 3, (), generator=ctx.rng.torch))}


def _text_classification(ctx: Any) -> dict[str, Any]:
    import torch

    label = int(torch.randint(0, 3, (), generator=ctx.rng.torch))
    return {"text": ("amber " if label == 0 else "blue " if label == 1 else "coral ") * 3,
            "label": label}


def _tagged_text(ctx: Any) -> dict[str, Any]:
    import torch

    picked = int(torch.randint(0, 4, (), generator=ctx.rng.torch))
    words = ["oak", "elm", "ash", "fir"]
    return {"text": " ".join(words), "spans": [{"start": sum(len(w) + 1 for w in words[:picked]),
            "end": sum(len(w) + 1 for w in words[:picked]) + len(words[picked]), "label": "tree"}]}


def _sequence_regression(ctx: Any, *, per_step: bool = False, features: int = 2,
                         target_size: int = 3, length: int = 8) -> dict[str, Any]:
    import torch

    x = torch.randn(length, features, generator=ctx.rng.torch)
    target = torch.stack((x[:, 0], x[:, 1]), dim=-1) if per_step else \
        torch.stack((x[-1, 0], x[-1, 1], x[-1].sum()))[:target_size]
    return {"inputs": synth.Tensor(x), "targets": synth.Tensor(target)}


def _forecast(ctx: Any) -> dict[str, Any]:
    import torch

    x = torch.randn(8, generator=ctx.rng.torch)
    y = torch.stack((x[-1], x[-1] * 0.5, x[-1] * -0.25))
    return {"inputs": synth.Tensor(x), "targets": synth.Tensor(y)}


def _recurrent_predictor_data(ctx: Any) -> dict[str, Any]:
    return _sequence_regression(ctx, features=2, target_size=3)


def _temporal_convnet_data(ctx: Any) -> dict[str, Any]:
    return _sequence_regression(ctx, features=2, target_size=3, length=8)


def _diagonal_ssm_data(ctx: Any) -> dict[str, Any]:
    return _sequence_regression(ctx, per_step=True, features=2, target_size=2, length=8)


def _seq2seq(ctx: Any) -> dict[str, Any]:
    import torch

    a = int(torch.randint(0, 10, (), generator=ctx.rng.torch))
    return {"source": f"{a} {a + 1} {a + 2}", "target": f"{a + 2} {a + 1} {a}"}


def _image_reconstruction(ctx: Any) -> dict[str, Any]:
    import torch

    image = torch.rand((1, 16, 16), generator=ctx.rng.torch)
    return {"x": synth.Tensor(image)}


def _conv_autoencoder(ctx: Any) -> dict[str, Any]:
    import torch

    return {"x": synth.Tensor(torch.rand((1, 16, 16), generator=ctx.rng.torch))}


def _latent_image(ctx: Any) -> dict[str, Any]:
    import torch

    return {"x": synth.Tensor(torch.rand((1, 16, 16), generator=ctx.rng.torch) * 2 - 1)}


def _density(ctx: Any) -> dict[str, Any]:
    import torch

    z = torch.randn(2, generator=ctx.rng.torch)
    return {"x": synth.Tensor(torch.stack((z[0], z[1] + 0.5 * z[0])))}


def _waveform(ctx: Any) -> dict[str, Any]:
    import torch

    t = torch.arange(64, dtype=torch.float32) / 8000
    label = int(ctx.index) % 3
    return {"audio": synth.Audio(torch.sin(2 * torch.pi * (220 + int(ctx.index) % 80) * t), sr=8000),
            "label": label}


def _ctc_sample(ctx: Any) -> dict[str, Any]:
    import torch

    tok = 1 + int(ctx.index) % 3
    x = torch.sin(torch.arange(64, dtype=torch.float32) * (tok + 1) * 0.12)
    return {"features": synth.Tensor(x), "targets": [tok]}


def _paired_image_text(ctx: Any) -> dict[str, Any]:
    import torch

    ident = int(ctx.index) % 4
    x = torch.zeros((1, 16, 16)); x[:, ident * 4:(ident + 1) * 4, :] = 1
    return {"image": synth.Image(x), "text": f"shape {ident}"}


def _graph_node(ctx: Any) -> dict[str, Any]:
    import torch

    n = 4
    x = torch.randn(n, 2, generator=ctx.rng.torch)
    x[:, 0] += int(ctx.index) % 2
    edge = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    return {"x": synth.Tensor(x), "edge_index": edge.tolist(),
            "y": [int(xi[0] > 0) for xi in x], "train_mask": [True, True, False, False]}


def _boxes(ctx: Any) -> dict[str, Any]:
    import torch

    image = torch.rand((1, 32, 32), generator=ctx.rng.torch)
    x = 4 + int(ctx.index) % 8
    return {"image": synth.Image(image), "boxes": [[float(x), 6., float(x + 12), 24.]], "labels": [0]}


def _audio_controls(ctx: Any) -> dict[str, Any]:
    import torch

    pitch = torch.full((4,), 200. + int(ctx.index) % 50)
    loudness = torch.full((4,), 0.2 + (int(ctx.index) % 3) * 0.1)
    t = torch.arange(128, dtype=torch.float32) / 8000
    audio = loudness[0] * torch.sin(2 * torch.pi * pitch[0] * t)
    return {"audio": synth.Audio(audio, sr=8000), "pitch": pitch, "loudness": loudness,
            "sample_rate": 8000, "lengths": 128}


def _transition(ctx: Any) -> dict[str, Any]:
    import torch

    s = torch.randn(4, generator=ctx.rng.torch)
    ns = s * 0.8
    action = int(ctx.index) % 3
    return {"state": synth.Tensor(s), "action": action, "reward": float(1.0 - s.square().mean()),
            "next_state": synth.Tensor(ns), "terminated": int(ctx.index) % 7 == 6}


def _ppo_transition(ctx: Any) -> dict[str, Any]:
    import torch

    observation = torch.randn(4, generator=ctx.rng.torch)
    return {"obs": synth.Tensor(observation), "observation": synth.Tensor(observation),
            "next_observation": synth.Tensor(torch.randn(4, generator=ctx.rng.torch)),
            "reward": 1.0 if int(ctx.index) % 2 == 0 else -0.25}


def _segmentation(ctx: Any) -> dict[str, Any]:
    import torch

    image = torch.rand((1, 16, 16), generator=ctx.rng.torch)
    mask = torch.zeros((16, 16), dtype=torch.long)
    mask[4:12, 4:12] = 1
    return {"pixels": synth.Tensor(image), "mask": synth.Tensor(mask)}


def _sequence_wave(ctx: Any) -> dict[str, Any]:
    import torch

    x = torch.randint(0, 16, (32,), generator=ctx.rng.torch)
    return {"tokens": x.tolist()}


def _gan_real(ctx: Any) -> dict[str, Any]:
    import torch

    return {"x": synth.Tensor(torch.rand((1, 16, 16), generator=ctx.rng.torch) * 2 - 1)}


def _recommendation(ctx: Any) -> dict[str, Any]:
    import torch

    return {"user": int(torch.randint(0, 8, (), generator=ctx.rng.torch)),
            "item": int(torch.randint(0, 12, (), generator=ctx.rng.torch)),
            "label": float(int(ctx.index) % 2)}


def _deepfm(ctx: Any) -> dict[str, Any]:
    import torch

    x = torch.randint(0, 4, (3,), generator=ctx.rng.torch)
    return {"x": x.tolist(), "label": float(int(x.sum()) % 2)}


class _SynthPPOEnv:
    def __init__(self, records):
        from topologies.synth.valuetypes import Type

        self.records = [{k: Type.unwrap(v) for k, v in r.items()} for r in records]
        self.position = 0

    def reset(self):
        self.position = 0
        return self.records[0]["observation"].float()

    def step(self, action):
        record = self.records[self.position % len(self.records)]
        self.position += 1
        done = self.position % len(self.records) == 0
        return record["next_observation"].float(), float(record["reward"]), done, False, {"action": int(action)}


_ENTRY_SPECS = {
    "efficient-attention": ("masked-lm", {"vocab_size": 32, "context_length": 8,
        "d_model": 16, "num_layers": 1, "num_heads": 2, "projection_rank": 2,
        "feature_count": 8, "bucket_count": 2, "mode": "linear", "dropout": 0.0}, _modernbert_tokens),
    "bart": ("seq2seq-lm", {"vocab_size": 32, "context_length": 16, "d_model": 16,
        "num_layers": 1, "num_decoder_layers": 1, "num_heads": 2, "d_ff": 32}, None),
    "t5": ("seq2seq-lm", {"vocab_size": 32, "context_length": 16, "d_model": 16,
        "num_layers": 1, "num_decoder_layers": 1, "num_heads": 2, "d_kv": 4,
        "d_ff": 32, "num_sentinels": 4}, _tokens),
    "ul2": ("seq2seq-lm", {"vocab_size": 32, "context_length": 16, "d_model": 16,
        "num_layers": 2, "num_heads": 2, "d_kv": 8, "d_ff": 24, "num_experts": 4}, _tokens),
    "marian": ("seq2seq-lm", {"vocab_size": 32, "context_length": 16, "d_model": 16,
        "encoder_layers": 1, "decoder_layers": 1, "heads": 2, "ff": 24}, _tokens),
    "byt5": ("seq2seq-lm", {"vocab_size": 384, "context_length": 128, "d_model": 16,
        "num_layers": 2, "num_decoder_layers": 1, "num_heads": 2, "d_kv": 8, "d_ff": 32}, _seq2seq),
    "charformer": ("seq2seq-lm", {"vocab_size": 258, "context_length": 32, "d_model": 16,
        "num_layers": 1, "num_heads": 2, "intermediate_size": 32,
        "max_block_size": 3, "downsample_rate": 2}, _seq2seq),
    "canine": ("canine-c", {"context_length": 32, "d_model": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "hash_buckets": 64, "num_hashes": 4, "downsample_rate": 4, "local_block_size": 8}, None),
    "modernbert": ("masked-lm", {"vocab_size": 32, "context_length": 16,
        "d_model": 16, "num_layers": 2, "num_heads": 2, "intermediate_size": 24,
        "local_window": 4, "global_attn_every_n_layers": 2, "pad_token_id": 0,
        "mask_token_id": 4}, _modernbert_tokens),
    "longformer": ("masked-lm", {"vocab_size": 32, "context_length": 64,
        "d_model": 16, "num_layers": 2, "num_heads": 2,
        "intermediate_size": 24, "attention_window": [8, 8],
        "pad_token_id": 0, "mask_token_id": 4}, _modernbert_tokens),
    "bigbird": ("masked-lm", {"vocab_size": 32, "context_length": 32,
        "hidden_size": 16, "num_layers": 2, "num_heads": 2,
        "intermediate_size": 32, "block_size": 4, "window_blocks": 1,
        "num_random_blocks": 1}, _modernbert_tokens),
    "bigbird-attention": ("attention", {"hidden_size": 8, "num_heads": 2,
        "block_size": 2, "window_blocks": 1, "num_random_blocks": 1,
        "dropout": 0.0, "seed": 17}, None),
    "spanbert": ("spanbert-pretraining", {"vocab_size": 32, "context_length": 8,
        "d_model": 16, "num_layers": 1, "num_heads": 4, "mask_token_id": 4,
        "max_span_length": 200}, _spanbert_tokens),
    "xlnet": ("permutation-lm", {"vocab_size": 32, "context_length": 8, "d_model": 16,
        "num_layers": 2, "num_heads": 4, "mem_len": 8, "pad_token_id": 0}, _permutation_tokens),
    "mlp": ("tabular-classify", {"in_features": 6, "out_features": 2, "hidden_dims": [12]}, _tabular),
    "compact-convnet": ("vision-classify", {"in_channels": 1, "num_classes": 2, "widths": [4, 8]}, _image),
    "autoencoder": ("autoencoder", {"in_features": 8, "latent_dim": 2, "hidden_dims": [8]}, _reconstruction),
    "transformer-lm": ("causal-lm", {"vocab_size": 16, "context_length": 9, "d_model": 16, "num_layers": 1, "num_heads": 2}, _tokens),
    "gpt-2": ("causal-lm", {"vocab_size": 16, "context_length": 9, "d_model": 16, "num_layers": 1, "num_heads": 2}, _tokens),
    "s4": ("causal-lm", {"vocab_size": 16, "d_model": 8, "state_size": 4,
        "n_layers": 1, "max_seq": 8, "dropout": 0.0}, _tokens),
    "dss-s5-mamba": ("causal-lm", {"vocab_size": 32, "d_model": 16, "state_size": 8,
        "num_layers": 1, "mode": "dss", "dropout": 0.0}, _tokens),
    "mamba2-gated-deltanet-h3": ("causal-lm", {"vocab_size": 64, "context_length": 12,
        "d_model": 16, "state_size": 8, "num_layers": 1, "mode": "mamba2",
        "dropout": 0.0}, _tokens),
    "hyena": ("causal-lm", {"vocab_size": 16, "context_length": 12, "d_model": 8,
        "num_layers": 1, "filter_dim": 8}, _tokens),
    "rwkv": ("causal-lm", {"vocab_size": 16, "context_length": 8, "d_model": 8,
        "num_layers": 1, "num_heads": 2, "mode": "rwkv4"}, _tokens),
    "retnet": ("causal-lm", {"vocab_size": 16, "context_length": 8, "d_model": 8,
        "num_layers": 1, "num_heads": 2, "mode": "chunkwise", "chunk_size": 2}, _tokens),
    "xlstm": ("causal-lm", {"vocab_size": 16, "context_length": 8, "d_model": 8,
        "num_layers": 1, "mode": "slstm"}, _tokens),
    "jamba-griffin-hawk": ("causal-lm", {"vocab_size": 16, "context_length": 8, "d_model": 8,
        "num_layers": 2, "num_heads": 2, "mode": "jamba"}, _tokens),
    "rag-fid": ("seq2seq-lm", {"vocab_size": 64, "context_length": 16, "d_model": 16,
        "num_heads": 2, "num_passages": 2, "mode": "rag"}, None),
    "retro": ("causal-lm", {"vocab_size": 64, "context_length": 8, "d_model": 16,
        "num_heads": 2, "chunk_size": 2}, _tokens),
    "sentence-encoder": ("sentence-contrastive", {"vocab_size": 64, "context_length": 16,
        "d_model": 16, "pool": "mean"}, None),
    "static-embeddings": ("negative-sampling", {"vocab_size": 64, "d_model": 16,
        "ngram_buckets": 8, "mode": "skipgram"}, None),
    "transitional-text": ("text-classify", {"vocab_size": 64, "context_length": 16,
        "d_model": 16, "num_classes": 3, "sentence_width": 2, "mode": "multichannel"}, None),
    "attention-nmt": ("seq2seq-lm", {"vocab_size": 64, "context_length": 8, "d_model": 16,
        "mode": "bahdanau"}, None),
    "lenet": ("lenet-classify", {"in_channels": 1, "num_classes": 10, "width": 1}, None),
    "alexnet": ("alexnet-classify", {"in_channels": 3, "num_classes": 10, "width": 1}, None),
    "vgg": ("vgg-classify", {"in_channels": 3, "num_classes": 10, "depth": 11, "width": 1,
        "batch_norm": False}, None),
    "inception": ("inception-classify", {"in_channels": 3, "num_classes": 10, "width": 1,
        "mode": "v1"}, None),
    "resnet-variants": ("resnet-variant-classify", {"in_channels": 3, "num_classes": 10,
        "width": 1, "mode": "v15"}, None),
    "densenet": ("densenet-classify", {"in_channels": 3, "num_classes": 10, "depth": 121,
        "growth": 2}, None),
    "efficient-conv": ("efficient-conv-classify", {"in_channels": 3, "num_classes": 10,
        "width": 4, "mode": "mobilenet-v1"}, None),
    "modern-conv": ("modern-conv-classify", {"in_channels": 3, "num_classes": 10,
        "width": 4, "mode": "convnext-v2"}, None),
    "deit": ("deit-classify", {"in_channels": 3, "num_classes": 10, "d_model": 16,
        "mode": "distilled"}, None),
    "swin": ("swin-classify", {"in_channels": 3, "num_classes": 10, "width": 8,
        "mode": "swin"}, None),
    "token-mixer": ("token-mixer-classify", {"in_channels": 3, "num_classes": 10,
        "width": 8, "mode": "mixer"}, None),
    "vision-ssl": ("vision-ssl-classify", {"in_channels": 3, "num_classes": 10,
        "width": 8, "mode": "dino"}, None),
    "masked-image": ("masked-image-classify", {"in_channels": 3, "num_classes": 10,
        "width": 8, "mode": "mae"}, None),
    "contrastive-vision": ("contrastive-vision-classify", {"in_channels": 3, "num_classes": 10,
        "width": 8, "mode": "clip"}, None),
    "two-stage-detect": ("two-stage-classify", {"in_channels": 3, "num_classes": 10,
        "width": 8, "mode": "faster"}, None),
    "single-stage-detect": ("single-stage-classify", {"in_channels": 3, "num_classes": 10,
        "width": 8, "mode": "yolo"}, None),
    "set-detect": ("set-detect-classify", {"in_channels": 3, "num_classes": 10,
        "width": 8, "mode": "detr"}, None),
    "detect-ops": ("detect-ops-classify", {"in_channels": 3, "num_classes": 10, "width": 8}, None),
    "segment": ("segment-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "fcn"}, None),
    "prompt-seg": ("prompt-seg-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "sam"}, None),
    "panoptic": ("panoptic-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "panoptic-fpn"}, None),
    "pose": ("pose-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "openpose"}, None),
    "vision-neck": ("vision-neck-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "panet"}, None),
    "attn2d": ("attn2d-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "cbam"}, None),
    "vision-norm": ("vision-norm-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "group"}, None),
    "pos2d": ("pos2d-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "relative"}, None),
    "video-conv": ("video-conv-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "c3d"}, None),
    "video-transformer": ("video-transformer-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "timesformer"}, None),
    "video-ssl": ("video-ssl-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "cvrl"}, None),
    "point-cloud": ("point-cloud-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "pointnet"}, None),
    "voxel-detect": ("voxel-detect-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "voxelnet"}, None),
    "volume-seg": ("volume-seg-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "unet3d"}, None),
    "neural-field": ("neural-field-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "nerf"}, None),
    "implicit-surface": ("implicit-surface-classify", {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "occupancy"}, None),
    "gpt-j-neox-pythia": ("causal-lm", {"vocab_size": 16, "context_length": 9, "d_model": 16, "num_layers": 1, "num_heads": 2, "family": "neox"}, _tokens),
    "gpt-decoder-family": ("causal-lm", {"vocab_size": 16, "context_length": 9,
        "family": "gpt3", "d_model": 16, "num_layers": 1, "num_heads": 2}, _tokens),
    "llama": ("causal-lm", {"vocab_size": 16, "context_length": 9, "d_model": 16,
        "num_layers": 1, "num_heads": 2, "num_kv_heads": 2, "intermediate_size": 32}, _tokens),
    "mistral": ("causal-lm", {"vocab_size": 16, "context_length": 9, "d_model": 16,
        "num_layers": 1, "num_heads": 2, "num_kv_heads": 1, "intermediate_size": 32,
        "sliding_window": 4, "chunk_size": 3}, _tokens),
    "qwen": ("causal-lm", {"vocab_size": 16, "context_length": 9, "d_model": 16,
        "num_layers": 1, "num_heads": 2, "num_kv_heads": 1, "intermediate_size": 32,
        "qk_norm": True}, _tokens),
    "gemma": ("causal-lm", {"vocab_size": 16, "context_length": 9, "hidden_size": 16,
        "num_layers": 1, "num_heads": 2, "num_kv_heads": 1, "head_dim": 8,
        "intermediate_size": 24, "qk_norm": True}, _tokens),
    "phi": ("causal-lm", {"vocab_size": 16, "context_length": 9, "hidden_size": 16,
        "num_layers": 1, "num_heads": 2, "num_kv_heads": 1, "head_dim": 8,
        "intermediate_size": 24}, _tokens),
    "mixtral": ("mixtral-causal-lm", {"vocab_size": 64, "context_length": 12, "d_model": 16,
        "num_layers": 2, "num_heads": 4, "num_kv_heads": 2, "intermediate_size": 32,
        "num_experts": 4}, _tokens),
    "transformer-xl": ("transformer-xl-causal-lm", {"vocab_size": 32, "context_length": 8, "d_model": 16,
        "num_layers": 2, "num_heads": 4, "mem_len": 8}, _transformer_xl_tokens),
    "bert-encoder": ("masked-lm", {"vocab_size": 16, "context_length": 9, "d_model": 16,
                                     "num_layers": 1, "num_heads": 2}, _masked_lm),
    "roberta": ("masked-lm", {"vocab_size": 16, "context_length": 9, "d_model": 16,
                              "num_layers": 1, "num_heads": 2}, _roberta_mlm),
    "albert": ("albert-pretraining", {"vocab_size": 16, "context_length": 9, "d_model": 16,
                                       "embedding_size": 8, "num_layers": 2, "num_heads": 2}, _albert_documents),
    "distilbert-encoder": ("distilbert-pretraining", {"vocab_size": 31, "context_length": 8,
        "d_model": 16, "num_layers": 2, "num_heads": 4, "intermediate_size": 64,
        "pad_token_id": 0}, _distilbert_masked_sample),
    "electra": ("electra-pretraining", {"vocab_size": 16, "context_length": 8,
        "generator_hidden_size": 8, "discriminator_hidden_size": 16,
        "generator_layers": 1, "discriminator_layers": 1, "num_heads": 4,
        "pad_token_id": 0, "mask_token_id": 4}, _electra_tokens),
    "deberta": ("masked-lm", {"vocab_size": 32, "context_length": 8,
        "hidden_size": 16, "layers": 1, "heads": 4, "mask_token_id": 4}, _deberta_masked_sample),
    "deberta-v3": ("electra-pretraining", {"vocab_size": 16, "context_length": 8,
        "hidden_size": 16, "layers": 1, "heads": 4, "generator_hidden_size": 8,
        "generator_heads": 2, "generator_layers": 1, "pad_token_id": 0,
        "mask_token_id": 4}, _electra_tokens),
    "recurrent-lm": ("causal-lm", {"vocab_size": 16, "context_length": 9, "hidden_size": 16, "num_layers": 1, "cell": "gru"}, _tokens),
    "tabular-mlp": ("tabular-classify", {"numeric_features": 3, "categorical_cardinalities": [], "out_features": 3, "hidden_dims": [8]}, _tabular_mixed),
    "resnet": ("vision-classify", {"in_channels": 1, "num_classes": 3, "layers": [1, 1, 1, 1], "widths": [4, 8, 16, 32]}, _image),
    "mobilenet-v2": ("vision-classify", {"in_channels": 1, "num_classes": 3, "width_multiplier": 0.25}, _image),
    "convnext": ("vision-classify", {"in_channels": 1, "num_classes": 3, "depths": [1, 1, 1, 1], "widths": [4, 8, 16, 32]}, _image),
    "vision-transformer": ("vision-classify", {"in_channels": 1, "num_classes": 3, "img_size": [16, 16], "patch_size": 4, "embed_dim": 12, "depth": 1, "num_heads": 3}, _image),
    "transformer-classifier": ("text-classify", {"vocab_size": 64, "num_classes": 3, "context_length": 32, "d_model": 8, "num_layers": 1, "num_heads": 2}, _text_classification),
    "transformer-tagger": ("token-classify", {"vocab_size": 64, "num_labels": 3, "context_length": 32, "d_model": 8, "num_layers": 1, "num_heads": 2}, _tagged_text),
    "text-cnn": ("text-classify", {"vocab_size": 64, "num_classes": 3, "embedding_dim": 8, "filters": 4, "kernel_sizes": [2, 3], "context_length": 32}, _text_classification),
    "recurrent-predictor": ("regression", {"in_features": 2, "hidden_size": 8, "out_features": 3}, _recurrent_predictor_data),
    "temporal-convnet": ("regression", {"in_features": 2, "out_features": 3, "channels": [4, 4], "head": "last"}, _temporal_convnet_data),
    "nbeats": ("regression", {"backcast_length": 8, "forecast_length": 3, "hidden_width": 8, "stacks": 1, "blocks_per_stack": 1}, _forecast),
    "diagonal-ssm-model": ("regression", {"input_dim": 2, "width": 4, "state_size": 3, "output_dim": 2}, _diagonal_ssm_data),
    "seq2seq": ("seq2seq-lm", {"source_vocab_size": 64, "target_vocab_size": 64, "source_context_length": 16, "target_context_length": 16, "bos_id": 1, "eos_id": 2, "pad_id": 0, "d_model": 8, "num_layers": 1, "num_heads": 2}, _seq2seq),
    "conv-autoencoder": ("autoencoder", {"in_channels": 1, "image_size": [16, 16], "latent_channels": 2, "features": [4, 8]}, _conv_autoencoder),
    "vae": ("vae", {"in_features": 8, "latent_dim": 2, "hidden_dims": [8]}, _reconstruction),
    "conv-vae": ("vae", {"in_channels": 1, "image_size": [16, 16], "latent_channels": 2, "features": [4, 8]}, _image_reconstruction),
    "vqvae": ("vqvae", {"in_channels": 1, "image_size": [16, 16], "codebook_size": 4, "embedding_dim": 2, "features": [4, 8]}, _image_reconstruction),
    "actor-critic": ("ppo", {"observation_dim": 4, "action_count": 3, "hidden_dims": [8]}, _ppo_transition),
    "audio-crnn": ("audio-classify", {"num_classes": 3, "input_mode": "waveform", "sample_rate": 8000, "n_fft": 16, "hop_length": 4, "hidden_size": 8, "conv_channels": 4}, _waveform),
    "ctc-acoustic-model": ("ctc-asr", {"vocab_size": 5, "conv_channels": 4, "hidden_size": 8, "rnn_layers": 1, "blank_id": 0}, _ctc_sample),
    "conv-asr": ("ctc-asr", {"vocab_size": 5, "in_channels": 1, "width": 8, "mode": "quartznet", "blank_id": 4}, _ctc_sample),
    "dcgan-generator": ("gan", {"latent_dim": 4, "out_channels": 1, "image_size": 16, "features": 4}, _gan_real),
    "dcgan-discriminator": ("gan", {"in_channels": 1, "image_size": 16, "features": 4}, _gan_real),
    "ddsp-synthesizer": ("audio-reconstruct", {"sample_rate": 8000, "n_harmonics": 3, "control_rate": 100, "fir_size": 3}, _audio_controls),
    "deepfm": ("recommender-bce", {"field_cardinalities": [4, 4, 4], "embedding_dim": 2, "hidden_dims": [8]}, _deepfm),
    "diffusion-unet": ("diffusion", {"in_channels": 1, "features": [4, 8], "time_dim": 8}, _latent_image),
    "graph-network": ("graph-supervised", {"in_features": 2, "hidden_features": 4, "num_classes": 2, "layers": 2}, _graph_node),
    "image-text-dual-encoder": ("contrastive-pairs", {"vocab_size": 64, "context_length": 16, "embedding_dim": 8, "in_channels": 1, "width": 4, "text_width": 8, "text_heads": 2, "text_layers": 1}, _paired_image_text),
    "matrix-factorization": ("recommender-bce", {"num_users": 8, "num_items": 12, "embedding_dim": 4}, _recommendation),
    "q-network": ("td-value", {"observation_dim": 4, "num_actions": 3, "hidden_dims": [8]}, _transition),
    "realnvp": ("density-estimation", {"features": 2, "coupling_layers": 2, "hidden_dims": [8]}, _density),
    "retinanet": ("object-detection", {"in_channels": 1, "num_classes": 2, "feature_width": 8}, _boxes),
    "unet": ("segmentation", {"in_channels": 1, "out_channels": 2, "features": [4, 8]}, _segmentation),
    "wavenet": ("waveform-lm", {"quantization_channels": 16, "residual_channels": 4, "layers": 2}, _sequence_wave),
}


def dataset_for_entry(ident: str, root: str | Path, seed: int = 11, *, size: int = 32):
    """Generate an entry-specific synth dataset; resolve recipe views with ``ds.for_recipe``.

    Example: ``ds = dataset_for_entry("mlp", root="runs/mlp"); train = ds.for_recipe("tabular-classify", split="train")``.
    """
    if ident == "efficient-attention":
        return _ModernBERTWorkflowDataset(size=size, seed=seed)
    if ident == "t5":
        return _T5WorkflowDataset(size=size, seed=seed)
    if ident == "ul2":
        return _UL2WorkflowDataset(size=size, seed=seed)
    if ident == "marian":
        return _MarianWorkflowDataset(size=size, seed=seed)
    if ident == "bart":
        return _BARTWorkflowDataset(size=size, seed=seed)
    if ident == "byt5":
        return _ByT5WorkflowDataset(size=size, seed=seed)
    if ident == "charformer":
        return _CharformerWorkflowDataset(size=size, seed=seed)
    if ident == "canine":
        return _CanineWorkflowDataset(size=size, seed=seed)
    if ident == "rag-fid":
        return _RagFidWorkflowDataset(size=size, seed=seed)
    if ident == "sentence-encoder":
        return _SentenceEncoderWorkflowDataset(size=size, seed=seed)
    if ident == "static-embeddings":
        return _StaticEmbeddingsWorkflowDataset(size=size, seed=seed)
    if ident == "transitional-text":
        return _TransitionalTextWorkflowDataset(size=size, seed=seed)
    if ident == "attention-nmt":
        return _AttentionNMTWorkflowDataset(size=size, seed=seed)
    if ident == "lenet":
        return _LeNetWorkflowDataset(size=size, seed=seed)
    if ident == "alexnet":
        return _AlexNetWorkflowDataset(size=size, seed=seed)
    if ident == "vgg":
        return _VGGWorkflowDataset(size=size, seed=seed)
    if ident == "inception":
        return _InceptionWorkflowDataset(size=size, seed=seed)
    if ident == "resnet-variants":
        return _ResidualVariantWorkflowDataset(size=size, seed=seed)
    if ident == "densenet":
        return _DenseNetWorkflowDataset(size=size, seed=seed)
    if ident == "efficient-conv":
        return _EfficientConvWorkflowDataset(size=size, seed=seed)
    if ident == "modern-conv":
        return _ModernConvWorkflowDataset(size=size, seed=seed)
    if ident == "deit":
        return _DeiTWorkflowDataset(size=size, seed=seed)
    if ident == "swin":
        return _SwinWorkflowDataset(size=size, seed=seed)
    if ident == "token-mixer":
        return _TokenMixerWorkflowDataset(size=size, seed=seed)
    if ident == "vision-ssl":
        return _VisionSSLWorkflowDataset(size=size, seed=seed)
    if ident == "masked-image":
        return _MaskedImageWorkflowDataset(size=size, seed=seed)
    if ident == "contrastive-vision":
        return _ContrastiveVisionWorkflowDataset(size=size, seed=seed)
    if ident == "two-stage-detect":
        return _TwoStageWorkflowDataset(size=size, seed=seed)
    if ident == "single-stage-detect":
        return _SingleStageWorkflowDataset(size=size, seed=seed)
    if ident == "set-detect":
        return _SetDetectWorkflowDataset(size=size, seed=seed)
    if ident == "detect-ops":
        return _DetectOpsWorkflowDataset(size=size, seed=seed)
    if ident == "segment":
        return _SegmentWorkflowDataset(size=size, seed=seed)
    if ident == "prompt-seg":
        return _PromptSegWorkflowDataset(size=size, seed=seed)
    if ident == "panoptic":
        return _PanopticWorkflowDataset(size=size, seed=seed)
    if ident == "pose":
        return _PoseWorkflowDataset(size=size, seed=seed)
    if ident == "vision-neck":
        return _VisionNeckWorkflowDataset(size=size, seed=seed)
    if ident == "attn2d":
        return _Attn2DWorkflowDataset(size=size, seed=seed)
    if ident == "vision-norm":
        return _VisionNormWorkflowDataset(size=size, seed=seed)
    if ident == "pos2d":
        return _Pos2DWorkflowDataset(size=size, seed=seed)
    if ident == "video-conv":
        return _VideoConvWorkflowDataset(size=size, seed=seed)
    if ident == "video-transformer":
        return _VideoTransformerWorkflowDataset(size=size, seed=seed)
    if ident == "video-ssl":
        return _VideoSSLWorkflowDataset(size=size, seed=seed)
    if ident == "point-cloud":
        return _PointCloudWorkflowDataset(size=size, seed=seed)
    if ident == "voxel-detect":
        return _VoxelDetectWorkflowDataset(size=size, seed=seed)
    if ident == "volume-seg":
        return _VolumeSegWorkflowDataset(size=size, seed=seed)
    if ident == "neural-field":
        return _NeuralFieldWorkflowDataset(size=size, seed=seed)
    if ident == "implicit-surface":
        return _ImplicitSurfaceWorkflowDataset(size=size, seed=seed)
    if ident == "albert":
        return _AlbertPretrainingDataset(seed=seed, size=size)
    if ident == "spanbert":
        constructor = _ENTRY_SPECS[ident][1]
        dataset = synth.Spec(ident, generator=_spanbert_tokens, size=size,
            splits={"train": 0.75, "val": 0.25}, seed=seed, root=root,
            monitor="none").run()
        return _SpanBERTPretrainingDataset(dataset, seed=seed,
                                          vocab_size=constructor["vocab_size"])
    if ident == "distilbert-encoder":
        constructor = _ENTRY_SPECS[ident][1]
        return _DistilBertPretrainingDataset(seed=seed, size=size,
            vocab_size=constructor["vocab_size"], context_length=constructor["context_length"],
            pad_token_id=constructor["pad_token_id"])
    if ident == "electra":
        constructor = _ENTRY_SPECS[ident][1]
        return _ElectraPretrainingDataset(seed=seed, size=size,
            vocab_size=constructor["vocab_size"], context_length=constructor["context_length"])
    try:
        _recipe, _constructor, generator = _ENTRY_SPECS[ident]
    except KeyError:
        raise ValueError(f"No catalog workflow registered for {ident!r}; choose one of {sorted(_ENTRY_SPECS)}") from None
    return synth.Spec(ident, generator=generator, size=size, splits={"train": 0.75, "val": 0.25},
                      seed=seed, root=root, monitor="none").run()


class _ModernBERTWorkflowDataset:
    def __init__(self, size: int, seed: int):
        self.size, self.seed = size, seed

    def for_recipe(self, recipe: str, split: str = "train"):
        import torch
        from torch.utils.data import DataLoader
        from topologies.data.adapters import DictDataset

        if recipe != "masked-lm" or split not in {"train", "val"}:
            raise ValueError("efficient-attention workflow accepts masked-lm train/val")
        count = max(1, int(self.size * (0.75 if split == "train" else 0.25)))
        data = [_modernbert_tokens(type("Context", (), {"index": torch.tensor(i), "rng": synth.RNG(self.seed + i)})()) for i in range(count)]
        rows = [{"input_ids": row["tokens"], "labels": row["labels"]} for row in data]
        return DataLoader(DictDataset(rows), batch_size=4, shuffle=False)


class _MarianWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.pairs = [([4 + (seed + i) % 20, 5], [7 + (seed + i) % 20, 8]) for i in range(size)]

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        from torch.utils.data import DataLoader, Subset
        from topologies.training.marian_data import PretokenizedTranslationDataset, TranslationCollator
        if recipe != "seq2seq-lm" or split not in {"train", "val"}:
            raise ValueError("Marian workflow accepts recipe='seq2seq-lm' and train/val splits")
        cut = max(1, int(len(self.pairs) * .75))
        selected = self.pairs[:cut] if split == "train" else self.pairs[cut:] or self.pairs[-1:]
        ds = PretokenizedTranslationDataset(selected)
        return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=TranslationCollator())


class _CharformerWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.texts = [f"café {seed + index} 世界" for index in range(size)]

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        from torch.utils.data import DataLoader
        from topologies.training.charformer_data import ByteSeq2SeqCollator, DenoisingByteDataset

        if recipe != "seq2seq-lm":
            raise ValueError("Charformer data supports only the seq2seq-lm recipe")
        cut = max(1, int(len(self.texts) * 0.75))
        values = self.texts[:cut] if split == "train" else self.texts[cut:] or self.texts[-1:]
        return DataLoader(DenoisingByteDataset(values), batch_size=4, shuffle=False,
                          collate_fn=ByteSeq2SeqCollator())


class _ByT5WorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.texts = [f"span denoising sample {seed + index}: café 世界 λ" for index in range(size)]
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        from torch.utils.data import DataLoader, Dataset
        from topologies.training.byt5_data import ByteSpanDenoisingCollator

        if recipe != "seq2seq-lm" or split not in {"train", "val"}:
            raise ValueError("ByT5 workflow accepts recipe='seq2seq-lm' and train/val splits")
        cut = max(1, int(len(self.texts) * 0.75))
        texts = self.texts[:cut] if split == "train" else self.texts[cut:] or self.texts[-1:]

        class TextDataset(Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, index):
                return texts[index]

        return DataLoader(TextDataset(), batch_size=4, shuffle=False,
                          collate_fn=ByteSpanDenoisingCollator(seed=self.seed + (0 if split == "train" else 1)))


class _T5WorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.rows = [[5 + ((seed + i + j) % 20) for j in range(12)] for i in range(size)]
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        from torch.utils.data import DataLoader
        from topologies.training.t5_data import PretokenizedDenoisingDataset, SpanDenoisingCollator

        if recipe != "seq2seq-lm" or split not in {"train", "val"}:
            raise ValueError("T5 workflow accepts recipe='seq2seq-lm' and train/val splits")
        cut = max(1, int(len(self.rows) * .75))
        rows = self.rows[:cut] if split == "train" else self.rows[cut:] or self.rows[-1:]
        return DataLoader(PretokenizedDenoisingDataset(rows), batch_size=4, shuffle=False,
            collate_fn=SpanDenoisingCollator(sentinel_ids=(31, 30, 29, 28), eos_token_id=1,
                seed=self.seed + (0 if split == "train" else 1)))


class _UL2WorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.rows = [[5 + ((seed + index + j) % 20) for j in range(12)] for index in range(size)]
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        from torch.utils.data import DataLoader
        from topologies.training.ul2_data import UL2DenoisingCollator
        from topologies.training.t5_data import PretokenizedDenoisingDataset

        if recipe != "seq2seq-lm" or split not in {"train", "val"}:
            raise ValueError("UL2 workflow accepts recipe='seq2seq-lm' and train/val splits")
        cut = max(1, int(len(self.rows) * .75))
        rows = self.rows[:cut] if split == "train" else self.rows[cut:] or self.rows[-1:]
        mode = "RSX"[self.seed % 3] if split == "train" else "RSX"[(self.seed + 1) % 3]
        return DataLoader(PretokenizedDenoisingDataset(rows), batch_size=4, shuffle=False,
            collate_fn=UL2DenoisingCollator(mode=mode, sentinel_ids=(31, 30, 29, 28),
                eos_token_id=1, pad_token_id=0, seed=self.seed + (0 if split == "train" else 1)))


class _BARTWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.rows = [[0, *[5 + ((seed + index + j) % 20) for j in range(3)], 13,
                      *[5 + ((seed + index + j + 3) % 20) for j in range(4)], 13,
                      *[5 + ((seed + index + j + 7) % 20) for j in range(3)], 2]
                     for index in range(size)]
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        from torch.utils.data import DataLoader
        from topologies.training.bart_data import BARTDenoisingCollator, PretokenizedBARTDataset

        if recipe != "seq2seq-lm" or split not in {"train", "val"}:
            raise ValueError("BART workflow accepts recipe='seq2seq-lm' and train/val splits")
        cut = max(1, int(len(self.rows) * .75))
        rows = self.rows[:cut] if split == "train" else self.rows[cut:] or self.rows[-1:]
        return DataLoader(PretokenizedBARTDataset(rows), batch_size=4, shuffle=False,
            collate_fn=BARTDenoisingCollator(mask_token_id=30, bos_token_id=0, eos_token_id=2,
                pad_token_id=1, mask_ratio=.3, poisson_lambda=3.0, full_stop_id=13,
                permute_sentences=1.0, seed=self.seed + (0 if split == "train" else 1)))


class _ImplicitSurfaceWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "implicit-surface-classify" or split not in {"train", "val"}:
            raise ValueError("implicit-surface workflow accepts recipe='implicit-surface-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _NeuralFieldWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "neural-field-classify" or split not in {"train", "val"}:
            raise ValueError("neural-field workflow accepts recipe='neural-field-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VolumeSegWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "volume-seg-classify" or split not in {"train", "val"}:
            raise ValueError("volume-seg workflow accepts recipe='volume-seg-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VoxelDetectWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "voxel-detect-classify" or split not in {"train", "val"}:
            raise ValueError("voxel-detect workflow accepts recipe='voxel-detect-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _PointCloudWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "point-cloud-classify" or split not in {"train", "val"}:
            raise ValueError("point-cloud workflow accepts recipe='point-cloud-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VideoSSLWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "video-ssl-classify" or split not in {"train", "val"}:
            raise ValueError("video-ssl workflow accepts recipe='video-ssl-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VideoTransformerWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "video-transformer-classify" or split not in {"train", "val"}:
            raise ValueError("video-transformer workflow accepts recipe='video-transformer-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VideoConvWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "video-conv-classify" or split not in {"train", "val"}:
            raise ValueError("video-conv workflow accepts recipe='video-conv-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _Pos2DWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "pos2d-classify" or split not in {"train", "val"}:
            raise ValueError("pos2d workflow accepts recipe='pos2d-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VisionNormWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "vision-norm-classify" or split not in {"train", "val"}:
            raise ValueError("vision-norm workflow accepts recipe='vision-norm-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _Attn2DWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "attn2d-classify" or split not in {"train", "val"}:
            raise ValueError("attn2d workflow accepts recipe='attn2d-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VisionNeckWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "vision-neck-classify" or split not in {"train", "val"}:
            raise ValueError("vision-neck workflow accepts recipe='vision-neck-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _PoseWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "pose-classify" or split not in {"train", "val"}:
            raise ValueError("pose workflow accepts recipe='pose-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _PanopticWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "panoptic-classify" or split not in {"train", "val"}:
            raise ValueError("panoptic workflow accepts recipe='panoptic-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _PromptSegWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "prompt-seg-classify" or split not in {"train", "val"}:
            raise ValueError("prompt-seg workflow accepts recipe='prompt-seg-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _SegmentWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "segment-classify" or split not in {"train", "val"}:
            raise ValueError("segment workflow accepts recipe='segment-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _DetectOpsWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "detect-ops-classify" or split not in {"train", "val"}:
            raise ValueError("detect-ops workflow accepts recipe='detect-ops-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _SetDetectWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "set-detect-classify" or split not in {"train", "val"}:
            raise ValueError("set-detect workflow accepts recipe='set-detect-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _SingleStageWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "single-stage-classify" or split not in {"train", "val"}:
            raise ValueError("single-stage workflow accepts recipe='single-stage-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _TwoStageWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "two-stage-classify" or split not in {"train", "val"}:
            raise ValueError("two-stage workflow accepts recipe='two-stage-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _ContrastiveVisionWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "contrastive-vision-classify" or split not in {"train", "val"}:
            raise ValueError("contrastive-vision workflow accepts recipe='contrastive-vision-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _MaskedImageWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "masked-image-classify" or split not in {"train", "val"}:
            raise ValueError("masked-image workflow accepts recipe='masked-image-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VisionSSLWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "vision-ssl-classify" or split not in {"train", "val"}:
            raise ValueError("vision-ssl workflow accepts recipe='vision-ssl-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _TokenMixerWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "token-mixer-classify" or split not in {"train", "val"}:
            raise ValueError("token-mixer workflow accepts recipe='token-mixer-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _SwinWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "swin-classify" or split not in {"train", "val"}:
            raise ValueError("swin workflow accepts recipe='swin-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _DeiTWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "deit-classify" or split not in {"train", "val"}:
            raise ValueError("deit workflow accepts recipe='deit-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _ModernConvWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "modern-conv-classify" or split not in {"train", "val"}:
            raise ValueError("modern-conv workflow accepts recipe='modern-conv-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _EfficientConvWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "efficient-conv-classify" or split not in {"train", "val"}:
            raise ValueError("efficient-conv workflow accepts recipe='efficient-conv-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _DenseNetWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "densenet-classify" or split not in {"train", "val"}:
            raise ValueError("densenet workflow accepts recipe='densenet-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _ResidualVariantWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "resnet-variant-classify" or split not in {"train", "val"}:
            raise ValueError("resnet-variants workflow accepts recipe='resnet-variant-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _InceptionWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "inception-classify" or split not in {"train", "val"}:
            raise ValueError("inception workflow accepts recipe='inception-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _VGGWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "vgg-classify" or split not in {"train", "val"}:
            raise ValueError("vgg workflow accepts recipe='vgg-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 224, 224, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _AlexNetWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "alexnet-classify" or split not in {"train", "val"}:
            raise ValueError("alexnet workflow accepts recipe='alexnet-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 3, 227, 227, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _LeNetWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if recipe != "lenet-classify" or split not in {"train", "val"}:
            raise ValueError("lenet workflow accepts recipe='lenet-classify' and train/val splits")
        generator = torch.Generator().manual_seed(self.seed + (0 if split == "train" else 1))
        count = max(1, int(self.size * 0.75)) if split == "train" else max(1, self.size - int(self.size * 0.75))
        pixels = torch.randn(count, 1, 32, 32, generator=generator)
        labels = torch.randint(0, 10, (count,), generator=generator)
        return DataLoader(TensorDataset(pixels, labels), batch_size=4, shuffle=False,
            collate_fn=lambda rows: {"pixels": torch.stack([row[0] for row in rows]), "labels": torch.stack([row[1] for row in rows])})


class _AttentionNMTWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.rows = [
            {
                "src": [1 + (seed + index + step) % 20 for step in range(4)],
                "tgt": [1 + (seed + index + step) % 20 for step in range(4)],
                "labels": [1 + (seed + index + step) % 20 for step in range(4)],
            }
            for index in range(size)
        ]

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader

        if recipe != "seq2seq-lm" or split not in {"train", "val"}:
            raise ValueError("attention-nmt workflow accepts recipe='seq2seq-lm' and train/val splits")
        cut = max(1, int(len(self.rows) * 0.75))
        selected = self.rows[:cut] if split == "train" else self.rows[cut:] or self.rows[-1:]
        tensors = [{key: torch.tensor(value) for key, value in row.items()} for row in selected]
        return DataLoader(tensors, batch_size=4, shuffle=False)


class _TransitionalTextWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.rows = [
            {
                "tokens": [1 + (seed + index + step) % 20 for step in range(8)],
                "labels": (seed + index) % 3,
            }
            for index in range(size)
        ]

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader

        if recipe != "text-classify" or split not in {"train", "val"}:
            raise ValueError("transitional-text workflow accepts recipe='text-classify' and train/val splits")
        cut = max(1, int(len(self.rows) * 0.75))
        selected = self.rows[:cut] if split == "train" else self.rows[cut:] or self.rows[-1:]
        tensors = [
            {"tokens": torch.tensor(row["tokens"]), "labels": torch.tensor(row["labels"])}
            for row in selected
        ]
        return DataLoader(tensors, batch_size=4, shuffle=False)


class _StaticEmbeddingsWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.rows = [
            {
                "target": 1 + (seed + index) % 20,
                "context": 1 + (seed + index + 3) % 20,
                "negatives": [1 + (seed + index + 5 + offset) % 20 for offset in range(2)],
            }
            for index in range(size)
        ]

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader

        if recipe != "negative-sampling" or split not in {"train", "val"}:
            raise ValueError("static-embeddings workflow accepts recipe='negative-sampling' and train/val splits")
        cut = max(1, int(len(self.rows) * 0.75))
        selected = self.rows[:cut] if split == "train" else self.rows[cut:] or self.rows[-1:]
        tensors = [
            {
                "target": torch.tensor(row["target"]),
                "context": torch.tensor(row["context"]),
                "negatives": torch.tensor(row["negatives"]),
            }
            for row in selected
        ]
        return DataLoader(tensors, batch_size=4, shuffle=False)


class _SentenceEncoderWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.rows = [
            {
                "left": [1 + (seed + index + step) % 20 for step in range(4)],
                "right": [1 + (seed + index + step + 3) % 20 for step in range(4)],
            }
            for index in range(size)
        ]

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader

        if recipe != "sentence-contrastive" or split not in {"train", "val"}:
            raise ValueError("sentence-encoder workflow accepts recipe='sentence-contrastive' and train/val splits")
        cut = max(1, int(len(self.rows) * 0.75))
        selected = self.rows[:cut] if split == "train" else self.rows[cut:] or self.rows[-1:]
        tensors = [{key: torch.tensor(value) for key, value in row.items()} for row in selected]
        return DataLoader(tensors, batch_size=4, shuffle=False)


class _RagFidWorkflowDataset:
    def __init__(self, size: int, seed: int) -> None:
        self.rows = [
            {
                "src": [1 + (seed + index + step) % 20 for step in range(4)],
                "tgt": [1 + (seed + index + step) % 20 for step in range(3)],
                "labels": [1 + (seed + index + step) % 20 for step in range(3)],
            }
            for index in range(size)
        ]

    def for_recipe(self, recipe: str, *, split: str = "train", **kwargs):
        import torch
        from torch.utils.data import DataLoader

        if recipe != "seq2seq-lm" or split not in {"train", "val"}:
            raise ValueError("RAG/FiD workflow accepts recipe='seq2seq-lm' and train/val splits")
        cut = max(1, int(len(self.rows) * 0.75))
        selected = self.rows[:cut] if split == "train" else self.rows[cut:] or self.rows[-1:]
        tensors = [{key: torch.tensor(value) for key, value in row.items()} for row in selected]
        return DataLoader(tensors, batch_size=4, shuffle=False)


class _CanineWorkflowDataset:
    def __init__(self, size, seed):
        self.texts = [f"CANINE text {index} café 世界 नमस्ते" for index in range(size)]
        self.seed = seed

    def for_recipe(self, recipe, *, split="train", **kwargs):
        from torch.utils.data import DataLoader
        from topologies.training.canine_data import CanineCCollator, UnicodeTextDataset

        if recipe != "canine-c" or split not in {"train", "val"}:
            raise ValueError("CANINE workflow accepts recipe='canine-c' and train/val splits")
        cut = max(1, int(len(self.texts) * 0.75))
        values = self.texts[:cut] if split == "train" else self.texts[cut:] or self.texts[-1:]
        return DataLoader(UnicodeTextDataset(values), batch_size=4, shuffle=False,
                          collate_fn=CanineCCollator(seed=self.seed + (0 if split == "train" else 1)))


def workflow_for_entry(ident: str) -> tuple[str, dict[str, Any]]:
    """Return the registry-matching recipe and tiny constructor kwargs for an entry."""
    try:
        recipe, constructor, _generator = _ENTRY_SPECS[ident]
    except KeyError:
        raise ValueError(f"No catalog workflow registered for {ident!r}; choose one of {sorted(_ENTRY_SPECS)}") from None
    return recipe, dict(constructor)


def generator_for_entry(ident: str):
    """Return the explicit, record-level synth generator used for an entry."""
    try:
        return _ENTRY_SPECS[ident][2]
    except KeyError:
        raise ValueError(f"No catalog workflow registered for {ident!r}; choose one of {sorted(_ENTRY_SPECS)}") from None


def recipe_kwargs_for_entry(ident: str) -> dict[str, Any]:
    """Return explicit converter options needed by the entry's data contract."""
    if ident in {"transformer-classifier", "text-cnn"}:
        return {"tokenizer": "char", "max_len": 16}
    if ident == "transformer-tagger":
        return {"tokenizer": "char", "max_len": 16, "tags": ["O", "B-tree", "I-tree"]}
    if ident == "seq2seq":
        return {"tokenizer": "char", "max_len": 12, "src_field": "source", "tgt_field": "target"}
    if ident in {"recurrent-predictor", "temporal-convnet", "nbeats", "diagonal-ssm-model"}:
        return {"input_field": "inputs", "target_field": "targets"}
    if ident == "conv-autoencoder":
        return {"preserve_shape": True}
    if ident == "conv-vae":
        return {"preserve_shape": True}
    if ident == "vqvae":
        return {"preserve_shape": True}
    if ident in {"dcgan-generator", "dcgan-discriminator", "diffusion-unet"}:
        return {"preserve_shape": True}
    if ident in {"ctc-acoustic-model", "conv-asr"}:
        return {"feature_field": "features", "target_field": "targets"}
    if ident == "image-text-dual-encoder":
        return {"tokenizer": "char", "max_len": 8}
    if ident == "deepfm":
        return {"input_mode": "fields"}
    if ident == "graph-network":
        return {"level": "node"}
    if ident == "retinanet":
        return {"num_classes": 2}
    if ident not in _ENTRY_SPECS:
        raise ValueError(f"No catalog workflow registered for {ident!r}; choose one of {sorted(_ENTRY_SPECS)}")
    return {}


def trainer_options_for_entry(ident: str, model: Any, dataset: Any) -> dict[str, Any]:
    """Build explicit named recipe components for GAN, PPO, and TD-value workflows."""
    import copy

    if ident == "transformer-xl":
        return {"recipe_config": {"segment_length": 3}}
    if ident == "distilbert-encoder":
        from topologies.models.bert_encoder import BERTEncoder
        from topologies.training.recipes.lm import distilbert_special_token_ids

        controls = distilbert_special_token_ids(model.vocab_size)
        teacher = BERTEncoder(vocab_size=model.vocab_size, context_length=model.context_length,
            d_model=model.d_model, num_layers=2 * model.num_layers, num_heads=model.num_heads,
            intermediate_size=model.intermediate_size, pad_token_id=model.pad_token_id)
        model.initialize_from_teacher(teacher)
        return {"components": {"teacher_model": teacher},
                "recipe_config": {"vocab_size": model.vocab_size, "pad_token_id": 0,
                    "unknown_token_id": controls[1], "cls_token_id": controls[2],
                    "sep_token_id": controls[3], "mask_token_id": controls[4],
                    "special_token_ids": controls}}

    if ident == "electra":
        from topologies.training.recipes.lm import electra_pretraining_special_token_ids

        controls = electra_pretraining_special_token_ids(model.vocab_size)
        return {"recipe_config": {"vocab_size": model.vocab_size,
                    "pad_token_id": controls[0], "unknown_token_id": controls[1],
                    "cls_token_id": controls[2], "sep_token_id": controls[3],
                    "mask_token_id": controls[4], "special_token_ids": controls}}

    if ident == "actor-critic":
        return {"components": {"env": _SynthPPOEnv(dataset.view("train").decoded())},
                "recipe_config": {"obs_dim": 4, "act_dim": 3, "rollout_steps": 4,
                                  "epochs_per_rollout": 1}}
    if ident == "q-network":
        return {"components": {"target_model": copy.deepcopy(model)}}
    if ident == "matrix-factorization":
        return {"forward_map": lambda module, batch: module(batch["user"], batch["item"])}
    if ident == "deepfm":
        return {"forward_map": lambda module, batch: module(batch["x"]),
                "recipe_config": {"input_mode": "fields"}}
    if ident in {"dcgan-generator", "dcgan-discriminator"}:
        from topologies.models.dcgan import DCGANDiscriminator, DCGANGenerator

        generator = model if ident == "dcgan-generator" else DCGANGenerator(
            latent_dim=4, out_channels=1, image_size=16, features=4)
        return {"model_override": generator,
                "components": {"discriminator": DCGANDiscriminator(
                    in_channels=1, image_size=16, features=4)},
                "optimizers": {"generator": "adam", "discriminator": "adam"},
                "recipe_config": {"latent_dim": 4, "image_channels": 1,
                                  "image_size": 16}}
    if ident == "ddsp-synthesizer":
        return {"recipe_config": {"sample_rate": 8000, "n_fft": 16, "hop_length": 4}}
    if ident == "diffusion-unet":
        return {"recipe_config": {"num_timesteps": 8}}
    if ident == "ctc-acoustic-model":
        return {"recipe_config": {"blank_id": 0}}
    if ident == "conv-asr":
        return {"recipe_config": {"blank_id": 4}}
    if ident == "retinanet":
        return {"recipe_config": {"num_classes": 2}}
    if ident not in _ENTRY_SPECS:
        raise ValueError(f"No catalog workflow registered for {ident!r}")
    return {}


def evaluation_mode_for_entry(ident: str) -> str:
    """Return ``fit_history`` for PPO/GAN fit-loop objectives, else ``evaluate``.

    PPO and alternating GAN recipes produce algorithm-specific fit metrics and do not
    define a held-out ``evaluate`` step in their current contracts.
    """
    if ident in {"actor-critic", "dcgan-generator", "dcgan-discriminator"}:
        return "fit_history"
    if ident not in _ENTRY_SPECS:
        raise ValueError(f"No catalog workflow registered for {ident!r}")
    return "evaluate"


__all__ = [
    "dataset_for_entry",
    "evaluation_mode_for_entry",
    "generator_for_entry",
    "recipe_kwargs_for_entry",
    "trainer_options_for_entry",
    "workflow_for_entry",
]
