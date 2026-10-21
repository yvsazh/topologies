"""Explicit local-data adapters; importing this module requires no optional packages.

These helpers never download data. Pillow, mido, and Gymnasium are imported only
by the feature that needs them. Audio resampling is never implicit.
"""
from __future__ import annotations

import csv
import json
import struct
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_pcm_wav(path: str | Path) -> dict[str, Any]:
    """Decode uncompressed PCM WAV to normalized float32 ``[channels, samples]``.

    Supports unsigned 8-bit and signed little-endian 16/24/32-bit PCM. The
    original rate and valid frame count are returned; no resampling occurs.
    """
    import torch

    path = Path(path)
    with wave.open(str(path), "rb") as reader:
        channels, sample_rate, frames = reader.getnchannels(), reader.getframerate(), reader.getnframes()
        width, compression = reader.getsampwidth(), reader.getcomptype()
        if compression != "NONE":
            raise ValueError(f"unsupported compressed WAV ({compression}); decode PCM explicitly first")
        payload = reader.readframes(frames)
    if frames < 1 or sample_rate < 1 or channels < 1:
        raise ValueError(f"WAV must contain at least one valid PCM frame: {path}")
    if width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported PCM sample width: {width} bytes")
    if len(payload) != frames * channels * width:
        raise ValueError(f"truncated PCM WAV payload: {path}")
    if width == 1:
        values = [(value - 128) / 128.0 for value in payload]
    elif width == 2:
        values = [value[0] / 32768.0 for value in struct.iter_unpack("<h", payload)]
    elif width == 3:
        values = []
        for offset in range(0, len(payload), 3):
            integer = int.from_bytes(payload[offset:offset + 3], "little", signed=True)
            values.append(integer / 8388608.0)
    else:
        values = [value[0] / 2147483648.0 for value in struct.iter_unpack("<i", payload)]
    audio = torch.tensor(values, dtype=torch.float32).reshape(frames, channels).transpose(0, 1).contiguous()
    return {"audio": audio, "sample_rate": sample_rate, "length": frames,
            "channels": channels, "sample_width": width, "path": path}


def collate_wavs(samples: Sequence[dict[str, Any]], *, target_sample_rate: int | None = None,
                 resampler: Callable[[Any, int, int], Any] | None = None,
                 target_pad_id: int = -100) -> dict[str, Any]:
    """Pad variable WAV tensors to ``[B,C,T]`` and return CTC lengths.

    Different rates are rejected unless both ``target_sample_rate`` and an
    explicit ``resampler(audio, source_rate, target_rate)`` are supplied.
    Padding is zero and is excluded by ``lengths``. Labels are padded separately.
    """
    import torch

    if not samples:
        raise ValueError("cannot collate an empty WAV batch")
    if target_sample_rate is not None and target_sample_rate < 1:
        raise ValueError("target_sample_rate must be positive")
    prepared = []
    rates, channels = set(), set()
    for sample in samples:
        audio, rate = sample["audio"], int(sample["sample_rate"])
        if audio.ndim != 2 or audio.shape[1] != int(sample["length"]):
            raise ValueError("WAV sample must contain channel-first audio and its exact valid frame length")
        if target_sample_rate is not None and rate != target_sample_rate:
            if resampler is None:
                raise ValueError("resampling requested but no explicit resampler callable was provided")
            audio = resampler(audio, rate, target_sample_rate)
            rate = target_sample_rate
        rates.add(rate); channels.add(int(audio.shape[0]))
        prepared.append((audio, rate, sample.get("label")))
    if len(rates) != 1:
        raise ValueError("WAV batch has mixed sample rates; provide an explicit resampling policy")
    if len(channels) != 1:
        raise ValueError("WAV batch has mixed channel counts")
    lengths = torch.tensor([int(audio.shape[1]) for audio, _rate, _label in prepared], dtype=torch.long)
    max_length = int(lengths.max().item())
    batch = torch.zeros((len(prepared), next(iter(channels)), max_length), dtype=torch.float32)
    for index, (audio, _rate, _label) in enumerate(prepared):
        batch[index, :, :audio.shape[1]] = audio
    labels = [label for _audio, _rate, label in prepared]
    result = {"audio": batch, "lengths": lengths, "sample_rate": next(iter(rates)),
              "channels": next(iter(channels))}
    if all(label is not None for label in labels):
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        target_width = max(1, int(target_lengths.max().item()))
        targets = torch.full((len(labels), target_width), target_pad_id, dtype=torch.long)
        for index, label in enumerate(labels):
            targets[index, :len(label)] = torch.as_tensor(label, dtype=torch.long)
        result.update(targets=targets, target_lengths=target_lengths)
    return result


@dataclass
class ImageFolder:
    """Local class-folder index with class IDs frozen from a training tree."""
    records: list[tuple[Path, int]]
    class_to_id: dict[str, int]
    transform: Callable[[Any], Any] | None = None

    @property
    def classes(self) -> list[str]:
        return [name for name, _index in sorted(self.class_to_id.items(), key=lambda pair: pair[1])]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, label = self.records[index]
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("local image-folder decoding requires Pillow; install topologies[vision]") from exc
        image = Image.open(path).convert("RGB")
        width, height = image.size
        if self.transform is not None:
            image = self.transform(image)
        else:
            import torch
            image = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            # PIL reports width/height; tobytes() is HWC RGB in row-major order.
            image = image.reshape(height, width, 3).permute(2, 0, 1).to(torch.float32).div_(255.0)
        return {"pixels": image, "labels": label, "path": str(path)}


def image_folder(root: str | Path, *, class_to_id: dict[str, int] | None = None,
                 transform: Callable[[Any], Any] | None = None,
                 unknown_class: str = "error") -> ImageFolder:
    """Index ``root/class/image``; fit sorted mapping on train or supply frozen mapping.

    Unseen evaluation classes fail by default; ``unknown_class='ignore'`` assigns
    -1 without adding a class to the training vocabulary.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"image folder is not a directory: {root}")
    classes = sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if unknown_class not in ("error", "ignore"):
        raise ValueError("unknown_class must be 'error' or 'ignore'")
    mapping = dict(class_to_id) if class_to_id is not None else {name: index for index, name in enumerate(classes)}
    if len(set(mapping.values())) != len(mapping) or any(not isinstance(value, int) or value < 0 for value in mapping.values()):
        raise ValueError("class_to_id must map unique class names to nonnegative integer IDs")
    records = []
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for class_name in classes:
        if class_name not in mapping and unknown_class == "error":
            raise ValueError(f"unseen image-folder class {class_name!r}; pass training class_to_id explicitly")
        label = mapping.get(class_name, -1)
        records.extend((file, label) for file in sorted((root / class_name).iterdir())
                       if file.is_file() and not file.name.startswith(".") and file.suffix.lower() in supported)
    return ImageFolder(records, mapping, transform)


def read_csv_rows(path: str | Path, *, encoding: str = "utf-8", delimiter: str = ",") -> list[dict[str, str]]:
    """Read local CSV records with headers; no fitting or dataset download occurs."""
    with Path(path).open(encoding=encoding, newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def fit_csv_pipeline(train_rows: Sequence[dict[str, Any]], pipeline: Any) -> Any:
    """Fit the canonical ``PreprocessingPipeline`` on train records only."""
    from topologies.data.preprocessing import PreprocessingPipeline
    if not isinstance(pipeline, PreprocessingPipeline):
        raise TypeError("pipeline must be topologies.data.preprocessing.PreprocessingPipeline")
    return pipeline.fit(list(train_rows), mode="training")


def encode_text_documents(documents: Sequence[str], tokenizer: Any, *, bos_id: int, eos_id: int,
                          pad_id: int, max_length: int | None = None) -> dict[str, Any]:
    """Encode documents with BOS/EOS, padding, explicit valid lengths, and masks."""
    import torch
    sequences = []
    for document in documents:
        body = tokenizer.encode(document) if hasattr(tokenizer, "encode") else tokenizer(document)
        sequence = [int(bos_id), *(int(token) for token in body), int(eos_id)]
        if max_length is not None:
            if max_length < 2:
                raise ValueError("max_length must leave room for BOS and EOS")
            sequence = sequence[:max_length]
            sequence[-1] = int(eos_id)
        sequences.append(sequence)
    width = max((len(sequence) for sequence in sequences), default=0)
    tokens = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    for index, sequence in enumerate(sequences):
        tokens[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[index, :len(sequence)] = True
    return {"tokens": tokens, "mask": mask, "lengths": lengths}


def text_windows_by_split(splits: dict[str, Sequence[int]], *, block_size: int, stride: int | None = None) -> dict[str, list[list[int]]]:
    """Create per-split windows independently; no block may cross split boundaries."""
    if block_size < 1 or (stride is not None and stride < 1):
        raise ValueError("block_size and stride must be positive")
    step = stride or block_size
    result = {}
    for name, tokens in splits.items():
        values = [int(token) for token in tokens]
        result[name] = [values[start:start + block_size]
                        for start in range(0, len(values) - block_size + 1, step)]
    return result


def paired_caption_manifest(path: str | Path) -> list[dict[str, str]]:
    """Validate local JSONL image/caption pairs, unique pair IDs, and split isolation."""
    path = Path(path)
    seen_ids, seen_images, records = set(), {}, []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid paired-caption JSONL at line {line_number}") from exc
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key]
                                                for key in ("id", "image", "caption", "split")):
            raise ValueError(f"paired-caption record at line {line_number} needs id/image/caption/split strings")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError(f"invalid paired-caption split at line {line_number}: {row['split']!r}")
        image = (path.parent / row["image"]).resolve()
        if not image.is_relative_to(path.parent.resolve()):
            raise ValueError(f"paired-caption image path escapes manifest directory at line {line_number}")
        if not image.is_file():
            raise FileNotFoundError(f"paired-caption image does not exist: {image}")
        if row["id"] in seen_ids:
            raise ValueError(f"paired-caption ID reused across rows/splits: {row['id']!r}")
        if image in seen_images and seen_images[image] != row["split"]:
            raise ValueError(f"paired image identity leaks across splits: {image}")
        seen_ids.add(row["id"]); seen_images[image] = row["split"]
        records.append({**row, "image_path": str(image)})
    return records


def paired_retrieval_batch(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return paired IDs in aligned order for an in-batch contrastive objective."""
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("retrieval batch requires unique pair IDs")
    return {"pair_ids": ids, "image_paths": [record["image_path"] for record in records],
            "captions": [record["caption"] for record in records],
            "positive_indices": list(range(len(ids)))}


def midi_to_event_tokens(path: str | Path, codec: Any) -> dict[str, Any]:
    """Convert monophonic, fixed-tempo, codec-grid MIDI note/rest data to EventCodec tokens.

    Unsupported controller/pitch-bend/program messages, tempo changes, overlapping
    notes, off-grid timing, or nonrepresentable velocities fail rather than being
    silently dropped or misrepresented. ``mido`` is optional and lazily imported.
    """
    try:
        import mido
    except ImportError as exc:
        raise ImportError("MIDI import requires optional package mido; install it with `pip install mido`") from exc
    midi = mido.MidiFile(str(path))
    if midi.type == 2:
        raise ValueError("asynchronous type-2 MIDI tracks cannot be represented as one event sequence")
    tempos = {message.tempo for track in midi.tracks for message in track if message.type == "set_tempo"}
    if len(tempos) > 1:
        raise ValueError("MIDI tempo changes are unsupported by this fixed-tempo event codec")
    tempo = next(iter(tempos), 500000)
    note_events = []
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            if message.is_meta or message.type in ("track_name", "end_of_track"):
                continue
            if message.type not in ("note_on", "note_off"):
                raise ValueError(f"unsupported MIDI event type {message.type!r}")
            is_on = message.type == "note_on" and message.velocity > 0
            note_events.append((absolute, int(message.note), int(message.velocity) if is_on else 0, is_on))
    note_events.sort(key=lambda item: (item[0], item[3]))
    active, motif, cursor = None, [], 0
    ticks_per_beat = codec.ticks_per_beat
    seconds_per_tick = tempo / 1_000_000 / midi.ticks_per_beat
    codec_ticks_per_second = ticks_per_beat / (tempo / 1_000_000)
    for tick, pitch, velocity, is_on in note_events:
        if is_on:
            if active is not None:
                raise ValueError("polyphonic/overlapping MIDI notes are unsupported")
            start = round(tick * seconds_per_tick * codec_ticks_per_second)
            if abs(start - tick * seconds_per_tick * codec_ticks_per_second) > 1e-7:
                raise ValueError("MIDI onset is off the configured event-codec tick grid")
            if start > cursor:
                if start - cursor > codec.max_duration:
                    raise ValueError("MIDI rest exceeds EventCodec max_duration; exact rest representation is unsupported")
                motif.append({"rest_ticks": start - cursor})
            active = (pitch, velocity, start)
        else:
            if active is None or active[0] != pitch:
                raise ValueError("unmatched MIDI note-off or unsupported overlapping notes")
            duration_float = tick * seconds_per_tick * codec_ticks_per_second - active[2]
            duration = round(duration_float)
            if duration < 1 or abs(duration_float - duration) > 1e-7:
                raise ValueError("MIDI note duration is off the configured event-codec tick grid")
            if duration > codec.max_duration:
                raise ValueError("MIDI note duration exceeds EventCodec max_duration; exact note hold is unsupported")
            velocity_id = min(127, max(1, int(round((active[1] - 1) / 8) * 8 + 1)))
            if velocity_id != active[1]:
                raise ValueError("MIDI velocity is not exactly representable in the configured event codec")
            motif.append({"pitch": pitch, "duration_ticks": duration, "velocity": active[1]})
            cursor = round(tick * seconds_per_tick * codec_ticks_per_second)
            active = None
    if active is not None:
        raise ValueError("MIDI note has no matching note-off")
    return {"tokens": codec.encode(motif), "tempo_microseconds_per_beat": tempo,
            "normalized_tempo_microseconds_per_beat": 500000,
            "source_ticks_per_beat": midi.ticks_per_beat, "codec_ticks_per_beat": ticks_per_beat,
            "event_codec": codec.metadata(len(motif) + 2)}


class CartPoleAdapter:
    """Thin seeded Gymnasium bridge; does not execute rollouts or training loops."""
    def __init__(self, env_id: str = "CartPole-v1", *, render_mode: str | None = None):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError("CartPole integration requires Gymnasium; install it with `pip install 'gymnasium[classic-control]'`") from exc
        self.environment = gym.make(env_id, render_mode=render_mode)

    def reset(self, *, seed: int) -> dict[str, Any]:
        observation, info = self.environment.reset(seed=seed)
        return {"observation": observation, "info": info, "seed": seed}

    def step(self, action: int) -> dict[str, Any]:
        observation, reward, terminated, truncated, info = self.environment.step(action)
        return {"observation": observation, "reward": float(reward), "terminated": bool(terminated),
                "truncated": bool(truncated), "episode_done": bool(terminated or truncated),
                "bootstrap": not bool(terminated), "info": info}

    def close(self) -> None:
        self.environment.close()
