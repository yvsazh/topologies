"""FSDD lazy WAV index with deterministic speaker-disjoint partitions."""
import re
import wave

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    NAME = re.compile(r"^(?P<digit>[0-9])_(?P<speaker>[^_]+)_(?P<take>[0-9]+)\.wav$")

    def load_data(self, paths):
        root = paths[0]
        recordings = []
        for path in sorted(root.rglob("*.wav")):
            match = self.NAME.match(path.name)
            if match:
                recordings.append((path, match.group("digit"), match.group("speaker")))
        speakers = sorted({speaker for _path, _digit, speaker in recordings})
        if len(speakers) < 2:
            raise ValueError("speaker-disjoint FSDD splits require at least two speaker IDs")
        # Provider protocol holds takes 0-4 for its official test set. For validation,
        # reserve one speaker among the official training takes to prevent speaker leakage.
        val_speakers = {speakers[-1]}
        heldout = {"train": set(speakers) - val_speakers, "validation": val_speakers,
                   "test": set(speakers)}
        if self.split is not None and self.split not in heldout:
            raise ValueError("FSDD adapter requires train, validation, or test split")
        records = []
        for path, digit, speaker in recordings:
            take = int(self.NAME.match(path.name).group("take"))
            if self.split is None:
                pass
            elif self.split == "test":
                if take > 4:
                    continue
            elif take < 5 or speaker not in heldout[self.split]:
                continue
            with wave.open(str(path), "rb") as stream:
                records.append({"path": path, "label": int(digit), "speaker": speaker,
                                "sample_rate": stream.getframerate(), "length": stream.getnframes(),
                                "channels": stream.getnchannels(), "sample_width": stream.getsampwidth()})
        return records

    @staticmethod
    def read_pcm(record, *, expected_sample_rate=None):
        path = record["path"]
        with wave.open(str(path), "rb") as stream:
            rate, frames, channels, width = stream.getframerate(), stream.getnframes(), stream.getnchannels(), stream.getsampwidth()
            if expected_sample_rate is not None and rate != expected_sample_rate:
                raise ValueError(f"sample rate {rate} != expected {expected_sample_rate}; resampling is not implicit")
            payload = stream.readframes(frames)
        if len(payload) != frames * channels * width:
            raise ValueError(f"truncated PCM WAV payload: {path}")
        return {"pcm_bytes": payload, "sample_rate": rate, "length": frames, "channels": channels, "sample_width": width}
