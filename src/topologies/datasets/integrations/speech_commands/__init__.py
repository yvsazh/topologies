"""Speech Commands lazy WAV index driven by provider validation/testing lists."""
import wave

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        root = paths[0]
        test_ids = set((root / "testing_list.txt").read_text().splitlines())
        validation_ids = set((root / "validation_list.txt").read_text().splitlines())
        records = []
        known = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}
        for path in sorted(root.rglob("*.wav")):
            relative = path.relative_to(root).as_posix()
            if self.split == "test" and relative not in test_ids:
                continue
            if self.split == "validation" and relative not in validation_ids:
                continue
            if self.split == "train" and (relative in test_ids or relative in validation_ids):
                continue
            label = path.parent.name
            if label == "_background_noise_":
                continue
            with wave.open(str(path), "rb") as stream:
                records.append({"path": path, "label": label if label in known else "unknown",
                                "sample_rate": stream.getframerate(), "length": stream.getnframes(),
                                "channels": stream.getnchannels(), "sample_width": stream.getsampwidth()})
        return records

    @staticmethod
    def read_pcm(record, *, expected_sample_rate=None):
        path = record["path"]
        with wave.open(str(path), "rb") as stream:
            rate, count, channels, width = stream.getframerate(), stream.getnframes(), stream.getnchannels(), stream.getsampwidth()
            if expected_sample_rate is not None and rate != expected_sample_rate:
                raise ValueError("sample rate mismatch; choose an explicit resampler")
            payload = stream.readframes(count)
        if len(payload) != count * channels * width:
            raise ValueError(f"truncated PCM WAV payload: {path}")
        return {"pcm_bytes": payload, "sample_rate": rate, "length": count, "channels": channels, "sample_width": width}
