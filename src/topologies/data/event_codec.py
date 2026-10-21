"""Deterministic quantized symbolic-music event codec (no MIDI dependency)."""
import json


class EventCodec:
    VERSION = "symbolic-events-v1"
    BOS, EOS, PAD = "BOS", "EOS", "PAD"

    def __init__(self, ticks_per_beat=24, max_duration=96, max_shift=96):
        if min(ticks_per_beat, max_duration, max_shift) < 1: raise ValueError("clock and event limits must be positive")
        self.ticks_per_beat, self.max_duration, self.max_shift = ticks_per_beat, max_duration, max_shift
        self.events = [self.BOS, self.EOS, self.PAD]
        self.events += [f"NOTE:{p}:{d}:{v}" for p in range(128) for d in range(1, max_duration + 1) for v in range(1, 128, 8)]
        self.events += [f"REST:{d}" for d in range(1, max_duration + 1)]
        self.events += [f"SHIFT:{d}" for d in range(1, max_shift + 1)]
        self.to_id = {e: i for i, e in enumerate(self.events)}

    def quantize(self, seconds, seconds_per_beat):
        if seconds < 0 or seconds_per_beat <= 0: raise ValueError("invalid physical time")
        return round(seconds / seconds_per_beat * self.ticks_per_beat)

    def encode(self, motif):
        """motif iterable of {pitch,duration_ticks,velocity} or {rest_ticks}; durations split."""
        result = [self.to_id[self.BOS]]
        for event in motif:
            if "rest_ticks" in event:
                n = int(event["rest_ticks"])
                if n <= 0: raise ValueError("rest duration must be positive ticks")
                while n:
                    d = min(n, self.max_duration); result.append(self.to_id[f"REST:{d}"]); n -= d
            elif "shift_ticks" in event:
                n = int(event["shift_ticks"])
                if n <= 0 or n > self.max_shift: raise ValueError("time shift outside configured tick range")
                result.append(self.to_id[f"SHIFT:{n}"])
            else:
                p, n, v = int(event["pitch"]), int(event["duration_ticks"]), int(event.get("velocity", 80))
                if not 0 <= p < 128 or not 1 <= v < 128 or n <= 0: raise ValueError("illegal pitch/velocity/duration")
                v = min(127, max(1, int(round((v - 1) / 8) * 8 + 1)))
                while n:
                    d = min(n, self.max_duration); result.append(self.to_id[f"NOTE:{p}:{d}:{v}"]); n -= d
        result.append(self.to_id[self.EOS]); return result

    def decode(self, tokens):
        out = []
        for i in tokens:
            if not 0 <= int(i) < len(self.events): raise ValueError("token outside codec vocabulary")
            e = self.events[int(i)]
            if e in (self.BOS, self.PAD): continue
            if e == self.EOS: break
            parts = e.split(":")
            if parts[0] == "NOTE": out.append({"pitch": int(parts[1]), "duration_ticks": int(parts[2]), "velocity": int(parts[3])})
            elif parts[0] == "REST": out.append({"rest_ticks": int(parts[1])})
            elif parts[0] == "SHIFT": out.append({"shift_ticks": int(parts[1])})
        return out

    def metadata(self, context_length):
        return {"codec_version": self.VERSION, "ticks_per_beat": self.ticks_per_beat,
                "context_length": int(context_length), "vocabulary_size": len(self.events),
                "time_unit": "quantized ticks; not a note-count sequence"}

    def save_metadata(self, path, context_length):
        with open(path, "w", encoding="utf8") as f: json.dump(self.metadata(context_length), f, indent=2)
