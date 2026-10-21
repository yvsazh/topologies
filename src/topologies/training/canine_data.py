import torch
from torch.utils.data import Dataset


class UnicodeTextDataset(Dataset):
    def __init__(self, texts):
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must be a non-empty sequence of strings")
        self.texts = tuple(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


class CanineCCollator:
    def __init__(self, cls_codepoint=2, mask_codepoint=0xF0000, pad_codepoint=0, mask_probability=0.15, max_predictions=320, seed=0):
        if not 0 <= mask_probability <= 1:
            raise ValueError("mask_probability must be in [0, 1]")
        if isinstance(max_predictions, bool) or not isinstance(max_predictions, int) or max_predictions <= 0:
            raise ValueError("max_predictions must be a positive integer")
        self.cls_codepoint, self.mask_codepoint, self.pad_codepoint = cls_codepoint, mask_codepoint, pad_codepoint
        self.mask_probability, self.max_predictions = mask_probability, max_predictions
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, texts):
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("collator input must be a non-empty sequence of strings")
        if not any(text for text in texts):
            raise ValueError("at least one input string must contain a character")
        rows = [[self.cls_codepoint, *(ord(character) for character in text)] for text in texts]
        width = max(map(len, rows))
        codepoints = torch.full((len(rows), width), self.pad_codepoint, dtype=torch.long)
        targets = torch.full_like(codepoints, self.pad_codepoint)
        valid = torch.zeros_like(codepoints, dtype=torch.bool)
        selected = torch.zeros_like(valid)
        for row_index, (row, text) in enumerate(zip(rows, texts)):
            size = len(row)
            codepoints[row_index, :size] = torch.tensor(row)
            targets[row_index, :size] = torch.tensor(row)
            valid[row_index, :size] = True
            spans = []
            start = 0
            while start < len(text):
                while start < len(text) and text[start].isspace():
                    start += 1
                end = start
                while end < len(text) and not text[end].isspace():
                    end += 1
                if end > start:
                    spans.append((start + 1, end + 1))
                start = end + 1
            if spans and self.mask_probability > 0:
                order = torch.randperm(len(spans), generator=self.generator).tolist()
                budget = min(self.max_predictions, max(1, round(len(text) * self.mask_probability)))
                masked = 0
                for span_index in order:
                    begin, end = spans[span_index]
                    selected[row_index, begin:end] = True
                    masked += end - begin
                    if masked >= budget:
                        break
        if not bool(selected.any()) and self.mask_probability > 0:
            row = next(index for index, text in enumerate(texts) if text)
            selected[row, 1] = True
        codepoints[selected] = self.mask_codepoint
        return {"codepoints": codepoints, "targets": targets, "prediction_mask": selected, "input_mask": valid}
