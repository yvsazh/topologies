
import torch
from torch.utils.data import Dataset


class PretokenizedTranslationDataset(Dataset):
    def __init__(self, pairs, family="marian", source_language=None, target_language=None, language_tokens=None,
                 decoder_start_token_id=None, pad_token_id=0):
        self.pairs = list(pairs)
        self.family, self.source_language, self.target_language = family, source_language, target_language
        self.language_tokens = language_tokens or {}
        self.decoder_start_token_id = pad_token_id if decoder_start_token_id is None else decoder_start_token_id
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        source, target = list(source), list(target)
        if self.family == "m2m100":
            source = [self.language_tokens[self.source_language], *source]
            decoder_start = self.language_tokens[self.target_language]
            target = [decoder_start, *target]
        elif self.family == "nllb":
            decoder_start = self.language_tokens[self.target_language]
            target = [decoder_start, *target]
        elif self.family == "marian":
            decoder_start = self.decoder_start_token_id
        else:
            raise ValueError(f"unknown translation family {self.family!r}")
        decoder = [decoder_start, *target[:-1]] if self.family == "marian" else target
        return {"src": source, "tgt": decoder, "labels": target}


class TranslationCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        width = max(len(row["src"]) for row in rows)
        target_width = max(len(row["tgt"]) for row in rows)
        label_width = max(len(row["labels"]) for row in rows)
        src = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        tgt = torch.full((len(rows), target_width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), label_width), -100, dtype=torch.long)
        for i, row in enumerate(rows):
            src[i, :len(row["src"])] = torch.tensor(row["src"])
            tgt[i, :len(row["tgt"])] = torch.tensor(row["tgt"])
            labels[i, :len(row["labels"])] = torch.tensor(row["labels"])
        return {"src": src, "tgt": tgt, "labels": labels, "src_mask": src.ne(self.pad_token_id), "tgt_mask": tgt.ne(self.pad_token_id)}
