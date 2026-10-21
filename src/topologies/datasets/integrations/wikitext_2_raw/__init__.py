"""Raw WikiText document loader and train-fitted token sequence utilities."""

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        # Preserve blank lines and article headers instead of silently discarding corpus structure.
        root = paths[0]
        files = {"train": "wiki.train.raw", "validation": "wiki.valid.raw", "test": "wiki.test.raw"}
        def load(split):
            path = next(root.rglob(files[split])) if root.is_dir() else root
            return {"documents": path.read_text(encoding="utf-8").splitlines()}
        if self.split is None:
            return {split: load(split) for split in files}
        if self.split not in files:
            raise ValueError(f"unknown WikiText split: {self.split}")
        return load(self.split)

    @staticmethod
    def fit_vocabulary(train_documents, *, max_tokens=None):
        from collections import Counter
        counts = Counter(token for document in train_documents for token in document.split())
        vocabulary = {"<pad>": 0, "<bos>": 1, "<unk>": 2}
        for token, _count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
            if max_tokens is not None and len(vocabulary) >= max_tokens:
                break
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
        return vocabulary

    @staticmethod
    def encode(documents, vocabulary, *, max_length):
        """Return integer tokens and true-token mask; BOS counts toward max_length."""
        if max_length < 1:
            raise ValueError("max_length must be positive")
        rows, masks = [], []
        for text in documents:
            tokens = [1] + [vocabulary.get(token, 2) for token in text.split()]
            tokens = tokens[:max_length]
            length = len(tokens)
            rows.append(tokens + [0] * (max_length - length))
            masks.append([True] * length + [False] * (max_length - length))
        return rows, masks
