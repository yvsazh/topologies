"""MovieLens ratings with deterministic held-out-user recommendation protocol."""
import hashlib

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        root = paths[0]
        rating_files = list(root.rglob("u.data"))
        if len(rating_files) != 1:
            raise ValueError("MovieLens archive must contain exactly one u.data")
        rows = []
        for line in rating_files[0].read_text().splitlines():
            user, item, rating, timestamp = line.split("\t")
            rows.append({"user": user, "item": item, "rating": float(rating), "timestamp": int(timestamp)})
        users = sorted({row["user"] for row in rows}, key=lambda user: hashlib.sha256(user.encode()).digest())
        if len(users) < 2:
            raise ValueError("held-out-user split requires at least two users")
        heldout = set(users[max(1, int(len(users) * 0.8)):])
        train_rows = [row for row in rows if row["user"] not in heldout]
        test_rows = [row for row in rows if row["user"] in heldout]
        user_ids = {user: index for index, user in enumerate(sorted({row["user"] for row in train_rows}))}
        item_ids = {item: index for index, item in enumerate(sorted({row["item"] for row in train_rows}))}
        if self.split == "train":
            selected = train_rows
        elif self.split == "test":
            selected = test_rows
        else:
            selected = rows
        encoded = [{"user_id": user_ids.get(row["user"], -1), "item_id": item_ids.get(row["item"], -1),
                    "user": row["user"], "item": row["item"], "rating": row["rating"], "timestamp": row["timestamp"]}
                   for row in selected]
        seen = {}
        for row in train_rows:
            seen.setdefault(row["user"], set()).add(row["item"])
        return {"interactions": encoded, "train_user_ids": user_ids, "train_item_ids": item_ids,
                "negative_candidate_items": sorted(item_ids), "train_seen_by_user": seen,
                "negative_policy": "candidates use only items observed in training; exclude each user's training positives; held-out positives are never consulted"}

    @staticmethod
    def negative_candidates(user, item_pool, training_seen):
        return [item for item in item_pool if item not in training_seen.get(user, set())]
