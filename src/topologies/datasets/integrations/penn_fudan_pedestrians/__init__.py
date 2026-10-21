"""Penn-Fudan instance-mask index and boxes in original image coordinates."""
import hashlib

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        root = paths[0]
        images = sorted(root.rglob("PNGImages/*.png"))
        records = []
        for image in images:
            mask = root / "PedMasks" / f"{image.stem}_mask.png"
            if not mask.is_file():
                raise ValueError(f"Penn-Fudan instance mask missing for {image.name}")
            records.append({"image": image, "mask": mask})
        if not records:
            raise ValueError("no Penn-Fudan image/mask pairs found")
        # No canonical split: stable identifier hashing creates reproducible held-outs.
        records.sort(key=lambda row: row["image"].name)
        bucket = lambda record: hashlib.sha256(record["image"].name.encode("utf-8")).digest()[0] % 10
        if self.split is None:
            return records
        if self.split not in ("train", "validation", "test"):
            raise ValueError("Penn-Fudan split must be train, validation, or test")
        if self.split == "test":
            return [row for row in records if bucket(row) == 0]
        if self.split == "validation":
            return [row for row in records if bucket(row) == 1]
        return [row for row in records if bucket(row) not in (0, 1)]

    @staticmethod
    def boxes_from_instance_mask(mask_path):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Penn-Fudan mask decoding requires topologies[vision] (Pillow) and NumPy") from exc
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        boxes, instance_ids = [], []
        for instance_id in np.unique(mask):
            if int(instance_id) in (0, 255):
                continue
            ys, xs = np.where(mask == instance_id)
            if not len(xs):
                continue
            # Exclusive x2/y2 follows common xyxy box convention; coordinates are not resized.
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
            instance_ids.append(int(instance_id))
        return {"boxes_xyxy": boxes, "instance_ids": instance_ids, "original_size": (mask.shape[1], mask.shape[0])}
