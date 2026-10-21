"""Oxford-IIIT Pet path-index adapter and paired image/trimap reader."""

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        image_root, annotation_root = paths
        image_files = {path.stem: path for path in image_root.rglob("*.jpg")}
        annotations = annotation_root / "annotations"
        def load_partition(partition):
            listing = annotations / ("test.txt" if partition == "test" else "trainval.txt")
            if not listing.is_file():
                raise FileNotFoundError(f"Oxford-IIIT Pet split manifest missing: {listing}")
            ids = [line.split()[0] for line in listing.read_text().splitlines() if line.strip() and not line.startswith("#")]
            if partition in ("train", "validation"):
                ids = [name for index, name in enumerate(ids) if (index % 5 != 0) == (partition == "train")]
            records = []
            for name in ids:
                image = image_files.get(name)
                mask = annotations / "trimaps" / f"{name}.png"
                if image is None or not mask.is_file():
                    raise ValueError(f"Oxford-IIIT Pet image/trimap pair missing for {name}")
                records.append({"image": image, "mask": mask})
            return records
        if self.split is None:
            return {partition: load_partition(partition) for partition in ("train", "validation", "test")}
        return load_partition(self.split)

    @staticmethod
    def read_pair(record, *, size=None):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Oxford-IIIT Pet image decoding requires topologies[vision] (Pillow)") from exc
        image = Image.open(record["image"]).convert("RGB")
        mask = Image.open(record["mask"]).convert("L")
        if image.size != mask.size:
            raise ValueError("pet image and trimap dimensions differ")
        if size:
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize(size, resampling.BILINEAR)
            mask = mask.resize(size, resampling.NEAREST)
        # Provider values: 1=foreground, 2=background, 3=boundary -> 1,0,2.
        pixels = mask.load()
        for y in range(mask.height):
            for x in range(mask.width):
                value = pixels[x, y]
                if value not in (1, 2, 3):
                    raise ValueError(f"unexpected Oxford-IIIT Pet trimap label {value}")
                pixels[x, y] = {1: 1, 2: 0, 3: 2}[value]
        return image, mask
