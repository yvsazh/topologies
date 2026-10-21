"""Export a dataset version: ``jsonl`` | ``csv`` | ``txt`` | ``parquet`` | ``hf`` | ``imagefolder``.

    ds.export("out/", format="jsonl")                 # out/train.jsonl, out/val.jsonl
    ds.export("out/", format="csv")
    ds.export("out/", format="txt", field="text")     # one text per line (LM corpora)
    ds.export("out/", format="parquet")               # needs topologies[synth-parquet]
    ds.export("out/", format="hf")                    # datasets.DatasetDict.save_to_disk
    ds.export("out/", format="imagefolder")           # out/<split>/<label>/<id>.png

``meta=True`` keeps ``_id`` / ``_attrs``; by default only schema fields are written.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from topologies.synth.errors import SynthError, unknown
from topologies.synth.records import strip_meta
from topologies.synth.store import dumps

FORMATS = ("jsonl", "csv", "txt", "parquet", "hf", "imagefolder")


def _rows(ds: Any, split: str, meta: bool) -> list[dict]:
    return [dict(r) if meta else strip_meta(r) for r in ds.view(split)]


def export(ds: Any, out: Path, *, format: str = "jsonl", meta: bool = False,
           field: str | None = None, splits: list[str] | None = None) -> list[Path]:
    if format not in FORMATS:
        raise unknown("export format", format, FORMATS, stage="export")
    out.mkdir(parents=True, exist_ok=True)
    names = splits or list(ds.splits)
    written: list[Path] = []
    if format == "jsonl":
        for s in names:
            p = out / f"{s}.jsonl"
            with p.open("w", encoding="utf-8") as fh:
                for r in _rows(ds, s, meta):
                    fh.write(dumps(r) + "\n")
            written.append(p)
    elif format == "csv":
        for s in names:
            rows = _rows(ds, s, meta)
            keys: list[str] = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            p = out / f"{s}.csv"
            with p.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: v if isinstance(v, (str, int, float)) or v is None
                                else json.dumps(v, ensure_ascii=False) for k, v in r.items()})
            written.append(p)
    elif format == "txt":
        f = field or _text_field(ds)
        for s in names:
            p = out / f"{s}.txt"
            with p.open("w", encoding="utf-8") as fh:
                for r in ds.view(s):
                    v = r.get(f)
                    if v is not None:
                        fh.write(str(v).replace("\n", " ") + "\n")
            written.append(p)
    elif format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SynthError("Parquet export needs pyarrow.", stage="export",
                             hint="pip install 'topologies[synth-parquet]'") from e
        for s in names:
            rows = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) and
                         not _flat_list(v) else v) for k, v in r.items()}
                    for r in _rows(ds, s, meta)]
            p = out / f"{s}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), p)
            written.append(p)
    elif format == "hf":
        try:
            import datasets as hfd
        except ImportError as e:
            raise SynthError("Hugging Face export needs the datasets package.", stage="export",
                             hint="pip install 'topologies[nlp]'") from e
        dd = hfd.DatasetDict({s: hfd.Dataset.from_list(_rows(ds, s, meta)) for s in names})
        dd.save_to_disk(str(out))
        written.append(out)
    elif format == "imagefolder":
        written += _imagefolder(ds, out, names, field)
    return written


def _flat_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)


def _text_field(ds: Any) -> str:
    schema = ds.schema
    if schema is not None and schema.text_fields():
        return schema.text_fields()[0]
    return "text"


def _imagefolder(ds: Any, out: Path, names: list[str], field: str | None) -> list[Path]:
    schema = ds.schema
    imf = field or (schema.of_kind("image")[0] if schema is not None and
                    schema.of_kind("image") else "image")
    lf = (schema.label_field() if schema is not None else None) or "label"
    written = []
    for s in names:
        for r in ds.view(s):
            v = r.get(imf)
            label = str(r.get(lf, "unlabeled"))
            target = out / s / label
            target.mkdir(parents=True, exist_ok=True)
            if isinstance(v, dict) and "$blob" in v:
                src = ds.store.blob_path(v, ds.version_id)
                dst = target / f"{r['_id']}.{v.get('type', 'png')}"
                shutil.copy2(src, dst)
            else:
                from topologies.synth.lib.image import save_png

                dst = target / f"{r['_id']}.png"
                save_png(v, dst)
            written.append(dst)
    return written


def push_hf(ds: Any, repo: str, *, private: bool = True, token_env: str = "HF_TOKEN") -> Any:
    """Upload to the Hugging Face Hub (explicit call only; token from ``$HF_TOKEN``)."""
    import os

    try:
        import datasets as hfd
    except ImportError as e:
        raise SynthError("push_hf needs the datasets package.", stage="export",
                         hint="pip install 'topologies[nlp]'") from e
    token = os.environ.get(token_env)
    dd = hfd.DatasetDict({s: hfd.Dataset.from_list(_rows(ds, s, False)) for s in ds.splits})
    return dd.push_to_hub(repo, private=private, token=token)


__all__ = ["FORMATS", "export", "push_hf"]
