"""``synth.preview``: look at (and listen to) a generator before generating a dataset.

    synth.preview(gen, n=8)                          # console: one block per sample
    synth.preview(gen, n=8, html="preview.html")     # + a self-contained page (no CDN):
                                                     #   <audio> players, PNG images, tables,
                                                     #   the node tree, timings per node
    synth.preview(spec, n=4, open=True)              # a Spec: its attributes and seed

Each sample shows its fields through their value types (text lines, tensor
shape and stats, audio duration with a waveform sparkline, image thumbnails in
block characters, graphs, ...), its attributes and its latents. A field that
cannot be rendered shows an error line instead: previews never break.
"""

from __future__ import annotations

import html as _html
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _type_of(name: str, value: Any, schema: Any) -> Any:
    from topologies.synth.valuetypes import Type, infer_type

    if isinstance(value, Type) and value.is_value:
        return value.spec()
    if schema is not None and name in schema:
        return schema[name]
    return infer_type(value)


def _console(name: str, value: Any, schema: Any) -> str:
    try:
        return _type_of(name, value, schema).preview(value)
    except Exception as e:  # noqa: BLE001 - observability never breaks generation
        return f"<preview failed: {type(e).__name__}: {e}>"


def _html_cell(name: str, value: Any, schema: Any) -> str:
    try:
        return _type_of(name, value, schema).preview_html(value)
    except Exception as e:  # noqa: BLE001
        return f"<span class='err'>preview failed: {_html.escape(str(e))}</span>"


class Preview:
    """Rendered preview: ``.records``, ``.text``, ``.html`` (when requested)."""

    def __init__(self, records: list[dict[str, Any]], text: str, html: str | None,
                 path: Path | None, timings: dict[str, float]) -> None:
        self.records, self.text, self.html, self.path, self.timings = records, text, html, \
            path, timings

    def __repr__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.records[i]


_CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e1e0d9;--card:rgba(127,127,127,.07)}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#ffffff;--mut:#c3c2b7;
--line:#2c2c2a}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;
padding:16px;max-width:1100px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:18px 0 6px}
.m{color:var(--mut);font-size:12px}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:10px 12px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%}td{border-top:1px solid var(--line);padding:5px 8px;
vertical-align:top}td.k{width:160px;color:var(--mut);font-family:ui-monospace,monospace}
pre{margin:0;white-space:pre-wrap;font:12px/1.4 ui-monospace,monospace}
pre.t{font:13px/1.45 system-ui,sans-serif}img.px{image-rendering:pixelated;
border:1px solid var(--line)}audio{height:32px;max-width:100%}.err{color:#d33}
"""


def _render_html(title: str, records: list[dict[str, Any]], schema: Any, tree: str,
                 timings: dict[str, float], seconds: float) -> str:
    e = _html.escape
    cards = []
    for r in records:
        rows = []
        for k, v in r.items():
            if k.startswith("_"):
                continue
            rows.append(f"<tr><td class='k'>{e(k)}</td><td>{_html_cell(k, v, schema)}</td></tr>")
        meta = []
        if r.get("_attrs"):
            meta.append("attributes: " + ", ".join(f"{k}={v!r}" for k, v in r["_attrs"].items()))
        if r.get("_latent"):
            meta.append("latents: " + ", ".join(f"{k}={v!r}" for k, v in r["_latent"].items()))
        cards.append(f"<div class='card'><div class='m'>record {r.get('_index')}"
                     f"{' · ' + e(' · '.join(meta)) if meta else ''}</div>"
                     f"<table>{''.join(rows)}</table></div>")
    slow = sorted(timings.items(), key=lambda kv: -kv[1])[:20]
    trows = "".join(f"<tr><td class='k'>{e(p or '<root>')}</td><td>{t * 1000:.2f} ms</td></tr>"
                    for p, t in slow)
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{e(title)}</title><style>{_CSS}</style></head><body>"
            f"<h1>{e(title)}</h1><div class='m'>{len(records)} records in {seconds:.3f}s</div>"
            f"{''.join(cards)}<h2>node tree</h2><div class='card'><pre>{e(tree)}</pre></div>"
            f"<h2>time per node (all previewed records)</h2><div class='card'><table>{trows}"
            f"</table></div></body></html>")


def preview(node_or_spec: Any, n: int = 8, *, seed: int | None = None,
            attributes: Any = None, html: str | Path | None = None, open: bool = False,
            start: int = 0, echo: bool = True) -> Preview:
    """Render ``n`` records of a generator or spec (see module docstring)."""
    from topologies.synth.core import GeneratorFn, Node, Record, record
    from topologies.synth.introspect import describe
    from topologies.synth.streaming import Stream

    is_spec = hasattr(node_or_spec, "gen_fingerprint") and hasattr(node_or_spec, "generator")
    if is_spec:
        st = Stream.from_spec(node_or_spec, n=start + n)
        if seed is not None:
            st = Stream(node_or_spec.generator, start + n, seed=seed,
                        attributes=node_or_spec.attributes, sampler=node_or_spec.sampler_spec,
                        schema=node_or_spec.schema)
        title = f"preview {node_or_spec.name}"
    else:
        # Value nodes are valid preview inputs, including tensors and whole sequences.
        generator = node_or_spec
        if isinstance(generator, Node) and not isinstance(generator, (Record, GeneratorFn)):
            generator = record(value=generator)
        st = Stream(generator, start + n, seed=seed or 0, attributes=attributes)
        title = f"preview {st.gen.node_kind()}"
    t0 = time.perf_counter()
    st.genr.ev.take_timings()
    records = st.sample(n, start=start)
    seconds = time.perf_counter() - t0
    timings, _ = st.genr.ev.take_timings()
    schema = st.schema
    lines = [f"{title}: {n} records in {seconds:.3f}s"]
    for r in records:
        lines.append(f"── record {r.get('_index')}" + (
            f"  attrs: {', '.join(f'{k}={v!r}' for k, v in r['_attrs'].items())}"
            if r.get("_attrs") else ""))
        width = max((len(k) for k in r if not k.startswith("_")), default=4)
        for k, v in r.items():
            if k.startswith("_"):
                continue
            lines.append(f"   {k:<{width}}  {_console(k, v, schema)}")
        if r.get("_latent"):
            lat = ", ".join(f"{k}={v!r}" for k, v in list(r["_latent"].items())[:6])
            lines.append(f"   {'latent':<{width}}  {lat}")
    text = "\n".join(lines)
    page = path = None
    if html is not None or open:
        tree = describe(st.gen, echo=False)
        page = _render_html(title, records, schema, tree, timings, seconds)
        path = Path(html) if html is not None else Path("preview.html")
        path.write_text(page, encoding="utf-8")
        if open:
            import webbrowser

            webbrowser.open(path.resolve().as_uri())
    if echo:
        try:
            print(text + (f"\nhtml: {path}" if path is not None else ""))
        except Exception:  # noqa: BLE001, S110 - display never breaks previews
            pass
    return Preview(records, text, page, path, timings)


def preview_records(records: list[Mapping[str, Any]], schema: Any = None) -> str:
    """Console lines for existing records (``ds.show`` with value types)."""
    from topologies.synth.valuetypes import from_jsonable

    out = []
    for r in records:
        for k, v in r.items():
            if not str(k).startswith("_"):
                out.append(f"{k}: {_console(k, from_jsonable(v), schema)}")
        out.append("")
    return "\n".join(out)


__all__ = ["Preview", "preview", "preview_records"]
