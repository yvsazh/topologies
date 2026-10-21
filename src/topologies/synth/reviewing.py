"""Human inspection sessions: look at records, accept / reject / relabel them.

    s = synth.review(ds, n=100, strategy="stratified")      # or "uncertain" / "random"
    s.run()                    # terminal: [a]ccept [r]eject [l]abel [c]omment [s]kip [q]uit
    s.serve(open_browser=True) # local page on 127.0.0.1 (stdlib http.server), same keys
    s.record(record_id, "reject", comment="span offsets are off")   # programmatic
    s.save()                   # versions/<id>/gold.jsonl  (annotations, merged by id)
    fixed = s.apply()          # new version: rejected records dropped, relabels applied

Strategies: ``stratified`` (equal numbers per label / ``by`` attribute),
``uncertain`` (verifier scores closest to their threshold first), ``random``.
Annotated records are the record plus ``_gold: {accept, label, comment, reviewer, at}``;
they also serve as a small human-checked reference set for fidelity and TSTR.
"""

from __future__ import annotations

import json
import random
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from topologies.experiments.store import now_iso
from topologies.synth.errors import SynthError, unknown

STRATEGIES = ("stratified", "uncertain", "random")
VERDICTS = ("accept", "reject", "relabel")


def _uncertainty(lin: dict) -> float:
    best = 1e9
    for v in (lin.get("verdicts") or {}).values():
        s = v.get("score")
        if isinstance(s, (int, float)):
            best = min(best, abs(float(s) - 3.5))
    return best


def pick(ds: Any, n: int, strategy: str = "stratified", *, by: str | None = None,
         split: str | None = None, seed: int = 0) -> list[dict]:
    if strategy not in STRATEGIES:
        raise unknown("review strategy", strategy, STRATEGIES, stage="review")
    recs = list(ds.view(split))
    rng = random.Random(seed)
    if strategy == "random" or not recs:
        return rng.sample(recs, min(n, len(recs)))
    if strategy == "stratified":
        key = by or (ds.schema.label_field() if ds.schema is not None else None)
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            groups[str(r.get(key, (r.get("_attrs") or {}).get(key)) if key else "")].append(r)
        for g in groups.values():
            rng.shuffle(g)
        out: list[dict] = []
        while len(out) < min(n, len(recs)):
            for k in sorted(groups):
                if groups[k] and len(out) < n:
                    out.append(groups[k].pop())
        return out
    lins = {r["_id"]: ds.lineage(r["_id"]) for r in recs}
    return sorted(recs, key=lambda r: (_uncertainty(lins[r["_id"]]), r["_id"]))[:n]


class ReviewSession:
    def __init__(self, ds: Any, records: Sequence[dict], *, reviewer: str | None = None) -> None:
        self.ds, self.records, self.reviewer = ds, list(records), reviewer
        self.verdicts: dict[str, dict[str, Any]] = {}
        self._by_id = {r["_id"]: r for r in self.records}
        self.label_field = (ds.schema.label_field() if getattr(ds, "schema", None) is not None
                            else None) or "label"

    def __len__(self) -> int:
        return len(self.records)

    @property
    def done(self) -> int:
        return len(self.verdicts)

    def record(self, record_id: str, verdict: str, *, label: Any = None,
               comment: str | None = None) -> None:
        if record_id not in self._by_id:
            raise unknown("record in this session", record_id, list(self._by_id), stage="review")
        if verdict not in VERDICTS:
            raise unknown("verdict", verdict, VERDICTS, stage="review")
        if verdict == "relabel" and label is None:
            raise SynthError("relabel needs label=...", stage="review")
        self.verdicts[record_id] = {"accept": verdict != "reject", "verdict": verdict,
                                    "label": label, "comment": comment,
                                    "reviewer": self.reviewer, "at": now_iso()}

    def gold(self) -> list[dict]:
        return [{**self._by_id[i], "_gold": v} for i, v in self.verdicts.items()]

    def path(self) -> Path:
        return Path(self.ds.dir) / "gold.jsonl"

    def save(self) -> Path:
        """Merge into ``gold.jsonl`` of the version (annotations; records are untouched)."""
        p = self.path()
        existing = {g["_id"]: g for g in load_gold(p)} if p.is_file() else {}
        for g in self.gold():
            existing[g["_id"]] = g
        from topologies.synth.store import dumps, write_text_atomic

        write_text_atomic(p, "".join(dumps(g) + "\n" for g in existing.values()))
        return p

    def apply(self) -> Any:
        """New version: rejected records dropped, relabels applied."""
        rej = {i for i, v in self.verdicts.items() if v["verdict"] == "reject"}
        relab = {i: v["label"] for i, v in self.verdicts.items() if v["verdict"] == "relabel"}
        f = self.label_field

        def fix(r: dict) -> dict | None:
            if r["_id"] in rej:
                return None
            if r["_id"] in relab:
                return {**r, f: relab[r["_id"]]}
            return r

        return self.ds.edit(fix, note=f"review: -{len(rej)} rejected, {len(relab)} relabeled")

    # -- terminal UI --------------------------------------------------------------
    def run(self, *, input_fn: Callable[[str], str] = input,
            output: Callable[[str], None] = print) -> ReviewSession:
        from topologies.synth.report import format_records

        todo = [r for r in self.records if r["_id"] not in self.verdicts]
        for k, r in enumerate(todo):
            output(format_records([r], title=f"[{self.done + 1}/{len(self)}]"))
            while True:
                key = input_fn("[a]ccept [r]eject [l]abel [c]omment [s]kip [q]uit > ").strip()
                key = key[:1].lower()
                if key == "a":
                    self.record(r["_id"], "accept")
                elif key == "r":
                    self.record(r["_id"], "reject", comment=input_fn("why? > ").strip() or None)
                elif key == "l":
                    self.record(r["_id"], "relabel", label=input_fn("label > ").strip())
                elif key == "c":
                    v = self.verdicts.get(r["_id"]) or {"accept": True, "verdict": "accept",
                                                        "label": None, "reviewer": self.reviewer,
                                                        "at": now_iso()}
                    v["comment"] = input_fn("comment > ").strip()
                    self.verdicts[r["_id"]] = v
                    continue
                elif key == "q":
                    self.save()
                    return self
                elif key != "s":
                    continue
                break
            if (k + 1) % 10 == 0:
                self.save()
        self.save()
        return self

    # -- local web page -------------------------------------------------------------------
    def serve(self, *, port: int = 0, open_browser: bool = False, block: bool = True,
              timeout: float | None = None) -> Any:
        """Serve the review page on 127.0.0.1; returns the server (``block=False``)."""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        session = self
        finished = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                return None

            def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path.startswith("/records"):
                    self._send(200, json.dumps({"records": session.records,
                                                "verdicts": session.verdicts,
                                                "label_field": session.label_field},
                                               ensure_ascii=False, default=str))
                else:
                    self._send(200, _PAGE, "text/html")

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if self.path == "/verdict":
                        session.record(body["id"], body["verdict"], label=body.get("label"),
                                       comment=body.get("comment"))
                        session.save()
                        self._send(200, json.dumps({"ok": True, "done": session.done}))
                    elif self.path == "/done":
                        session.save()
                        self._send(200, json.dumps({"ok": True}))
                        finished.set()
                    else:
                        self._send(404, json.dumps({"error": "not found"}))
                except (SynthError, KeyError, ValueError) as e:
                    self._send(400, json.dumps({"error": str(e)[:300]}))

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        self.url = url
        if open_browser:
            import webbrowser

            webbrowser.open(url)
        if not block:
            return server
        print(f"review page: {url}  (finish on the page or Ctrl+C)")
        try:
            finished.wait(timeout)
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            self.save()
        return self


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>synth review</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;
background:#fbfbfa;color:#1d1d1b}
@media (prefers-color-scheme: dark){body{background:#171716;color:#ececea}
.card{background:#232321!important;border-color:#3a3a37!important}}
.card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin:12px 0}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0}.attrs{opacity:.7;font-size:13px}
button{font:inherit;padding:6px 14px;margin-right:6px;border-radius:7px;border:1px solid #888;
cursor:pointer}.done{opacity:.45}#bar{position:sticky;top:0;padding:8px 0;background:inherit}
</style></head><body><div id="bar"><b>synth review</b> <span id="count"></span>
<button onclick="finish()">Finish</button></div><div id="list"></div>
<script>
let data;
async function load(){data=await (await fetch('records')).json();render();}
function render(){const L=document.getElementById('list');L.innerHTML='';
 document.getElementById('count').textContent=Object.keys(data.verdicts).length+'/'+
 data.records.length+' reviewed';
 for(const r of data.records){const d=document.createElement('div');
  d.className='card'+(data.verdicts[r._id]?' done':'');const f={...r};delete f._id;
  const a=f._attrs||{};delete f._attrs;
  d.innerHTML='<pre></pre><div class="attrs"></div><div style="margin-top:8px">'+
   '<input placeholder="reason / new label" style="font:inherit;margin-right:6px">'+
   '<button>Accept (a)</button><button>Reject (r)</button><button>Relabel (l)</button></div>';
  d.querySelector('pre').textContent=JSON.stringify(f,null,1);
  d.querySelector('.attrs').textContent=Object.entries(a).map(([k,v])=>k+'='+v).join(' · ');
  const b=d.querySelectorAll('button'),inp=d.querySelector('input');
  b[0].onclick=()=>send(r._id,'accept');
  b[1].onclick=()=>send(r._id,'reject',null,inp.value||null);
  b[2].onclick=()=>{if(inp.value)send(r._id,'relabel',inp.value);};
  L.appendChild(d);}}
async function send(id,verdict,label,comment){await fetch('verdict',{method:'POST',
 body:JSON.stringify({id,verdict,label,comment})});
 data.verdicts[id]={verdict};render();}
async function finish(){await fetch('done',{method:'POST',body:'{}'});
 document.body.innerHTML='<p>Saved. You can close this tab.</p>';}
document.addEventListener('keydown',e=>{const next=data&&data.records.find(r=>!data.verdicts[r._id]);
 if(!next||e.target.tagName==='INPUT')return;if(e.key==='a')send(next._id,'accept');if(e.key==='r')send(next._id,'reject');
 if(e.target.tagName==='INPUT')return;});
load();
</script></body></html>"""


def review(ds: Any, n: int = 100, *, strategy: str = "stratified", by: str | None = None,
           split: str | None = None, seed: int = 0, reviewer: str | None = None,
           web: bool = False, **kw: Any) -> ReviewSession:
    """Start a review session (``web=True`` serves the page immediately)."""
    if isinstance(ds, str):
        from topologies.synth.dataset import load

        ds = load(ds)
    s = ReviewSession(ds, pick(ds, n, strategy, by=by, split=split, seed=seed),
                      reviewer=reviewer)
    if web:
        s.serve(**kw)
    return s


def load_gold(src: Any) -> list[dict]:
    """Gold records from a path, a dataset version, a session or a list."""
    if src is None:
        return []
    if isinstance(src, ReviewSession):
        return src.gold()
    if hasattr(src, "dir") and hasattr(src, "view"):
        p = Path(src.dir) / "gold.jsonl"
        return load_gold(p) if p.is_file() else []
    if isinstance(src, (str, Path)):
        from topologies.synth.store import iter_jsonl

        p = Path(src)
        if p.is_dir():
            p = p / "gold.jsonl"
        if not p.is_file():
            raise SynthError(f"No gold file at {p}.", stage="review")
        return [_norm_gold(r) for r in iter_jsonl(p)]
    if isinstance(src, Iterable):
        return [_norm_gold(dict(r)) for r in src]
    raise SynthError(f"Cannot read gold records from {type(src).__name__}.", stage="review")


def _norm_gold(r: dict) -> dict:
    """Accept ``_gold`` dicts, or plain ``accept`` / ``verdict`` / ``human`` keys."""
    if "_gold" in r:
        return r
    for key in ("accept", "human", "verdict"):
        if key in r:
            v = r.pop(key)
            ok = v if isinstance(v, bool) else str(v).lower() in ("accept", "yes", "true", "1",
                                                                   "ok")
            return {**r, "_gold": {"accept": bool(ok), "verdict": "accept" if ok else "reject"}}
    raise SynthError("Gold record lacks a human verdict ('_gold', 'accept' or 'verdict').",
                     stage="review")


__all__ = ["STRATEGIES", "ReviewSession", "load_gold", "pick", "review"]


from topologies.synth.ops._callable import make_callable

make_callable(__name__, "review")
