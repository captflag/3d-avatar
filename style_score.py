"""Score the stylized style: how much of the person it keeps.

Each dev face's build (out/TAG-<name>.glb, from score_run.py) is drawn as the
studio's stylized portrait (app.html?style=stylized, its own picture export) in
headless Chrome, wearing the test set's labelled haircut, idle motion off. The
example (out/example/front.png, if present, built as out/TAG-example.glb) rides
along in the short curly cut. Scored against the person's photo:

  identity_s   SFace cosine, as worn (higher is better; same person 0.363)
  identity_sf  the same with hair greyed out of both
  dlib_s       dlib distance (lower is better; same person under 0.6)
  photo_err    mean |render / photo - 1| over the face proportions (PROPORTIONS)
  ideal_err    mean |render / photo - IDEAL| : how far from the reference
               cartoon avatar's own changes to its photo

SFace and dlib are internal yardsticks only, never shipped (identity_score.py).

    python style_score.py TAG [--off paint,light] [--against VARIANT]

`--off` leaves parts of the style out (lib/scene-setup.js STYLIZED_PARTS, and
`hair`: the build's own, bald) to measure what each costs. Writes
out/scores/style-TAG[-off].json.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

# Face proportions, landmark pairs over the eye-corner spacing.
PROPORTIONS = {"eye_open": [(159, 145), (386, 374)], "eye_len": [(33, 133), (362, 263)], "mouth_w": [(61, 291)],
               "lips_h": [(0, 17)], "nose_w": [(129, 358)], "nose_len": [(168, 2)], "jaw_w": [(172, 397)],
               "face_h": [(10, 152)], "brow_eye": [(105, 159), (334, 386)]}
# The reference cartoon avatar's proportions over its photo's (measured
# 2026-11-04 on out/example/reference-output.png against front.png): eyes
# opened half again, face shorter, nose, mouth and lips narrower and thinner.
IDEAL = {"eye_open": 1.53, "eye_len": 1.05, "mouth_w": 0.90, "lips_h": 0.75, "nose_w": 0.91, "nose_len": 0.94,
         "jaw_w": 0.985, "face_h": 0.91, "brow_eye": 0.89}

DRIVER = """<!doctype html><meta charset="utf-8">
<iframe id="app" style="width:1300px;height:1000px;border:0"></iframe>
<script>
const log = (m) => fetch('/__log', { method: 'POST', body: String(m) });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const frame = document.getElementById('app');
  for (const job of await (await fetch('/__jobs')).json()) {
    try {
      await new Promise((r) => { frame.onload = r; frame.src = job.page; });
      const doc = () => frame.contentDocument;
      for (let i = 0; i < 600 && !doc().querySelector('#cuts button'); i++) await sleep(100);
      if (job.cut) doc().querySelector('#cuts button[data-cut="' + job.cut + '"]').click();
      await sleep(1500);
      doc().getElementById('tosave').click();
      const img = doc().getElementById('finalshot');
      for (let i = 0; i < 100 && !(img.src || '').startsWith('data:'); i++) await sleep(100);
      await fetch('/__result/' + job.id, { method: 'POST', body: img.src });
    } catch (err) {
      await log(job.page + ': ' + (err && err.stack || err));
      await fetch('/__result/' + job.id, { method: 'POST', body: '' });
    }
  }
})();
</script>"""


def portraits(jobs: list[dict]) -> list[pathlib.Path | None]:
    """Draw each job ({page, cut, png}) in headless Chrome; the PNGs written, None where one failed."""
    from synth import browser
    results = queue.Queue()

    class Page(SimpleHTTPRequestHandler):
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".svg": "image/svg+xml"}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(HERE), **kwargs)

        def log_message(self, *args):
            pass

        def send(self, data: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/__drive":
                return self.send(DRIVER.encode(), "text/html; charset=utf-8")
            if self.path == "/__jobs":
                return self.send(json.dumps([{"id": i, "page": j["page"], "cut": j["cut"]} for i, j in enumerate(jobs)]).encode(),
                                 "application/json")
            return super().do_GET()

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.startswith("/__result/"):
                results.put((int(self.path.rsplit("/", 1)[1]), data))
            elif self.path == "/__log":
                print("page:", data.decode(errors="replace"), file=sys.stderr)
            self.send(b"", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    done: dict[int, pathlib.Path | None] = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        page = subprocess.Popen([browser(None), "--headless=new", f"--user-data-dir={profile}", "--no-first-run",
                                 "--no-default-browser-check", "--disable-extensions", "--ignore-gpu-blocklist",
                                 "--window-size=1400,1100", f"http://127.0.0.1:{server.server_port}/__drive"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            while len(done) < len(jobs):
                i, data = results.get(timeout=240)
                if data.startswith(b"data:image/png;base64,"):
                    jobs[i]["png"].parent.mkdir(parents=True, exist_ok=True)
                    jobs[i]["png"].write_bytes(base64.b64decode(data.split(b",", 1)[1]))
                    done[i] = jobs[i]["png"]
                else:
                    done[i] = None
        finally:
            page.terminate()
            page.wait()
            server.shutdown()
    return [done[i] for i in range(len(jobs))]


def proportions(path: pathlib.Path) -> dict | None:
    from PIL import Image
    from score_run import landmarks
    p = landmarks(np.asarray(Image.open(path).convert("RGB")))
    if p is None:
        return None
    spacing = float(np.linalg.norm(p[33] - p[263]))
    return {k: float(np.mean([np.linalg.norm(p[a] - p[b]) for a, b in pairs]) / spacing) for k, pairs in PROPORTIONS.items()}


def faces() -> list[dict]:
    from score_run import TESTSET, load_faces
    people = [{"name": f["name"], "photo": HERE / f["photo"], "cut": f.get("hair_cut")}
              for f in load_faces(TESTSET / "test-set.json", "dev")]
    if (OUT / "example" / "front.png").exists():
        people.append({"name": "example", "photo": OUT / "example" / "front.png", "cut": "hair_short_curly"})
    return people


def score(tag: str, off: list[str]) -> dict:
    import identity_score as ids
    from score_run import hairless, upright
    variant = tag + ("-" + "-".join(off) if off else "")
    people = [f for f in faces() if (OUT / f"{tag}-{f['name']}.glb").exists()]
    query = "&off=" + ",".join(["motion", *off])
    jobs = [{"page": f"/app.html?model=/out/{tag}-{f['name']}.glb&style=stylized{query}",
             "cut": None if f["cut"] in (None, "bald") or "hair" in off else f["cut"],
             "png": OUT / "renders" / "style" / f"{f['name']}-{variant}.png"} for f in people]
    rows = {}
    for f, png in zip(people, portraits(jobs)):
        if png is None:
            rows[f["name"]] = None
            continue
        photo = upright(f["photo"])
        row = {"identity_s": ids.similarity(photo, png), "identity_sf": ids.similarity(hairless(photo), hairless(png))}
        a, b = ids.dlib_embedding(photo), ids.dlib_embedding(png)
        row["dlib_s"] = None if a is None or b is None else float(np.linalg.norm(a - b))
        have, want = proportions(png), proportions(photo)
        if have and want:
            ratio = {k: have[k] / want[k] for k in PROPORTIONS}
            row["photo_err"] = float(np.mean([abs(r - 1) for r in ratio.values()]))
            row["ideal_err"] = float(np.mean([abs(ratio[k] - IDEAL[k]) for k in PROPORTIONS]))
            row["ratios"] = {k: round(v, 3) for k, v in ratio.items()}
        rows[f["name"]] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
    path = OUT / "scores" / f"style-{variant}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


COLUMNS = ("identity_s", "identity_sf", "dlib_s", "photo_err", "ideal_err")


def mean(rows: dict, key: str, dev_only: bool = True) -> float | None:
    # A render with no face found scores nan; left out, and counted in `missing`.
    vals = [r[key] for n, r in rows.items() if r and r.get(key) is not None and np.isfinite(r[key])
            and not (dev_only and n == "example")]
    return float(np.mean(vals)) if vals else None


def table(rows: dict, base: dict | None) -> str:
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    lines = ["face      " + "".join(f"{c:>20s}" for c in COLUMNS)]
    for name, row in [*rows.items(), ("mean dev", {c: mean(rows, c) for c in COLUMNS})]:
        cells = []
        for c in COLUMNS:
            v = None if row is None else row.get(c)
            b = None if base is None else (({c: mean(base, c) for c in COLUMNS} if name == "mean dev" else base.get(name)) or {}).get(c)
            cells.append(f"{fmt(b)} -> {fmt(v)}" if base is not None else f"{fmt(v):>20s}")
        lines.append(f"{name:10s}" + "".join(f"{x:>20s}" for x in cells))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag")
    parser.add_argument("--off", default="", help="comma-separated parts of the style to leave out")
    parser.add_argument("--against", help="a variant scored before (style-<VARIANT>.json), to print beside this one")
    args = parser.parse_args()
    rows = score(args.tag, [o for o in args.off.split(",") if o])
    base = None
    if args.against:
        base = json.loads((OUT / "scores" / f"style-{args.against}.json").read_text(encoding="utf-8"))
    print(table(rows, base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
