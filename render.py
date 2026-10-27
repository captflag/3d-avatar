"""Render avatars as the studio does, with no browser window.

The studio page (studio.html) is loaded in a frame inside headless Chrome or
Edge, served from this folder by a local server, and its own renderPng draws
each GLB: the same three.js, lighting (lib/scene-setup.js) and framing as a 2D
download from the studio. Each GLB is drawn at the photo's head pose (p) and
level (l), head only unless --body.

    python render.py out/a.glb out/b.glb --out out/renders     # -> a-p.png a-l.png ...

three.js comes from unpkg, as in the studio, so this needs the network.
`render(jobs)` is the same from Python (score_run.py uses it).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
# Per render: a head takes 2-4 s; the first also waits for three.js to load.
STALL_SECONDS = 180

DRIVER = """<!doctype html><meta charset="utf-8">
<iframe id="studio" src="/" style="width:1200px;height:900px;border:0"></iframe>
<script>
const log = (m) => fetch('/__log', { method: 'POST', body: String(m) });
window.onerror = (m) => { log(m); };
const frame = document.getElementById('studio');
frame.onload = async () => {
  try {
    const page = frame.contentWindow;
    for (let i = 0; i < 300 && !page.__studio; i++) await new Promise((r) => setTimeout(r, 100));
    if (!page.__studio) throw new Error('the studio page did not load');
    for (const job of await (await fetch('/__jobs')).json()) {
      const png = await page.__studio.renderPng(job.url, job.body, job.pose, job.size);
      await fetch('/__result/' + job.id, { method: 'POST', body: png });
    }
  } catch (err) { await log(err && err.stack || err); }
  await fetch('/__done', { method: 'POST' });
};
</script>"""


def render(jobs: list[tuple[pathlib.Path, pathlib.Path, bool]], size: int = 900, body: bool = False,
           browser_path: str | None = None) -> None:
    """Render each (glb, png, photo_pose) job, writing the PNG. Raises if the
    page stalls, the browser exits early, or a render did not come back."""
    from synth import browser

    jobs = [(pathlib.Path(g).resolve(), pathlib.Path(p), bool(pose)) for g, p, pose in jobs]
    results, done = queue.Queue(), threading.Event()

    class Page(SimpleHTTPRequestHandler):
        # Windows' registry can map .js to text/plain, which a module script refuses.
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                          ".svg": "image/svg+xml"}

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
            path = self.path.split("?")[0]
            if path == "/":
                return self.send((HERE / "studio.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/__drive":
                return self.send(DRIVER.encode(), "text/html; charset=utf-8")
            if path == "/__jobs":
                listed = [{"id": i, "url": f"/__glb/{i}", "pose": pose, "size": size, "body": body}
                          for i, (_, _, pose) in enumerate(jobs)]
                return self.send(json.dumps(listed).encode(), "application/json")
            if path.startswith("/__glb/"):
                return self.send(jobs[int(path.rsplit("/", 1)[1])][0].read_bytes(), "model/gltf-binary")
            return super().do_GET()

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.startswith("/__result/"):
                results.put((int(self.path.rsplit("/", 1)[1]), data))
            elif self.path == "/__log":
                print("page:", data.decode(errors="replace"), file=sys.stderr)
            elif self.path == "/__done":
                done.set()
            self.send(b"", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        page = subprocess.Popen(
            [browser(browser_path), "--headless=new", f"--user-data-dir={profile}", "--no-first-run",
             "--no-default-browser-check", "--disable-extensions", "--ignore-gpu-blocklist",
             "--window-size=1280,1000", f"http://127.0.0.1:{server.server_port}/__drive"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            got, waited = 0, time.time()
            while got < len(jobs):
                if page.poll() is not None:
                    raise RuntimeError("the browser exited before finishing")
                if time.time() - waited > STALL_SECONDS:
                    raise TimeoutError(f"no render for {STALL_SECONDS} s; see the page: lines above")
                try:
                    i, png = results.get(timeout=1.0)
                except queue.Empty:
                    if done.is_set():
                        raise RuntimeError(f"the page finished with {got} of {len(jobs)} renders; "
                                           "see the page: lines above")
                    continue
                if not png.startswith(b"\x89PNG"):
                    raise RuntimeError(f"render {i} came back empty")
                jobs[i][1].parent.mkdir(parents=True, exist_ok=True)
                jobs[i][1].write_bytes(png)
                got, waited = got + 1, time.time()
        finally:
            page.terminate()
            page.wait()
            server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("glb", nargs="+", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "out" / "renders")
    parser.add_argument("--views", default="pl", help="p: the photo's head pose, l: level (default both)")
    parser.add_argument("--size", type=int, default=900, help="render width in pixels (default 900)")
    parser.add_argument("--body", action="store_true", help="full body instead of the head")
    parser.add_argument("--browser", help="path to chrome.exe or msedge.exe")
    args = parser.parse_args()
    jobs = [(g, args.out / f"{g.stem}-{v}.png", v == "p") for g in args.glb for v in args.views]
    began = time.time()
    render(jobs, args.size, args.body, args.browser)
    print(f"{len(jobs)} renders in {time.time() - began:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
