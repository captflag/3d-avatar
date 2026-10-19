"""Synthetic training data for a learned head-shape model (ADR-001, item 4).

The face fit in morph.py is linear and explains about half to two thirds of how
a face differs from average. The plan is a learned correction on top of it,
trained only on data we own. This makes that data: random heads built from the
same MakeHuman CC0 targets the fit uses, dressed in the catalogue's CC0 hair and
clothes, rendered as photo-like images, each saved with the exact shape behind it.

Each sample draws:
  - face targets from the folders morph.fit uses, as sliders: a few moved, one
    end of each (incr or decr, up or down ...), left and right nearly alike;
  - a male/female blend of MakeHuman's macro targets (morph.body_for);
  - skin tone and mottling, lip colour, iris, haircut and hair colour, brows,
    an outfit;
  - a camera: yaw within 20 degrees, pitch within 10, 24-85 mm equivalent, the
    face filling a third to two thirds of the frame and a little off centre;
  - lighting: key, fill and rim direction, strength and colour, environment,
    exposure, and a gradient backdrop.

The render is the studio's own: three.js with lib/scene-setup.js (synth.html),
in headless Chrome. Worker processes build each head as a GLB; a local server
hands them to the page, which posts back the PNG.

Writes NNNNNN.png and NNNNNN.json. The JSON holds every non-zero target weight
(names as in morph.load_targets(), absent means 0), the macro blend, the camera
in model units (1 unit is about 10 cm; +x is the subject's left, +y up, +z out
of the face), and the rest of the draw. `vertices_for(female, weights)` rebuilds
the exact mesh.

ponytail: the shape prior is a guess (sparse, near-symmetric, half-normal), not
the distribution of real fits; neutral expression only; plain gradient
backdrops. Revisit each once a trained model shows which one it trips on.
"""

from __future__ import annotations

import argparse
import colorsys
import functools
import json
import math
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

import compose
import morph
import skin
import skinned_glb
from appearance import IRIS_PALETTE, hairline_tint
from catalogue import build_item, catalogue, items_by_id, load_base, load_base_uv, submesh_uv
from facetexture import head_features, rasterize
from rig import load_skeleton, weight_table

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "data" / "synth"

# Opposite ends of one MakeHuman slider. A sample moves a slider towards one end
# or leaves it, so no two targets cancel into the same face under two labels.
OPPOSITE = {"decr": "incr", "down": "up", "in": "out", "backward": "forward",
            "concave": "convex", "compress": "uncompress"}
# Share of sliders moved per face, and the spread of how far. Fits of real
# photos move about 100 of the 270 targets, the strongest to 0.6-1.0.
ACTIVE = 0.4
SPREAD = 0.45
# How far a right-side slider strays from its left-side twin.
ASYMMETRY = 0.1

BALD = 0.1
HAIR_RGB = [(22, 18, 16), (48, 34, 26), (86, 58, 40), (128, 94, 62), (158, 126, 86),
            (204, 174, 124), (124, 58, 34), (150, 148, 144), (212, 210, 204)]

CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
# Longest wait for one render, the first included (three.js from the CDN,
# shader compiles), before giving up on the browser.
STALL_SECONDS = 180


def target_names() -> list[str]:
    return [name for name, _, _ in morph.load_targets()]


@functools.cache
def sliders() -> list[tuple[int | None, int]]:
    """Each face slider as (index of its negative end or None, its positive end)."""
    names = target_names()
    index = {n: i for i, n in enumerate(names)}
    out = []
    for i, name in enumerate(names):
        stem, _, end = name.rpartition("-")
        if f"{stem}-{OPPOSITE.get(end)}" in index:
            continue                                   # taken with its positive end
        negative = next((index[f"{stem}-{lo}"] for lo, hi in OPPOSITE.items()
                         if hi == end and f"{stem}-{lo}" in index), None)
        out.append((negative, i))
    return out


def shape(rng: np.random.Generator) -> np.ndarray:
    """Face target weights: a few sliders moved, one end each, sides alike."""
    names, axes = target_names(), sliders()
    value = np.where(rng.random(len(axes)) < ACTIVE, rng.normal(0.0, SPREAD, len(axes)), 0.0)
    slot = {names[pos]: k for k, (_, pos) in enumerate(axes)}
    for k, (_, pos) in enumerate(axes):
        left = slot.get(names[pos].replace("/r-", "/l-"))
        if left is not None and left != k:
            value[k] = value[left] + (value[left] != 0) * rng.normal(0.0, ASYMMETRY)
    # Rounded before use, so the weights written out are exactly the ones applied.
    value = np.clip(value, -1.0, 1.0).round(4)
    weights = np.zeros(len(names))
    for (negative, positive), v in zip(axes, value):
        if v >= 0 or negative is None:
            weights[positive] = abs(v)
        else:
            weights[negative] = -v
    return weights


@functools.cache
def bodies() -> tuple[np.ndarray, np.ndarray]:
    raw, _ = load_base()
    return morph.body_for(raw, "male"), morph.body_for(raw, "female")


def vertices_for(female: float, weights) -> np.ndarray:
    """The exact base mesh of a sample: macro blend, then face targets."""
    male, woman = bodies()
    return morph.apply((1.0 - female) * male + female * woman, weights)


@functools.cache
def fixed():
    """What every head shares: UV islands, the lip mask, the skin grain."""
    raw, groups = load_base()
    uv_all, uv_groups = load_base_uv()
    faces, uv_faces = groups["body"], uv_groups["body"]
    in_head = compose.head_island(raw, groups, faces, uv_faces)
    head_uv = compose.island_uv(uv_all, uv_faces[in_head])
    # MakeHuman paints no lips; its lip-volume targets say where they are.
    lip = np.zeros(len(raw))
    for which in ("upperlip", "lowerlip"):
        idx, delta = morph.read_target(morph.TARGET_DIR / "mouth" / f"mouth-{which}-volume-incr.target")
        reach = np.linalg.norm(delta, axis=1)
        lip[idx] = np.maximum(lip[idx], np.clip(reach / (0.3 * reach.max()), 0.0, 1.0))
    lips = cv2.GaussianBlur(rasterize(lip, faces[in_head], head_uv, uv_faces[in_head], 1024, fill=0.0),
                            (0, 0), 2.0)
    return in_head, head_uv, lips, Image.fromarray(skin.pore_normals(1024))


def lab_to_rgb(lab) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB).reshape(3)


def tint(rng: np.random.Generator) -> list[float]:
    """A light's colour: warm tungsten through white to cool shade, sRGB 0-255."""
    t = rng.uniform(-1.0, 1.0)
    end = np.array([255, 190, 130]) if t < 0 else np.array([200, 218, 255])
    return [round(float(c), 1) for c in 255 + abs(t) * (end - 255)]


def light(rng, azimuth, elevation, intensity) -> dict:
    """A directional light, aimed by degrees round the face (0 is straight ahead)."""
    az, el = math.radians(azimuth), math.radians(elevation)
    return {"azimuth": round(azimuth, 2), "elevation": round(elevation, 2),
            "direction": [math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)],
            "intensity": round(intensity, 3), "colour": tint(rng)}


def build(index: int, seed: int, size: int, folder: pathlib.Path) -> dict:
    """One random head, written to folder/<id>.glb. Returns everything drawn."""
    rng = np.random.default_rng([seed, index])
    _, groups = load_base()
    uv_all, uv_groups = load_base_uv()
    in_head, head_uv, lips, pores = fixed()
    names = target_names()

    female = round(float(rng.beta(0.6, 0.6)), 4)
    weights = shape(rng)
    vertices = vertices_for(female, weights)

    # --- dress: an outfit, brows, lashes and, mostly, a haircut ---
    items = items_by_id()
    haircuts = [i["id"] for i in catalogue()["items"] if i["category"] == "hair" and "mhclo" in i]
    outfits = [i["id"] for i in catalogue()["items"] if i["category"] == "outfit"]
    cut = None if rng.random() < BALD else str(rng.choice(haircuts))
    hair_rgb = np.clip(np.array(HAIR_RGB[rng.integers(len(HAIR_RGB))]) * rng.uniform(0.85, 1.15), 0, 255)
    brow_rgb = hair_rgb * rng.uniform(0.7, 1.0)
    wear = [str(rng.choice(outfits)), str(rng.choice(["eyebrows", "eyebrows_thin"])), "eyelashes"]
    if cut:
        wear += ["hair_underlayer", cut]
    placed = compose.trim_face_border(compose.place_face(morph.canonical_face(), vertices, groups))
    worn = []
    for item in (items[i] for i in wear):
        kind = item.get("colour")
        colour = (hair_rgb * float(item.get("colour_scale", 1.0)) if kind == "from_photo"
                  else brow_rgb if kind == "from_brows" else None)
        if item["category"] == "hair" and "group" in item:
            item = {**item, "hairline": compose.hairline_for(placed, vertices, groups,
                                                             float(item.get("hairline_lift", 0.0)))}
            item.pop("y_min", None)
        worn.append((item, build_item(item, None if colour is None else colour.astype(np.uint8), vertices)))

    # Skin MakeHuman marks as under the outfit is left out: a still head
    # cannot open a gap at the collar.
    faces, uv_faces = groups["body"], uv_groups["body"]
    hidden = np.zeros(len(vertices), dtype=bool)
    for _, mesh in worn:
        if mesh.metadata.get("hides") is not None:
            hidden[mesh.metadata["hides"][mesh.metadata["hides"] < len(hidden)]] = True
    shown = ~hidden[faces].all(axis=1)

    # --- skin: a tone from very dark to very light, lips a shade darker and redder ---
    tone = np.array([rng.uniform(28, 80) * 2.55, 128 + rng.uniform(7, 18), 128 + rng.uniform(10, 26)])
    spread = np.array([rng.uniform(4, 9), rng.uniform(1.5, 3), rng.uniform(1.5, 3)])
    lip_shift = np.array([-rng.uniform(8, 25), rng.uniform(6, 16), rng.uniform(-4, 4)])
    face_lab = cv2.cvtColor(skin.albedo(1024, tone, spread, seed=int(rng.integers(1 << 30))),
                            cv2.COLOR_RGB2LAB).astype(np.float64)
    face_lab += lips[..., None] * lip_shift
    head_texture = Image.fromarray(cv2.cvtColor(np.clip(face_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    head_faces, head_uv_faces = faces[in_head & shown], uv_faces[in_head & shown]
    haircut = next((m for item, m in worn if item["id"] == cut), None)
    if haircut is not None:
        head_texture = hairline_tint(head_texture, vertices, head_faces, head_uv, head_uv_faces,
                                     haircut.vertices, hair_rgb)
    body_texture = Image.fromarray(skin.albedo(1024, tone, spread, seed=int(rng.integers(1 << 30))))
    iris_name = str(rng.choice(list(IRIS_PALETTE)))
    eyes, eye_uv, eye_texture = compose.build_eyes(vertices, groups, np.array(IRIS_PALETTE[iris_name], float))

    # --- rig and write, in the rest pose so the head sits where it was built ---
    skeleton = load_skeleton(vertices)
    joints, bone_weights = weight_table(skeleton, len(vertices))
    parts = []
    for name, keep, uv, texture in (("head", in_head & shown, head_uv, head_texture),
                                    ("body", ~in_head & shown, uv_all, body_texture)):
        mesh, mesh_uv = submesh_uv(faces[keep], uv_faces[keep], vertices, uv=uv)
        src = mesh.metadata["base_index"]
        parts.append(skinned_glb.Part(
            name, mesh.vertices, mesh.faces, joints[src], bone_weights[src],
            uv=compose.gltf_uv(mesh_uv), texture=texture, normals=mesh.vertex_normals,
            normal_texture=pores, roughness=compose.ROUGHNESS["skin"],
            normal_scale=compose.PORE_STRENGTH[name]))
    for item, mesh in worn:
        roughness = item.get("roughness", compose.ROUGHNESS.get(item["category"], 0.8))
        parts.append(compose.rigged_part(item["id"], mesh, roughness, vertices, groups,
                                         joints, bone_weights, len(skeleton)))
    src = eyes.metadata["base_index"]
    parts.append(skinned_glb.Part("eyes", eyes.vertices, eyes.faces, joints[src], bone_weights[src],
                                  uv=compose.gltf_uv(eye_uv), texture=eye_texture,
                                  normals=eyes.vertex_normals, roughness=compose.ROUGHNESS["eyes"]))
    name = f"{index:06d}"
    skinned_glb.export(parts, skeleton, folder / f"{name}.glb", pose=compose.rest_pose())

    # --- camera: orbit the face, then frame it ---
    feature = head_features(vertices, groups)
    eyes_mid = (feature["left_eye"] + feature["right_eye"]) / 2.0
    centre = (eyes_mid + feature["mouth"]) / 2.0              # about the nose
    yaw, pitch = rng.uniform(-20, 20), rng.uniform(-10, 10)
    focal = float(np.exp(rng.uniform(np.log(24), np.log(85))))
    fov = 2.0 * math.degrees(math.atan(12.0 / focal))      # a full-frame sensor's 24 mm height
    # Hairline to chin is about 1.55 times eyes to chin.
    frame = 1.55 * (eyes_mid[1] - feature["chin"][1]) / rng.uniform(0.35, 0.6)
    distance = frame / (2.0 * math.tan(math.radians(fov) / 2.0))
    y, p = math.radians(yaw), math.radians(pitch)
    position = centre + distance * np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)])
    look_at = centre + np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.0]) * frame

    key_side = rng.choice([-1.0, 1.0])
    top = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0, 0.5), rng.uniform(0.15, 0.95))
    bottom = np.clip(np.array(top) * rng.uniform(0.5, 1.2), 0, 1)
    r4 = lambda v: [round(float(x), 4) for x in v]
    return {
        "id": name, "seed": seed, "index": index, "image": f"{name}.png", "size": size,
        "female": female,
        "targets": {names[i]: float(w) for i, w in enumerate(weights) if w > 0},
        "camera": {"yaw": round(yaw, 3), "pitch": round(pitch, 3), "focal_mm": round(focal, 2),
                   "fov": round(fov, 4), "focal_px": round(size / 2 / math.tan(math.radians(fov) / 2), 2),
                   "distance": round(distance, 4), "centre": r4(centre),
                   "position": r4(position), "look_at": r4(look_at), "up": [0, 1, 0]},
        "light": {"key": light(rng, key_side * rng.uniform(0, 75), rng.uniform(-5, 60), rng.uniform(0.8, 3.0)),
                  "fill": light(rng, -key_side * rng.uniform(30, 100), rng.uniform(-10, 30), rng.uniform(0.05, 0.8)),
                  "rim": light(rng, rng.uniform(140, 220), rng.uniform(10, 60), rng.uniform(0.0, 2.0)),
                  "environment": round(rng.uniform(0.2, 1.2), 3)},
        "exposure": round(rng.uniform(0.7, 1.4), 3),
        "background": {"top": r4(np.array(top) * 255), "bottom": r4(bottom * 255)},
        "skin_lab": r4(tone), "skin_rgb": [int(c) for c in lab_to_rgb(tone)],
        "lip_shift_lab": r4(lip_shift),
        "hair": cut, "hair_rgb": [int(c) for c in hair_rgb], "iris": iris_name,
        "wearing": wear,
    }


def browser(given: str | None) -> str:
    for path in (given, *CHROME, shutil.which("chrome"), shutil.which("chromium")):
        if path and pathlib.Path(path).exists():
            return path
    raise FileNotFoundError("no Chrome or Edge found; pass --browser")


def generate(out: pathlib.Path, count: int, size: int, seed: int, workers: int,
             chrome: str | None) -> None:
    """Build `count` heads in worker processes, render each in headless Chrome,
    and file image and draw side by side in `out`. Numbering carries on from
    what is already there, so a second run adds rather than overwrites."""
    out.mkdir(parents=True, exist_ok=True)
    start = len(list(out.glob("*.json")))
    built, rendered, failed = queue.Queue(), queue.Queue(), []
    # Built heads waiting for the renderer, each a few MB on disk.
    ahead = threading.Semaphore(2 * workers)
    handed, given = [0], {}
    lock = threading.Lock()

    with tempfile.TemporaryDirectory() as glbs, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile, \
            Pool(workers) as pool:
        glbs = pathlib.Path(glbs)

        class Page(SimpleHTTPRequestHandler):
            # Windows' registry can map .js to text/plain, which a module script refuses.
            extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript"}

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
                if self.path == "/next":
                    with lock:
                        more = handed[0] < count
                        handed[0] += more
                    job = built.get() if more else None
                    if job:
                        given[job["id"]] = job
                    return self.send(json.dumps(job).encode(), "application/json")
                if self.path.startswith("/glb/"):
                    return self.send((glbs / f"{pathlib.Path(self.path).name}.glb").read_bytes(),
                                     "model/gltf-binary")
                return super().do_GET()

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.startswith("/result/"):
                    rendered.put((pathlib.Path(self.path).name, body))
                elif self.path == "/log":
                    print("page:", body.decode(errors="replace"), file=sys.stderr)
                self.send(b"", "text/plain")

        def feed():
            for i in range(start, start + count):
                ahead.acquire()
                pool.apply_async(build, (i, seed, size, glbs), callback=built.put,
                                 error_callback=failed.append)

        threading.Thread(target=feed, daemon=True).start()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        page = subprocess.Popen(
            [browser(chrome), "--headless=new", f"--user-data-dir={profile}", "--no-first-run",
             "--no-default-browser-check", "--disable-extensions", "--ignore-gpu-blocklist",
             f"http://127.0.0.1:{server.server_port}/synth.html?size={size}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        began = time.time()
        try:
            for done in range(1, count + 1):
                waited = time.time()
                while True:
                    if failed:
                        raise failed[0]
                    if page.poll() is not None:
                        raise RuntimeError("the browser exited before finishing")
                    if time.time() - waited > STALL_SECONDS:
                        raise TimeoutError(f"no render for {STALL_SECONDS} s; see the page: lines above")
                    try:
                        name, png = rendered.get(timeout=1.0)
                        break
                    except queue.Empty:
                        pass
                (out / f"{name}.png").write_bytes(png)
                draw = given.pop(name)
                (out / f"{name}.json").write_text(json.dumps(draw, indent=1), encoding="utf-8")
                (glbs / f"{name}.glb").unlink()
                ahead.release()
                if done % 25 == 0 or done == count:
                    spent = time.time() - began
                    print(f"{done}/{count}  {spent:.0f} s  {spent / done:.2f} s each", flush=True)
        finally:
            page.terminate()
            page.wait()
            server.shutdown()


def check(out: pathlib.Path, need: float = 0.95) -> bool:
    """Does MediaPipe find a face in the renders? The data is useless if not."""
    import mediapipe as mp
    from generate import get_landmarker

    images = sorted(out.glob("*.png"))
    if not images:
        print(f"no images in {out}")
        return False
    missed = [p.stem for p in images
              if not get_landmarker().detect(mp.Image.create_from_file(str(p))).face_landmarks]
    found = 1.0 - len(missed) / len(images)
    print(f"MediaPipe found a face in {len(images) - len(missed)}/{len(images)} ({found:.1%}), "
          f"need {need:.0%}" + (f"; missed {', '.join(missed[:20])}" if missed else ""))
    return found >= need


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic heads with known shape, for training.")
    parser.add_argument("-n", "--count", type=int, default=0, help="how many samples to add")
    parser.add_argument("-o", "--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=512, help="image width and height, pixels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="processes building heads")
    parser.add_argument("--browser", help="path to chrome.exe or msedge.exe")
    parser.add_argument("--check", action="store_true",
                        help="afterwards, check MediaPipe finds a face in 95%% of the folder's images")
    args = parser.parse_args()

    if args.count:
        generate(args.out, args.count, args.size, args.seed, args.workers, args.browser)
    return 0 if not args.check or check(args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
