"""Build, render and score every test face in one go, no browser window.

    python score_run.py TAG [--against BASE [--gate METRIC]] [--noise] [--split dev|lockbox|all]
                            [--skip-build] [--skip-render] [--set data/testset/test-set.json]

For each face in the test set: builds out/TAG-<name>.glb (compose.py, one
process per face), renders it at the photo's head pose and level
(render.py, out/renders/<name>-TAG{p,l}.png), and scores it:

  identity   SFace cosine, render against the photo (identity_score.py), at
             the pose and level; against the person's other photos too where
             the set lists them (others_p), which carries no self-match bias.
  rank1      1 where the render is closer to its own person's photos than to
  margin     anyone else's, and by how much (own - best other); needs two or
             more people in the set.
  lm_*       landmark error, % of the eye spacing, render against photo after
             a similarity fit on the landmarks expressions hardly move: per
             region on the front photo (lm_front_outline, _nose, _eyes,
             _mouth, _brows), and over all of them against the held-out photo
             (lm_held), rendered at that photo's own head pose. The held-out
             photo is never built from, so it judges shape the front photo
             cannot fake.
  hair       the angle between the strands the photo shows and those the
             render shows, aligned on the face, where both read clearly
             (0 = same way, 45 = unrelated, 90 = across). None where the hair
             reads too little.

Scores go to out/scores/TAG.json; with --against, each is printed beside the
other tag's. Builds pin BLAS to one thread and renders seed their noise, so a
rerun gives the same numbers: a difference is the change, not the renderer.

How far a score moves with the photo itself is another matter: --noise also
builds each face from three barely different copies of its photo (JPEG
re-encoded, 1% resized) and writes each score's spread per face to
out/scores/noise.json. --gate METRIC then decides keep or drop against BASE:
the mean must improve, no face may get worse by more than 2 of its own
spreads, and identity at the pose may not fall by more than its spread. From
ten people up, a sign test and a Wilcoxon test (p < 0.05) must agree as well;
below ten the verdict says it is provisional. Exits 1 when the gate fails.

The test set (TESTSET.md has the photo protocol and the consent form) is a
JSON list in data/testset/, git-ignored:
  [{"name": "p01", "photo": "data/testset/p01/front.jpg", "gender": "male",
    "consent_id": "c01", "split": "dev",
    "sides": ["data/testset/p01/left35.jpg", "data/testset/p01/right35.jpg"],
    "held_out": "data/testset/p01/left20.jpg",
    "others": ["data/testset/p01/o1.jpg", "data/testset/p01/o2.jpg"]}, ...]
`others` and `held_out` are scored against, never built from; `sides` are
turned photos built from (compose.py --side, as guided capture sends them).
Keep them apart, or the score judges the avatar on its own input. A face
whose consent_id is missing from data/testset/consent.csv, withdrawn, without
the "score" scope or past its retain_until date is left out, and said so.

The identity model's training data is research-only: this is an internal
yardstick and never ships (identity_score.py).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
TESTSET = HERE / "data" / "testset"
# Landmarks that move least with expression, to align a render onto its photo.
STABLE = [168, 6, 1, 2, 10, 152, 234, 454, 33, 263]
# MediaPipe's own landmark regions, for the per-region shape error.
REGIONS = {"outline": ["FACE_OVAL"], "nose": ["NOSE"], "eyes": ["LEFT_EYE", "RIGHT_EYE"],
           "mouth": ["LIPS"], "brows": ["LEFT_EYEBROW", "RIGHT_EYEBROW"]}
# Scores where lower is better; the rest are better higher.
LOWER = ("hair", "lm_")
PROVISIONAL_BELOW = 10


def load_faces(path: pathlib.Path, split: str) -> list[dict]:
    """The test set's faces in `split` whose consent is on record and current."""
    if not path.exists():
        sys.exit(f"no test set at {path}; TESTSET.md says how to make one")
    register = path.parent / "consent.csv"
    rows = {}
    if register.exists():
        with register.open(encoding="utf-8", newline="") as f:
            rows = {r["consent_id"].strip(): r for r in csv.DictReader(f)}
    today = dt.date.today().isoformat()
    faces = []
    for face in json.loads(path.read_text(encoding="utf-8")):
        r = rows.get(str(face.get("consent_id", "")).strip())
        why = ("no consent on record" if r is None
               else "consent withdrawn" if (r.get("withdrawn") or "").strip()
               else "no consent to scoring" if "score" not in (r.get("scopes") or "")
               else "past its retain_until date" if (r.get("retain_until") or "") < today
               else None)
        if why:
            print(f"left out {face.get('name')}: {why}", file=sys.stderr)
        elif split == "all" or face.get("split", "dev") == split:
            faces.append(face)
    if not faces:
        sys.exit("no faces to score")
    return faces


def noise_copies(face: dict) -> list[dict]:
    """The face built from barely different copies of its photo (--noise)."""
    from PIL import Image, ImageOps
    folder = OUT / "noise"
    folder.mkdir(parents=True, exist_ok=True)
    photo = ImageOps.exif_transpose(Image.open(HERE / face["photo"])).convert("RGB")
    w, h = photo.size
    copies = []
    for k, (scale, quality) in enumerate([(1.0, 90), (0.99, None), (1.01, 95)]):
        im = photo.resize((round(w * scale), round(h * scale)), Image.LANCZOS) if scale != 1 else photo
        path = folder / f"{face['name']}~{k}.{'jpg' if quality else 'png'}"
        im.save(path, **({"quality": quality} if quality else {}))
        copies.append({**face, "name": f"{face['name']}~{k}", "photo": str(path)})
    return copies


def held_glb(tag: str, face: dict) -> pathlib.Path:
    """The face's GLB set to turn its head as the held-out photo shows it."""
    import pygltflib
    from compose import photo_head_pose
    from generate import fit_face
    _, marks, result, photo = fit_face(HERE / face["held_out"])
    pose = photo_head_pose(result, marks, photo)
    g = pygltflib.GLTF2().load(str(OUT / f"{tag}-{face['name']}.glb"))
    g.meshes[0].extras = {**(g.meshes[0].extras or {}), "photoHeadPose": pose}
    path = OUT / "held" / f"{tag}-{face['name']}.glb"
    path.parent.mkdir(parents=True, exist_ok=True)
    g.save(str(path))
    return path


def build(tag: str, face: dict) -> str:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(HERE)}
    sides = [arg for side in face.get("sides", []) for arg in ("--side", str(HERE / side))]
    run = subprocess.run([sys.executable, str(HERE / "compose.py"), str(HERE / face["photo"]),
                          "-o", str(OUT / f"{tag}-{face['name']}.glb"), "--gender", face.get("gender", "male"),
                          *sides], cwd=HERE, env=env, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"{face['name']} did not build:\n{run.stderr[-2000:]}")
    return face["name"]


def upright(path: pathlib.Path) -> pathlib.Path:
    """The photo as the build reads it (generate.stand_up: EXIF orientation,
    then whole quarter turns until the head is up), saved once under
    out/upright/. Scored as it lies, a sideways phone photo matched nothing."""
    import hashlib
    from generate import stand_up
    copy = OUT / "upright" / f"{hashlib.md5(str(path.resolve()).encode()).hexdigest()[:16]}.png"
    if not copy.exists() or copy.stat().st_mtime < path.stat().st_mtime:
        copy.parent.mkdir(parents=True, exist_ok=True)
        stand_up(path)[0].save(copy)
    return copy


def flat(path) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    back = Image.new("RGBA", im.size, (200, 200, 200, 255))
    back.alpha_composite(im)
    return np.asarray(back.convert("RGB"))


def landmarks(rgb: np.ndarray) -> np.ndarray | None:
    import mediapipe as mp
    from generate import get_landmarker
    found = get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not found.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in found.face_landmarks[0]])


def hair_error(photo: np.ndarray, render: np.ndarray) -> float | None:
    """Mean angle, degrees, between photo and render strands (see the module doc)."""
    import cv2
    import appearance
    from garments import _strand_field

    pm, rm = landmarks(photo), landmarks(render)
    if pm is None or rm is None:
        return None
    h, w = photo.shape[:2]
    fit, _ = cv2.estimateAffinePartial2D(rm[STABLE].astype(np.float32), pm[STABLE].astype(np.float32))
    warped = cv2.warpAffine(render, fit, (w, h), borderValue=(200, 200, 200))
    eye = float(np.linalg.norm(pm[33] - pm[263]))
    strand_p, clear_p = _strand_field(photo, appearance.segment(photo) == appearance.HAIR, eye)
    parts = appearance.segment(warped)
    skin = (parts == appearance.FACE_SKIN) | (parts == appearance.BODY_SKIN)
    backdrop = np.abs(warped.astype(int) - 200).sum(axis=2) < 12
    brows = float(pm[[105, 334], 1].mean())
    hair_r = ~backdrop & ~skin & (np.arange(h)[:, None] < brows)
    strand_r, clear_r = _strand_field(warped, hair_r, eye)
    both = np.minimum(clear_p, clear_r)
    seen = both > 0.3
    if seen.sum() < 50:
        return None
    apart = np.abs((strand_p - strand_r + np.pi / 2) % np.pi - np.pi / 2)
    return float(np.degrees((apart * both)[seen].sum() / both[seen].sum()))


def landmark_error(photo: np.ndarray, render: np.ndarray) -> dict | None:
    """Per region and over all of them, % of the eye spacing (see the module doc)."""
    import cv2
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    pm, rm = landmarks(photo), landmarks(render)
    if pm is None or rm is None:
        return None
    fit, _ = cv2.estimateAffinePartial2D(rm[STABLE].astype(np.float32), pm[STABLE].astype(np.float32))
    placed = rm @ fit[:, :2].T + fit[:, 2]
    off = np.linalg.norm(placed - pm, axis=1) / np.linalg.norm(pm[33] - pm[263]) * 100
    marks = {region: sorted({i for part in parts for c in getattr(C, f"FACE_LANDMARKS_{part}")
                             for i in (c.start, c.end)}) for region, parts in REGIONS.items()}
    out = {region: float(off[idx].mean()) for region, idx in marks.items()}
    out["all"] = float(off[sorted({i for idx in marks.values() for i in idx})].mean())
    return out


def score(tag: str, faces: list[dict]) -> dict:
    import identity_score as ids
    out = {}
    unit = lambda e: e.ravel() / np.linalg.norm(e)
    # Each person's photos to tell them apart by: their other photos, else the front one.
    gallery = {}
    for face in faces:
        person = face["name"].split("~")[0]
        if person not in gallery:
            found = ([ids.embedding(upright(HERE / o)) for o in face.get("others", [])]
                     or [ids.embedding(upright(HERE / face["photo"]))])
            gallery[person] = [e for e in found if e is not None]
    for face in faces:
        name, photo = face["name"], upright(HERE / face["photo"])
        row = {}
        reference = ids.embedding(photo)
        others = [ids.embedding(upright(HERE / o)) for o in face.get("others", [])]
        for view in "pl":
            png = OUT / "renders" / f"{name}-{tag}{view}.png"
            got = ids.embedding(png)
            if got is None or reference is None:
                row[f"identity_{view}"] = None
                continue
            cos = lambda e: float(np.dot(got.ravel(), e.ravel()) / (np.linalg.norm(got) * np.linalg.norm(e)))
            row[f"identity_{view}"] = round(cos(reference), 4)
            if others:
                row[f"others_{view}"] = round(float(np.mean([cos(e) for e in others if e is not None])), 4)
        got = ids.embedding(OUT / "renders" / f"{name}-{tag}p.png")
        person = name.split("~")[0]
        if got is not None and gallery[person] and len(gallery) > 1:
            near = {p: float(np.mean([unit(got) @ unit(e) for e in es])) for p, es in gallery.items() if es}
            best_other = max(v for p, v in near.items() if p != person)
            row["rank1"] = float(near[person] > best_other)
            row["margin"] = round(near[person] - best_other, 4)
        render_p = flat(OUT / "renders" / f"{name}-{tag}p.png")
        for region, value in (landmark_error(flat(photo), render_p) or {}).items():
            if region != "all":
                row[f"lm_front_{region}"] = round(value, 2)
        held_png = OUT / "renders" / f"{name}-{tag}h.png"
        if face.get("held_out") and held_png.exists():
            held = landmark_error(flat(upright(HERE / face["held_out"])), flat(held_png))
            row["lm_held"] = None if held is None else round(held["all"], 2)
        row["hair"] = hair_error(flat(photo), render_p)
        row["hair"] = None if row["hair"] is None else round(row["hair"], 1)
        out[name] = row
    return out


def summary(scores: dict) -> dict:
    keys = sorted({k for row in scores.values() for k in row})
    return {k: round(float(np.mean(v)), 4) for k in keys
            if (v := [row[k] for row in scores.values() if row.get(k) is not None])}


def table(tag: str, scores: dict, base: str | None, base_scores: dict | None) -> str:
    cols = [("identity_p", "identity pose"), ("identity_l", "identity level"), ("others_p", "other photos"),
            ("margin", "margin"), ("lm_front_outline", "outline %"), ("lm_front_nose", "nose %"),
            ("lm_held", "held-out %"), ("hair", "hair angle")]
    cell = lambda v: "     -" if v is None else f"{v:6.3f}" if abs(v) < 2 else f"{v:6.1f}"
    lines = [f"{'face':24s}" + "".join(f"{label:>17s}" for _, label in cols)]
    rows = list(scores.items()) + [("mean", summary(scores))]
    base_rows = dict(base_scores or {}, mean=summary(base_scores or {}))
    for name, row in rows:
        cells = []
        for key, _ in cols:
            now, then = row.get(key), base_rows.get(name, {}).get(key) if base_scores else None
            text = cell(now)
            if then is not None and now is not None:
                text = f"{cell(then).strip()} -> {cell(now).strip()}"
            cells.append(f"{text:>17s}")
        lines.append(f"{name[:24]:24s}" + "".join(cells))
    lines.append(f"(tag {tag}" + (f", against {base})" if base else ")") +
                 "  identity, margin: higher is better; landmark %, hair angle: lower is better")
    return "\n".join(lines)


def spreads(scores: dict) -> dict:
    """Per face and score, the spread across the face and its noise copies."""
    out = {}
    for person in {n.split("~")[0] for n in scores}:
        group = [row for n, row in scores.items() if n.split("~")[0] == person]
        keys = {k for row in group for k in row}
        out[person] = {k: round(float(np.std(v, ddof=1)), 5) for k in keys
                       if len(v := [row[k] for row in group if row.get(k) is not None]) > 1}
    return out


def gate(metric: str, scores: dict, base: dict, noise: dict) -> bool:
    """Keep (True) or drop the change, by `metric` (see the module doc)."""
    from scipy.stats import binomtest, wilcoxon
    sign = -1 if metric.startswith(LOWER) else 1
    names = [n for n in scores if scores[n].get(metric) is not None and base.get(n, {}).get(metric) is not None]
    if not names:
        print(f"gate: no face has {metric} in both runs")
        return False
    unsure = [n for n in names if metric not in noise.get(n, {})]
    if unsure:
        print(f"gate: no noise spread for {', '.join(unsure)}; run with --noise first")
        return False
    d = np.array([sign * (scores[n][metric] - base[n][metric]) for n in names])
    sigma = np.array([max(noise[n][metric], 1e-4) for n in names])
    worst = int(np.argmin(d / sigma))
    checks = {"mean improves": d.mean() > 0,
              "no face worse than 2 spreads": (d / sigma).min() >= -2}
    if metric != "identity_p":
        idn = [n for n in scores if scores[n].get("identity_p") is not None
               and base.get(n, {}).get("identity_p") is not None and "identity_p" in noise.get(n, {})]
        if idn:
            drop = np.mean([scores[n]["identity_p"] - base[n]["identity_p"] for n in idn])
            checks["identity not falling"] = drop >= -np.mean([noise[n]["identity_p"] for n in idn])
    lines = [f"gate on {metric} ({'lower' if sign < 0 else 'higher'} is better), {len(names)} faces:",
             f"  mean change {sign * d.mean():+.4f}, median {sign * np.median(d):+.4f}, "
             f"better on {(d > 0).sum()}, worse on {(d < 0).sum()}",
             f"  worst: {names[worst]} {sign * d[worst]:+.4f}, {abs(d[worst]) / sigma[worst]:.1f} of its spreads "
             f"{'better' if d[worst] >= 0 else 'worse'}"]
    if len(names) >= PROVISIONAL_BELOW:
        moved = d[d != 0]
        p_sign = binomtest(int((moved > 0).sum()), len(moved), 0.5, alternative="greater").pvalue if len(moved) else 1.0
        p_wil = wilcoxon(d, alternative="greater").pvalue if len(moved) else 1.0
        checks["sign test p < 0.05"], checks["Wilcoxon p < 0.05"] = p_sign < 0.05, p_wil < 0.05
        lines.append(f"  sign test p {p_sign:.3f}, Wilcoxon p {p_wil:.3f}")
    for check, ok in checks.items():
        lines.append(f"  {'pass' if ok else 'FAIL'}  {check}")
    kept = all(checks.values())
    lines.append(f"  -> {'keep' if kept else 'drop'}" +
                 (f" (provisional: under {PROVISIONAL_BELOW} people)" if len(names) < PROVISIONAL_BELOW else ""))
    print("\n".join(lines))
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag")
    parser.add_argument("--against", help="a tag scored before, to print beside this one")
    parser.add_argument("--gate", metavar="METRIC", help="keep or drop against --against, by this score")
    parser.add_argument("--noise", action="store_true", help="also measure each score's spread per face")
    parser.add_argument("--set", type=pathlib.Path, default=TESTSET / "test-set.json")
    parser.add_argument("--split", default="dev", choices=["dev", "lockbox", "all"])
    parser.add_argument("--skip-build", action="store_true", help="render and score GLBs already built")
    parser.add_argument("--skip-render", action="store_true", help="score renders already made")
    parser.add_argument("--workers", type=int, default=max(1, min(5, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()
    people = load_faces(args.set, args.split)
    faces = people + [c for f in people for c in noise_copies(f)] if args.noise else people
    began = time.time()
    if not args.skip_build:
        with ThreadPoolExecutor(args.workers) as pool:
            for name in pool.map(lambda f: build(args.tag, f), faces):
                print(f"built {name}", flush=True)
    if not args.skip_render:
        from render import render
        render([(OUT / f"{args.tag}-{f['name']}.glb", OUT / "renders" / f"{f['name']}-{args.tag}{v}.png", v == "p")
                for f in faces for v in "pl"] +
               [(held_glb(args.tag, f), OUT / "renders" / f"{f['name']}-{args.tag}h.png", True)
                for f in people if f.get("held_out")])
    scores = score(args.tag, faces)
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    if args.noise:
        (OUT / "scores" / "noise.json").write_text(json.dumps(spreads(scores), indent=1), encoding="utf-8")
        scores = {n: row for n, row in scores.items() if "~" not in n}
    (OUT / "scores" / f"{args.tag}.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")
    base_scores = None
    if args.against:
        path = OUT / "scores" / f"{args.against}.json"
        base_scores = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if base_scores is None:
            print(f"no scores for {args.against}; score it first", file=sys.stderr)
    print(table(args.tag, scores, args.against, base_scores))
    print(f"{len(faces)} faces in {time.time() - began:.0f} s")
    if args.gate:
        noise_path = OUT / "scores" / "noise.json"
        noise = json.loads(noise_path.read_text(encoding="utf-8")) if noise_path.exists() else {}
        return 0 if base_scores and gate(args.gate, scores, base_scores, noise) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
