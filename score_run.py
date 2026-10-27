"""Build, render and score every test face in one go, no browser window.

    python score_run.py TAG [--against BASE] [--skip-build] [--set uploads/test-set.json]

For each face in the test set: builds out/TAG-<name>.glb (compose.py, one
process per face), renders it at the photo's head pose and level
(render.py, out/renders/<name>-TAG{p,l}.png), and scores it:

  identity   SFace cosine, render against the photo (identity_score.py), at
             the pose and level; against the person's other photos too where
             the set lists them, which carries no self-match bias.
  hair       the angle between the strands the photo shows and those the
             render shows, aligned on the face, where both read clearly
             (0 = same way, 45 = unrelated, 90 = across). None where the hair
             reads too little.

Scores go to out/scores/TAG.json; with --against, each is printed beside the
other tag's. Builds pin BLAS to one thread and renders seed their noise, so a
rerun gives the same numbers: a difference is the change, not noise.

The test set is a JSON list, kept beside the photos (git-ignored):
  [{"name": "ref-man", "photo": "uploads/ref-man.png", "gender": "male",
    "others": ["uploads/ref-man-2.jpg"], "sides": ["uploads/ref-man-left.jpg"]}, ...]
`others` are photos scored against, never built from; `sides` are turned
photos built from (compose.py --side, as guided capture sends them). Keep
the two apart, or the score judges the avatar on its own input.
Without a set file, the five faces this project was tuned on are used.

The identity model's training data is research-only: this is an internal
yardstick and never ships (identity_score.py).
"""

from __future__ import annotations

import argparse
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
DEFAULT_SET = [
    {"name": "ref-man", "photo": "uploads/ref-man.png", "gender": "male"},
    {"name": "ref-woman", "photo": "uploads/ref-woman.png", "gender": "female"},
    {"name": "photo-10", "photo": "uploads/photo-10.jpg", "gender": "male"},
    {"name": "20261019-175001-images", "photo": "uploads/20261019-175001-images.jfif", "gender": "male"},
    {"name": "photo-16", "photo": "uploads/photo-16.jpg", "gender": "male",
     "others": ["uploads/sk-21.png", "uploads/sk-22.png"]},
]
# Landmarks that move least with expression, to align a render onto its photo.
STABLE = [168, 6, 1, 2, 10, 152, 234, 454, 33, 263]


def build(tag: str, face: dict) -> str:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(HERE)}
    sides = [arg for side in face.get("sides", []) for arg in ("--side", str(HERE / side))]
    run = subprocess.run([sys.executable, str(HERE / "compose.py"), str(HERE / face["photo"]),
                          "-o", str(OUT / f"{tag}-{face['name']}.glb"), "--gender", face.get("gender", "male"),
                          *sides], cwd=HERE, env=env, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"{face['name']} did not build:\n{run.stderr[-2000:]}")
    return face["name"]


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


def score(tag: str, faces: list[dict]) -> dict:
    import identity_score as ids
    out = {}
    for face in faces:
        name, photo = face["name"], HERE / face["photo"]
        row = {}
        reference = ids.embedding(photo)
        others = [ids.embedding(HERE / o) for o in face.get("others", [])]
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
        row["hair"] = hair_error(flat(photo), flat(OUT / "renders" / f"{name}-{tag}p.png"))
        row["hair"] = None if row["hair"] is None else round(row["hair"], 1)
        out[name] = row
    return out


def summary(scores: dict) -> dict:
    keys = sorted({k for row in scores.values() for k in row})
    return {k: round(float(np.mean(v)), 4) for k in keys
            if (v := [row[k] for row in scores.values() if row.get(k) is not None])}


def table(tag: str, scores: dict, base: str | None, base_scores: dict | None) -> str:
    cols = [("identity_p", "identity pose"), ("identity_l", "identity level"), ("others_p", "other photos"),
            ("hair", "hair angle")]
    cell = lambda v: "     -" if v is None else f"{v:6.3f}" if abs(v) < 2 else f"{v:6.1f}"
    lines = [f"{'face':24s}" + "".join(f"{label:>22s}" for _, label in cols)]
    rows = list(scores.items()) + [("mean", summary(scores))]
    base_rows = dict(base_scores or {}, mean=summary(base_scores or {}))
    for name, row in rows:
        cells = []
        for key, _ in cols:
            now, then = row.get(key), base_rows.get(name, {}).get(key) if base_scores else None
            text = cell(now)
            if then is not None and now is not None:
                text = f"{cell(then).strip()} -> {cell(now).strip()}"
            cells.append(f"{text:>22s}")
        lines.append(f"{name[:24]:24s}" + "".join(cells))
    lines.append(f"(tag {tag}" + (f", against {base})" if base else ")") +
                 "  identity: higher is better; hair angle: lower is better")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag")
    parser.add_argument("--against", help="a tag scored before, to print beside this one")
    parser.add_argument("--set", type=pathlib.Path, default=HERE / "uploads" / "test-set.json")
    parser.add_argument("--skip-build", action="store_true", help="render and score GLBs already built")
    parser.add_argument("--skip-render", action="store_true", help="score renders already made")
    parser.add_argument("--workers", type=int, default=max(1, min(5, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()
    faces = json.loads(args.set.read_text(encoding="utf-8")) if args.set.exists() else DEFAULT_SET
    began = time.time()
    if not args.skip_build:
        with ThreadPoolExecutor(args.workers) as pool:
            for name in pool.map(lambda f: build(args.tag, f), faces):
                print(f"built {name}", flush=True)
    if not args.skip_render:
        from render import render
        render([(OUT / f"{args.tag}-{f['name']}.glb", OUT / "renders" / f"{f['name']}-{args.tag}{v}.png", v == "p")
                for f in faces for v in "pl"])
    scores = score(args.tag, faces)
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    (OUT / "scores" / f"{args.tag}.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")
    base_scores = None
    if args.against:
        path = OUT / "scores" / f"{args.against}.json"
        base_scores = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if base_scores is None:
            print(f"no scores for {args.against}; score it first", file=sys.stderr)
    print(table(args.tag, scores, args.against, base_scores))
    print(f"{len(faces)} faces in {time.time() - began:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
