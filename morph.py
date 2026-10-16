"""Fit the head's shape to the person in the photo.

This is the missing piece behind most of REALISM-GAP.md: until now every avatar
had the same skull, and the photo was laid over it. Brow ridge, nose bridge,
cheekbones and jaw were nobody's in particular, which is why an avatar read as
"a photo of X on a dummy" rather than as X, and why the texture never quite
lined up.

A morphable head model fixes that, and we already own one. MakeHuman ships
hundreds of shape targets for the face — nose width and bridge, jaw, chin,
cheekbones, eye spacing, ear shape — each a set of per-vertex offsets on the very
base mesh we dress. They are CC0, each file says so, which sidesteps the trap the
research models fall into: FLAME's licence forbids commercial use and forbids
training networks on it, and SMPL-X is a paid commercial licence.

The fit is a bounded linear least squares: find the blend of targets that moves
the skull's landmark points the way this face differs from an average one.
"Average" is MediaPipe's own canonical face, placed on the head exactly as the
person's face is, so whatever error the placement makes cancels out of the
difference rather than being baked into the shape.

ponytail: linear blend of hand-authored targets, landmarks only, depth trusted
less than width and height. That captures proportions, not pores or asymmetric
folds. The upgrade is a learned displacement on top of this same basis, trained
on our own CC0 synthetic renders.
"""

from __future__ import annotations

import pathlib

import numpy as np
import trimesh
from scipy.optimize import lsq_linear

HERE = pathlib.Path(__file__).resolve().parent
TARGET_DIR = HERE / "assets" / "targets"
CANONICAL_OBJ = HERE / "assets" / "canonical_face_model.obj"

# Identity, not expression or age: these folders shape who someone is.
FACE_FOLDERS = ["nose", "mouth", "eyes", "chin", "cheek", "forehead", "head", "eyebrows", "ears"]

# Monocular depth from one photo is a guess; widths and heights are measured.
DEPTH_TRUST = 0.35
# Pull weights towards zero so the fit explains the face with as little change
# as it can, instead of stacking opposing targets to chase landmark noise.
REGULARISE = 0.002

_targets = None
_canonical = None


def load_targets():
    """Every face target as (name, vertex indices, offsets)."""
    global _targets
    if _targets is not None:
        return _targets

    targets = []
    for folder in FACE_FOLDERS:
        for path in sorted((TARGET_DIR / folder).glob("*.target")):
            idx, delta = [], []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                idx.append(int(parts[0]))
                delta.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if idx:
                targets.append((f"{folder}/{path.stem}",
                                np.array(idx, dtype=np.int64),
                                np.array(delta, dtype=np.float64)))
    _targets = targets
    return targets


def canonical_face() -> trimesh.Trimesh:
    """MediaPipe's average face, as a mesh in the same frame as a fitted one."""
    global _canonical
    if _canonical is None:
        vertices, faces = [], []
        for line in CANONICAL_OBJ.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
        _canonical = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces),
                                     process=False)
    return _canonical


def correspondences(base: np.ndarray, body_faces: np.ndarray,
                    placed_canonical: np.ndarray) -> np.ndarray:
    """For each of the 468 landmarks, the skull vertex it sits on.

    Found once against the average face, so it does not depend on whoever is in
    the photo.
    """
    body_vertices = np.unique(body_faces)
    candidates = base[body_vertices]
    head = candidates[:, 1] > placed_canonical[:, 1].min() - 0.6
    pool, pool_idx = candidates[head], body_vertices[head]
    d = np.linalg.norm(placed_canonical[:, None, :] - pool[None, :, :], axis=2)
    return pool_idx[d.argmin(axis=1)]


def fit(base: np.ndarray, corr: np.ndarray,
        person: np.ndarray, average: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Target weights in [0, 1] that move the skull like this face differs from average."""
    targets = load_targets()
    n = len(corr)

    # Each column is what one target does to the landmark vertices.
    lookup = {int(v): i for i, v in enumerate(corr)}
    basis = np.zeros((3 * n, len(targets)))
    for k, (_, idx, delta) in enumerate(targets):
        for v, d in zip(idx, delta):
            row = lookup.get(int(v))
            if row is not None:
                basis[3 * row:3 * row + 3, k] = d

    residual = (person[:n] - average[:n]).reshape(-1)

    # Trust depth less than width and height.
    trust = np.tile([1.0, 1.0, DEPTH_TRUST], n)
    a = basis * trust[:, None]
    b = residual * trust

    # Ridge as extra rows, then bounded so every weight stays a real MakeHuman
    # setting between off and fully applied.
    a = np.vstack([a, np.sqrt(REGULARISE) * np.eye(len(targets))])
    b = np.concatenate([b, np.zeros(len(targets))])
    result = lsq_linear(a, b, bounds=(0.0, 1.0), method="trf", lsmr_tol="auto")
    return result.x, [name for name, _, _ in targets]


def apply(base: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The base mesh with the fitted shape applied — helpers included, so the
    eyes and teeth move with the face that holds them."""
    shaped = base.copy()
    for w, (_, idx, delta) in zip(weights, load_targets()):
        if w > 1e-4:
            shaped[idx] += w * delta
    return shaped


def explained(base: np.ndarray, corr: np.ndarray, person: np.ndarray,
              average: np.ndarray, weights: np.ndarray) -> float:
    """How much of the face's difference from average the fit accounts for, 0-1."""
    n = len(corr)
    want = (person[:n] - average[:n])
    got = apply(base, weights)[corr] - base[corr]
    before = np.linalg.norm(want)
    after = np.linalg.norm(want - got)
    return float(1.0 - after / before) if before > 0 else 0.0
