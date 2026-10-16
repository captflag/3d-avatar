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

Expression is fitted in the same solve and then left off the mesh. Without it a
smile was read as identity — a wider mouth, higher cheeks — and kept. MakeHuman's
expression units (CC0 too) are extra columns, each capped by the MediaPipe
blendshape scores that measure the same movement, so a unit the photo shows no
sign of cannot borrow from who the person is.

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

# MakeHuman's expression units, each with the MediaPipe blendshapes that score
# the same movement. Sided units take either side's score: the cap is only a
# ceiling, and the fit decides each side. Units no blendshape measures (nostril
# flare, platysma) are left out rather than given a free hand, and so is
# mouth-upward-retraction: as a second smile it overshot the lift on a broad
# grin and identity cancelled it with turned-down corners, a frown at rest.
EXPRESSION_UNITS = {
    "mouth-corner-puller": ["mouthSmileLeft", "mouthSmileRight"],
    "mouth-retraction": ["mouthStretchLeft", "mouthStretchRight",
                         "mouthDimpleLeft", "mouthDimpleRight"],
    "mouth-depression": ["mouthFrownLeft", "mouthFrownRight"],
    "mouth-open": ["jawOpen"],
    "mouth-part-later": ["mouthUpperUpLeft", "mouthUpperUpRight"],
    "mouth-depression-retraction": ["mouthLowerDownLeft", "mouthLowerDownRight"],
    "mouth-elevation": ["mouthShrugLower", "mouthShrugUpper"],
    "mouth-compression": ["mouthPressLeft", "mouthPressRight"],
    "mouth-pursing": ["mouthPucker"],
    "mouth-protusion": ["mouthFunnel"],
    "eye-left-slit": ["eyeSquintLeft", "eyeSquintRight"],
    "eye-right-slit": ["eyeSquintLeft", "eyeSquintRight"],
    "eye-left-closure": ["eyeBlinkLeft", "eyeBlinkRight"],
    "eye-right-closure": ["eyeBlinkLeft", "eyeBlinkRight"],
    "eye-left-opened-up": ["eyeWideLeft", "eyeWideRight"],
    "eye-right-opened-up": ["eyeWideLeft", "eyeWideRight"],
    "eyebrows-left-down": ["browDownLeft", "browDownRight"],
    "eyebrows-right-down": ["browDownLeft", "browDownRight"],
    "eyebrows-left-inner-up": ["browInnerUp"],
    "eyebrows-right-inner-up": ["browInnerUp"],
    "eyebrows-left-extern-up": ["browOuterUpLeft", "browOuterUpRight"],
    "eyebrows-right-extern-up": ["browOuterUpLeft", "browOuterUpRight"],
    "nose-left-elevation": ["noseSneerLeft", "noseSneerRight"],
    "nose-right-elevation": ["noseSneerLeft", "noseSneerRight"],
}
EXPRESSION_DIR = TARGET_DIR / "expression"
# Each unit comes in three regional sculpts; blended equally, as body_for does.
REGIONS = ("african", "asian", "caucasian")
# A blendshape score below this is noise, not an expression to take off.
SHOWN = 0.1
# A unit may go up to this many times its blendshape score, and no further.
EXPRESSION_GAIN = 1.5
# Lighter than identity's: where a smile and a wide mouth explain the same
# landmarks, the smile the photo was measured to have is the likelier cause.
EXPRESSION_REGULARISE = 0.0005
# How alike (|cosine| at the landmarks) an identity target and a shown
# expression unit must be for the identity target to sit the fit out.
SAME_AXIS = 0.6
# How far across the face, in mesh units, to look for the skin in front of a
# landmark's vertex (see `front_of`).
FRONT_REACH = 0.02

# MediaPipe's lip contours from the right corner to the left, corners left out:
# the same position in each list is the same place along the mouth.
INNER_UPPER = [191, 80, 81, 82, 13, 312, 311, 310, 415]
INNER_LOWER = [95, 88, 178, 87, 14, 317, 402, 318, 324]
OUTER_UPPER = [185, 40, 39, 37, 0, 267, 269, 270, 409]
OUTER_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375]

_targets = None
_expressions = None
_canonical = None


def read_target(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """A MakeHuman .target file: vertex indices and their offsets."""
    idx, delta = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        idx.append(int(parts[0]))
        delta.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(idx, dtype=np.int64), np.array(delta, dtype=np.float64).reshape(-1, 3)


def load_targets():
    """Every face target as (name, vertex indices, offsets)."""
    global _targets
    if _targets is not None:
        return _targets

    targets = []
    for folder in FACE_FOLDERS:
        for path in sorted((TARGET_DIR / folder).glob("*.target")):
            idx, delta = read_target(path)
            if len(idx):
                targets.append((f"{folder}/{path.stem}", idx, delta))
    _targets = targets
    return targets


def load_expressions():
    """Every expression unit as (name, vertex indices, offsets), regions blended."""
    global _expressions
    if _expressions is None:
        _expressions = []
        for unit in EXPRESSION_UNITS:
            idx, delta = zip(*(read_target(EXPRESSION_DIR / region / f"{unit}.target")
                               for region in REGIONS))
            # The regions move overlapping vertices; sum them per vertex.
            merged, slot = np.unique(np.concatenate(idx), return_inverse=True)
            offsets = np.zeros((len(merged), 3))
            np.add.at(offsets, slot, np.concatenate(delta) / len(REGIONS))
            _expressions.append((unit, merged, offsets))
    return _expressions


def body_for(base: np.ndarray, gender: str) -> np.ndarray:
    """The base mesh made into an adult man or woman, before any face fit.

    The raw base mesh is neither: MakeHuman always applies its macro targets on
    top, and we had been skipping them, so every avatar wore the same
    androgynous body and "female" only changed the outfit. For a young adult of
    average muscle and weight the universal target is empty; the male/female
    difference lives in the three regional targets, which MakeHuman blends
    equally by default. We do the same, so no ancestry is assumed — the face fit
    supplies the person's own features afterwards.
    """
    sex = "female" if gender == "female" else "male"
    shaped = base.copy()
    for region in ("african", "asian", "caucasian"):
        idx, delta = read_target(TARGET_DIR / "macrodetails" / f"{region}-{sex}-young.target")
        shaped[idx] += delta / 3.0
    return shaped


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


def front_of(base: np.ndarray, body_faces: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """For each landmark vertex, the front-most skin within FRONT_REACH of it
    across the face.

    MediaPipe's average face sits its eyelid landmarks about a centimetre
    deeper than MakeHuman's lids, so their nearest vertices are on the rim
    inside the lid, which MakeHuman's blink and squint barely move. The skin a
    camera sees there is the lid in front. Used for expression only; identity
    keeps the correspondences it was tuned with.

    The inner edge of each lip takes its outer edge's vertex. MakeHuman's mouth
    is closed, so a parted mouth's inner contour has no skin of its own, and
    the front-most vertex there was the upper lip for some landmarks and the
    lower lip for their neighbours: neutralising then crossed the two lips over.
    """
    body = np.unique(body_faces)
    pool = body[base[body, 1] > base[corr, 1].min() - FRONT_REACH]
    across = np.linalg.norm(base[corr, None, :2] - base[None, pool, :2], axis=2)
    depth = np.where(across < FRONT_REACH, base[pool, 2][None, :], -np.inf)
    front = pool[depth.argmax(axis=1)]
    front[INNER_UPPER], front[INNER_LOWER] = front[OUTER_UPPER], front[OUTER_LOWER]
    return front


def fit(base: np.ndarray, corr: np.ndarray, person: np.ndarray, average: np.ndarray,
        scores: dict | None = None,
        front: np.ndarray | None = None) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Identity weights in [0, 1] that move the skull like this face differs
    from average, and the expression fitted alongside them.

    Returns the identity weights, their names, and the expression as
    {unit: weight}, which the mesh is built without. A unit is fitted only as
    far as the photo's blendshape `scores` show its movement, measured on the
    skin at `front` (see `front_of`). With no scores the fit is identity alone.
    """
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

    # Expression columns, each capped by the photo's own blendshape scores.
    scores = scores or {}
    measured = [max(scores.get(s, 0.0) for s in shapes) for shapes in EXPRESSION_UNITS.values()]
    shown = [k for k, m in enumerate(measured) if m >= SHOWN]
    ceiling = {k: min(1.0, EXPRESSION_GAIN * measured[k]) for k in shown}
    if shown:
        units = load_expressions()
        at = corr if front is None else front
        moves = [_landmark_motion(base, at, units[k]) for k in shown]
        basis = np.hstack([basis, np.column_stack(moves)])

    residual = (person[:n] - average[:n]).reshape(-1)

    # Trust depth less than width and height.
    trust = np.tile([1.0, 1.0, DEPTH_TRUST], n)
    a = basis * trust[:, None]
    b = residual * trust

    # Along a shown unit's own axis the landmarks cannot tell who someone is
    # from what they were doing, so identity targets that nearly repeat or
    # nearly reverse it sit this fit out. Repeating it, a smile became a mouth
    # shaped that way; reversing it, identity cancelled the lift it had just let
    # expression take, and the resting face frowned.
    t = len(targets)
    used = np.ones(a.shape[1], dtype=bool)
    if shown:
        unit = a / np.maximum(np.linalg.norm(a, axis=0), 1e-12)
        used[:t] = np.abs(unit[:, :t].T @ unit[:, t:]).max(axis=1) < SAME_AXIS

    # Ridge as extra rows, then bounded so every weight stays a real MakeHuman
    # setting between off and fully applied.
    ridge = np.array([REGULARISE] * t + [EXPRESSION_REGULARISE] * len(shown))[used]
    upper = np.array([1.0] * t + [ceiling[k] for k in shown])[used]
    a = np.vstack([a[:, used], np.diag(np.sqrt(ridge))])
    b = np.concatenate([b, np.zeros(len(ridge))])
    weights = np.zeros(len(used))
    weights[used] = lsq_linear(a, b, bounds=(0.0, upper), method="trf", lsmr_tol="auto").x

    expression = dict.fromkeys(EXPRESSION_UNITS, 0.0)
    names = list(EXPRESSION_UNITS)
    for k, w in zip(shown, weights[t:]):
        expression[names[k]] = float(w)
    return weights[:t], [name for name, _, _ in targets], expression


def _landmark_motion(base: np.ndarray, at: np.ndarray, target) -> np.ndarray:
    """What one target does to the vertices `at`, flattened as a basis column."""
    _, idx, delta = target
    motion = np.zeros_like(base)
    motion[idx] = delta
    return motion[at].reshape(-1)


def expression_offsets(base: np.ndarray, at: np.ndarray, expression: dict) -> np.ndarray:
    """How the fitted expression moves each landmark. Subtract it to neutralise."""
    if not any(expression.values()):
        return np.zeros((len(at), 3))
    units = load_expressions()
    shaped = apply(base, [expression.get(name, 0.0) for name, _, _ in units], units)
    return shaped[at] - base[at]


def close_lips(vertices: np.ndarray) -> np.ndarray:
    """Landmarks with parted lips brought together, for a resting face.

    No expression unit explains an upper lip lifted off the teeth, so after the
    fitted expression comes off, an open smile still has a gap between its lips,
    and the teeth in it were baked onto the closed lips of the mesh. Each lip
    moves half the gap, inner and outer edge together so it keeps its
    thickness. Lips already closed are left alone. The texture sampler looks
    each lip up separately (FaceSampler's `lips`), so meeting exactly does not
    fold the warp.
    """
    v = vertices.copy()
    shut = np.maximum(v[INNER_UPPER, 1] - v[INNER_LOWER, 1], 0.0) / 2.0
    for upper, lower in ((INNER_UPPER, INNER_LOWER), (OUTER_UPPER, OUTER_LOWER)):
        v[upper, 1] -= shut
        v[lower, 1] += shut
    return v


def apply(base: np.ndarray, weights, targets=None) -> np.ndarray:
    """The base mesh with the fitted shape applied — helpers included, so the
    eyes and teeth move with the face that holds them. Identity targets unless
    told otherwise."""
    shaped = base.copy()
    for w, (_, idx, delta) in zip(weights, targets or load_targets()):
        if w > 1e-4:
            shaped[idx] += w * delta
    return shaped


def explained(base: np.ndarray, corr: np.ndarray, person: np.ndarray,
              average: np.ndarray, weights: np.ndarray,
              expression: dict | None = None, front: np.ndarray | None = None) -> float:
    """How much of the face's difference from average the fit accounts for,
    identity and expression together, 0-1."""
    n = len(corr)
    want = (person[:n] - average[:n])
    got = apply(base, weights)[corr] - base[corr]
    if expression:
        got = got + expression_offsets(base, corr if front is None else front, expression)
    before = np.linalg.norm(want)
    after = np.linalg.norm(want - got)
    return float(1.0 - after / before) if before > 0 else 0.0
