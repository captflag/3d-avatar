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
# Blendshape scores that say the mouth is moving, and how hard a broad smile
# pulls mouth and cheek shape towards average (see `fit`).
MOUTH_SCORES = ("mouthSmileLeft", "mouthSmileRight", "jawOpen", "mouthStretchLeft", "mouthStretchRight")
# Below this smile score the mouth's press and retraction count as its shape.
SET_SMILE = 0.15
MOUTH_DAMPING = 300.0

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
# The distances a face is recognised by: eye spacing, face and jaw width, face
# height, nose to chin (canonical MediaPipe landmarks). Fitted as distances as
# well as through their landmarks: among 468 landmarks the few that set them
# were outvoted, and every avatar came out with eyes 4-6% too far apart and a
# chin 5-11% too short, which reads as a child's face. Forehead width (54-284)
# was tried for temples rendering 2-5% of the eye spacing too narrow: it moved
# them 0.1-0.4 and left two of four faces worse, so it is not among them.
# Cheekbone (50-280) and mid-cheek (123-352) width: across five faces the
# render's cheeks were 5-9% of the eye spacing off in either direction, a
# broad face oval and a narrow one broad, with no cheek target at its limit.
PROPORTIONS = [(33, 263), (234, 454), (172, 397), (10, 152), (2, 152), (50, 280), (123, 352)]
# Each distance counts as much as this many landmarks fitted exactly.
PROPORTION_WEIGHT = 10.0
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
        front: np.ndarray | None = None,
        reliable: np.ndarray | None = None,
        depth: np.ndarray | None = None) -> tuple[np.ndarray, list[str], dict[str, float]]:
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
    # A pressed, drawn-in mouth on a face that is not smiling is how the
    # mouth sits, not something it is doing: taken off as expression, a
    # photo's set mouth rendered as a flat neutral line (corners 7% of the
    # eye spacing higher than the photo's). Below SET_SMILE the press and
    # retraction units are left to the identity targets.
    smiling = max((scores.get(s, 0.0) for s in ("mouthSmileLeft", "mouthSmileRight")), default=0.0)
    if smiling < SET_SMILE:
        for k, name in enumerate(EXPRESSION_UNITS):
            if name in ("mouth-compression", "mouth-retraction"):
                measured[k] = 0.0
    shown = [k for k, m in enumerate(measured) if m >= SHOWN]
    ceiling = {k: min(1.0, EXPRESSION_GAIN * measured[k]) for k in shown}
    if shown:
        units = load_expressions()
        at = corr if front is None else front
        moves = [_landmark_motion(base, at, units[k]) for k in shown]
        basis = np.hstack([basis, np.column_stack(moves)])

    residual = (person[:n] - reference(base, corr, average)).reshape(-1)

    # Trust depth less than width and height, and any landmark the caller
    # marks as unreliable (a jaw outline lying on a beard) less again.
    # Depth counts a third where one photo only guessed it, fully where side
    # views measured it (`depth`, per landmark; views.fuse).
    trust = np.ones((n, 3))
    trust[:, 2] = DEPTH_TRUST if depth is None else np.asarray(depth, dtype=np.float64)[:n]
    trust = trust.reshape(-1)
    if reliable is not None:
        trust = trust * np.repeat(np.asarray(reliable, dtype=np.float64)[:n], 3)
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

    # The wider the smile, the less the photo says about the mouth at rest. The
    # expression units cannot reach a broad grin, and what they leave was taken
    # as identity: on a beaming portrait three mouth-width targets sat at their
    # limit and the "resting" mouth came out wide and full. So mouth and cheek
    # shape is pulled towards average in proportion to how much the mouth moves.
    rows, want = proportion_rows(base, corr, person, basis)
    a = np.vstack([a, PROPORTION_WEIGHT * rows])
    b = np.concatenate([b, PROPORTION_WEIGHT * want])

    ridges = np.full(t, REGULARISE)
    moving = max((scores.get(s, 0.0) for s in MOUTH_SCORES), default=0.0)
    damp = 1.0 + MOUTH_DAMPING * np.clip((moving - 0.3) / 0.6, 0.0, 1.0) ** 2
    for k, (name, _, _) in enumerate(targets):
        if name.startswith(("mouth/", "cheek/")):
            ridges[k] *= damp

    # Ridge as extra rows, then bounded so every weight stays a real MakeHuman
    # setting between off and fully applied.
    ridge = np.concatenate([ridges, [EXPRESSION_REGULARISE] * len(shown)])[used]
    upper = np.array([1.0] * t + [ceiling[k] for k in shown])[used]
    a = np.vstack([a[:, used], np.diag(np.sqrt(ridge))])
    b = np.concatenate([b, np.zeros(len(ridge))])
    lower = np.zeros(len(upper))
    if depth is not None:
        # Placement sets depth by the nose tip alone (compose.place_face), so
        # with depth measured, a face whose nose stands further out reads as
        # one whose every other landmark sits further back, and the targets
        # bent the whole face to match. A free shift in depth, which the mesh
        # never gets, takes up where the face sits instead.
        shift = np.zeros(len(a))
        shift[:3 * n] = np.tile([0.0, 0.0, 1.0], n) * trust
        a = np.hstack([a, shift[:, None]])
        upper = np.append(upper, np.inf)
        lower = np.append(lower, -np.inf)
    # bvls, not trf: with the cheek rows added, trf stopped short of the
    # optimum and left weights of 0.2 on targets nothing asked for (a face
    # differing from the skull only by a depth shift bent head-fat-decr to
    # 0.23); bvls solves the bounded problem exactly.
    solved = lsq_linear(a, b, bounds=(lower, upper), method="bvls").x
    weights = np.zeros(len(used))
    weights[used] = solved[:used.sum()]

    expression = dict.fromkeys(EXPRESSION_UNITS, 0.0)
    names = list(EXPRESSION_UNITS)
    for k, w in zip(shown, weights[t:]):
        expression[names[k]] = float(w)
    return weights[:t], [name for name, _, _ in targets], expression


def reference(base: np.ndarray, corr: np.ndarray, average: np.ndarray) -> np.ndarray:
    """Where the fit measures the photo's landmarks from: the skull's own, in
    the picture's plane.

    Measured from MediaPipe's average face, the fit carried only how the photo
    differs from average, and wherever MakeHuman's base differs from that
    average stayed in every result: its chin sits 6% of the eye spacing higher
    and its jaw corners 5% further in. Depth is still measured from the
    average: the landmarks snap to the nearest skin, a centimetre off at the
    eyelids, and photo depth is a guess anyway.
    """
    at = average[:len(corr)].copy()
    at[:, :2] = base[corr][:, :2]
    return at


def proportion_rows(base: np.ndarray, corr: np.ndarray, person: np.ndarray,
                    basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each of PROPORTIONS as a least-squares row: how every column of `basis`
    lengthens it, in the picture's plane, and how much longer the photo's is."""
    rows, want = [], []
    for i, j in PROPORTIONS:
        span = (base[corr[j]] - base[corr[i]]) * [1.0, 1.0, 0.0]
        length = np.linalg.norm(span)
        rows.append((span / length) @ (basis[3 * j:3 * j + 3] - basis[3 * i:3 * i + 3]))
        want.append(np.linalg.norm((person[j] - person[i])[:2]) - length)
    return np.array(rows), np.array(want)


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


# Where each lip turns into the mouth: the base mesh's vertices on the edge
# between the lips' skin and the inside of the mouth, corner to corner (the
# head's UV island ends there). Upper and lower share the corners.
UPPER_RIM = [527, 526, 3730, 528, 524, 525, 533, 5359, 532, 531, 539, 538, 537, 7282, 7283, 7276,
             7277, 11961, 7278, 7270, 7269, 7273, 10398, 7271, 7272]
LOWER_RIM = [527, 3709, 529, 530, 536, 535, 534, 540, 7279, 7280, 7281, 7275, 7274, 10377, 7272]
# How far round the rims the seal is carried (mesh units, ~3 mm).
SEAL_REACH = 0.03
# Every lip landmark: corners, outer and inner contours (MediaPipe's 40).
LIP_LANDMARKS = [61, 291, 78, 308] + OUTER_UPPER + OUTER_LOWER + INNER_UPPER + INNER_LOWER


def lip_corr(base: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """`corr` with the inner lip landmarks and the corners on the lips' rims.

    The nearest vertex to MediaPipe's average face put the corners on 528,
    3.6 mm in and 2.4 mm up from where the rims meet (527), and the inner
    lips on the lips' fronts, 2.4-3.8 mm off the seam: corrected to the
    photo there, the lips' fronts were pulled onto its lip line. Each inner
    landmark takes the rim vertex at its own place across the mouth."""
    out = corr.copy()
    for marks, rim in ((INNER_UPPER, UPPER_RIM), (INNER_LOWER, LOWER_RIM)):
        across = base[rim, 0]
        for i in marks:
            out[i] = rim[int(np.argmin(np.abs(across - base[corr[i], 0])))]
    for i in (61, 78, 291, 308):
        out[i] = UPPER_RIM[0] if base[corr[i], 0] * base[UPPER_RIM[0], 0] > 0 else UPPER_RIM[-1]
    return out


def seal_lips(vertices: np.ndarray, body_faces: np.ndarray) -> np.ndarray:
    """The lips' inner rims brought together, for a resting mouth.

    MakeHuman's mouth is modelled a little open: its rims stand 1 mm apart
    at the middle and 2.7 mm towards the corners, and on fitted heads the
    teeth showed through there from a third of the way out to each corner
    (355 pixels of teeth on ref-woman's head seen from the front). Moving the
    landmark vertices did not close it: those sit on the lips' fronts, and
    meeting there dented the lower lip. Each rim is moved to the middle of
    the two, and the lip it belongs to with it, fading over SEAL_REACH; the
    ARKit mouth shapes open it again as before."""
    v = vertices.copy()
    up, low = v[UPPER_RIM], v[LOWER_RIM]
    along = lambda chain, x: np.stack([np.interp(x, chain[:, 0], chain[:, k]) for k in range(3)], axis=1)
    to_up = (up + along(low, up[:, 0])) / 2.0 - up
    to_low = (low + along(up, low[:, 0])) / 2.0 - low
    seam = np.concatenate([up + to_up, low + to_low])
    seam = seam[np.argsort(seam[:, 0])]
    skin = np.unique(body_faces)
    p = v[skin]
    above = p[:, 1] > np.interp(p[:, 0], seam[:, 0], seam[:, 1])
    for rim, move, side in ((up, to_up, above), (low, to_low, ~above)):
        w = np.exp(-(np.linalg.norm(p[side, None] - rim[None], axis=2) / SEAL_REACH) ** 2)
        v[skin[side]] += (w @ move) / np.maximum(w.sum(axis=1, keepdims=True), 1e-12) * w.max(axis=1, keepdims=True)
    # The rims themselves exactly on the seam: the falloff averages each
    # rim's move with its neighbours' and left them 0.45 mm apart.
    v[UPPER_RIM], v[LOWER_RIM] = up + to_up, low + to_low
    return v


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


# The free-form correction after the target fit (`correct`): how wide each
# landmark's pull spreads, in eye-corner spacings, and how much it is damped
# against landmark noise (a ridge on the radial basis; 0 fits every landmark exactly).
CORRECT_REACH = 0.12
CORRECT_DAMPING = 0.3
# Depth is corrected only near the face's midline, within this many eye-corner
# spacings either side (a Gaussian): a profile lives there, and there the
# landmarks are fixed points on the skin. Towards the outline MediaPipe's
# landmarks follow the silhouette as the head turns, and on a rendered test
# head their fused depth came out 5-6% of the eye spacing too deep: corrected
# to it, the temples sank a centimetre and the chin stood 8 mm proud.
CORRECT_MIDLINE = 0.35


def correct(vertices: np.ndarray, base: np.ndarray, corr: np.ndarray, person: np.ndarray,
            average: np.ndarray, weights: np.ndarray, expression: dict | None = None,
            front: np.ndarray | None = None, reliable: np.ndarray | None = None,
            depth: np.ndarray | None = None) -> np.ndarray:
    """The fitted mesh moved the rest of the way to the photo's landmarks.

    MakeHuman's targets are a few hundred broad sliders: what the fit leaves
    is the part of this face no combination of them reaches (on a rendered
    test head the nose stayed 3-4 mm short). That remainder, at each landmark,
    is spread over the mesh by a Gaussian radial basis, damped so landmark
    noise is smoothed rather than baked in, and fading to nothing a few
    centimetres off the face. Every vertex near the face moves with it, the
    eyeballs and teeth among them, so lids and eyes stay together. Each
    landmark counts across the picture as far as the fit trusted it
    (`reliable`), and in depth only as far as side views measured it
    (`depth`, 0-1; views.fuse): a single photo's depth guess is left to the
    targets, which keep it to a plausible face.
    """
    n = len(corr)
    want = person[:n] - reference(base, corr, average)
    got = vertices[corr] - base[corr]
    if expression:
        got = got + expression_offsets(base, corr if front is None else front, expression)
    left = want - got
    trust = np.ones((n, 3))
    at = vertices[corr]
    spacing = float(np.linalg.norm(at[33] - at[263]))
    if depth is not None:
        midline = np.exp(-(at[:, 0] / (CORRECT_MIDLINE * spacing)) ** 2)
        trust[:, 2] = np.asarray(depth, dtype=np.float64)[:n] * midline
    else:
        trust[:, 2] = 0.0
    if reliable is not None:
        trust *= np.asarray(reliable, dtype=np.float64)[:n, None]
    # Where the face sits in depth is set by placement, not measured: no shift.
    left[:, 2] -= (left[:, 2] * trust[:, 2]).sum() / max(trust[:, 2].sum(), 1e-12)

    reach = CORRECT_REACH * spacing
    kernel = lambda a, b: np.exp(-(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) / reach) ** 2)
    near = kernel(at, at)
    moved = vertices.copy()
    close = np.where(np.linalg.norm(vertices - at.mean(axis=0), axis=1)
                     < np.linalg.norm(at - at.mean(axis=0), axis=1).max() + 3 * reach)[0]
    spread = kernel(vertices[close], at)
    for axis in range(3):
        # Ridge weighted by trust: an untrusted landmark's pull is mostly damped away.
        damping = CORRECT_DAMPING / np.maximum(trust[:, axis], 1e-3)
        alpha = np.linalg.solve(near + np.diag(damping), left[:, axis])
        moved[close, axis] += spread @ alpha
    return moved


# The jaw's outline from cheek to cheek round the chin (MediaPipe ids), fitted
# as the silhouette the camera sees (`fit_outline`). The points level with the
# ears are left out: there the silhouette is the ear.
JAW_OUTLINE = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
OUTLINE_ROUNDS = 3
# How wide each point's pull spreads (eye-corner spacings) and how it is
# damped, as in `correct`.
OUTLINE_REACH = 0.25
OUTLINE_DAMPING = 0.3
# Silhouettes are drawn at this many pixels per eye-corner spacing.
OUTLINE_SCALE = 300.0


def head_only(vertices: np.ndarray, faces: np.ndarray, weights: dict) -> np.ndarray:
    """The faces of the head itself, not the neck: those whose every vertex is
    bound mostly to bones other than the neck's and spine's (`weights`, the
    rig's {bone: [(vertex, weight)]}). Seen from the front, the jaw's lower
    edge is then a silhouette rather than a fold over the neck."""
    away = np.zeros(len(vertices))
    total = np.zeros(len(vertices))
    for bone, entries in weights.items():
        idx = np.array([v for v, _ in entries], dtype=np.int64)
        w = np.array([x for _, x in entries], dtype=np.float64)
        ok = idx < len(vertices)
        np.add.at(total, idx[ok], w[ok])
        if bone.startswith(("neck", "spine", "clavicle", "shoulder", "breast", "pectoral")):
            np.add.at(away, idx[ok], w[ok])
    head = (total > 0) & (away < 0.5 * total)
    return faces[head[faces].all(axis=1)]


def _silhouette(vertices: np.ndarray, faces: np.ndarray, rotation: np.ndarray, centre: np.ndarray,
                scale: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """The faces drawn as one filled shape, seen along -z after `rotation`, and
    every vertex's place in that picture (x right, y down)."""
    import cv2
    p = (vertices - centre) @ rotation.T
    xy = np.stack([p[:, 0] * scale + size / 2, -p[:, 1] * scale + size / 2], axis=1)
    mask = np.zeros((size, size), np.uint8)
    for tri in np.round(xy[faces]).astype(np.int32):
        cv2.fillConvexPoly(mask, tri, 1)
    return mask, xy


def _edge_along(mask: np.ndarray, targets: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per target, the silhouette's last pixel on the ray from `centre` through
    it, and how far short of the target that is (negative: beyond it)."""
    gaps, edges = [], []
    for t in targets:
        d = t - centre
        length = float(np.linalg.norm(d))
        r = np.arange(0.0, 2.0 * length, 0.5)
        pts = np.round(centre + r[:, None] * d / length).astype(int)
        ok = (pts >= 0).all(axis=1) & (pts[:, 0] < mask.shape[1]) & (pts[:, 1] < mask.shape[0])
        inside = np.zeros(len(r), bool)
        inside[ok] = mask[pts[ok, 1], pts[ok, 0]] > 0
        last = r[np.where(inside)[0].max()] if inside.any() else 0.0
        gaps.append(length - last)
        edges.append(centre + last * d / length)
    return np.array(gaps), np.array(edges)


def fit_outline(vertices: np.ndarray, head_faces: np.ndarray, person: np.ndarray,
                rotation: np.ndarray) -> np.ndarray:
    """The mesh's jaw moved until its silhouette, seen as the photo saw the
    head, runs through the photo's jaw outline.

    The landmark fit and `correct` move the vertices the outline landmarks were
    matched to, but those are not where the jaw's edge shows: corrected that
    way, the outline vertices reached the photo's (5.5 -> 3.0% of the eye
    spacing) and the outline read off the render did not follow. Here the head
    is turned by the photo's `rotation` (MediaPipe's, camera = upright @ R.T),
    drawn, and each jaw landmark (the photo's own outline, exactly where it was
    in the picture) is met along the ray from the nose tip: the vertex on the
    silhouette there is moved the gap in the picture's plane, spread like
    `correct`'s, and drawn again. Every landmark counts in full, beard or not:
    on the two bearded test faces MediaPipe's jaw runs along the jaw under
    the beard (the segmented face edge lies 0.10-0.19 face widths outside it),
    and fitted at the outline trust the rest of the fit uses (0.15 there), the
    jaw stayed 3.2-3.8% off where full trust took it to 3.1."""
    centre = person[:468].mean(axis=0)
    spacing = float(np.linalg.norm(person[33] - person[263]))
    scale = OUTLINE_SCALE / spacing
    size = int(4 * OUTLINE_SCALE)
    in_view = lambda q: np.stack([q[:, 0] * scale + size / 2, -q[:, 1] * scale + size / 2], axis=1)
    targets = in_view((person[JAW_OUTLINE] - centre) @ rotation.T)
    nose = in_view((person[[1]] - centre) @ rotation.T)[0]
    rays = (targets - nose) / np.linalg.norm(targets - nose, axis=1, keepdims=True)
    on_head = np.unique(head_faces)
    reach = OUTLINE_REACH * spacing
    kernel = lambda a, b: np.exp(-(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) / reach) ** 2)
    v = vertices.copy()
    for _ in range(OUTLINE_ROUNDS):
        mask, xy = _silhouette(v, head_faces, rotation, centre, scale, size)
        gaps, edges = _edge_along(mask, targets, nose)
        at = v[on_head[np.argmin(np.linalg.norm(xy[on_head][None] - edges[:, None], axis=2), axis=1)]]
        moved_px = gaps[:, None] * rays
        move = np.stack([moved_px[:, 0], -moved_px[:, 1], np.zeros(len(at))], axis=1) / scale @ rotation
        move[:, 2] = 0.0
        alpha = np.linalg.solve(kernel(at, at) + OUTLINE_DAMPING * np.eye(len(at)), move)
        near = np.where(np.linalg.norm(v - at.mean(axis=0), axis=1)
                        < np.linalg.norm(at - at.mean(axis=0), axis=1).max() + 3 * reach)[0]
        v[near] += kernel(v[near], at) @ alpha
    return v


# The lip targets that thin or thicken the lips, and how far they are set
# by the photo's lip height against the skull's (`lip_volume`).
LIP_TARGETS = ("mouth/mouth-upperlip-volume-{}", "mouth/mouth-lowerlip-volume-{}",
               "mouth/mouth-upperlip-height-{}", "mouth/mouth-lowerlip-height-{}")
LIP_GAIN = 2.5


def lip_volume(weights: np.ndarray, names: list[str], base: np.ndarray, corr: np.ndarray,
               person: np.ndarray) -> np.ndarray:
    """Lip fullness set by the photo's lip height against the skull's own.

    The landmarks see the lips' outline, not their bulk: fitted to a thin-
    lipped photo, the fit pulled the outline in with the lip height targets
    (and set the opposing pair against each other, 0.44 against 0.38) and
    left the lips as full as the skull's, which under the light read as a
    pouting mouth; on the render the lips were detected 11% of the eye
    spacing taller than the photo's. The height ratio (outer upper to outer
    lower lip, in the picture's plane) sets the volume and height targets
    outright, the opposing pair cleared.
    """
    ratio = (np.linalg.norm((person[0] - person[17])[:2])
             / max(np.linalg.norm((base[corr[0]] - base[corr[17]])[:2]), 1e-9))
    amount = float(np.clip(LIP_GAIN * abs(1.0 - ratio), 0.0, 1.0))
    more, less = ("incr", "decr") if ratio > 1.0 else ("decr", "incr")
    out = np.asarray(weights, dtype=np.float64).copy()
    index = {n: i for i, n in enumerate(names)}
    for pattern in LIP_TARGETS:
        if pattern.format(more) in index:
            out[index[pattern.format(more)]] = amount
        if pattern.format(less) in index:
            out[index[pattern.format(less)]] = 0.0
    return out


def explained(base: np.ndarray, corr: np.ndarray, person: np.ndarray,
              average: np.ndarray, weights: np.ndarray,
              expression: dict | None = None, front: np.ndarray | None = None) -> float:
    """How much of the face's difference from average the fit accounts for,
    identity and expression together, 0-1."""
    n = len(corr)
    want = person[:n] - reference(base, corr, average)
    got = apply(base, weights)[corr] - base[corr]
    if expression:
        got = got + expression_offsets(base, corr if front is None else front, expression)
    before = np.linalg.norm(want)
    after = np.linalg.norm(want - got)
    return float(1.0 - after / before) if before > 0 else 0.0


# Apple ARKit's 52 face blendshapes, in its own order. MediaPipe's
# FaceLandmarker scores the same names, and audio-to-expression models emit
# them, so an avatar carrying these is driven by either without a remap.
# "Left" is the subject's own left, which on this mesh is +x.
ARKIT_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]
# The symmetric mouth units, split at the midline into ARKit's sided shapes:
# (ARKit name, {unit: share}). A smile pulls the corners out as well as up:
# MakeHuman's corner puller alone widened the mouth 3-12% at full weight,
# where a broad smile in a photo is 20-33% wider than the face at rest, so it
# takes the retraction (the corners drawn outwards) too.
SPLIT_UNITS = [("mouthSmile", {"mouth-corner-puller": 1.0, "mouth-retraction": 1.0}),
               ("mouthFrown", {"mouth-depression": 1.0}),
               ("mouthDimple", {"mouth-retraction": 0.5}), ("mouthStretch", {"mouth-retraction": 1.0}),
               ("mouthPress", {"mouth-compression": 1.0}),
               ("mouthLowerDown", {"mouth-depression-retraction": 1.0}),
               ("mouthUpperUp", {"mouth-part-later": 1.0})]
# Half the width, in mesh units (~10 cm each), over which a split unit hands
# over from one side to the other.
MIDLINE = 0.08
# Full eye turns, degrees: up and down, towards and away from the nose.
EYE_LOOK = {"Up": 20.0, "Down": 25.0, "In": 30.0, "Out": 30.0}
# The lids follow the gaze: looking up opens them (share of the eye-wide
# unit), looking down lowers them (share of the blink). Without it, an eye
# turned up rolled its iris away under a lid that stayed put.
LID_FOLLOWS = {"Up": ("opened-up", 1.0), "Down": ("closure", 0.3)}
# Shapes MakeHuman has no unit for, in mesh units at full weight.
JAW_SHIFT = 0.1          # jaw sideways and forward
MOUTH_SHIFT = 0.1        # both lips sideways
LIP_ROLL = (0.02, 0.05)  # a lip rolled in: towards the other lip, and back
CHEEK_PUFF = (0.08, 0.06)  # along the skin's normal, and outwards
CHIN_RAISE = (0.0, 0.04, 0.015)  # lower lip and chin pushed up and forward
# MakeHuman's nostril raise is a twitch beside ARKit's sneer.
SNEER_GAIN = 2.0
CHEEK_SQUINT = 0.03      # cheek pushed up under the eye
TONGUE_OUT = (0.0, -0.04, 0.35)


def arkit_shapes(vertices: np.ndarray, corr: np.ndarray, groups: dict) -> np.ndarray:
    """The 52 ARKit blendshapes as (52, vertices, 3) offsets on the base mesh.

    Built from MakeHuman's CC0 expression units wherever one does the same
    movement: blink, squint, wide eyes, brows, sneer, jaw open, pucker, funnel
    as they are, and the symmetric mouth units split at the midline into the
    sided shapes (SPLIT_UNITS). The rest have no unit and are made here: eye
    looks as rotations of the eyeballs about their centres, jaw and mouth
    shifts carried by how much the jaw-open and pucker units move each vertex,
    lip rolls above and below the mouth's line, the chin raiser, cheek puff and
    squint round MediaPipe's cheek landmarks (`corr` maps them to vertices),
    and the tongue slid forward, its tip furthest.
    """
    n = len(vertices)
    unit = {}
    for name, idx, offsets in load_expressions():
        dense = np.zeros((n, 3))
        keep = idx < n
        dense[idx[keep]] = offsets[keep]
        unit[name] = dense

    def reach(name):
        """How much a unit moves each vertex, 0-1."""
        m = np.linalg.norm(unit[name], axis=1)
        return m / max(float(m.max()), 1e-12)

    def near(points, radius):
        d = np.linalg.norm(vertices[:, None, :] - np.atleast_2d(points)[None, :, :], axis=2).min(axis=1)
        return np.exp(-(d / radius) ** 2)

    smooth = lambda t: (lambda c: c * c * (3.0 - 2.0 * c))(np.clip(t, 0.0, 1.0))
    left = smooth((vertices[:, 0] + MIDLINE) / (2.0 * MIDLINE))[:, None]
    lips, jaw = reach("mouth-pursing"), reach("mouth-open")
    mouth_y = float((vertices[:, 1] * lips).sum() / lips.sum())
    below = smooth((mouth_y + 0.03 - vertices[:, 1]) / 0.06)[:, None]
    above = 1.0 - below

    shapes = {
        "jawOpen": unit["mouth-open"],
        "jawForward": jaw[:, None] * [0.0, 0.0, JAW_SHIFT],
        "jawLeft": jaw[:, None] * [JAW_SHIFT, 0.0, 0.0],
        "jawRight": jaw[:, None] * [-JAW_SHIFT, 0.0, 0.0],
        "mouthClose": -unit["mouth-open"] * lips[:, None] * below,
        "mouthFunnel": unit["mouth-protusion"],
        "mouthPucker": unit["mouth-pursing"],
        "mouthLeft": lips[:, None] * [MOUTH_SHIFT, 0.0, 0.0],
        "mouthRight": lips[:, None] * [-MOUTH_SHIFT, 0.0, 0.0],
        "mouthRollLower": lips[:, None] * below * [0.0, LIP_ROLL[0], -LIP_ROLL[1]],
        "mouthRollUpper": lips[:, None] * above * [0.0, -LIP_ROLL[0], -LIP_ROLL[1]],
        # MakeHuman's "elevation" raises the upper lip and leaves the chin:
        # the lower shrug (the chin raiser) is the jaw's own region pushed up.
        "mouthShrugLower": jaw[:, None] * below * CHIN_RAISE,
        "mouthShrugUpper": unit["mouth-elevation"],
        "browInnerUp": unit["eyebrows-left-inner-up"] + unit["eyebrows-right-inner-up"],
    }
    normals = trimesh.Trimesh(vertices, groups["body"], process=False).vertex_normals
    # Each cheek's middle: halfway from the mouth corner to the face's edge,
    # clear of the lower lid and the nostril, which a wider puff pushed about.
    cheeks = (vertices[corr[[61, 291]]] + vertices[corr[[234, 454]]]) / 2.0
    outwards = np.c_[np.sign(vertices[:, 0]), np.zeros((n, 2))]
    shapes["cheekPuff"] = near(cheeks, 0.2)[:, None] * (normals * CHEEK_PUFF[0] + outwards * CHEEK_PUFF[1])

    for side, share in (("Left", left), ("Right", 1.0 - left)):
        s = side.lower()
        shapes[f"eyeBlink{side}"] = unit[f"eye-{s}-closure"]
        shapes[f"eyeSquint{side}"] = unit[f"eye-{s}-slit"]
        shapes[f"eyeWide{side}"] = unit[f"eye-{s}-opened-up"]
        shapes[f"browDown{side}"] = unit[f"eyebrows-{s}-down"]
        shapes[f"browOuterUp{side}"] = unit[f"eyebrows-{s}-extern-up"]
        shapes[f"noseSneer{side}"] = unit[f"nose-{s}-elevation"] * SNEER_GAIN
        for arkit, mix in SPLIT_UNITS:
            shapes[arkit + side] = sum(unit[name] * k for name, k in mix.items()) * share
        # Under the eye on this side (MediaPipe 346 and 117, whichever is here).
        under = [i for i in (346, 117) if (vertices[corr[i], 0] > 0) == (side == "Left")]
        shapes[f"cheekSquint{side}"] = near(vertices[corr[under]], 0.2)[:, None] * [0.0, CHEEK_SQUINT, 0.01]

        ball = np.unique(groups[f"helper-{s[0]}-eye"])
        centre = vertices[ball].mean(axis=0)
        nose = np.array([-np.sign(centre[0]), 0.0, 0.0])
        for look, towards in (("Up", [0.0, 1.0, 0.0]), ("Down", [0.0, -1.0, 0.0]), ("In", nose), ("Out", -nose)):
            # Turning the front of the eye (+z) towards `towards`.
            axis = np.cross([0.0, 0.0, 1.0], towards)
            turn = trimesh.transformations.rotation_matrix(np.radians(EYE_LOOK[look]), axis)[:3, :3]
            offsets = np.zeros((n, 3))
            offsets[ball] = (vertices[ball] - centre) @ turn.T - (vertices[ball] - centre)
            if look in LID_FOLLOWS:
                lid, share = LID_FOLLOWS[look]
                offsets = offsets + share * unit[f"eye-{s}-{lid}"]
            shapes[f"eyeLook{look}{side}"] = offsets

    tongue = np.unique(groups["helper-tongue"])
    z = vertices[tongue, 2]
    tip = ((z - z.min()) / max(float(np.ptp(z)), 1e-9)) ** 1.5
    shapes["tongueOut"] = np.zeros((n, 3))
    shapes["tongueOut"][tongue] = tip[:, None] * TONGUE_OUT

    assert len(ARKIT_NAMES) == 52 and set(shapes) == set(ARKIT_NAMES)
    return np.stack([shapes[name] for name in ARKIT_NAMES]).astype(np.float32)


# MediaPipe's score for each blendshape on our own avatars at weights 0, 1/4,
# 1/2, 3/4 and 1: three test avatars rendered by the studio, averaged, made
# monotone. It depends on how the avatar is baked and lit: measure it again
# after changing either. MediaPipe's response is far from a straight line (a smile scores
# 0.65 at half weight and 0.81 at full), and even a neutral face scores some
# shapes (browOuterUpLeft 0.45). A photo's score is turned back through this
# into the weight that makes our avatar score the same.
ARKIT_RESPONSE = {
    "eyeSquintLeft": [0.15, 0.25, 0.32, 0.38, 0.41], "eyeSquintRight": [0.07, 0.16, 0.25, 0.32, 0.39],
    "mouthSmileLeft": [0.02, 0.37, 0.7, 0.82, 0.89], "mouthSmileRight": [0.02, 0.34, 0.68, 0.82, 0.9],
    "mouthLowerDownLeft": [0.0, 0.01, 0.09, 0.26, 0.48], "mouthLowerDownRight": [0.0, 0.02, 0.12, 0.33, 0.57],
    "mouthUpperUpLeft": [0.06, 0.38, 0.73, 0.75, 0.79], "mouthUpperUpRight": [0.08, 0.43, 0.79, 0.82, 0.85],
    "browOuterUpLeft": [0.38, 0.46, 0.59, 0.68, 0.75], "browOuterUpRight": [0.23, 0.29, 0.39, 0.49, 0.57],
    # jawOpen re-measured after the lips were sealed and the head squared
    # (photo-16, ref-woman, sk-21): the old row, [0, 0.01, 0.13, 0.27, 0.35],
    # turned a photo's barely parted lips (0.08) into weight 0.39, which
    # rendered as a mouth open on its teeth (0.45). Per avatar it ranges
    # from 0.33 to 0.85 at full weight.
    "jawOpen": [0.03, 0.19, 0.35, 0.55, 0.66], "mouthFunnel": [0.0, 0.02, 0.03, 0.06, 0.18],
    "mouthPucker": [0.12, 0.28, 0.57, 0.84, 0.95], "browInnerUp": [0.24, 0.28, 0.33, 0.41, 0.48],
}
# Shapes MediaPipe scores on photos but hardly at all on our avatars (frown,
# press, brow down, sneer ...): their score is the weight, less what a neutral
# avatar scores.
ARKIT_REST = {"mouthShrugLower": 0.04, "mouthShrugUpper": 0.14}
# Never taken from a photo: the eyes, which the shape fit already gives the
# photo's opening (gaze and blink readings were MediaPipe's own lean, and
# squint, whose response tops out at 0.51, moved the eye landmarks further
# from the photo on all three test faces), and the tongue, which MediaPipe
# does not score.
PHOTO_SKIPS = ("eyeLook", "eyeBlink", "eyeWide", "eyeSquint", "tongueOut")


def photo_expression(scores: dict) -> dict:
    """The blendshape weights that give the avatar the face the photo was
    making, from MediaPipe's scores for the photo (same 52 names)."""
    out = {}
    for name in ARKIT_NAMES:
        if name.startswith(PHOTO_SKIPS) or name not in scores:
            continue
        score = scores[name]
        if name in ARKIT_RESPONSE:
            curve = np.asarray(ARKIT_RESPONSE[name]) + np.arange(5) * 1e-6   # strictly rising
            weight = float(np.interp(score, curve, [0.0, 0.25, 0.5, 0.75, 1.0]))
        else:
            rest = ARKIT_REST.get(name, 0.0)
            weight = (score - rest) / (1.0 - rest)
        if weight >= SHOWN:
            out[name] = round(min(weight, 1.0), 3)
    return out
