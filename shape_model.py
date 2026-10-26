"""A learned correction to the face-shape fit, trained on our own synthetic heads.

morph.fit reads the face from MediaPipe's landmarks with a bounded linear
solve over MakeHuman's CC0 face targets. It leaves known, repeatable errors:
the outline sits a few percent of the eye spacing off and the nose comes out
long. synth.py renders heads whose target weights are known exactly, so the
fit can be run on them and what it gets wrong measured and learned.

The model predicts the residual: the target weights the fit should have found
minus the ones it did, from what the fit itself saw (the placed landmarks'
offsets from the skull's, the fitted weights, the head pose MediaPipe read,
the body chosen). It is a ridge regression; the weights it returns stay real
MakeHuman settings in [0, 1], so the rest of the pipeline (lips and lids
corrected free-form, rig, clothes) runs as before.

    python shape_model.py extract data/synth      # fit every sample, cache
    python shape_model.py train                   # fit the ridge, report test error
    python shape_model.py real uploads/a.png:male ...   # plausibility on real photos
    python shape_model.py jaw-bias data/synth     # MediaPipe's jaw against the true edge
    python shape_model.py                         # self-check (needs no data)

Trained only on synth.py's CC0 renders; numpy only, no pretrained weights.

ponytail: linear in the fit's own inputs. The synthetic shape prior is a guess,
so whatever the model learns about it (rather than about the fit) transfers to
real faces only as far as that prior resembles real faces; the `real` check
reports how far predictions stray from what the training set ever asked for.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

import morph

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "data" / "shape" / "fits.npz"
MODEL = HERE / "data" / "shape" / "model.npz"

# The jaw and cheek outline, ear to ear under the eyes, and the nose's length
# as REALISM-GAP.md measures it (bridge 168 to the point under the tip, 2).
OUTLINE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
           288, 361, 323, 454]
NOSE = (168, 2)
EYES = (33, 263)
# Lip and lid landmarks, which compose corrects free-form after the fit.
LIP_MARKS = (morph.INNER_UPPER + morph.INNER_LOWER + morph.OUTER_UPPER + morph.OUTER_LOWER
             + [61, 291, 78, 308])
# The face's skin: head vertices within this far of a landmark's (1 cm).
FACE_REACH = 0.1
# How strongly the best reachable weights stay at the fit's where the skin
# does not say otherwise (a ridge; the face's own columns weigh 1-100).
KEEP = 1e-3
# Held out for testing: every sample whose number ends in this digit.
TEST_DIGIT = 0
RIDGE = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000)


@functools.cache
def skull(gender: str):
    """What compose derives from the body alone: base mesh, the average face
    placed on it, landmark correspondences and the skin in front of them."""
    from catalogue import load_base
    from compose import place_face
    raw, groups = load_base()
    base = morph.body_for(raw, gender)
    average = place_face(morph.canonical_face(), base, groups).vertices
    corr = morph.correspondences(base, groups["body"], average[:468])
    front = morph.front_of(base, groups["body"], corr)
    return base, average, corr, front


def normalise(vertices: np.ndarray, gender: str) -> np.ndarray:
    """A true head put in the frame a photo of it is fitted in: scaled and
    moved so its eyes, nose, mouth and chin sit on this body's, as
    compose.place_face puts a photo's face. A photo shows neither the head's
    size nor where it sits on the neck, so neither counts as error."""
    from catalogue import load_base
    from facetexture import head_features
    _, groups = load_base()
    keys = ("left_eye", "right_eye", "nose", "mouth", "chin")
    have, want = head_features(vertices, groups), head_features(skull(gender)[0], groups)
    src, dst = np.array([have[k] for k in keys]), np.array([want[k] for k in keys])
    s_mean, d_mean = src[:, :2].mean(axis=0), dst[:, :2].mean(axis=0)
    scale = ((src[:, :2] - s_mean) * (dst[:, :2] - d_mean)).sum() / ((src[:, :2] - s_mean) ** 2).sum()
    shift = np.r_[d_mean - scale * s_mean, dst[2, 2] - scale * src[2, 2]]
    return scale * vertices + shift


@functools.cache
def face_basis(gender: str):
    """The face's skin vertices, and the face targets on them, reduced by QR
    so a bounded solve against them is quick."""
    import compose
    from catalogue import load_base, load_base_uv
    _, groups = load_base()
    _, uv_groups = load_base_uv()
    base, _, corr, _ = skull(gender)
    head = np.unique(groups["body"][compose.head_island(base, groups, groups["body"], uv_groups["body"])])
    skin = head[np.linalg.norm(base[head][:, None] - base[corr][None], axis=2).min(axis=1) < FACE_REACH]
    targets = morph.load_targets()
    row = np.full(len(base), -1)
    row[skin] = np.arange(len(skin))
    a = np.zeros((len(skin), 3, len(targets)))
    for k, (_, idx, delta) in enumerate(targets):
        keep = row[idx] >= 0
        a[row[idx[keep]], :, k] = delta[keep]
    q, r = np.linalg.qr(a.reshape(-1, len(targets)))
    return skin, q, r


def reachable(gender: str, truth: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """The target weights on this body that come closest, over the face's
    skin, to the true face `truth` (normalised): what the fit should have
    found, given the body it was handed. Targets the skin cannot tell apart
    (ears, the back of the head) keep the `fitted` weights."""
    from scipy.optimize import lsq_linear
    skin, q, r = face_basis(gender)
    b = q.T @ (truth[skin] - skull(gender)[0][skin]).reshape(-1)
    a = np.vstack([r, np.sqrt(KEEP) * np.eye(len(r))])
    return lsq_linear(a, np.r_[b, np.sqrt(KEEP) * fitted], bounds=(0.0, 1.0), method="bvls").x


def fit_photo(path: pathlib.Path, gender: str) -> dict:
    """The face-shape fit exactly as compose.compose runs it on one photo (no
    side views): returns everything the mesh and the model are built from.
    Keep in step with compose.compose."""
    import appearance
    from compose import SKIN_LANDMARKS, place_face
    from catalogue import load_base
    from generate import expression_of, fit_face

    _, groups = load_base()
    base, average, corr, front = skull(gender)
    face, landmarks, result, photo = fit_face(path)
    person = place_face(face, base, groups).vertices
    pw, ph = photo.size
    cheeks = np.array([[landmarks[i].x * pw, landmarks[i].y * ph]
                       for i in SKIN_LANDMARKS if i < len(landmarks)])
    scores = expression_of(result)
    trust, _ = appearance.outline_trust(np.asarray(photo.convert("RGB")), landmarks, cheeks, scores)
    weights, names, expression = morph.fit(base, corr, person, average, scores, front, trust)
    weights = morph.lip_volume(weights, names, base, corr, person)
    return {"gender": gender, "person": person[:468], "weights": weights,
            "expression": np.array([expression[u] for u in morph.EXPRESSION_UNITS]),
            "trust": np.asarray(trust, dtype=np.float64)[:468], "pose": head_pose(result)}


def head_pose(result) -> np.ndarray:
    """MediaPipe's reading of the head's turn: yaw, pitch, roll in degrees."""
    from scipy.spatial.transform import Rotation
    if not result.facial_transformation_matrixes:
        return np.zeros(3)
    u, _, vt = np.linalg.svd(np.asarray(result.facial_transformation_matrixes[0])[:3, :3])
    return Rotation.from_matrix(u @ vt).as_euler("yxz", degrees=True)


def mesh(fitted: dict, weights: np.ndarray, outline: bool = True) -> np.ndarray:
    """The skull with these weights, then lips and lids corrected free-form,
    the lips sealed and (`outline`) the jaw fitted as a silhouette, as compose
    builds it."""
    from scipy.spatial.transform import Rotation
    from appearance import EYE_CONTOURS
    from catalogue import load_base
    from rig import load_weights
    base, average, corr, front = skull(fitted["gender"])
    expression = dict(zip(morph.EXPRESSION_UNITS, fitted["expression"]))
    marks = np.zeros(len(corr))
    marks[LIP_MARKS] = 1.0
    marks[list(EYE_CONTOURS[0]) + list(EYE_CONTOURS[1])] = 1.0
    vertices = morph.correct(morph.apply(base, weights), base, corr,
                             morph.close_lips(fitted["person"]),
                             average, weights, expression, front, fitted["trust"] * marks)
    vertices = morph.seal_lips(vertices, load_base()[1]["body"])
    if outline:
        resting = fitted["person"] - morph.expression_offsets(base, front, expression)
        faces = load_base()[1]["body"]
        vertices = morph.fit_outline(vertices, morph.head_only(vertices, faces, load_weights()), resting,
                                     Rotation.from_euler("yxz", fitted["pose"], degrees=True).as_matrix())
    return vertices


def features(fitted: dict) -> np.ndarray:
    """What the fit saw and what it made of it, as one row."""
    base, average, corr, _ = skull(fitted["gender"])
    seen = fitted["person"] - morph.reference(base, corr, average)
    return np.concatenate([seen.reshape(-1), fitted["weights"], fitted["pose"] / 20.0,
                           [fitted["gender"] == "female"]])


def predict(fitted: dict, model: dict | None = None) -> np.ndarray:
    """Target weights for this face: the fit's, corrected, in [0, 1]. Needs
    `fitted` to hold gender, person (the placed landmarks), weights (after
    morph.lip_volume) and pose (head_pose)."""
    m = model or load()
    return np.clip(fitted["weights"] + residual(m, features(fitted)), 0.0, 1.0)


@functools.cache
def _loaded(path: str) -> dict:
    with np.load(path) as f:
        return dict(f)


def load(path: pathlib.Path = MODEL) -> dict:
    return _loaded(str(path))


# --- data ---

def _extract_one(json_path: str):
    from generate import NoFaceFound
    draw = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    gender = "female" if draw["female"] >= 0.5 else "male"
    try:
        fitted = fit_photo(pathlib.Path(json_path).with_suffix(".png"), gender)
    except NoFaceFound:
        return None
    from synth import vertices_for
    names = [n for n, _, _ in morph.load_targets()]
    truth = np.array([draw["targets"].get(n, 0.0) for n in names])
    best = reachable(gender, normalise(vertices_for(draw["female"], truth), gender), fitted["weights"])
    return int(draw["index"]), draw["female"], truth, best, fitted


def extract(folder: pathlib.Path, workers: int) -> None:
    """Fit every sample in `folder` and cache fits and truths in CACHE."""
    # One BLAS thread per worker: the fit's solves are small, and threads
    # fighting over the cores made each one six times slower.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    paths = sorted(str(p) for p in folder.glob("*.json"))
    rows = []
    with Pool(workers) as pool:
        for k, row in enumerate(pool.imap(_extract_one, paths, chunksize=8), 1):
            if row:
                rows.append(row)
            if k % 250 == 0:
                print(f"{k}/{len(paths)}", flush=True)
    index, female, truth, best, fits = zip(*rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, index=np.array(index), female=np.array(female), truth=np.array(truth),
                        best=np.array(best),
                        **{key: np.array([f[key] for f in fits]) for key in fits[0]})
    print(f"{len(rows)} fits of {len(paths)} samples (no face found in {len(paths) - len(rows)}) -> {CACHE}")


def fits(cache: pathlib.Path = CACHE) -> tuple[dict, list[dict]]:
    """The cache as (per-sample arrays, one fitted dict per sample)."""
    with np.load(cache) as f:
        data = dict(f)
    keys = ("gender", "person", "weights", "expression", "trust", "pose")
    return data, [dict(zip(keys, row)) for row in zip(*(data[k] for k in keys))]


# --- training and measuring ---

def ridge(x: np.ndarray, y: np.ndarray, lam: float) -> dict:
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = (x - mean) / scale
    coef = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y.mean(axis=0)))
    return {"mean": mean, "scale": scale, "coef": coef, "bias": y.mean(axis=0)}


def residual(m: dict, x: np.ndarray) -> np.ndarray:
    """The ridge's prediction for feature rows `x`."""
    return ((x - m["mean"]) / m["scale"]) @ m["coef"] + m["bias"]


def jaw_edge(fitted: dict, got: np.ndarray, want: np.ndarray, corr: np.ndarray) -> float:
    """How far the jaw's silhouette, as the photo saw the head, is from the
    true head's: mean gap along rays from the nose through the jaw outline
    landmarks, % of the eye spacing. The edge, not the vertices on it."""
    from scipy.spatial.transform import Rotation
    from catalogue import load_base
    from rig import load_weights
    rotation = Rotation.from_euler("yxz", fitted["pose"], degrees=True).as_matrix()
    faces = morph.head_only(want, load_base()[1]["body"], load_weights())
    centre = want[corr].mean(axis=0)
    scale = morph.OUTLINE_SCALE / np.linalg.norm(want[corr[EYES[0]]] - want[corr[EYES[1]]])
    size = int(4 * morph.OUTLINE_SCALE)
    true_mask, xy = morph._silhouette(want, faces, rotation, centre, scale, size)
    got_mask, _ = morph._silhouette(got, faces, rotation, centre, scale, size)
    nose, through = xy[corr[1]], xy[corr[morph.JAW_OUTLINE]]
    reach = lambda mask: _edge_along(mask, through, nose)
    return float(np.abs(reach(true_mask) - reach(got_mask)).mean() / morph.OUTLINE_SCALE * 100)


def _edge_along(mask: np.ndarray, through: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Per point, how far from `centre` the silhouette ends along the ray through it."""
    gaps, _ = morph._edge_along(mask, through, centre)
    return np.linalg.norm(through - centre, axis=1) - gaps


def errors(fitted: dict, truth_vertices: np.ndarray, weights: np.ndarray, corr: np.ndarray,
           outline: bool = True) -> dict:
    """How far the mesh with these weights is from the true one, in % of the
    true eye spacing, after lining the faces' landmark vertices up by their mean."""
    got = mesh(fitted, weights, outline)
    want = truth_vertices
    got = got - got[corr].mean(axis=0) + want[corr].mean(axis=0)
    spacing = np.linalg.norm(want[corr[EYES[0]]] - want[corr[EYES[1]]])
    off = (got[corr] - want[corr]) / spacing * 100
    nose = lambda v: np.linalg.norm(v[corr[NOSE[0]]] - v[corr[NOSE[1]]])
    return {"face": np.linalg.norm(off, axis=1).mean(),
            "face_xy": np.linalg.norm(off[:, :2], axis=1).mean(),
            "outline": np.linalg.norm(off[OUTLINE], axis=1).mean(),
            "outline_xy": np.linalg.norm(off[OUTLINE, :2], axis=1).mean(),
            "jaw_edge": jaw_edge(fitted, got, want, corr),
            "nose_long": (nose(got) - nose(want)) / spacing * 100}


def signed(weights: np.ndarray) -> np.ndarray:
    """Weights per MakeHuman slider, one end negative: the comparable form."""
    from synth import sliders
    return np.array([weights[..., pos] - (weights[..., neg] if neg is not None else 0.0)
                     for neg, pos in sliders()]).T


def report(rows: list[dict], label: str) -> str:
    keys = rows[0].keys()
    mean = {k: np.mean([r[k] for r in rows]) for k in keys}
    return (f"{label:<16} face {mean['face']:.2f} (xy {mean['face_xy']:.2f})  "
            f"outline {mean['outline']:.2f} (xy {mean['outline_xy']:.2f})  jaw edge {mean['jaw_edge']:.2f}  "
            f"nose long {mean['nose_long']:+.2f} (|{np.mean([abs(r['nose_long']) for r in rows]):.2f}|)")


def train(cache: pathlib.Path = CACHE, measure: int = 200) -> None:
    """Fit the ridge on the training samples, pick its strength on a validation
    tenth, and measure fit and fit + model on `measure` held-out samples."""
    from synth import vertices_for
    data, fitted = fits(cache)
    x = np.array([features(f) for f in fitted])
    y = data["best"] - data["weights"]
    test = data["index"] % 10 == TEST_DIGIT
    val = data["index"] % 10 == (TEST_DIGIT + 1) % 10
    fit_rows = ~test & ~val

    corrected = lambda m, rows: np.clip(data["weights"][rows] + residual(m, x[rows]), 0, 1)
    slider_error = lambda w, rows: np.abs(signed(w) - signed(data["best"][rows])).mean()
    lam = min(RIDGE, key=lambda l: slider_error(corrected(ridge(x[fit_rows], y[fit_rows], l), val), val))
    print(f"ridge {lam}; learning curve, validation slider error (fit alone "
          f"{slider_error(data['weights'][val], val):.4f}):")
    for n in (250, 500, 1000, 2000, 4000, 8000, 16000):
        if n > fit_rows.sum():
            break
        rows = np.where(fit_rows)[0][:n]
        print(f"  n={n:5d}  {slider_error(corrected(ridge(x[rows], y[rows], lam), val), val):.4f}")
    model = ridge(x[~test], y[~test], lam)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL, **model, ridge=lam)
    _loaded.cache_clear()
    print(f"{(~test).sum()} train, {test.sum()} test; slider error (mean |weight - best reachable|, "
          f"each MakeHuman slider signed): fit {slider_error(data['weights'][test], test):.4f}  "
          f"fit+model {slider_error(corrected(model, test), test):.4f}")

    rows = {f"{label}{tail}": [] for tail in ("", " -outl") for label in ("fit", "fit+model", "best")}
    clear = []
    for k in np.where(test)[0][:measure]:
        f = fitted[k]
        truth = normalise(vertices_for(float(data["female"][k]), data["truth"][k]), f["gender"])
        corr = skull(f["gender"])[2]
        for label, w in (("fit", f["weights"]), ("fit+model", predict(f, model)), ("best", data["best"][k])):
            rows[label].append(errors(f, truth, w, corr))
            rows[label + " -outl"].append(errors(f, truth, w, corr, outline=False))
        clear.append(abs(data["female"][k] - 0.5) > 0.3)
    print(f"vertex error on {len(clear)} held-out heads, % of eye spacing (best: the floor on this body; "
          f"-outl: without morph.fit_outline):")
    for label, r in rows.items():
        print(report(r, label))
    print(f"of which {sum(clear)} clearly male or female (blend under 0.2 or over 0.8):")
    for label, r in rows.items():
        print(report([e for e, c in zip(r, clear) if c], label))


def landmark_gap(fitted: dict, weights: np.ndarray) -> np.ndarray:
    """Per landmark, how far the mesh's landmark vertex is from the photo's
    landmark in the picture's plane, % of the eye spacing: the fit's own
    error measure, which needs no ground truth."""
    base, average, corr, _ = skull(fitted["gender"])
    want = fitted["person"] - morph.reference(base, corr, average)
    got = mesh(fitted, weights)[corr] - base[corr]
    return np.linalg.norm((want - got)[:, :2], axis=1) / np.linalg.norm(base[corr[EYES[0]]] - base[corr[EYES[1]]]) * 100


def change(fitted: dict, weights: np.ndarray) -> dict:
    """What the model does to one face, needing no ground truth."""
    corr = skull(fitted["gender"])[2]
    a, b = mesh(fitted, fitted["weights"])[corr], mesh(fitted, weights)[corr]
    spacing = np.linalg.norm(a[EYES[0]] - a[EYES[1]])
    nose = lambda v: np.linalg.norm(v[NOSE[0]] - v[NOSE[1]])
    shift = np.linalg.norm((b - b.mean(0)) - (a - a.mean(0)), axis=1) / spacing * 100
    before, after = landmark_gap(fitted, fitted["weights"]), landmark_gap(fitted, weights)
    return {"sliders": np.abs(signed(weights) - signed(fitted["weights"])).mean(),
            "face": shift.mean(), "outline": shift[OUTLINE].mean(),
            "nose": (nose(b) - nose(a)) / spacing * 100,
            "gap": before.mean(), "gap_after": after.mean(),
            "outline_gap": before[OUTLINE].mean(), "outline_gap_after": after[OUTLINE].mean()}


def real(photos: list[str], measure: int = 100) -> None:
    """On real photos (path:gender), what the model changes, beside the same
    numbers on held-out synthetic heads: a plausibility check."""
    data, fitted = fits()
    show = lambda c: (f"|slider change| {c['sliders']:.3f}  moves face {c['face']:.2f}, outline {c['outline']:.2f}, "
                      f"nose length {c['nose']:+.2f}  landmark gap {c['gap']:.2f} -> {c['gap_after']:.2f}, "
                      f"outline {c['outline_gap']:.2f} -> {c['outline_gap_after']:.2f}")
    synth = [change(fitted[k], predict(fitted[k])) for k in np.where(data["index"] % 10 == TEST_DIGIT)[0][:measure]]
    print(f"{'synthetic (mean)':<28} " + show({k: np.mean([c[k] for c in synth]) for k in synth[0]}))
    for spec in photos:
        path, _, gender = spec.rpartition(":")
        f = fit_photo(pathlib.Path(path), gender)
        print(f"{pathlib.Path(path).name:<28} " + show(change(f, predict(f))))


def jaw_bias(folder: pathlib.Path, count: int = 100) -> None:
    """Where MediaPipe's jaw outline lies on held-out renders, against the true
    head's silhouette drawn through the render's own perspective camera:
    signed gap along rays from the nose, % of the eye spacing (positive:
    MediaPipe's outline outside the true edge). morph.fit_outline fits the
    jaw's silhouette to that outline."""
    import cv2
    import mediapipe as mp
    from catalogue import load_base
    from generate import get_landmarker
    from rig import load_weights
    from synth import vertices_for
    names = [n for n, _, _ in morph.load_targets()]
    body = load_base()[1]["body"]
    rows = []
    for path in sorted(folder.glob(f"*{TEST_DIGIT}.json"))[:count]:
        draw = json.loads(path.read_text(encoding="utf-8"))
        found = get_landmarker().detect(mp.Image.create_from_file(str(path.with_suffix(".png"))))
        if not found.face_landmarks:
            continue
        size, cam = draw["size"], draw["camera"]
        marks = np.array([[p.x * size, p.y * size] for p in found.face_landmarks[0]])
        v = vertices_for(draw["female"], np.array([draw["targets"].get(n, 0.0) for n in names]))
        eye, at = np.array(cam["position"]), np.array(cam["look_at"])
        back = (eye - at) / np.linalg.norm(eye - at)
        right = np.cross(cam["up"], back)
        right /= np.linalg.norm(right)
        p = (v - eye) @ np.stack([right, np.cross(back, right), back]).T
        focal = size / 2 / np.tan(np.radians(cam["fov"]) / 2)
        px = 4 * np.stack([size / 2 - focal * p[:, 0] / p[:, 2], size / 2 + focal * p[:, 1] / p[:, 2]], axis=1)
        mask = np.zeros((4 * size, 4 * size), np.uint8)       # drawn 4x over
        for tri in np.round(px[morph.head_only(v, body, load_weights())]).astype(np.int32):
            cv2.fillConvexPoly(mask, tri, 1)
        gaps, _ = morph._edge_along(mask, 4 * marks[morph.JAW_OUTLINE], 4 * marks[1])
        rows.append(-gaps / (4 * np.linalg.norm(marks[33] - marks[263])) * 100)
    rows = np.array(rows)
    print(f"{len(rows)} renders: MediaPipe's jaw outline against the true silhouette, % of the eye spacing: "
          f"mean {rows.mean():+.2f}, mean |gap| {np.abs(rows).mean():.2f}")
    print("per landmark, chin in the middle: " + " ".join(f"{m:+.1f}" for m in rows.mean(axis=0)))


def demo() -> None:
    """Self-check, no data needed: ridge recovers a linear residual, the
    slider form folds opposite ends together, a body is already in its own
    frame, and the best reachable weights of a head built from targets are
    those targets."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = x @ rng.normal(size=(5, 3)) + 2.0
    m = ridge(x, y, 1e-6)
    assert np.allclose(residual(m, x), y, atol=1e-4)
    from synth import sliders
    neg, pos = next((n, p) for n, p in sliders() if n is not None)
    w = np.zeros(len(morph.load_targets()))
    w[neg] = 0.3
    assert signed(w[None])[0][[p for _, p in sliders()].index(pos)] == -0.3
    base = skull("male")[0]
    assert np.allclose(normalise(base, "male"), base)
    names = [n for n, _, _ in morph.load_targets()]
    w = np.zeros(len(names))
    w[names.index("nose/nose-scale-vert-incr")] = 0.5
    w[names.index("chin/chin-width-decr")] = 0.4
    assert np.abs(reachable("male", morph.apply(base, w), np.zeros_like(w)) - w).max() < 0.02
    print("shape_model self-check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd")
    e = sub.add_parser("extract")
    e.add_argument("folder", type=pathlib.Path)
    e.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--measure", type=int, default=200, help="held-out heads to build meshes for")
    j = sub.add_parser("jaw-bias")
    j.add_argument("folder", type=pathlib.Path)
    j.add_argument("-n", type=int, default=100)
    r = sub.add_parser("real")
    r.add_argument("photos", nargs="+", help="photo:male or photo:female")
    args = parser.parse_args()
    if args.cmd == "extract":
        extract(args.folder, args.workers)
    elif args.cmd == "train":
        train(measure=args.measure)
    elif args.cmd == "jaw-bias":
        jaw_bias(args.folder, args.n)
    elif args.cmd == "real":
        real(args.photos)
    else:
        demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
