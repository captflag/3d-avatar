"""Reading colour from a photo whose own colour cannot be trusted.

Every photo carries a colour cast: warm indoor light, a cool overcast sky, a
camera's guess. Read straight, a cast moves everything: blue eyes in a warm
photo measure grey (LAB b* -1 against a neutral 0), and skin comes out too pink
or too yellow. The face carries its own reference for white — the whites of the
eyes — so the cast can be measured on the person and taken out.

The sclera is not paper-white: healthy eyes read a little warm, and a small,
soft photo bleeds lid and lash into it. So the cast is measured against a
slightly warm "natural white", only the brightest sclera pixels count, and it
is applied in part and capped.

ponytail: the sclera is a few hundred pixels in a typical selfie; a photo with
the eyes nearly shut gives no reference and nothing is corrected.
"""

from __future__ import annotations

import pathlib
import threading

import cv2
import numpy as np

from delight import EYE_LEFT, EYE_RIGHT

HERE = pathlib.Path(__file__).resolve().parent
# MediaPipe's multiclass selfie segmenter, Apache-2.0 per its model card:
# background, hair, body skin, face skin, clothes, accessories.
SEGMENTER_TASK = HERE / "assets" / "selfie_multiclass_256x256.tflite"
HAIR, BODY_SKIN, FACE_SKIN, CLOTHES = 1, 2, 3, 4

_segmenter = None
_segmenter_lock = threading.Lock()

# What a healthy sclera measures, as LAB a*, b* offsets from neutral.
NATURAL_WHITE = np.array([1.0, 4.0])
# The largest cast we believe, in LAB units; beyond this the sclera reading is
# more likely lid and lash than lighting.
MAX_CAST = 10.0

# Natural iris colours, sRGB. Eye colour in a selfie is a few soft pixels, so
# the measurement is snapped halfway to the nearest natural colour: enough to
# read as blue, green or brown, while keeping the person's own shade.
IRIS_PALETTE = {
    "dark brown": (58, 36, 24), "brown": (98, 60, 34), "hazel": (126, 98, 56),
    "green": (96, 116, 70), "grey": (120, 126, 128), "blue-grey": (96, 118, 138),
    "blue": (72, 112, 158),
}


def _lab(pixels: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(pixels.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


def _rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB).reshape(3)


def _eye_masks(landmarks, size):
    w, h = size
    eye = np.zeros((h, w), np.uint8)
    for ring in (EYE_LEFT, EYE_RIGHT):
        cv2.fillPoly(eye, [np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ring], np.int32)], 1)
    iris = np.zeros((h, w), np.uint8)
    for centre, ring in ((468, range(469, 473)), (473, range(474, 478))):
        cx, cy = landmarks[centre].x * w, landmarks[centre].y * h
        r = np.mean([np.hypot(landmarks[i].x * w - cx, landmarks[i].y * h - cy) for i in ring])
        cv2.circle(iris, (int(cx), int(cy)), max(int(r * 1.05), 1), 1, -1)
    return eye.astype(bool), iris.astype(bool)


def colour_cast(photo_rgb: np.ndarray, landmarks) -> np.ndarray:
    """The photo's colour cast as LAB (a*, b*), measured on the whites of the
    eyes; zero when they cannot be measured."""
    if len(landmarks) <= 477:
        return np.zeros(2)
    h, w = photo_rgb.shape[:2]
    eye, iris = _eye_masks(landmarks, (w, h))
    sclera = _lab(photo_rgb[eye & ~iris])
    if len(sclera) < 40:
        return np.zeros(2)
    bright = sclera[sclera[:, 0] >= np.percentile(sclera[:, 0], 60)]
    cast = np.median(bright[:, 1:], axis=0) - 128.0 - NATURAL_WHITE
    return np.clip(cast, -MAX_CAST, MAX_CAST)


def white_balance(rgb: np.ndarray, cast: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """`rgb` with `strength` of the measured cast taken out of every pixel."""
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[..., 1:] -= strength * cast
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def iris_colour(photo_rgb: np.ndarray, landmarks, cast: np.ndarray | None = None,
                strength: float = 0.7, default=(70, 48, 32)):
    """Iris colour with the photo's cast taken out, snapped halfway to the
    nearest natural eye colour. Returns (rgb, palette name)."""
    if len(landmarks) <= 477:
        return np.array(default, dtype=np.float64), None
    h, w = photo_rgb.shape[:2]
    eye, iris = _eye_masks(landmarks, (w, h))
    pixels = _lab(photo_rgb[eye & iris])
    if len(pixels) < 10:
        return np.array(default, dtype=np.float64), None
    lo, hi = np.percentile(pixels[:, 0], [20, 90])            # no pupil, no catchlight
    measured = np.median(pixels[(pixels[:, 0] >= lo) & (pixels[:, 0] <= hi)], axis=0)
    if cast is None:
        cast = colour_cast(photo_rgb, landmarks)
    measured[1:] -= strength * cast

    names = list(IRIS_PALETTE)
    palette = _lab(np.array([IRIS_PALETTE[n] for n in names]))
    # Hue decides the family; lightness only separates dark from light shades.
    distance = np.linalg.norm((palette - measured) * np.array([0.5, 1.0, 1.0]), axis=1)
    nearest = int(distance.argmin())
    snapped = (measured + palette[nearest]) / 2.0
    return _rgb(snapped).astype(np.float64), names[nearest]


# ------------------------------------------------------------------ hair ------

def segment(photo_rgb: np.ndarray) -> np.ndarray | None:
    """Per-pixel part labels (HAIR, FACE_SKIN, ...) for the photo, or None
    when the segmenter model is not installed."""
    global _segmenter
    if not SEGMENTER_TASK.exists():
        return None
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options

    with _segmenter_lock:
        if _segmenter is None:
            _segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(SEGMENTER_TASK)),
                output_category_mask=True, output_confidence_masks=False))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(photo_rgb[..., :3]))
        mask = _segmenter.segment(image).category_mask.numpy_view()
    return np.array(mask).reshape(photo_rgb.shape[:2])


def read_hair(photo_rgb: np.ndarray, landmarks, parts: np.ndarray | None) -> dict | None:
    """What the photo says about the hair, measured in face heights.

    `top`: how far the hair rises above the top of the face mesh.
    `bottom`: where it ends relative to the chin (negative is above the chin).
    `fringe`: share of the forehead it covers. `amount`: hair area against
    face-skin area, near zero when bald. `colour`: its base colour, read from
    the hair pixels themselves, below the highlights and above the shadow.
    """
    if parts is None or len(landmarks) <= 454:
        return None
    h, w = parts.shape
    hair = parts == HAIR
    top, chin = landmarks[10], landmarks[152]
    face_h = max(abs(chin.y - top.y) * h, 1.0)
    face_w = max(abs(landmarks[454].x - landmarks[234].x) * w, 1.0)
    cx, ty, cy = top.x * w, top.y * h, chin.y * h

    near = np.zeros_like(hair)
    near[:, max(int(cx - 0.9 * face_w), 0):int(cx + 0.9 * face_w)] = True
    ys, xs = np.nonzero(hair & near)
    amount = float((hair & near).sum() / max((parts == FACE_SKIN).sum(), 1))
    reading = {"amount": round(amount, 2), "top": None, "bottom": None, "fringe": 0.0, "colour": None}
    if len(ys) < 50:
        return reading

    brow_y = (landmarks[66].y + landmarks[296].y) / 2 * h
    span = brow_y - ty
    forehead = hair[max(int(ty + 0.1 * span), 0):max(int(brow_y - 0.1 * span), 1),
                    max(int(cx - 0.25 * face_w), 0):int(cx + 0.25 * face_w)]
    core = cv2.erode(hair.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & near
    pixels = photo_rgb[core] if core.sum() >= 50 else photo_rgb[hair & near]
    lab = cv2.cvtColor(pixels.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    lo, hi = np.percentile(lab[:, 0], [20, 60])
    base = np.median(lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)], axis=0)

    reading.update(top=round(float((ty - ys.min()) / face_h), 2),
                   bottom=round(float((ys.max() - cy) / face_h), 2),
                   fringe=round(float(forehead.mean()) if forehead.size else 0.0, 2),
                   colour=_rgb(base))
    return reading


def hair_style(reading: dict | None, default: str) -> str | None:
    """The catalogue haircut closest to what the photo shows; None for bald.

    Thresholds measured on the test photos (hair bottom against the chin, in
    face heights): short hair ends 0.3 to 0.65 above the chin. Hair reaching the
    jaw reads as a bob, below the chin as long. A close crop rises about a third
    of a face height above the face mesh; fuller short hair about half.

    ponytail: four measurements and six haircuts. The long-hair branch has not
    yet been checked against a real long-haired photo.
    """
    if reading is None or reading["top"] is None:
        return default if reading is None else None
    if reading["amount"] < 0.08:
        return None                                   # bald or shaved
    fringe = reading["fringe"] > 0.35
    if reading["bottom"] > 0.15:
        return "hair_long01"
    if reading["bottom"] > -0.15:
        return "hair_bob02" if fringe else "hair_bob01"
    if fringe:
        return "hair_short03"
    return "hair_short01" if reading["top"] < 0.4 else "hair_short02"
