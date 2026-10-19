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
    `fringe`: share of the forehead it covers. `sides`: how far it stands out
    past the face's outline at the temples, in face widths (median over the
    rows from the top of the face to the eyes). `amount`: hair area against
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
    reading = {"amount": round(amount, 2), "top": None, "bottom": None, "fringe": 0.0, "sides": None,
               "colour": None}
    if len(ys) < 50:
        return reading

    brow_y = (landmarks[66].y + landmarks[296].y) / 2 * h
    span = brow_y - ty
    forehead = hair[max(int(ty + 0.1 * span), 0):max(int(brow_y - 0.1 * span), 1),
                    max(int(cx - 0.25 * face_w), 0):int(cx + 0.25 * face_w)]
    core = cv2.erode(hair.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & near
    pixels = photo_rgb[core] if core.sum() >= 50 else photo_rgb[hair & near]
    lab = cv2.cvtColor(pixels.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    # The body of the hair, short of its brightest strands and of the scalp
    # showing through, which pull a plain median towards grey. The darkest
    # band (10-40%) read the shadows: rendered hair came out 7 L* darker than
    # the photo's, and a strawberry blonde, whose shadows are redder, copper.
    lo, hi = np.percentile(lab[:, 0], [25, 60])
    base = np.median(lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)], axis=0)

    outline = sorted((landmarks[234].x * w, landmarks[454].x * w))
    eye_y = (landmarks[159].y + landmarks[386].y) / 2 * h
    overhang = []
    for row in range(max(int(ty), 0), min(int(eye_y), h)):
        xs_row = np.nonzero(hair[row])[0]
        if len(xs_row):
            overhang.append(max(outline[0] - xs_row.min(), xs_row.max() - outline[1], 0.0) / face_w)

    reading.update(top=round(float((ty - ys.min()) / face_h), 2),
                   bottom=round(float((ys.max() - cy) / face_h), 2),
                   fringe=round(float(forehead.mean()) if forehead.size else 0.0, 2),
                   sides=round(float(np.median(overhang)), 2) if overhang else None,
                   colour=_rgb(base))
    return reading


def hair_style(reading: dict | None, default: str) -> str | None:
    """The catalogue haircut closest to what the photo shows; None for bald.

    Thresholds measured on the test photos (hair bottom against the chin, in
    face heights): short hair ends 0.3 to 0.65 above the chin. Hair reaching the
    jaw reads as a bob, below the chin as long. A close crop rises about a third
    of a face height above the face mesh; fuller short hair about half.

    ponytail: five measurements and seven haircuts. The long-hair branch has not
    yet been checked against a real long-haired photo, and `sides` takes the
    wider side, so a strongly turned head may read as fuller at the sides.
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
    if reading["top"] < 0.4:
        return "hair_short01"
    # Fuller on top: clipped close at the sides (0.08 face widths past the
    # face on photo-10) or full all round (0.13 to 0.16 on the references).
    if reading.get("sides") is not None and reading["sides"] < 0.11:
        return "hair_short_sides"
    return "hair_short02"


# ----------------------------------------------------------------- mouth ------

# Lips against the person's own cheek, in OpenCV LAB: darker and a little
# redder. Measured on the four test photos: L -7 to -30, a* +5 to +13, b* 0 to
# -11.
LIP_FROM_CHEEK = np.array([-15.0, 9.0, -4.0])


# MediaPipe's eye contours, each a closed loop round the opening.
EYE_CONTOURS = ([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
                [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466])


def fill_eyes(skin_rgb: np.ndarray, landmarks) -> np.ndarray:
    """The photo with each eye opening, lash line included, painted over with
    the lid skin round it.

    The eyeballs are generated, so nothing of the photo's eye belongs on skin;
    but the lids' rims sit inside the opening and the bake gave them the
    photo's white and iris. Closed by a blink, the lid stretched that across
    the eye as a grey smear, and MediaPipe read the eye as open.
    """
    if len(landmarks) <= max(max(c) for c in EYE_CONTOURS):
        return skin_rgb
    h, w = skin_rgb.shape[:2]
    opening = np.zeros((h, w), np.uint8)
    for contour in EYE_CONTOURS:
        poly = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in contour], np.int32)
        grow = max(int(0.12 * np.ptp(poly[:, 0])), 1)     # past the lashes, to lid skin
        eye = np.zeros((h, w), np.uint8)
        cv2.fillPoly(eye, [poly], 255)
        opening |= cv2.dilate(eye, np.ones((2 * grow + 1,) * 2, np.uint8))
    return cv2.inpaint(np.ascontiguousarray(skin_rgb), opening, 5, cv2.INPAINT_TELEA)


def fill_mouth(skin_rgb: np.ndarray, landmarks, cheek_pixels: np.ndarray | None = None
               ) -> tuple[np.ndarray, bool]:
    """The photo with an open mouth painted over in lip colour.

    The avatar is given a closed, resting mouth, but the photo it is painted
    from may be grinning: the bake fills the gap between the lips from
    whatever the photo shows there, so a closed mouth came out with teeth
    painted across it. The opening between the inner lip lines is filled from
    the lips around it (OpenCV inpainting), then pulled to a lip colour. A
    closed mouth in the photo is left alone.

    Given the cheeks, the lip colour is the person's own skin made lip-like
    (LIP_FROM_CHEEK), and the lips themselves are pulled halfway to it: a laugh
    stretches them into thin wet bands of inner lip and gum pink, and the
    resting mouth spread that over full lips, swollen and pink.
    """
    from delight import INNER_LIPS

    if len(landmarks) <= max(INNER_LIPS):
        return skin_rgb, False
    h, w = skin_rgb.shape[:2]
    poly = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in INNER_LIPS], np.int32)
    opening = np.zeros((h, w), np.uint8)
    cv2.fillPoly(opening, [poly], 255)
    mouth_w = max(np.ptp(poly[:, 0]), 1)
    if opening.sum() / 255 < 0.02 * mouth_w * mouth_w:       # lips already together
        return skin_rgb, False

    grow = max(int(0.04 * mouth_w), 1)
    opening = cv2.dilate(opening, np.ones((2 * grow + 1,) * 2, np.uint8))
    ring = cv2.dilate(opening, np.ones((4 * grow + 1,) * 2, np.uint8)) & ~opening
    lips = skin_rgb[ring > 0]
    filled = cv2.inpaint(np.ascontiguousarray(skin_rgb), opening, max(3, 2 * grow), cv2.INPAINT_TELEA)
    if len(lips):
        lab = cv2.cvtColor(filled, cv2.COLOR_RGB2LAB).astype(np.float64)
        if cheek_pixels is not None and len(cheek_pixels):
            xs = np.clip(cheek_pixels[:, 0].astype(int), 0, w - 1)
            ys = np.clip(cheek_pixels[:, 1].astype(int), 0, h - 1)
            target = np.median(lab[ys, xs], axis=0) + LIP_FROM_CHEEK
            outer = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in OUTER_LIPS], np.int32)
            band = np.zeros((h, w), np.uint8)
            cv2.fillPoly(band, [outer], 1)
            # Lips only, not the moustache over them: nothing far darker.
            band = (band > 0) & (opening == 0) & (lab[..., 0] > target[0] - 40.0)
            lab[band, 1:] = 0.5 * lab[band, 1:] + 0.5 * target[1:]
            lab[band, 0] = 0.7 * lab[band, 0] + 0.3 * target[0]
        else:
            lip_lab = _lab(lips)
            # Lips are the reddest thing round the mouth: on a small photo the
            # ring is mostly moustache and beard, and a plain median turned the
            # fill dark brown, which read as an open mouth. The darker half of
            # the red pixels, as the ring also catches the edges of bright teeth.
            red = lip_lab[lip_lab[:, 1] >= np.percentile(lip_lab[:, 1], 60)]
            lo, hi = np.percentile(red[:, 0], [10, 60])
            target = np.median(red[(red[:, 0] >= lo) & (red[:, 0] <= hi)], axis=0)
        inside = opening > 0
        lab[inside] = 0.2 * lab[inside] + 0.8 * target
        # The closed lips' seam samples the middle of this fill, and a closed
        # mouth has a dark line there; plain lip colour read as thick lips.
        depth = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        lab[..., 0] *= 1.0 - 0.45 * depth / max(float(depth.max()), 1.0)
        filled = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return filled, True


# ------------------------------------------------------------ colour cast -----

# Hue angle of skin in LAB (atan2(b*, a*), degrees): every skin tone falls in a
# narrow band between red and yellow. Measured on the test photos: 53 to 60.
SKIN_HUE = (40.0, 70.0)


def correct_cast(skin_rgb: np.ndarray, face_mask: np.ndarray):
    """Pull a face whose skin hue is impossible back into the natural range.

    The whites of the eyes were tried as a white reference and failed: in a
    small photo they pick up pink from the lids, and correcting by them turned
    light skin sallow. Skin itself is the safer reference, used only to catch
    what is clearly wrong: a green fluorescent cast, a blue one from shade. A
    face inside the band is left untouched; one outside is rotated to the
    nearest edge of it, chroma kept, and the whole image moved by that shift.
    Returns the image and the (a*, b*) shift applied.
    """
    if face_mask is None or face_mask.sum() < 200:
        return skin_rgb, (0.0, 0.0)
    lab = cv2.cvtColor(skin_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    face = lab[face_mask]
    lo, hi = np.percentile(face[:, 0], [30, 80])
    a, b = np.median(face[(face[:, 0] >= lo) & (face[:, 0] <= hi)][:, 1:], axis=0) - 128.0
    chroma = float(np.hypot(a, b))
    hue = float(np.degrees(np.arctan2(b, a)))
    if chroma < 4.0 or SKIN_HUE[0] <= hue <= SKIN_HUE[1]:
        return skin_rgb, (0.0, 0.0)            # natural, or too grey to judge
    target = np.radians(min(max(hue, SKIN_HUE[0]), SKIN_HUE[1]))
    # A blue cast also drains saturation; every skin measured is above 20.
    chroma = max(chroma, 14.0)
    shift = np.array([chroma * np.cos(target) - a, chroma * np.sin(target) - b])
    lab[..., 1:] += shift
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out, (round(float(shift[0]), 1), round(float(shift[1]), 1))


# -------------------------------------------------------------- hairline ------

def opaque_hair(mesh, per_face: int = 12) -> np.ndarray:
    """Points on a haircut's cards where its texture is opaque.

    Cards are alpha-cut, and their transparent tips hang well below where the
    hair visibly ends: judged by the geometry, a cut covered a band of forehead
    that shows bare in every render, and the hairline tint greyed it.
    """
    uv, texture = mesh.metadata.get("uv"), mesh.metadata.get("texture")
    if uv is None or texture is None or texture.mode not in ("RGBA", "P"):
        return np.asarray(mesh.vertices)
    alpha = np.asarray(texture.convert("RGBA"))[..., 3]
    h, w = alpha.shape
    weights = np.random.default_rng(0).dirichlet(np.ones(3), per_face)
    points = np.einsum("sk,fkd->fsd", weights, np.asarray(mesh.vertices)[mesh.faces]).reshape(-1, 3)
    uvs = np.einsum("sk,fkd->fsd", weights, uv[mesh.faces]).reshape(-1, 2)
    seen = alpha[np.clip(((1.0 - uvs[:, 1]) * (h - 1)).astype(int), 0, h - 1),
                 np.clip((uvs[:, 0] * (w - 1)).astype(int), 0, w - 1)]
    return points[seen > 127] if (seen > 127).sum() > 100 else np.asarray(mesh.vertices)


# How far past the hair's own silhouette, seen from the front (mesh units),
# the skin still counts as under the hair for the hairline tint. At 0.06 the
# tint ran a band of strands down the photo's own forehead below the hair's
# visible edge (a quarter to a third of the strip under it painted dark); at
# 0.015 that is a tenth, and no skin shows between the cards.
HAIR_COVER_REACH = 0.015


def hairline_tint(texture, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray,
                  uv_faces: np.ndarray, hair_points: np.ndarray, hair_rgb,
                  fade: float = 0.32, strength: float = 0.85, spare: np.ndarray | None = None):
    """Paint a soft hairline onto the skin under and just below the hair.

    Hair cards end in a row on the forehead, so the hair met the skin in a
    hard, ragged line: nothing in the texture said hair grows there. The hair's
    lower edge is found at each angle round the head (the lowest hair in each
    5-degree slice), and skin above that line is tinted towards a darker shade
    of the hair colour, fading to nothing `fade` below it (mesh units, ~10 cm
    each) — the fine hairs and root shadow along a real hairline.

    Distance to the nearest hair vertex was tried first and missed: the cards
    stand off the forehead, so the bare skin just below their edge is 0.4 to
    0.5 away from any of them, outside any band narrow enough to look right.

    `spare` (per vertex, 0-1) is how much of the photo each point already
    carries. Below the hair's edge, where the photo shows the forehead up to
    its own hairline, the tint is left off: mixed over tan skin, a dark hair
    colour only greyed it.
    """
    from PIL import Image
    from facetexture import rasterize

    used = np.unique(faces)
    centre = vertices[used][:, [0, 2]].mean(axis=0)

    def angle(points):
        return np.degrees(np.arctan2(points[:, 0] - centre[0], points[:, 2] - centre[1])) % 360.0

    bins = 72
    slot = (angle(hair_points) / (360.0 / bins)).astype(int) % bins
    edge = np.full(bins, np.nan)
    for b in range(bins):
        ys = hair_points[slot == b, 1]
        if len(ys) >= 3:
            edge[b] = np.percentile(ys, 3)
    if np.isnan(edge).all():
        return texture
    known = ~np.isnan(edge)                       # fill empty slices round the circle
    edge = np.interp(np.arange(bins), np.arange(bins)[known], edge[known], period=bins)
    edge = np.convolve(np.r_[edge[-2:], edge, edge[:2]], np.ones(5) / 5, mode="valid")

    line = np.interp(angle(vertices) / (360.0 / bins), np.arange(bins), edge, period=bins)
    t = np.clip(1.0 - (line - vertices[:, 1]) / fade, 0.0, 1.0)
    # Strong under the cards, so gaps between strands show hair colour rather
    # than bare skin, and falling off fastest near the edge's foot.
    tint = t ** 1.6 * strength
    if spare is not None:
        # Only where no hair lies in front, as the camera sees it: under the
        # cards the tint is what fills the gaps between strands, photo skin or
        # not. The line above cannot tell: the lowest hair at an angle can hang
        # below where the cut visibly ends. The hair's vertices are splatted
        # into a front-view depth map, wide enough to close gaps between cards.
        cell, reach = 0.01, HAIR_COVER_REACH
        lo = hair_points[:, :2].min(axis=0) - reach
        shape = np.ceil((hair_points[:, :2].max(axis=0) + reach - lo) / cell).astype(int) + 1
        front = np.full((shape[1], shape[0]), -1e9, np.float32)
        hx, hy = ((hair_points[:, :2] - lo) / cell).astype(int).T
        np.maximum.at(front, (hy, hx), hair_points[:, 2].astype(np.float32))
        r = int(round(reach / cell))
        front = cv2.dilate(front, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        vx, vy = ((vertices[:, :2] - lo) / cell).astype(int).T
        inside = (vx >= 0) & (vx < shape[0]) & (vy >= 0) & (vy < shape[1])
        covered = np.zeros(len(vertices), bool)
        covered[inside] = front[vy[inside], vx[inside]] > vertices[inside, 2]
        tint = tint * (1.0 - np.clip(spare, 0.0, 1.0) * ~covered)
    weight = rasterize(tint, faces, uv, uv_faces, texture.size[0], fill=0.0)
    # Fine hairs, not a smooth shadow: where the tint fades, it breaks into
    # scattered short strands that thin out downwards, as a real hairline
    # does. A smooth fade under the cards' hard lower edge read as a shelf.
    # The strands run up the head, which on the head's UV island is across the
    # texture: stretched down the image instead, they lay across the forehead
    # as a band of horizontal dashes.
    size = texture.size[0]
    tinted = tint[faces].mean(axis=1) > 0.05
    if tinted.sum() < 3:
        return texture
    corners = uv[uv_faces[tinted]].mean(axis=1)
    rise = np.linalg.lstsq(np.c_[corners, np.ones(len(corners))],
                           vertices[faces[tinted], 1].mean(axis=1), rcond=None)[0][:2]
    fine, long = max(size / 1400.0, 0.6), max(size / 350.0, 1.5)
    across = bool(abs(rise[0]) > abs(rise[1]))  # height changes along u: strands run along x
    hairs = cv2.GaussianBlur(np.random.default_rng(9).random((size, size)), (0, 0),
                             sigmaX=long if across else fine, sigmaY=fine if across else long)
    hairs = (hairs - hairs.mean()) / max(float(hairs.std()), 1e-9)
    solid = np.clip((weight / max(strength, 1e-6) - 0.55) / 0.35, 0.0, 1.0)
    wisps = np.clip(weight * (0.6 + 0.9 * hairs), 0.0, strength)
    weight = solid * weight + (1.0 - solid) * wisps
    root = np.asarray(hair_rgb, dtype=np.float64) * 0.7
    rgb = np.asarray(texture.convert("RGB")).astype(np.float64)
    out = rgb * (1 - weight[..., None]) + root * weight[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# ----------------------------------------------------------------- beard ------

# The face outline from ear to ear round the chin (canonical MediaPipe mesh).
JAW_OUTLINE = [93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]


# Landmarks a smile moves: the lips, the nostrils' wings and the folds from
# nose to mouth corner (canonical MediaPipe mesh).
SMILE_MOVED = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39,
               40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
               82, 81, 80, 191, 64, 294, 98, 327, 129, 358, 48, 278, 102, 331, 49, 279, 219, 439,
               115, 344, 216, 436, 206, 426, 92, 322, 186, 410, 57, 287, 43, 273, 212, 432, 202,
               422, 210, 430]


def outline_trust(photo_rgb: np.ndarray, landmarks, cheek_pixels: np.ndarray,
                  scores: dict | None = None, n: int = 468):
    """How far to trust each landmark for the head-shape fit, and how bearded
    the jaw is (0-1).

    Two things move landmarks off the bone. A laugh spreads the mouth and the
    nostrils: on one photo the mouth measured 57% wider than the average face
    and the nostrils 41%, while temples and cheekbones were 3% wider, and the
    fit explained that with the whole head's width. And MediaPipe draws the jaw
    outline round whatever silhouette the lower face has, which on a full beard
    is the beard's edge, a centimetre or two outside the jaw. So landmarks a
    smile moves count less the broader the smile, and the jaw outline counts
    little where the photo shows facial hair just inside it.
    """
    from facetexture import facial_hair
    from morph import MOUTH_SCORES

    trust = np.ones(n)
    if len(landmarks) <= max(JAW_OUTLINE):
        return trust, 0.0
    moving = max((((scores or {}).get(s, 0.0)) for s in MOUTH_SCORES), default=0.0)
    trust[SMILE_MOVED] = 1.0 - 0.85 * float(np.clip((moving - 0.3) / 0.6, 0.0, 1.0))
    h, w = photo_rgb.shape[:2]
    keep = facial_hair(photo_rgb, landmarks, cheek_pixels)
    centre = np.array([landmarks[1].x * w, landmarks[1].y * h])
    outline = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in JAW_OUTLINE])
    inside = outline + 0.12 * (centre - outline)
    xs = np.clip(inside[:, 0].astype(int), 0, w - 1)
    ys = np.clip(inside[:, 1].astype(int), 0, h - 1)
    beard = float(keep[ys, xs].mean())
    covered = np.clip((beard - 0.3) / 0.4, 0.0, 1.0)
    trust[JAW_OUTLINE] = np.minimum(trust[JAW_OUTLINE], 1.0 - 0.85 * covered)
    # Inside the beard too: MediaPipe stretches the whole lower face out to the
    # beard's edge, and the jowl landmarks under it, trusted fully, widened the
    # jaw past the beard itself. The lips are lips whatever grows round them.
    from morph import INNER_LOWER, INNER_UPPER, OUTER_LOWER, OUTER_UPPER
    points = np.array([[lm.x * w, lm.y * h] for lm in landmarks[:n]])
    on_hair = keep[np.clip(points[:, 1].astype(int), 0, h - 1), np.clip(points[:, 0].astype(int), 0, w - 1)]
    hairy = np.clip((on_hair - 0.3) / 0.4, 0.0, 1.0) * covered
    hairy[INNER_UPPER + INNER_LOWER + OUTER_UPPER + OUTER_LOWER + [61, 291, 78, 308]] = 0.0
    trust = np.minimum(trust, 1.0 - 0.85 * hairy)
    return trust, round(beard, 2)


def forehead_skin(parts: np.ndarray | None, landmarks) -> np.ndarray | None:
    """Photo-space map, 0-1, of the forehead skin the photo shows, up to the hairline.

    The face landmarks stop a finger's width below the hairline, so the bake
    took the top of the forehead from generated skin. The segmenter sees the
    real forehead; this keeps its face-skin class above the brows and between
    the temples, pulled in from the hair a little, as the part of the photo to
    bake there.
    """
    if parts is None or len(landmarks) <= 454:
        return None
    h, w = parts.shape
    face_w = abs(landmarks[454].x - landmarks[234].x) * w
    grow = max(int(round(0.03 * face_w)), 1)
    skin = cv2.erode((parts == FACE_SKIN).astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8))
    brow = np.mean([landmarks[i].y for i in (105, 334)]) * h
    lid = np.mean([landmarks[i].y for i in (159, 386)]) * h
    rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]
    above = np.clip((lid - rows) / max(lid - brow, 1.0), 0.0, 1.0)
    between = np.clip((0.42 * face_w - np.abs(cols - landmarks[9].x * w)) / (0.1 * face_w), 0.0, 1.0)
    return cv2.GaussianBlur(skin * above * between, (0, 0), max(0.01 * face_w, 0.8))


# --------------------------------------------------------------- clothes ------

def clothing_colour(photo_rgb: np.ndarray, landmarks, parts: np.ndarray | None):
    """The colour of the top the person is wearing, or None if none is seen.

    From the segmenter's clothes mask, keeping only the piece of cloth directly
    under the chin, so someone standing behind does not count. Read in LAB
    between the darker and lighter folds, so embroidery glints and shadow do
    not move it.
    """
    if parts is None or len(landmarks) <= 454:
        return None
    h, w = parts.shape
    chin_y = landmarks[152].y * h
    face_w = abs(landmarks[454].x - landmarks[234].x) * w
    cx = landmarks[152].x * w
    cloth = (parts == CLOTHES).astype(np.uint8)
    count, label = cv2.connectedComponents(cloth)
    probe = label[int(min(chin_y + 0.4 * face_w, h - 1)):int(min(chin_y + 1.2 * face_w, h)),
                  max(int(cx - 0.4 * face_w), 0):int(cx + 0.4 * face_w)]
    ids, hits = np.unique(probe[probe > 0], return_counts=True)
    if not len(ids) or hits.max() < 30:
        return None
    mine = cv2.erode((label == ids[hits.argmax()]).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lab = _lab(photo_rgb[mine])
    lo, hi = np.percentile(lab[:, 0], [20, 70])
    return _rgb(np.median(lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)], axis=0))


OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
BEARD_REACH = 0.8          # mesh units, ~8 cm: how far a beard is carried under the jaw


def beard_underside(head, head_uv: np.ndarray, head_texture, sampler, facial_hair_map: np.ndarray,
                    landmarks, photo_size: tuple[int, int], onto=None):
    """A texture with a beard carried round under the jaw.

    The photo sees a beard only where the jaw faces the camera. Under it the
    bake has nothing and leaves generated skin, so a full beard stopped at the
    jaw line with bare skin beneath. Each surface the camera did not see takes
    the beard of the nearest one it did, fading over BEARD_REACH, painted in
    the beard's own colour with strands. Lips and nostrils are dark too and are
    kept out.

    The beard is read from the head; it is painted onto `onto`, a (mesh, uv,
    texture) triple, or the head itself by default. The neck below the head's
    texture is the body's, and painted only on the head the beard stopped on
    that seam in a line from the ear down: the body's neck is painted by the
    same rule, so the two sides agree.

    A lifted, alpha-cut copy of the bearded skin (a "shell") did this first.
    Alpha cut is all or nothing per sample, and on a small photo it drew its
    edge as a torn line across the cheek and jaw however it was softened.

    Returns the texture, or the one given when there is no beard to carry.
    """
    from PIL import Image
    from scipy.spatial import cKDTree
    from facetexture import rasterize

    target, target_uv, texture = onto if onto is not None else (head, head_uv, head_texture)
    w, h = photo_size
    exclude = np.zeros((h, w), np.uint8)
    lips = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in OUTER_LIPS], np.int32)
    cv2.fillPoly(exclude, [lips], 1)
    from delight import NOSE_BASE
    nose = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in NOSE_BASE], np.int32)
    cv2.fillConvexPoly(exclude, cv2.convexHull(nose), 1)
    grow = max(int(0.15 * np.ptp(lips[:, 0])), 1)
    exclude = cv2.dilate(exclude, np.ones((2 * grow + 1,) * 2, np.uint8))
    beard_map = facial_hair_map * (1 - exclude)

    pixels, strength = sampler.sample(head.vertices)
    px = np.clip(pixels[:, 0].astype(int), 0, w - 1)
    py = np.clip(pixels[:, 1].astype(int), 0, h - 1)
    beard = np.where(strength > 0.01, beard_map[py, px], 0.0)
    # Seen squarely, well inside the face sheet: the photo's own beard is good
    # there. Turned away, at a grazing angle where the photo's pixels are its
    # blurred edge, or past the sheet's edge where the photo's beard fades out
    # (FaceSampler.trust), the carried beard takes over, blended across so no
    # line shows. Inside the sheet the photo decides, so the beard does not
    # creep up a clean-shaven cheek.
    facing = head.vertex_normals[:, 2]
    trust = sampler.trust(head.vertices)
    seen = (facing > 0.45) & (trust > 0.9)
    # Where the beard is, from wherever the photo reads it reliably, the side
    # of the jaw included: the colour is the beard's own (below), so a
    # grazing angle's blur does not matter here. Squarely seen beard only
    # lay over 3.5 cm from the back of the jaw and never reached it.
    bearded = (facing > 0.15) & (trust > 0.9) & (beard > 0.35)
    if bearded.sum() < 20:
        return texture
    # And the face sheet's outline, read just inside it: where the photo shows
    # the beard reaching the edge of the face, it carries on past it, towards
    # the ears and under the jaw. From the interior alone the side of the jaw
    # lay 5 to 6.5 cm off and stayed bare.
    centre = sampler.rim_pixels.mean(axis=0)
    inward = sampler.rim_pixels + 0.06 * (centre - sampler.rim_pixels)
    rim_beard = beard_map[np.clip(inward[:, 1].astype(int), 0, h - 1), np.clip(inward[:, 0].astype(int), 0, w - 1)]
    at_rim = rim_beard > 0.35
    source_points = np.vstack([head.vertices[bearded], sampler.rim_points[at_rim]])
    source_beard = np.concatenate([beard[bearded], rim_beard[at_rim]])
    sources = cKDTree(source_points)

    # The beard's colour, from the photo's beard where it is dense.
    head_size = head_texture.size[0]
    coverage = rasterize(np.where(seen, beard, 0.0), head.faces, head_uv, head.faces, head_size, fill=0.0)
    head_rgb = np.asarray(head_texture.convert("RGB")).astype(np.float64)
    core = coverage > 0.7
    tone = np.median(head_rgb[core], axis=0) if core.sum() > 50 else head_rgb[coverage > 0.4].mean(axis=0)

    # Only the target's faces within reach, so a whole body is not rasterised.
    near = sources.query(target.vertices)[0] < BEARD_REACH
    faces = target.faces[near[target.faces].any(axis=1)]
    if not len(faces):
        return texture
    normals = target.vertex_normals

    # Worked out per texel, not per vertex: on the neck's large triangles a
    # per-vertex weight drew their outlines as pale, angular patches.
    size = texture.size[0]
    covered = rasterize(np.ones(len(target.vertices)), faces, target_uv, faces, size, fill=0.0) > 0.5
    at = lambda values: rasterize(values, faces, target_uv, faces, size, fill=0.0)[covered]
    points = np.column_stack([at(target.vertices[:, k]) for k in range(3)])
    texel_facing = at(normals[:, 2])
    unknown = np.maximum(np.clip((0.45 - texel_facing) / 0.3, 0.0, 1.0), 1.0 - sampler.trust(points))
    # Density and height blended over the eight nearest sources: taken from
    # the single nearest, they jumped wherever it changed and drew the
    # beard's edge as streaks.
    distance, nearest = sources.query(points, k=min(8, len(source_points)))
    distance, nearest = distance.reshape(len(points), -1), nearest.reshape(len(points), -1)
    blend = 1.0 / (distance + 0.05) ** 2
    blend /= blend.sum(axis=1, keepdims=True)
    # How dense the beard is where it is read reliably. Interior sources lie
    # at the beard map's soft edge (about 0.4), and carried at that value the
    # beard came out faint past the face's outline.
    density = np.clip((blend * source_beard[nearest]).sum(axis=1) / 0.6, 0.0, 1.0)
    # Full for the first half of the reach, then fading: a full beard runs on
    # under the jaw to the neck at its full density.
    reach = np.clip((BEARD_REACH - distance[:, 0]) / (0.5 * BEARD_REACH), 0.0, 1.0)
    # Downwards only, past a centimetre or so: the beard runs on under the
    # jaw to the neck, 6 to 8 cm from the chin's front, but carried that far
    # up the side of the head it would reach the temples.
    height = (blend * source_points[nearest, 1]).sum(axis=1)
    rise = np.clip(1.0 - (points[:, 1] - height) / 0.15, 0.0, 1.0)
    paint = unknown * density * rise * reach * reach * (3.0 - 2.0 * reach)
    # Inside the nostrils and lips the sheet does not reach (they sit deeper),
    # so they read as unknown; the photo keeps them clear of beard, so must this.
    texel_pixels, _ = sampler.sample(points)
    paint *= 1.0 - exclude[np.clip(texel_pixels[:, 1].astype(int), 0, h - 1),
                           np.clip(texel_pixels[:, 0].astype(int), 0, w - 1)]
    weight = np.zeros((size, size))
    weight[covered] = paint
    # Bled a few texels past each island's edge: texture filtering reads
    # across it, and unpainted border texels drew the seam as a pale line.
    weight = np.where(covered, weight, cv2.dilate(weight.astype(np.float32), np.ones((7, 7), np.uint8)))

    pixels_rgba = np.asarray(texture.convert("RGBA")).astype(np.float64)
    rgb = pixels_rgba[..., :3]
    # Strands: noise drawn out along one texture axis, as light and dark hairs
    # (on the head's map forehead to chin runs along u).
    strands = cv2.GaussianBlur(np.random.default_rng(5).random((size, size)), (0, 0),
                               sigmaX=max(size / 400.0, 1.0), sigmaY=max(size / 2000.0, 0.4))
    strands = (strands - strands.min()) / max(float(np.ptp(strands)), 1e-9)
    hair = tone[None, None, :] * (1.0 + 0.35 * (strands[..., None] - 0.5))
    out = np.clip(rgb * (1.0 - weight[..., None]) + hair * weight[..., None], 0, 255).astype(np.uint8)
    if texture.mode == "RGBA":
        return Image.fromarray(np.dstack([out, pixels_rgba[..., 3].astype(np.uint8)]), "RGBA")
    return Image.fromarray(out, "RGB")


# ---------------------------------------------------------- skin surface -----

# Each brow's lower edge paired with the upper lid below it, outer to inner
# (canonical MediaPipe mesh): the band of skin under the brow.
UNDER_BROW = [(46, 161), (53, 160), (52, 159), (65, 158), (55, 157),
              (276, 388), (283, 387), (282, 386), (295, 385), (285, 384)]
# Each brow's upper edge paired with its lower: the brow itself. The photo
# shades the brow ridge too; lit again, a light brow came out 0.72 of the
# forehead's brightness against the photo's (and the texture's) 0.80.
BROW_LINE = [(70, 46), (63, 53), (105, 52), (66, 65), (107, 55),
             (300, 276), (293, 283), (334, 282), (296, 295), (336, 285)]


def under_brow(vertices: np.ndarray, landmark_vertices: np.ndarray) -> np.ndarray:
    """Per vertex, 0-1, how far it lies on the brow or the skin below it,
    down to the upper lid.

    `landmark_vertices` maps each MediaPipe landmark to the vertex it sits on.
    """
    at = lambda ids: vertices[landmark_vertices[list(ids)]]
    centres = np.vstack([(at(a) + at(b)) / 2.0 for a, b in (zip(*UNDER_BROW), zip(*BROW_LINE))])
    reach = 0.2 * np.linalg.norm(at([33])[0] - at([263])[0])
    near = np.linalg.norm(vertices[:, None, :] - centres[None, :, :], axis=2).min(axis=1)
    return np.exp(-(near / reach) ** 2)


def _neighbours(n: int, faces: np.ndarray):
    """Vertex adjacency (sparse, 0/1) and each vertex's neighbour count."""
    from scipy.sparse import coo_matrix

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0
    return adjacency, np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1.0)


def broad_normals(faces: np.ndarray, normals: np.ndarray, rounds: int) -> np.ndarray:
    """`normals` averaged over their neighbours `rounds` times: the surface's
    broad form, without the creases and hollows inside it."""
    adjacency, degree = _neighbours(len(normals), faces)
    out = np.asarray(normals, dtype=np.float64)
    for _ in range(rounds):
        out = (adjacency @ out + out) / (degree + 1.0)[:, None]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def skin_cavity(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """How far each vertex sits in a crease, 0-1, from the mesh's own shape.

    Ear folds, nostrils, the corners of the eyes, knuckles and the navel catch
    less light and collect colour; without it generated skin reads as moulded
    plastic. Measured as the umbrella Laplacian along the normal (positive
    where a vertex sits below its neighbours), against the local edge length,
    less the same averaged over the neighbourhood.
    """
    import trimesh

    adjacency, degree = _neighbours(len(vertices), faces)
    mean = (adjacency @ vertices) / degree[:, None]
    normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
    spread = np.sqrt((adjacency @ (vertices ** 2).sum(axis=1)) / degree - (mean ** 2).sum(axis=1)
                     + 1e-12)
    depth = ((mean - vertices) * normals).sum(axis=1) / np.maximum(spread, 1e-6)
    # Against the neighbourhood's own curvature: a finger or an arm is convex
    # all round, and its creases are only less convex than the skin beside them.
    smooth = lambda x: (adjacency @ x + x) / (degree + 1.0)
    local = depth.copy()
    for _ in range(4):
        local = smooth(local)
    crease = smooth(depth) - local
    return np.clip((crease - 0.02) / 0.15, 0.0, 1.0)


# Roughness of skin by region (glTF roughness): the T-zone is oilier, lips
# glossier, cheeks and the rest matte. One value everywhere read as a plastic
# sheen across the whole face.
SKIN_ROUGHNESS = {"skin": 0.6, "t_zone": 0.45, "lips": 0.35}


def skin_roughness(landmarks, photo_size: tuple[int, int]) -> np.ndarray:
    """A photo-space map of skin roughness, from the face's landmarks."""
    w, h = photo_size
    rough = np.full((h, w), SKIN_ROUGHNESS["skin"], np.float32)
    pts = lambda ids: np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids], np.int32)
    face_w = abs(landmarks[454].x - landmarks[234].x) * w
    # Forehead centre down the nose: between the inner brows, from the top of
    # the face to the nose tip's sides.
    t_zone = pts([10, 109, 107, 55, 193, 129, 98, 2, 327, 358, 417, 285, 336, 338])
    cv2.fillPoly(rough, [cv2.convexHull(t_zone)], SKIN_ROUGHNESS["t_zone"])
    cv2.fillPoly(rough, [pts(OUTER_LIPS)], SKIN_ROUGHNESS["lips"])
    return cv2.GaussianBlur(rough, (0, 0), max(0.03 * face_w, 1.0))


def nail_weight(vertices: np.ndarray, normals: np.ndarray, skeleton, joints: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
    """Per vertex, 0-1, how much it is fingernail.

    The base mesh has no nails, only fingertips: smooth skin to the end,
    which read as mittens. A nail is the back of the last finger joint's far
    half: vertices that bone mostly carries, facing away from the palm, past
    40% of the bone's length.
    """
    dominant = joints[np.arange(len(joints)), weights.argmax(axis=1)]
    head = lambda name: skeleton.heads[skeleton.index[name]]
    tail = lambda name: skeleton.tails[skeleton.index[name]]
    out = np.zeros(len(vertices))
    for side in ("L", "R"):
        knuckles = (head(f"finger2-1.{side}"), head(f"finger5-1.{side}"))
        palm = np.cross(knuckles[1] - knuckles[0], tail(f"finger3-3.{side}") - head(f"finger3-1.{side}"))
        palm *= np.sign(palm @ (tail(f"finger1-3.{side}") - (knuckles[0] + knuckles[1]) / 2))
        back = -palm / np.linalg.norm(palm)
        for finger in range(1, 6):
            bone = f"finger{finger}-3.{side}"
            rows = dominant == skeleton.index[bone]
            start, end = head(bone), tail(bone)
            axis = end - start
            along = ((vertices[rows] - start) @ axis) / max(float(axis @ axis), 1e-9)
            # The back of this joint: the hand's back taken square to the bone.
            # The model's fingers are curled, so the hand's own back direction
            # met the fingertips at a slant and found only a sliver of nail.
            unit = axis / max(np.linalg.norm(axis), 1e-9)
            facing_dir = back - (back @ unit) * unit
            facing_dir /= max(np.linalg.norm(facing_dir), 1e-9)
            facing = normals[rows] @ facing_dir
            out[rows] = np.clip((facing - 0.3) / 0.3, 0, 1) * np.clip((along - 0.3) / 0.15, 0, 1)
    return out


# Fingernails against the skin around them, in OpenCV LAB: lighter and pinker.
NAIL_SHIFT = np.array([12.0, 6.0, -4.0])
NAIL_ROUGHNESS = 0.3


def paint_nails(rgb: np.ndarray, nail: np.ndarray) -> np.ndarray:
    """Skin texture with nails painted where `nail` (texture-space, 0-1) says,
    and a faint darker fold round each one."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    fold = np.clip(4.0 * nail * (1.0 - nail), 0.0, 1.0)
    lab += NAIL_SHIFT[None, None, :] * nail[..., None]
    lab[..., 0] -= 10.0 * fold
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return np.dstack([out, rgb[..., 3:]]) if rgb.shape[-1] == 4 else out


# Ears against the skin around them, in OpenCV LAB: redder and a little
# darker, most at the rim. They are thin, and blood shows through; generated
# skin left them the head's flat tone, which under the rim light read grey.
EAR_SHIFT = np.array([-4.0, 9.0, 3.0])


def ear_weight(vertex_count: int) -> np.ndarray:
    """Per base-mesh vertex, 0-1, how much it is ear: from how far MakeHuman's
    own ear-shape targets move it, which touch the ears and nothing else."""
    from morph import TARGET_DIR, read_target
    reach = np.zeros(vertex_count)
    for path in sorted((TARGET_DIR / "ears").glob("*.target")):
        idx, delta = read_target(path)
        keep = idx < vertex_count
        np.maximum.at(reach, idx[keep], np.linalg.norm(delta[keep], axis=1))
    return np.clip(reach / max(np.percentile(reach[reach > 0], 60), 1e-6), 0.0, 1.0) if reach.any() else reach


def warm_ears(rgb: np.ndarray, ear: np.ndarray) -> np.ndarray:
    """Skin texture with the ears warmed where `ear` (texture-space, 0-1) says."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    lab += EAR_SHIFT[None, None, :] * ear[..., None]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return np.dstack([out, rgb[..., 3:]]) if rgb.shape[-1] == 4 else out
