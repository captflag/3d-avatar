"""Take the photo's lighting out of the skin before it is baked.

The face used to be painted exactly as shot: the photo's key light on one
cheek, its shadow under the nose and along the jaw, and a bright studio's
exposure. Lit again by the viewer, that shading doubled up (nostrils and brows
too dark) and a bright photo came out near-white. Real pipelines bake an albedo
— the skin's own colour — and let the renderer light it.

No network is needed for the first-order part of that. MediaPipe gives every
landmark a depth, so the face is a 3D surface in the camera's own frame and each
pixel of it has a normal. Skin is close to one colour across the face, so how
its brightness varies with that normal *is* the lighting: fit it with
first-order spherical harmonics — an ambient term plus one direction — and
divide it out, so what remains is shaded as if lit straight from the front.

Measured on three photos, second order was worse, not better: a face shows only
the front half of the sphere, the extra terms chase detail, and gains reached
11x at the jaw. The landmark mesh's normals are also rough at the scale of brow
and nostril, so the correction is smoothed to the broad scale and applied in
part: it lifts the shadowed side and the under-jaw falloff, not the features.

Then exposure. A photo can be blown out — the bright test photo's cheeks sit at
(250, 204, 182) — but real skin reflects at most about half the light that falls
on it. Skin brighter than that is capped, so the renderer has room to light it.

ponytail: one colour of skin assumed across the face, first-order lighting, no
shadows or occlusion (nostrils stay dark), no white balance. A learned albedo
model is the upgrade; this removes the broad shading that read as "pasted".
"""

from __future__ import annotations

import cv2
import numpy as np

# Brightest skin we let through, as linear luminance: light skin reflects about
# 0.4 to 0.5 of visible light.
SKIN_CEILING = 0.45
# The texture is lit again by the renderer, whose lights brighten skin by
# about a third in linear luminance (scene-setup; a flat grey skin rendered at
# 1.3-1.4 times its albedo at the cheek), so the skin is written that much darker than the
# photo shows it. Before, the de-lighting left the brightness wherever its
# gain put it and the renderer added its quarter on top: light faces hit the
# ceiling and happened to come out right, a tan face rendered 33 L* lighter
# than its photo.
RENDER_LIFT = 1.35
# How much of the fitted shading to remove (1 = all of it), and the most any
# pixel may be brightened or darkened.
STRENGTH = 0.7
MAX_GAIN, MIN_GAIN = 1.4, 0.75


def to_linear(srgb: np.ndarray) -> np.ndarray:
    c = np.asarray(srgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(linear: np.ndarray) -> np.ndarray:
    c = np.clip(linear, 0.0, 1.0)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(c * 255.0 + 0.5, 0, 255).astype(np.uint8)


def sh_basis(normals: np.ndarray) -> np.ndarray:
    """First-order spherical harmonics: ambient plus one direction."""
    return np.column_stack([np.ones(len(normals)), normals])


def face_normals(landmarks, faces: np.ndarray, size: tuple[int, int]):
    """Per-pixel surface normal of the face, in the camera's frame.

    The landmarks become a mesh in pixel units (x right, y down, depth away
    from the camera), rasterised with a depth test so the nose hides the
    cheek behind it. Returns normals (h, w, 3), zero where there is no face,
    and the mask of pixels that have one.
    """
    w, h = size
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks[:468]])
    tri = pts[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    # Make normals point at the camera (-z), whichever way the triangles wind.
    if np.median(face_n[:, 2]) > 0:
        face_n = -face_n
    vertex_n = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vertex_n, faces[:, k], face_n)
    vertex_n /= np.maximum(np.linalg.norm(vertex_n, axis=1, keepdims=True), 1e-9)

    normals = np.zeros((h, w, 3))
    depth = np.full((h, w), np.inf)
    for fi in np.where(face_n[:, 2] < 0)[0]:             # facing the camera
        a, b, c = tri[fi]
        x0, x1 = int(max(np.floor(min(a[0], b[0], c[0])), 0)), int(min(np.ceil(max(a[0], b[0], c[0])), w - 1))
        y0, y1 = int(max(np.floor(min(a[1], b[1], c[1])), 0)), int(min(np.ceil(max(a[1], b[1], c[1])), h - 1))
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-9:
            continue
        w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / denom
        w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z = w0 * a[2] + w1 * b[2] + w2 * c[2]
        nearer = inside & (z < depth[ys, xs])
        if not nearer.any():
            continue
        ty, tx = ys[nearer], xs[nearer]
        depth[ty, tx] = z[nearer]
        i, j, k = faces[fi]
        n = (w0[nearer, None] * vertex_n[i] + w1[nearer, None] * vertex_n[j]
             + w2[nearer, None] * vertex_n[k])
        normals[ty, tx] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return normals, np.isfinite(depth)


def fit_lighting(normals: np.ndarray, linear: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Lighting (4 x 3) that best explains `linear` colour from `normals`,
    assuming one albedo. Robust: pixels the fit cannot explain — brows, lips,
    beard, eyes, cast shadows — are dropped and the fit repeated."""
    basis = sh_basis(normals)
    keep = np.ones(len(basis), dtype=bool)
    for _ in range(rounds):
        coeffs = np.linalg.lstsq(basis[keep], linear[keep], rcond=None)[0]
        residual = np.abs(basis @ coeffs - linear).sum(axis=1)
        spread = np.median(residual[keep]) * 1.4826 + 1e-6
        keep = residual < 2.5 * spread
    return coeffs


# Eye openings and the inside of the lips: white of the eye and teeth are meant
# to be brighter than skin, so they are kept out of shine removal.
EYE_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def remove_shine(linear: np.ndarray, mask: np.ndarray, landmarks, size: tuple[int, int]):
    """Pull specular shine on the skin back towards the skin around it.

    A shiny forehead or nose in the photo is the light source's reflection, not
    skin colour. Lighting fits cannot take it out — it is not diffuse — and
    baked in, the renderer adds its own highlight on top, so the forehead and
    nose bridge came out near-white on a brown face. Shine is told apart by
    being brighter than the skin's local colour (a median over a patch about a
    tenth of the face across); such pixels are blended back towards that local
    colour, the more the brighter they are. Eyes and mouth are excluded.
    """
    w, h = size
    keep_out = np.zeros((h, w), dtype=np.uint8)
    for ring in (EYE_LEFT, EYE_RIGHT, INNER_LIPS):
        poly = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ring], dtype=np.int32)
        cv2.fillPoly(keep_out, [poly], 1)
    face_w = max(int(np.ptp(np.nonzero(mask)[1])) if mask.any() else w, 10)
    keep_out = cv2.dilate(keep_out, np.ones((max(face_w // 40, 3),) * 2, np.uint8))
    skin = mask & (keep_out == 0)
    if skin.sum() < 500:
        return linear, 0.0

    srgb = to_srgb(linear)
    k = max(face_w // 10, 5) | 1                       # odd, ~a tenth of the face
    local = to_linear(cv2.medianBlur(srgb, min(k, 255)))
    lum = linear @ np.array([0.2126, 0.7152, 0.0722])
    local_lum = local @ np.array([0.2126, 0.7152, 0.0722])
    excess = np.clip((lum / np.maximum(local_lum, 1e-4) - 1.08) / 0.3, 0.0, 1.0)
    excess = excess * excess * (3 - 2 * excess) * skin
    excess = cv2.GaussianBlur(excess, (0, 0), 1.0)
    out = linear * (1 - excess[..., None]) + local * excess[..., None]
    return out, round(float((excess > 0.5).sum() / skin.sum()), 3)


# Where the photo's light cannot reach: under the nose, round the nostrils and
# the alae, and the inner corners of the eyes. Landmark hulls, from the
# canonical MediaPipe mesh.
NOSE_BASE = [48, 64, 98, 97, 2, 326, 327, 294, 278, 219, 439, 166, 392, 79, 309]
EYE_CORNERS = [(133, 243, 244, 189), (362, 463, 464, 413)]
# The socket above and below each eye: the brow's lower edge, the lid's upper
# line, the lid's lower line (outer to inner corner), and the brow itself,
# which is hair and must not be lifted.
SOCKETS = [
    {"brow_low": [46, 53, 52, 65, 55], "lid_up": [246, 161, 160, 159, 158, 157, 173],
     "lid_low": [33, 7, 163, 144, 145, 153, 154, 155, 133], "brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]},
    {"brow_low": [276, 283, 282, 295, 285], "lid_up": [466, 388, 387, 386, 385, 384, 398],
     "lid_low": [263, 249, 390, 373, 374, 380, 381, 382, 362], "brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]},
]


def lift_crevices(srgb: np.ndarray, landmarks, size: tuple[int, int], strength: float = 0.6):
    """Brighten the photo's crevice shadows towards the skin around them.

    Shadow under the nose and in the eye corners is occlusion: no light got
    there. Baked in, it stays dark whatever the renderer does, and the renderer
    then darkens the same creases again with its own occlusion, so nostrils and
    eye corners read as holes. Inside those zones only, each pixel darker than
    the skin's local median is lifted by `strength` of the difference, so a
    nostril opening stays darker than skin but no longer black. Brows, lashes,
    lips and beard lie outside the zones and are left as they are.
    """
    w, h = size
    zone = np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in NOSE_BASE], np.int32))
    cv2.fillConvexPoly(zone, hull, 1)
    eye_w = abs(landmarks[263].x - landmarks[362].x) * w
    for ring in EYE_CORNERS:
        for i in ring:
            cv2.circle(zone, (int(landmarks[i].x * w), int(landmarks[i].y * h)), max(int(0.18 * eye_w), 2), 1, -1)

    def points(ids):
        return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids])

    for socket in SOCKETS:
        # Above the eye: the band between brow and lid, kept clear of both.
        brow, lid = points(socket["brow_low"]), points(socket["lid_up"])
        gap = float(lid[:, 1].mean() - brow[:, 1].mean())
        upper = np.vstack([brow + [0, 0.25 * gap], (lid - [0, 0.15 * gap])[::-1]])
        cv2.fillPoly(zone, [upper.astype(np.int32)], 1)
        # Below it: the hollow under the lower lid, about half an eye-height deep.
        low = points(socket["lid_low"])
        under = np.vstack([low + [0, 0.06 * eye_w], (low + [0, 0.32 * eye_w])[::-1]])
        cv2.fillPoly(zone, [under.astype(np.int32)], 1)
    keep_out = np.zeros((h, w), np.uint8)
    for ring in [EYE_LEFT, EYE_RIGHT, INNER_LIPS] + [s["brow"] for s in SOCKETS]:
        cv2.fillPoly(keep_out, [points(ring).astype(np.int32)], 1)
    soft = cv2.GaussianBlur((zone & (1 - keep_out)).astype(np.float64), (0, 0), max(0.08 * eye_w, 1.0))
    if soft.max() <= 0:
        return srgb, 0.0

    lab = cv2.cvtColor(srgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    k = max(int(0.6 * eye_w), 5) | 1
    local = cv2.medianBlur(lab[..., 0].astype(np.uint8), min(k, 255)).astype(np.float64)
    deficit = np.clip(local - lab[..., 0], 0.0, None)
    lift = strength * deficit * soft
    lab[..., 0] += lift
    # Write back only what was lifted: an 8-bit LAB round trip nudges every
    # pixel, and nothing outside the crevices should change at all.
    out = srgb.copy()
    changed = lift > 0.5
    out[changed] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)[changed]
    return out, round(float(lift[soft > 0.5].mean()) if (soft > 0.5).any() else 0.0, 1)


def delight(photo_rgb: np.ndarray, landmarks, faces: np.ndarray,
            cheek_pixels: np.ndarray, samples: int = 20000, seed: int = 0,
            exclude: np.ndarray | None = None):
    """The photo with the face's lighting divided out and its exposure capped.

    Only the face region changes, faded at its edge; the rest of the photo is
    returned as it was, scaled by the same exposure factor. Returns the new
    image, a small report of what was done, and the mask of face pixels.

    `exclude` (photo-space, 0-1) marks pixels that are not skin, such as a
    beard, to leave out of the lighting fit: a dark lower face read as shadow
    tells the fit the light comes from above, and the correction then lifts
    the beard to grey.
    """
    h, w = photo_rgb.shape[:2]
    linear = to_linear(photo_rgb)
    normals, mask = face_normals(landmarks, faces, (w, h))
    xs_c = np.clip(cheek_pixels[:, 0].astype(int), 0, w - 1)
    ys_c = np.clip(cheek_pixels[:, 1].astype(int), 0, h - 1)
    luma = np.array([0.2126, 0.7152, 0.0722])
    shown = float(np.median(linear[ys_c, xs_c] @ luma))     # the cheeks as the photo shows them

    gain = np.ones((h, w))
    report = {"lit_from": None, "gain_range": [1.0, 1.0]}
    ys, xs = np.nonzero(mask)
    if len(ys) > 500:
        fit = np.arange(len(ys)) if exclude is None else np.nonzero(exclude[ys, xs] < 0.3)[0]
        if len(fit) < 500:
            fit = np.arange(len(ys))
        pick = np.random.default_rng(seed).choice(fit, min(samples, len(fit)), replace=False)
        coeffs = fit_lighting(normals[ys[pick], xs[pick]], linear[ys[pick], xs[pick]])
        luminance = coeffs @ np.array([0.2126, 0.7152, 0.0722])          # (4,)
        shading = sh_basis(normals[ys, xs]) @ luminance
        frontal = float((sh_basis(np.array([[0.0, 0.0, -1.0]])) @ luminance)[0])
        if frontal > 1e-6:
            ratio = (frontal / np.maximum(shading, 1e-6)) ** STRENGTH
            gain[ys, xs] = np.clip(ratio, MIN_GAIN, MAX_GAIN)
        # Broad scale only, and faded out towards the face's edge, so neither a
        # feature nor the face's outline is drawn into the photo. Blurring the
        # gain alone would pull in the 1.0 outside the face; normalised
        # convolution keeps the edge honest.
        sigma = max(0.015 * w, 2.0)
        inside = mask.astype(np.float64)
        weight = cv2.GaussianBlur(inside, (0, 0), sigma)
        smooth = cv2.GaussianBlur(gain * inside, (0, 0), sigma) / np.maximum(weight, 1e-6)
        soft = cv2.GaussianBlur(inside, (0, 0), max(0.008 * w, 1.0))
        gain = 1.0 + (smooth - 1.0) * soft
        direction = luminance[1:4] / max(np.linalg.norm(luminance[1:4]), 1e-9)
        report = {"lit_from": [round(float(v), 2) for v in direction],
                  "gain_range": [round(float(gain[mask].min()), 2), round(float(gain[mask].max()), 2)]}
    linear = linear * gain[..., None]
    linear, report["shine_removed"] = remove_shine(linear, mask, landmarks, (w, h))

    # Exposure: the cheeks back to the brightness the photo showed, less the
    # renderer's lift, and capped at what real skin reflects. The lighting fit
    # only redistributes shading; the level is the photo's.
    cheek = float(np.median(linear[ys_c, xs_c] @ luma))
    scale = min(SKIN_CEILING, shown / RENDER_LIFT) / max(cheek, 1e-6)
    report["exposure"] = round(float(scale), 2)
    out, report["crevices_lifted"] = lift_crevices(to_srgb(linear * scale), landmarks, (w, h))
    return out, report, mask
