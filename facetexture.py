"""Bake the photo onto the body's texture map, aligned and blended.

Three problems, three steps.

1. Alignment. The head is the same generic skull for everyone, so its eyes,
   nose and mouth are not where this person's are. Projecting the photo
   straight on put the photo's eyes on the skull's cheeks. So the photo is
   warped first: its landmarks are moved to wherever the skull's own features
   fall under the fitted camera. The skull's features come from the base mesh
   itself — its eyeball and teeth helpers sit exactly where the eyes and mouth
   are — so no hand annotation is needed.

2. Detail. Colour per vertex cannot hold a face; a texture can. The base mesh
   ships UV coordinates, so the aligned photo is painted into that map at full
   resolution.

3. The seam. A photo pasted onto flat skin leaves an edge wherever the two
   meet. Poisson blending (OpenCV's seamlessClone) keeps the photo's detail
   inside the face but solves its colours so they meet the skin with no step.

ponytail: orthographic camera, a handful of correspondences, no delighting. A
morphable head model would fit the skull's shape instead of bending the photo
to suit it, and is the real fix for likeness. This makes the generic skull wear
the photo as well as it can.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator


# ----------------------------------------------------------------- camera ----

# --------------------------------------------------------- correspondences ---

def head_features(vertices: np.ndarray, groups: dict) -> dict[str, np.ndarray]:
    """Where the generic head keeps its eyes, nose, mouth and chin, in 3D.

    Read off the base mesh's own helper geometry: the eyeball helpers are the
    eyes and the teeth helpers are the mouth. The nose tip and chin are the
    furthest-forward and lowest points of the face between them.
    """
    def points(name):
        return vertices[np.unique(groups[name])]

    left_eye = points("helper-l-eye").mean(axis=0)
    right_eye = points("helper-r-eye").mean(axis=0)
    upper = points("helper-upper-teeth")
    lower = points("helper-lower-teeth")
    # Where the lips meet is the bite line: the lowest edge of the front upper
    # teeth against the highest of the front lower teeth. Checked against the
    # face's midline profile, where it falls in the dip between the two lips.
    # The centre of the teeth, used before, sat 5 mm above it; the rig's oris01
    # bone, which a reviewer suggested, sits 14 mm below it, on the lower lip.
    front_upper = upper[upper[:, 2] > upper[:, 2].max() - 0.08]
    front_lower = lower[lower[:, 2] > lower[:, 2].max() - 0.08]
    mouth = np.vstack([upper, lower]).mean(axis=0)
    mouth[1] = (front_upper[:, 1].min() + front_lower[:, 1].max()) / 2.0

    body = points("body")
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    midline = np.abs(body[:, 0]) < 0.25

    zone = body[midline & (body[:, 1] < eye_y) & (body[:, 1] > mouth[1])]
    nose = zone[zone[:, 2].argmax()]

    below = body[midline & (body[:, 1] < mouth[1]) & (body[:, 1] > mouth[1] - 1.2)]
    front = below[below[:, 2] > below[:, 2].max() - 0.35]
    chin = front[front[:, 1].argmin()]

    return {"left_eye": left_eye, "right_eye": right_eye, "nose": nose,
            "mouth": mouth, "chin": chin}


# ------------------------------------------------------------------ warp -----

# ------------------------------------------------------------------ skin -----

def skin_tone_lab(rgb: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Median skin colour, taken in LAB.

    Medians in RGB mix hue and brightness, so one shadowed sample drags the hue
    as well. LAB separates lightness from colour, so the result keeps the
    person's undertone even when part of the cheek is in shadow.
    """
    h, w = rgb.shape[:2]
    xs = np.clip(pixels[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pixels[:, 1].astype(int), 0, h - 1)
    samples = rgb[ys, xs].astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(samples, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    median = np.median(lab, axis=0).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(median, cv2.COLOR_LAB2RGB).reshape(3)


# --------------------------------------------------------------- sampler -----

class FaceSampler:
    """Map points on the head to pixels in the photo, through every landmark.

    The fitted face, placed on the skull by anatomy, is a sheet of 468 landmarks
    each of which knows its pixel in the photo. A point on the head is looked up
    by where it falls across that sheet, and its pixel interpolated from the
    three landmarks around it — a piecewise-linear warp with a triangle per
    landmark triangle, rather than one global camera or five control points.

    This is the dense landmark warp a reviewer asked for, done in 3D rather than
    in UV space, because this mesh's face is split across several UV islands —
    the inner lips even sit in an island of their own — and a warp drawn in UV
    space would tear along every seam.
    """

    def __init__(self, patch_vertices: np.ndarray, pixels: np.ndarray,
                 border: np.ndarray, depth_tolerance: float = 0.6,
                 keep: np.ndarray | None = None, lips=None):
        xy = patch_vertices[:, :2]
        self._pixels = LinearNDInterpolator(xy, pixels)
        # A closed mouth's lips sit a hair apart on the sheet, and a
        # triangulation across them blends the upper lip's pixels with the
        # lower's, through whatever the photo showed between them: an open
        # smile's teeth. Given the two inner lip contours, corner to corner,
        # each lip is looked up without the other's, split where they meet.
        self._lips = None
        if lips is not None:
            upper, lower = (np.asarray(row) for row in lips)
            seam = (xy[upper] + xy[lower]) / 2.0
            seam = seam[np.argsort(seam[:, 0])]
            sides = []
            for other in (np.setdiff1d(lower, upper), np.setdiff1d(upper, lower)):
                own = np.setdiff1d(np.arange(len(xy)), other)
                sides.append(LinearNDInterpolator(xy[own], pixels[own]))
            self._lips = (seam, *sides)
        self._depth = LinearNDInterpolator(xy, patch_vertices[:, 2])
        # How far inside the face each landmark is, 0 on the outline, for a
        # fade that leaves no edge.
        width = np.ptp(xy[:, 0])
        from_border = np.linalg.norm(xy[:, None, :] - xy[border][None, :, :], axis=2).min(axis=1)
        self._inside = LinearNDInterpolator(xy, np.clip(from_border / (0.18 * width), 0.0, 1.0))
        self.depth_tolerance = depth_tolerance
        # Photo-space map of pixels to keep at full strength even inside the
        # fade: a beard lives in exactly that outer band of the face.
        self._keep = keep

    def sample(self, points: np.ndarray):
        """Pixels for `points`, and how strongly each should take the photo (0-1)."""
        xy = points[:, :2]
        pixels = self._pixels(xy)
        if self._lips is not None:
            seam, above, below = self._lips
            across = (xy[:, 0] > seam[0, 0]) & (xy[:, 0] < seam[-1, 0])
            under = xy[:, 1] < np.interp(xy[:, 0], seam[:, 0], seam[:, 1])
            for side, chosen in ((above, across & ~under), (below, across & under)):
                if chosen.any():
                    pixels[chosen] = side(xy[chosen])
        depth = self._depth(xy)
        inside = self._inside(xy)
        valid = (~np.isnan(pixels).any(axis=1) & ~np.isnan(depth)
                 & (np.abs(points[:, 2] - depth) < self.depth_tolerance))
        strength = np.where(valid, np.nan_to_num(inside), 0.0)
        # Smoothstep, so the fade has no visible start or end.
        strength = strength * strength * (3.0 - 2.0 * strength)
        pixels = np.nan_to_num(pixels)
        if self._keep is not None:
            h, w = self._keep.shape
            kx = np.clip(pixels[:, 0].astype(int), 0, w - 1)
            ky = np.clip(pixels[:, 1].astype(int), 0, h - 1)
            strength = np.maximum(strength, np.where(valid, self._keep[ky, kx], 0.0))
        return pixels, strength


def facial_hair(photo_rgb: np.ndarray, landmarks, cheek_pixels: np.ndarray) -> np.ndarray:
    """Where the photo shows a beard or moustache, 0-1, in photo pixels.

    The bake fades the photo into generated skin across the outer band of the
    face so no edge shows, and that band is exactly where a beard grows: the
    first version shaved every bearded person it was given. Facial hair is told
    apart by being clearly darker than the person's own cheek skin, below the
    nose tip. Sideburns and stubble lighter than that threshold are left to the
    fade.

    ponytail: darkness only, so a hard jaw shadow on a clean-shaven face also
    survives the fade. Add a texture (local variance) test if that shows.
    """
    h, w = photo_rgb.shape[:2]
    lightness = cv2.cvtColor(photo_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float64)
    xs = np.clip(cheek_pixels[:, 0].astype(int), 0, w - 1)
    ys = np.clip(cheek_pixels[:, 1].astype(int), 0, h - 1)
    skin_l = float(np.median(lightness[ys, xs]))

    darker = np.clip((skin_l - lightness - 25.0) / 30.0, 0.0, 1.0)
    darker = darker * darker * (3.0 - 2.0 * darker)

    nose_y = landmarks[1].y * h
    chin_y = landmarks[152].y * h
    below = np.clip((np.arange(h)[:, None] - nose_y) / max(0.05 * (chin_y - nose_y), 1.0), 0.0, 1.0)
    keep = darker * below
    sigma = max(0.004 * w, 0.8)
    return cv2.GaussianBlur(keep, (0, 0), sigma)


# ------------------------------------------------------------------ bake -----

def bake(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, uv_faces: np.ndarray,
         sampler: "FaceSampler", face_centre: np.ndarray, face_radius: float,
         photo_rgb: np.ndarray, skin: np.ndarray, size: int = 1024,
         blend: str = "poisson") -> Image.Image:
    """Paint the aligned photo into the mesh's UV map and blend it into skin.

    `faces` and `uv_faces` are parallel: same triangle, one indexing positions
    and the other texture coordinates.
    """
    ph, pw = photo_rgb.shape[:2]
    skin = np.asarray(skin, dtype=np.float64)
    # `skin` is either one colour or a whole skin texture to paint the face into.
    textured_base = skin.ndim == 3

    painted = np.zeros((size, size, 3), dtype=np.float64)
    weight = np.zeros((size, size), dtype=np.float64)

    centroids = vertices[faces].mean(axis=1)
    candidates = np.where(np.linalg.norm(centroids - face_centre, axis=1) < face_radius * 2.0)[0]

    for fi in candidates:
        tri = vertices[faces[fi]]
        if np.cross(tri[1] - tri[0], tri[2] - tri[0])[2] <= 0:
            continue                            # facing away from the camera

        tri_uv = uv[uv_faces[fi]]
        px = tri_uv[:, 0] * (size - 1)
        py = (1.0 - tri_uv[:, 1]) * (size - 1)  # UV origin is bottom-left

        x0, x1 = int(np.floor(px.min())), int(np.ceil(px.max()))
        y0, y1 = int(np.floor(py.min())), int(np.ceil(py.max()))
        if x1 - x0 > size // 2 or y1 - y0 > size // 2:
            continue                            # straddles a UV seam
        x0, x1, y0, y1 = max(x0, 0), min(x1, size - 1), max(y0, 0), min(y1, size - 1)
        if x0 > x1 or y0 > y1:
            continue

        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        denom = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
        if abs(denom) < 1e-9:
            continue
        w0 = ((py[1] - py[2]) * (xs - px[2]) + (px[2] - px[1]) * (ys - py[2])) / denom
        w1 = ((py[2] - py[0]) * (xs - px[2]) + (px[0] - px[2]) * (ys - py[2])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -0.002) & (w1 >= -0.002) & (w2 >= -0.002)
        if not inside.any():
            continue

        position = w0[..., None] * tri[0] + w1[..., None] * tri[1] + w2[..., None] * tri[2]
        flat = position[inside]
        pixels, strength = sampler.sample(flat)
        ix = np.clip(pixels[:, 0].astype(int), 0, pw - 1)
        iy = np.clip(pixels[:, 1].astype(int), 0, ph - 1)

        ty, tx = ys[inside], xs[inside]
        painted[ty, tx] = photo_rgb[iy, ix]
        weight[ty, tx] = np.maximum(weight[ty, tx], strength)

    base = skin if textured_base else np.tile(skin, (size, size, 1))
    linear = painted * weight[..., None] + base * (1.0 - weight[..., None])
    linear_u8 = np.clip(linear, 0, 255).astype(np.uint8)
    base_u8 = np.clip(base, 0, 255).astype(np.uint8)

    if blend != "poisson":
        return Image.fromarray(linear_u8, mode="RGB")

    # Poisson blend: keep the photo's gradients inside the face but solve its
    # colours so they meet the skin with no step at the boundary.
    mask = (weight > 0.3).astype(np.uint8) * 255
    mask[:2, :] = mask[-2:, :] = 0
    mask[:, :2] = mask[:, -2:] = 0
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    if mask.any():
        ys_, xs_ = np.where(mask > 0)
        centre = (int((xs_.min() + xs_.max()) / 2), int((ys_.min() + ys_.max()) / 2))
        try:
            blended = cv2.seamlessClone(linear_u8, base_u8, mask, centre, cv2.NORMAL_CLONE)
        except cv2.error:
            blended = linear_u8                 # fall back to the soft blend
    else:
        blended = linear_u8

    return Image.fromarray(blended, mode="RGB")


def rasterize(values: np.ndarray, faces: np.ndarray, uv: np.ndarray, uv_faces: np.ndarray,
              size: int, fill: float = 1.0) -> np.ndarray:
    """Paint a per-vertex value into UV space, interpolated across each triangle.

    `values` is indexed like the vertices `faces` refers to. Texels no triangle
    covers keep `fill`.
    """
    image = np.full((size, size), fill, dtype=np.float64)
    for fi in range(len(faces)):
        tri_uv = uv[uv_faces[fi]]
        px = tri_uv[:, 0] * (size - 1)
        py = (1.0 - tri_uv[:, 1]) * (size - 1)
        x0, x1 = int(np.floor(px.min())), int(np.ceil(px.max()))
        y0, y1 = int(np.floor(py.min())), int(np.ceil(py.max()))
        if x1 - x0 > size // 2 or y1 - y0 > size // 2:
            continue
        x0, x1, y0, y1 = max(x0, 0), min(x1, size - 1), max(y0, 0), min(y1, size - 1)
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        denom = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
        if abs(denom) < 1e-9:
            continue
        w0 = ((py[1] - py[2]) * (xs - px[2]) + (px[2] - px[1]) * (ys - py[2])) / denom
        w1 = ((py[2] - py[0]) * (xs - px[2]) + (px[0] - px[2]) * (ys - py[2])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
        v = values[faces[fi]]
        image[ys[inside], xs[inside]] = (w0 * v[0] + w1 * v[1] + w2 * v[2])[inside]
    return image


def uv_island(uv_faces: np.ndarray, seed_uv: int) -> np.ndarray:
    """Which triangles share a UV island with the given UV index.

    Triangles are connected when they share a texture coordinate, so an island
    is a connected component of that graph. Used to give the head its own
    texture: in the shared atlas the whole face gets about 130 pixels across,
    and no amount of care in the bake survives that.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    corners = uv_faces.reshape(-1)
    n = int(corners.max()) + 1
    a = uv_faces[:, [0, 1, 2]].reshape(-1)
    b = uv_faces[:, [1, 2, 0]].reshape(-1)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    return label[uv_faces[:, 0]] == label[seed_uv]


def iris_colour(photo_rgb: np.ndarray, landmarks, default=(70, 48, 32)) -> np.ndarray:
    """The colour of the person's irises, from the ring between pupil and edge.

    Eight single pixels partway out, as before, mostly hit the upper lid's
    shadow and the dark limbal ring, and turned blue eyes grey. Now the whole
    annulus between pupil and rim is sampled, the top of it — usually under the
    lid — is skipped, and the darkest (pupil, lashes) and brightest (catchlight)
    pixels are dropped before the median, taken in LAB.
    """
    if len(landmarks) <= 477:
        return np.array(default, dtype=np.float64)
    h, w = photo_rgb.shape[:2]
    samples = []
    for centre, ring in ((468, range(469, 473)), (473, range(474, 478))):
        cx, cy = landmarks[centre].x * w, landmarks[centre].y * h
        radius = np.mean([np.hypot(landmarks[r].x * w - cx, landmarks[r].y * h - cy) for r in ring])
        if radius < 1.0:
            continue
        span = np.linspace(-radius, radius, 15)
        dx, dy = np.meshgrid(span, span)
        d = np.hypot(dx, dy) / radius
        use = (d >= 0.35) & (d <= 0.85) & (dy >= -0.3 * radius)
        xs = np.clip((cx + dx[use]).astype(int), 0, w - 1)
        ys = np.clip((cy + dy[use]).astype(int), 0, h - 1)
        samples.append(photo_rgb[ys, xs])
    if not samples:
        return np.array(default, dtype=np.float64)
    pixels = np.vstack(samples).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    lo, hi = np.percentile(lab[:, 0], [20, 90])
    median = np.median(lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)], axis=0)
    rgb = cv2.cvtColor(median.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB).reshape(3)
    return rgb.astype(np.float64)


BROW_LANDMARKS = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55,
                  300, 293, 334, 296, 336, 276, 283, 282, 295, 285]


def brow_colour(photo_rgb: np.ndarray, landmarks, default=(60, 44, 34)) -> np.ndarray:
    """The colour of the eyebrows, for tinting the brow cards.

    The brow landmarks trace the brow's outline, so a small patch round each is
    part brow, part skin; the darker half of the patch is the brow.
    """
    if len(landmarks) <= max(BROW_LANDMARKS):
        return np.array(default, dtype=np.float64)
    h, w = photo_rgb.shape[:2]
    reach = max(int(0.004 * w), 1)
    patches = []
    for i in BROW_LANDMARKS:
        x, y = int(landmarks[i].x * w), int(landmarks[i].y * h)
        patch = photo_rgb[max(y - reach, 0):y + reach + 1, max(x - reach, 0):x + reach + 1]
        patches.append(patch.reshape(-1, 3))
    pixels = np.vstack(patches).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    dark = lab[lab[:, 0] <= np.percentile(lab[:, 0], 50)]
    median = np.median(dark, axis=0).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(median, cv2.COLOR_LAB2RGB).reshape(3).astype(np.float64)


def bake_eyes(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, uv_faces: np.ndarray,
              centres: np.ndarray, iris: np.ndarray, size: int = 512) -> Image.Image:
    """Generated eyeballs: white sclera, the person's iris colour, a pupil.

    The photo's eye region cannot be painted onto an eyeball: it includes the
    eyelids and brow, and on a sphere it reads as skin. Avatar eyes are
    synthetic for that reason; only the iris colour comes from the photo.
    Each texel's direction from its eyeball's centre decides what it is, so the
    iris is a true circle however coarse the eyeball mesh.
    """
    sclera = np.array([236.0, 232.0, 226.0])
    pupil = np.array([14.0, 12.0, 12.0])
    image = np.tile(sclera, (size, size, 1))
    forward = np.array([0.0, 0.0, 1.0])

    def band(angle, edge, soft=1.5):
        return np.clip((edge + soft - angle) / (2 * soft), 0.0, 1.0)

    for fi in range(len(faces)):
        tri = vertices[faces[fi]]
        tri_uv = uv[uv_faces[fi]]
        px = tri_uv[:, 0] * (size - 1)
        py = (1.0 - tri_uv[:, 1]) * (size - 1)
        x0, x1 = max(int(np.floor(px.min())), 0), min(int(np.ceil(px.max())), size - 1)
        y0, y1 = max(int(np.floor(py.min())), 0), min(int(np.ceil(py.max())), size - 1)
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        denom = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
        if abs(denom) < 1e-9:
            continue
        w0 = ((py[1] - py[2]) * (xs - px[2]) + (px[2] - px[1]) * (ys - py[2])) / denom
        w1 = ((py[2] - py[0]) * (xs - px[2]) + (px[0] - px[2]) * (ys - py[2])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
        if not inside.any():
            continue
        pos = (w0[..., None] * tri[0] + w1[..., None] * tri[1] + w2[..., None] * tri[2])[inside]
        centre = centres[np.linalg.norm(pos[:, None, :] - centres[None, :, :], axis=2).argmin(axis=1)]
        direction = pos - centre
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        angle = np.degrees(np.arccos(np.clip(direction @ forward, -1.0, 1.0)))[:, None]

        # Sizes from anatomy: an iris about 11.7 mm across on a 24 mm eyeball
        # subtends roughly 29 degrees either side of the gaze, a relaxed pupil
        # about 10. Smaller, and too much white shows, which reads as a stare.
        rim_shade = 1.0 - 0.45 * np.clip((angle - 19.0) / 10.0, 0.0, 1.0)
        colour = sclera * (1.0 - band(angle, 28.5)) + iris * rim_shade * band(angle, 28.5)
        colour = colour * (1.0 - band(angle, 10.0)) + pupil * band(angle, 10.0)
        image[ys[inside], xs[inside]] = colour

    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
