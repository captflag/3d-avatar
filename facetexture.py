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

# Share of the sheet's fade band (0.18 of the face's width) over which a kept
# beard fades out at the sheet's edge: about 6% of the face.
KEEP_EDGE = 0.35


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
                 keep: np.ndarray | None = None, lips=None,
                 beyond: np.ndarray | None = None, lip_outline=None):
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
        # The photo's lips, as a mask in photo pixels, from a ring of landmark
        # indices round their outer edge: the bake paints them whichever way
        # they face (on_lips).
        self._lip_mask = None
        if lip_outline is not None:
            ring = np.asarray(pixels)[np.asarray(lip_outline)]
            mask = np.zeros((int(ring[:, 1].max()) + 2, int(ring[:, 0].max()) + 2), np.uint8)
            cv2.fillPoly(mask, [np.round(ring).astype(np.int32)], 1)
            self._lip_mask = mask
        # How far inside the face each landmark is, 0 on the outline, for a
        # fade that leaves no edge.
        width = np.ptp(xy[:, 0])
        from_border = np.linalg.norm(xy[:, None, :] - xy[border][None, :, :], axis=2).min(axis=1)
        self._inside = LinearNDInterpolator(xy, np.clip(from_border / (0.18 * width), 0.0, 1.0))
        self.depth_tolerance = depth_tolerance
        # Photo-space map of pixels to keep at full strength even inside the
        # fade: a beard lives in exactly that outer band of the face.
        self._keep = keep
        # The landmarks stop a finger's width short of the hairline, and the
        # forehead above them was generated skin: a grey band under the hair.
        # `beyond` is a photo-space map of skin the photo does show there; past
        # the sheet, points are mapped by an affine fitted to the upper face,
        # which expression barely moves, and taken where that map says skin.
        self._beyond = beyond
        # The sheet's outline, in 3D and in the photo: where the photo stops.
        self.rim_points = np.asarray(patch_vertices)[border]
        self.rim_pixels = np.asarray(pixels)[border]
        if beyond is not None:
            upper = xy[:, 1] > np.percentile(xy[:, 1], 66)
            self._affine = np.linalg.lstsq(np.c_[xy[upper], np.ones(upper.sum())], pixels[upper],
                                           rcond=None)[0]
            from scipy.spatial import cKDTree
            self._rim = cKDTree(xy[border])
            self._rim_depth = patch_vertices[border, 2]
            # A few centimetres: the forehead between the landmarks and the
            # hairline; `facing` in sample() stops it where the head turns away.
            self._reach = 0.3 * width

    def trust(self, points: np.ndarray) -> np.ndarray:
        """How far the photo can say what is at each point, 0-1: 1 well inside
        the sheet, falling to 0 at its edge over the same band a kept beard
        fades over, 0 off it."""
        xy = points[:, :2]
        depth = self._depth(xy)
        inside = np.nan_to_num(self._inside(xy))
        valid = ~np.isnan(depth) & (np.abs(points[:, 2] - np.nan_to_num(depth)) < self.depth_tolerance)
        return np.where(valid, np.clip(inside / KEEP_EDGE, 0.0, 1.0), 0.0)

    def on_lips(self, points: np.ndarray) -> np.ndarray:
        """Which points look up a pixel on the photo's lips (given lip_outline).

        The resting mouth's lips curl back into it, facing down and away from
        the camera, and the bake left that part generated skin: on
        samples/portrait.jpg, flat tan patches over 59% of the upper lip's
        inner side, which a smile played back turns outwards."""
        if self._lip_mask is None:
            return np.zeros(len(points), bool)
        pixels = np.nan_to_num(self._lookup(points[:, :2]), nan=-1.0).round().astype(int)
        h, w = self._lip_mask.shape
        inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        out = np.zeros(len(points), bool)
        out[inside] = self._lip_mask[pixels[inside, 1], pixels[inside, 0]] > 0
        return out

    def _lookup(self, xy: np.ndarray) -> np.ndarray:
        """Photo pixels for sheet positions, each lip looked up on its own side."""
        pixels = self._pixels(xy)
        if self._lips is not None:
            seam, above, below = self._lips
            across = (xy[:, 0] > seam[0, 0]) & (xy[:, 0] < seam[-1, 0])
            under = xy[:, 1] < np.interp(xy[:, 0], seam[:, 0], seam[:, 1])
            for side, chosen in ((above, across & ~under), (below, across & under)):
                if chosen.any():
                    pixels[chosen] = side(xy[chosen])
        return pixels

    def sample(self, points: np.ndarray, facing: np.ndarray | float | None = None):
        """Pixels for `points`, and how strongly each should take the photo (0-1).

        `facing` is how squarely each point's surface faces the camera (the
        normal's z). Past the sheet only a surface facing the camera takes the
        photo: where the head turns away over the top, a front view crushes
        the scalp onto a few rows of pixels at the hairline and smeared them.
        """
        xy = points[:, :2]
        pixels = self._lookup(xy)
        depth = self._depth(xy)
        inside = self._inside(xy)
        off_sheet = np.isnan(pixels).any(axis=1)
        valid = (~off_sheet & ~np.isnan(depth)
                 & (np.abs(points[:, 2] - depth) < self.depth_tolerance))
        strength = np.where(valid, np.nan_to_num(inside), 0.0)
        # Smoothstep, so the fade has no visible start or end.
        strength = strength * strength * (3.0 - 2.0 * strength)
        pixels = np.nan_to_num(pixels)
        # Both maps are in photo pixels.
        h, w = next((m.shape for m in (self._keep, self._beyond) if m is not None), (1, 1))
        kx = np.clip(pixels[:, 0].astype(int), 0, w - 1)
        ky = np.clip(pixels[:, 1].astype(int), 0, h - 1)
        if self._keep is not None:
            # Full strength through the fade band, but not up to the sheet's
            # very edge: a beard kept to the edge stopped there in a hard line
            # from the ear down the jaw. It fades over the outer 6% of the face.
            edge = np.clip(np.nan_to_num(inside) / KEEP_EDGE, 0.0, 1.0)
            strength = np.maximum(strength, np.where(valid, self._keep[ky, kx] * edge, 0.0))
        if self._beyond is not None:
            # Forehead skin the photo shows is taken in full on the sheet too:
            # the sheet's top edge is not the hairline.
            strength = np.maximum(strength, np.where(valid, self._beyond[ky, kx], 0.0))
        if self._beyond is not None and off_sheet.any():
            out = np.nonzero(off_sheet)[0]
            pixels[out] = np.c_[xy[out], np.ones(len(out))] @ self._affine
            distance, nearest = self._rim.query(xy[out])
            # Near the sheet's rim and not behind it: the scalp's top and back
            # also fall over the forehead in a front view.
            near = ((distance < self._reach)
                    & (np.abs(points[out, 2] - self._rim_depth[nearest]) < self.depth_tolerance))
            fade = np.clip((self._reach - distance) / (0.3 * self._reach), 0.0, 1.0)
            h, w = self._beyond.shape
            bx = np.clip(pixels[out, 0].astype(int), 0, w - 1)
            by = np.clip(pixels[out, 1].astype(int), 0, h - 1)
            square = 1.0 if facing is None else np.clip((np.broadcast_to(facing, len(points))[out] - 0.5) / 0.2, 0.0, 1.0)
            strength[out] = np.where(near, self._beyond[by, bx] * fade * square, 0.0)
        return pixels, strength


# A beard's grain, as a share of the face's width (facial_hair).
BEARD_GRAIN = 0.006


def facial_hair(photo_rgb: np.ndarray, landmarks, cheek_pixels: np.ndarray) -> np.ndarray:
    """Where the photo shows a beard or moustache, 0-1, in photo pixels.

    The bake fades the photo into generated skin across the outer band of the
    face so no edge shows, and that band is exactly where a beard grows: the
    first version shaved every bearded person it was given. Facial hair is told
    apart by being clearly darker than the person's own cheek skin, below the
    nose tip. Sideburns and stubble lighter than that threshold are left to the
    fade.

    Darker than the cheek on the same side: lit from one side, the shadowed
    half of a clean-shaven face sat 40 L* under the median of both cheeks and
    read as beard (jaw_beard 0.42 on three generated women, cheeks 50-70%
    apart), was kept out of the skin's smoothing and darkened as beard: brown
    patches. The reference runs across the face from one cheek's lightness to
    the other's.

    ponytail: darkness only, so a hard jaw shadow on a clean-shaven face also
    survives the fade. Add a texture (local variance) test if that shows.
    """
    h, w = photo_rgb.shape[:2]
    lightness = cv2.cvtColor(photo_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float64)
    xs = np.clip(cheek_pixels[:, 0].astype(int), 0, w - 1)
    ys = np.clip(cheek_pixels[:, 1].astype(int), 0, h - 1)
    nose_x = landmarks[1].x * w
    sides = [(xs < nose_x), (xs >= nose_x)]
    if all(s.any() for s in sides):
        (xl, ll), (xr, lr) = ((float(xs[s].mean()), float(np.median(lightness[ys[s], xs[s]]))) for s in sides)
        across = np.clip((np.arange(w)[None, :] - xl) / max(xr - xl, 1.0), 0.0, 1.0)
        skin_l = ll + (lr - ll) * across
    else:
        skin_l = float(np.median(lightness[ys, xs]))

    darker = np.clip((skin_l - lightness - 25.0) / 30.0, 0.0, 1.0)
    darker = darker * darker * (3.0 - 2.0 * darker)
    # And grainy, as hair is and a shadow is not: the fine-scale variation
    # (BEARD_GRAIN of the face's width) against the cheeks'. Darkness alone
    # took a side-lit woman's whole lower face for beard, and the skin's
    # smoothing kept it as dark patches.
    face_w = abs(landmarks[454].x - landmarks[234].x) * w
    fine = max(BEARD_GRAIN * face_w, 0.8)
    local = lightness - cv2.GaussianBlur(lightness, (0, 0), fine)
    grain = np.sqrt(cv2.GaussianBlur(local * local, (0, 0), 3 * fine))
    cheek_grain = float(np.median(grain[ys, xs]))
    textured = np.clip((grain - 1.5 * cheek_grain) / max(1.5 * cheek_grain, 1.0), 0.0, 1.0)
    # A full, dense beard is too even to be grainy (a black beard on dark skin:
    # grain 7 against its cheeks' 14) but far darker than any shadow: 50 L*
    # (of 255) under the same-side cheek counts as beard grain or none.
    dense = np.clip((skin_l - lightness - 50.0) / 30.0, 0.0, 1.0)
    darker = darker * np.maximum(textured, dense)

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
    photo32 = np.ascontiguousarray(photo_rgb, dtype=np.float32)
    skin = np.asarray(skin, dtype=np.float64)
    # `skin` is either one colour or a whole skin texture to paint the face into.
    textured_base = skin.ndim == 3

    painted = np.zeros((size, size, 3), dtype=np.float64)
    weight = np.zeros((size, size), dtype=np.float64)

    centroids = vertices[faces].mean(axis=1)
    candidates = np.where(np.linalg.norm(centroids - face_centre, axis=1) < face_radius * 2.0)[0]
    # How squarely the surface faces the camera, smooth across triangles.
    corners = vertices[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(vertex_normals, faces, face_normals[:, None, :].repeat(3, axis=1))
    toward = vertex_normals[:, 2] / np.maximum(np.linalg.norm(vertex_normals, axis=1), 1e-12)

    lips = sampler.on_lips(centroids[candidates])
    for fi, on_lips in zip(candidates, lips):
        tri = vertices[faces[fi]]
        if face_normals[fi, 2] <= 0 and not on_lips:
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
        f0, f1, f2 = toward[faces[fi]]
        pixels, strength = sampler.sample(flat, (w0 * f0 + w1 * f1 + w2 * f2)[inside])

        ty, tx = ys[inside], xs[inside]
        # Bilinear, not the nearest pixel: a small photo's face is a few
        # dozen pixels wide, spread over hundreds of texels, and nearest
        # lookups painted it as visible squares.
        painted[ty, tx] = cv2.remap(photo32, (pixels[:, :1] - 0.5).astype(np.float32),
                                    (pixels[:, 1:] - 0.5).astype(np.float32),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0]
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


def brow_depth(photo_rgb: np.ndarray, landmarks) -> np.ndarray:
    """How far down each brow's hair lies across its landmarks: 0 on the upper
    contour, 1 on the lower, right brow then left.

    MediaPipe's brow contours run high. Halfway between them is often the
    brow's top edge: on the test photos the hair is darkest 0.64 to 1.05 of the
    way down. Each brow is read where it is darkest, across the contours and a
    little past the lower one, and the median over its five landmark pairs kept.
    """
    h, w = photo_rgb.shape[:2]
    light = cv2.cvtColor(np.ascontiguousarray(photo_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)[..., 0]
    light = cv2.GaussianBlur(light.astype(np.float32), (0, 0), max(w / 600, 0.7))
    px = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in BROW_LANDMARKS]).reshape(2, 2, 5, 2)
    t = np.linspace(0.0, 1.5, 61)
    along = px[None, :, 0] + t[:, None, None, None] * (px[None, :, 1] - px[None, :, 0])
    dark = light[np.clip(along[..., 1].astype(int), 0, h - 1), np.clip(along[..., 0].astype(int), 0, w - 1)]
    return np.clip(np.median(t[dark.argmin(axis=0)], axis=1), 0.3, 1.2)


def brow_colour(photo_rgb: np.ndarray, landmarks, default=(60, 44, 34)) -> np.ndarray:
    """The colour of the eyebrows, for tinting the brow cards.

    Read in small patches along the line where each brow is darkest
    (`brow_depth`), and the darker half of them kept. Round the landmarks
    themselves, which run high, half the patch was forehead: a light brown brow
    read as (198, 147, 121), next to skin of (250, 204, 182), and the cards
    came out as pale streaks over the photo's darker brow.
    """
    if len(landmarks) <= max(BROW_LANDMARKS):
        return np.array(default, dtype=np.float64)
    h, w = photo_rgb.shape[:2]
    reach = max(int(0.004 * w), 1)
    px = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in BROW_LANDMARKS]).reshape(2, 2, 5, 2)
    line = px[:, 0] + brow_depth(photo_rgb, landmarks)[:, None, None] * (px[:, 1] - px[:, 0])
    patches = []
    for x, y in line.reshape(-1, 2).astype(int):
        patch = photo_rgb[max(y - reach, 0):y + reach + 1, max(x - reach, 0):x + reach + 1]
        patches.append(patch.reshape(-1, 3))
    pixels = np.vstack(patches).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    dark = lab[lab[:, 0] <= np.percentile(lab[:, 0], 50)]
    median = np.median(dark, axis=0).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(median, cv2.COLOR_LAB2RGB).reshape(3).astype(np.float64)


def bake_eyes(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, uv_faces: np.ndarray,
              centres: np.ndarray, iris: np.ndarray, size: int = 1024,
              sclera=(236.0, 232.0, 226.0)) -> Image.Image:
    """Generated eyeballs: sclera, the person's iris colour, a pupil.

    The photo's eye region cannot be painted onto an eyeball: it includes the
    eyelids and brow, and on a sphere it reads as skin. Avatar eyes are
    synthetic for that reason; only the iris colour comes from the photo.
    Each texel's direction from its eyeball's centre decides what it is, so the
    iris is a true circle however coarse the eyeball mesh.

    A flat disc of iris colour round a black pupil read as a doll's eye at any
    distance. Real ones carry radial fibres, a lighter collarette ring part way
    out, a dark limbal ring at the edge; the white dims and warms towards the
    corners, and the upper lid shades the top of the eye.
    """
    corner = np.array([214.0, 178.0, 170.0]) * np.asarray(sclera) / [236.0, 232.0, 226.0]   # pinkish, at the corners
    sclera = np.asarray(sclera, dtype=np.float64)
    pupil = np.array([14.0, 12.0, 12.0])
    image = np.tile(sclera, (size, size, 1))
    forward = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(11)
    # Fibres: high-frequency ripples round the gaze axis, a little wavy with
    # distance out, so they read as strands rather than spokes.
    spokes = [(int(rng.integers(18, 70)), rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.0), rng.uniform(-0.15, 0.15))
              for _ in range(14)]

    def band(angle, edge, soft=1.5):
        return np.clip((edge + soft - angle) / (2 * soft), 0.0, 1.0)

    def fibres(phi, angle):
        wave = sum(amp * np.sin(freq * phi + phase + bend * angle) for freq, phase, amp, bend in spokes)
        return wave / sum(amp for _, _, amp, _ in spokes)

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
        # subtends roughly 29 degrees either side of the gaze. The pupil is
        # set for daylight, about 8 (10, a relaxed pupil, looked dilated next
        # to a flat iris). Smaller iris, and too much white shows: a stare.
        phi = np.arctan2(direction[:, 1], direction[:, 0])[:, None]
        streak = fibres(phi, angle)
        shade = (1.0 + 0.28 * streak
                 + 0.12 * np.exp(-((angle - 15.0) / 2.0) ** 2)                 # collarette
                 - 0.55 * np.clip((angle - 25.5) / 3.0, 0.0, 1.0)               # limbal ring
                 - 0.12 * np.clip((11.0 - angle) / 3.0, 0.0, 1.0))              # darker by the pupil
        out = np.clip((angle - 40.0) / 45.0, 0.0, 1.0)
        white = sclera * (1.0 - 0.15 * out) * (1.0 - 0.3 * out * np.abs(np.cos(phi))) \
            + corner * 0.3 * out * np.abs(np.cos(phi))
        colour = white * (1.0 - band(angle, 28.5)) + iris * shade * band(angle, 28.5)
        colour = colour * (1.0 - band(angle, 8.0, soft=0.8)) + pupil * band(angle, 8.0, soft=0.8)
        # The upper lid's shadow over the top of the eye, a little under the lower.
        up = direction[:, 1:2]
        # Deeper than it was (0.4): the whites rendered as a flat bright
        # ellipse where the photo's are shaded under the lid and warm at the
        # corners. A little under the lower lid too.
        colour = colour * (1.0 - 0.6 * np.clip((up - 0.08) / 0.35, 0.0, 1.0)
                           - 0.25 * np.clip((-up - 0.25) / 0.3, 0.0, 1.0))
        image[ys[inside], xs[inside]] = colour

    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
