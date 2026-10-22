"""Depth from more than one photo: the face's landmarks seen front-on and turned.

From one photo MediaPipe measures where each landmark lies across the image and
only guesses how far it stands out of it, so the shape fit trusts depth at a
third (morph.DEPTH_TRUST) and a profile comes out average. Turned 30 to 45
degrees, the same landmarks show their depth as sideways movement in the
image. Every view is a turned, scaled and shifted picture of one 3D face; each
landmark's 3D position is solved for from all of them at once, the measured
image coordinates counting fully, each view's depth guess a little, and a view
counting for a landmark only as far as that landmark faces its camera.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import trimesh
from PIL import Image

import appearance
import facetexture

from generate import build_mesh, fit_face, load_canonical_faces

# How much a view's own depth estimate counts beside its measured image
# coordinates (MediaPipe's z is a guess: morph.DEPTH_TRUST).
GUESSED_DEPTH = 0.1
# Rounds of re-estimating each view's turn, scale and shift around the solved face.
ROUNDS = 8
# How far either way of MediaPipe's own estimate a side view's turn is searched,
# and in what steps (degrees). MediaPipe read renders turned 35 degrees as 28.6
# (relative to its front reading): a sixth too little turn makes every depth a
# fifth too deep, since depth is sideways shift over the sine of the turn.
TURN_SEARCH = (20.0, 0.25)


def view_of(photo_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One photo as fuse takes it: its landmarks as build_mesh places them
    before turning them upright (x right, y up, z towards the camera, in units
    of the image's height), the rotation that turns the upright face into
    this view (row vectors: camera = upright @ rotation.T), and the photo
    itself (RGB) for painting from. Raises NoFaceFound."""
    _, landmarks, result, photo = fit_face(photo_path)       # stood upright, as the landmarks are
    points = build_mesh(landmarks, load_canonical_faces(), photo, None).vertices
    rotation = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)[:3, :3]
    u, _, vt = np.linalg.svd(rotation)          # drop any scale: a pure rotation
    return np.asarray(points, dtype=np.float64), u @ vt, np.asarray(photo.convert("RGB"))


def fuse(views: list[tuple]) -> tuple[np.ndarray, np.ndarray, list]:
    """One upright 3D face from several views of it.

    `views`: (points, rotation, ...) per view as view_of gives them, the front
    photo first. Returns the fused points, in the front view's upright frame
    and scale (as upright(points) would give them); per landmark how much
    depth was measured, 0-1 (the best view's sideways sight of it); and each
    view's camera as (rotation, scale, shift): that view's points are
    scale * fused @ rotation.T + shift.
    """
    faces = load_canonical_faces()
    views = [view[:2] for view in views]
    front_points, front_rotation = views[0]
    # Start from the front view, turned upright: that is the frame and scale
    # the rest of the pipeline expects.
    centre = front_points.mean(axis=0)
    shape = (front_points - centre) @ front_rotation + centre
    origin = shape.mean(axis=0)
    shape = shape - origin

    rotations = [rotation for _, rotation in views]
    scales = [1.0] * len(views)
    shifts = [points.mean(axis=0) for points, _ in views]
    weight = np.array([1.0, 1.0, GUESSED_DEPTH])
    n = len(shape)
    measured = np.zeros(n)
    for _ in range(ROUNDS):
        # Which landmarks each view sees: the upright face's normals against
        # the direction back to that view's camera.
        normals = trimesh.Trimesh(shape, faces, process=False).vertex_normals
        lhs = np.zeros((n, 3, 3))
        rhs = np.zeros((n, 3))
        for k, (points, _) in enumerate(views):
            rotation = rotations[k]
            towards = np.array([0.0, 0.0, 1.0]) @ rotation   # camera direction in the upright frame
            seen = np.clip(normals @ towards, 0.0, 1.0) if k else np.ones(n)
            # points ~ scale * shape @ rotation.T + shift, per landmark.
            a = scales[k] * rotation * weight[:, None]           # (3, 3): rows are camera axes
            b = (points - shifts[k]) * weight
            w = seen[:, None, None]
            lhs += w * (a.T @ a)[None]
            rhs += seen[:, None] * (b @ a)
            if k:
                # Sideways sight of the landmark's depth: how much the view
                # turns its forward axis into the image's x.
                measured = np.maximum(measured, seen * abs(rotation[0, 2]))
        shape = np.linalg.solve(lhs + 1e-9 * np.eye(3)[None], rhs[..., None])[..., 0]

        # Each side view's turn, scale and shift, from its measured x and y:
        # the front fixes every landmark's x and y, so only one turn makes the
        # depths the two sides imply agree. The front view sets the frame.
        normals = trimesh.Trimesh(shape, faces, process=False).vertex_normals
        for k, (points, _) in enumerate(views):
            if k == 0:
                continue
            best = None
            reach, step = TURN_SEARCH
            for delta in np.radians(np.arange(-reach, reach + step / 2, step)):
                c, si = np.cos(delta), np.sin(delta)
                turn = np.array([[c, 0.0, si], [0.0, 1.0, 0.0], [-si, 0.0, c]])
                rotation = rotations[k] @ turn
                seen = np.clip(normals @ (np.array([0.0, 0.0, 1.0]) @ rotation), 0.0, 1.0)
                model = (shape @ rotation.T)[:, :2]
                w = seen[:, None]
                mc = (model * w).sum(axis=0) / max(w.sum(), 1e-9)
                tc = (points[:, :2] * w).sum(axis=0) / max(w.sum(), 1e-9)
                scale = float((w * (model - mc) * (points[:, :2] - tc)).sum() / max((w * (model - mc) ** 2).sum(), 1e-12))
                error = float((w * (scale * (model - mc) + tc - points[:, :2]) ** 2).sum())
                if best is None or error < best[0]:
                    best = (error, rotation, scale, tc - scale * mc)
            _, rotations[k], scales[k], xy = best
            shifts[k] = np.array([xy[0], xy[1], points[:, 2].mean() - scales[k] * (shape @ rotations[k].T)[:, 2].mean()])
    cameras = [(rotations[k], scales[k], shifts[k] - scales[k] * origin @ rotations[k].T)
               for k in range(len(views))]
    return shape + origin, np.clip(measured / 0.5, 0.0, 1.0), cameras


# How squarely a texel must face a turned photo's camera before it takes its
# colour (normal against the direction to the camera): from nothing at the
# first to fully at the second.
SIDE_FACING = (0.25, 0.6)
# Texels across which a turned photo's paint fades in, at a 2048 map.
SIDE_FEATHER = 24


def _look_up(image: np.ndarray, px: np.ndarray, py: np.ndarray, interpolation: int,
              **border) -> np.ndarray:
    """image at the points (px, py), any number of them: cv2.remap takes at
    most 32767 rows, so they go through in rows of 1024."""
    n = len(px)
    rows = -(-n // 1024)
    grid = lambda v: np.pad(v.astype(np.float32), (0, rows * 1024 - n)).reshape(rows, 1024)
    found = cv2.remap(np.ascontiguousarray(image, dtype=np.float32), grid(px), grid(py), interpolation, **border)
    return found.reshape(rows * 1024, -1)[:n].squeeze()


def paint_sides(texture: Image.Image, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray,
                uv_faces: np.ndarray, front: np.ndarray, to_fused: tuple[float, np.ndarray],
                sides: list[tuple[np.ndarray, tuple]],
                skip: np.ndarray | None = None) -> tuple[Image.Image, np.ndarray]:
    """Paint what the front photo could not see from the turned photos.

    The front photo reaches the face's outline and no further: the cheeks'
    sides, the jaw towards the ear and the ears themselves were generated
    skin. Each turned photo is projected onto the shaped head through the
    camera fuse found for it, and paints the texels that face it squarely,
    where the front photo did not (`front`, per vertex, 0-1), and where its
    own segmentation sees skin (not hair, not background). Its colour is
    matched to the front's on the cheek both photos see.

    `to_fused`: (scale, shift) with mesh = scale * fused + shift (how
    place_face put the fused face on the skull). `sides`: (photo RGB, camera)
    per turned photo, the camera as fuse returned it. `skip`, per vertex
    0-1: where not to paint at all. Returns the texture
    and, per texel, how much of it the turned photos painted.
    """
    size = texture.size[0]
    scale_p, shift_p = to_fused
    normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
    # Every texel's place on the head and its normal, from the mesh's.
    at = np.dstack([facetexture.rasterize(vertices[:, k], faces, uv, uv_faces, size, fill=np.nan)
                    for k in range(3)])
    facing_map = np.dstack([facetexture.rasterize(normals[:, k], faces, uv, uv_faces, size, fill=0.0)
                            for k in range(3)])
    front_map = facetexture.rasterize(front, faces, uv, uv_faces, size, fill=0.0)
    if skip is not None:
        front_map = np.maximum(front_map, facetexture.rasterize(skip, faces, uv, uv_faces, size, fill=0.0))
    on = ~np.isnan(at[..., 0])
    if not on.any():
        return texture, np.zeros((size, size))
    fused = (at[on] - shift_p) / scale_p
    painted = np.asarray(texture.convert("RGB"), dtype=np.float64)
    out = painted.copy()
    total = np.zeros(on.sum())

    for photo, (rotation, scale, shift) in sides:
        h, w = photo.shape[:2]
        seen = scale * fused @ rotation.T + shift
        # build_mesh's frame back to pixels: x right from the centre, y up, in image heights.
        px = seen[:, 0] * h + w / 2.0
        py = (0.5 - seen[:, 1]) * h
        towards = np.array([0.0, 0.0, 1.0]) @ rotation       # to this camera, in the fused frame
        facing = facing_map[on] @ towards
        lo, hi = SIDE_FACING
        weight = np.clip((facing - lo) / (hi - lo), 0.0, 1.0) * (1.0 - front_map[on])

        both = (front_map[on] > 0.6) & (facing > hi)
        parts = appearance.segment(photo)
        if parts is not None:
            skin = np.isin(parts, (appearance.FACE_SKIN, appearance.BODY_SKIN)).astype(np.uint8)
            # A few pixels in from the skin's edge: the hairline and the outline
            # against the background are where a small misfit picks up the wrong side.
            skin = cv2.erode(skin, np.ones((5, 5), np.uint8))
            on_skin = _look_up(skin, px, py, cv2.INTER_LINEAR, borderValue=0.0)
            weight *= on_skin
            both &= on_skin > 0.5
        colour = _look_up(photo, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)

        # Exposure and white balance as the front photo's, from the cheek both see.
        if both.sum() > 50:
            gain = np.median(painted[on][both], axis=0) / np.maximum(np.median(colour[both], axis=0), 1.0)
            colour = colour * np.clip(gain, 0.5, 2.0)

        # Where both turned photos see a texel, the one facing it more squarely wins.
        take = weight > total
        out[on] = np.where(take[:, None], colour, out[on])
        total = np.maximum(total, weight)

    strength = np.zeros((size, size))
    strength[on] = total
    strength = cv2.GaussianBlur(strength, (0, 0), SIDE_FEATHER * size / 2048.0)
    # The blur reaches texels the raw weight never painted: there `out` is
    # still the old texture, so they blend back to it unchanged.
    blended = painted * (1.0 - strength[..., None]) + out * strength[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), strength
