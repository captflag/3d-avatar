"""Skin texture for the body: colour variation and pores, matched to the face.

Until now the body was one flat colour. Real skin is never flat: it is mottled
at the scale of centimetres, a little redder in some places than others, and
covered in pores that catch light at grazing angles. A body without any of
that reads as moulded plastic however good the lighting is (REALISM-GAP.md,
causes 8 and 21).

This one is generated: procedural, so it carries no licence question at all.
MakeHuman's own skins and the community skin packs are CC0 photo textures
(fetch_assets.py fetches a set of them to assets/), but nothing here reads
them yet; taking their pore and mottling detail in place of the noise is the
next step, kept only if it scores better (REALISM-GAP.md).

Matching follows the advice we received, applied where it makes sense: the
skin's colour statistics are measured on the person's cheeks in LAB (mean and
spread, so both their tone and how much it varies), and the generated texture is
built to those statistics. Only cheek patches are measured, because whole-face
statistics are dominated by eyes, brows and lips, which are not skin.

ponytail: value noise, not a scanned skin; one pore scale everywhere; noise runs
straight across UV seams. Subtle enough that the seams do not show at normal
viewing distance. A licensed or captured skin texture replaces `albedo()` later.
"""

from __future__ import annotations

import cv2
import numpy as np


def cheek_stats(photo_rgb: np.ndarray, pixels: np.ndarray, radius: int = 6):
    """LAB mean and standard deviation over small patches on the cheeks."""
    h, w = photo_rgb.shape[:2]
    patches = []
    for x, y in pixels.astype(int):
        x0, x1 = max(x - radius, 0), min(x + radius + 1, w)
        y0, y1 = max(y - radius, 0), min(y + radius + 1, h)
        if x1 > x0 and y1 > y0:
            patches.append(photo_rgb[y0:y1, x0:x1].reshape(-1, 3))
    if not patches:
        return np.array([170.0, 140.0, 140.0]), np.array([6.0, 2.0, 2.0])
    samples = np.vstack(patches).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(samples, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    # Median for the centre, so one shadowed patch does not shift the tone; the
    # spread is robust too, from the interquartile range.
    q1, q3 = np.percentile(lab, [25, 75], axis=0)
    return np.median(lab, axis=0), (q3 - q1) / 1.35


def face_stats(photo_rgb: np.ndarray, mask: np.ndarray):
    """LAB median and spread of the skin over the whole face.

    Ten cheek patches are not the face: in a smile the cheeks bulge into the
    light and read a few levels brighter than the forehead, jaw and temples the
    generated skin actually borders. The whole face is used instead, keeping
    pixels in the middle of its lightness range and near its median colour, so
    brows, lashes, lips, beard, teeth and highlights drop out.
    """
    lab = cv2.cvtColor(photo_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)[mask].astype(np.float64)
    if len(lab) < 200:
        return None
    lo, hi = np.percentile(lab[:, 0], [30, 80])
    mid = lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)]
    chroma = np.linalg.norm(mid[:, 1:] - np.median(mid[:, 1:], axis=0), axis=1)
    skin = mid[chroma <= np.percentile(chroma, 80)]
    q1, q3 = np.percentile(skin, [25, 75], axis=0)
    return np.median(skin, axis=0), (q3 - q1) / 1.35


def _noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    grid = rng.standard_normal((cells, cells)).astype(np.float32)
    return cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)


def _fractal(size: int, rng: np.random.Generator, octaves) -> np.ndarray:
    total = sum(weight * _noise(size, cells, rng) for cells, weight in octaves)
    return total / (total.std() + 1e-9)


def albedo(size: int, mean_lab, spread_lab, seed: int = 7) -> np.ndarray:
    """A skin-coloured texture with the person's tone and natural mottling.

    The measured spread includes the photo's lighting, so it is scaled down and
    capped: mottling should be felt rather than seen.
    """
    rng = np.random.default_rng(seed)
    mottling = _fractal(size, rng, ((6, 1.0), (24, 0.55), (96, 0.25)))
    blotches = _fractal(size, rng, ((12, 1.0), (48, 0.4)))

    l_amp = min(float(spread_lab[0]) * 0.35, 6.0)
    ab_amp = min(float(max(spread_lab[1], spread_lab[2])) * 0.5, 2.5)

    lab = np.empty((size, size, 3), dtype=np.float32)
    lab[..., 0] = mean_lab[0] + l_amp * mottling
    # Slightly warmer where it is darker, as blood shows through thinner skin.
    lab[..., 1] = mean_lab[1] + ab_amp * blotches - 0.25 * l_amp * mottling
    lab[..., 2] = mean_lab[2] + 0.6 * ab_amp * _fractal(size, rng, ((10, 1.0), (40, 0.5)))
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def pore_normals(size: int, strength: float = 0.16, seed: int = 11) -> np.ndarray:
    """A tangent-space normal map of fine skin grain.

    Flat skin reflects light evenly and looks like plastic; fine grain breaks
    the highlight up. It has to stay at the finest scale and stay faint: one
    1024-pixel atlas covers the whole body, so the face gets roughly 150 pixels
    of it, and anything coarser than a pixel or two turns from pores into
    bark. The first version made exactly that mistake.

    ponytail: real pores need a tiled detail map (KHR_texture_transform) at many
    times the atlas density; this only takes the plastic sheen off.
    """
    rng = np.random.default_rng(seed)
    height = _fractal(size, rng, ((size // 2, 1.0), (size // 4, 0.3)))
    gy, gx = np.gradient(height.astype(np.float64))
    nx, ny = -gx * strength, -gy * strength
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx / length, ny / length, nz / length], axis=-1)
    return ((normal * 0.5 + 0.5) * 255.0).astype(np.uint8)
