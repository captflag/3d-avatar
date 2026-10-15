"""Wearables: the clothes, hair and eyes an avatar is assembled from.

The product we are competing with does not generate clothes either — it exports a
body plus separately authored wearables and hides the body underneath with
visibility masks (../avatarsdk-re/REPORT.md section 5). This is the same idea at
v1 scale: every item is cut from the base mesh's own helper shells, which are
modelled to fit the body, so nothing needs refitting.

Items live in catalogue.json. Adding one is a data change, not a code change.

ponytail: flat colours, no fabric textures, no per-garment UVs. Swapping a cut
shell for a real authored garment mesh later changes `build_item`, nothing else.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import trimesh

HERE = pathlib.Path(__file__).resolve().parent
CATALOGUE_JSON = HERE / "catalogue.json"
BODY_OBJ = HERE / "assets" / "body_base.obj"

_groups: dict[str, np.ndarray] | None = None
_vertices: np.ndarray | None = None
_catalogue: dict | None = None


def catalogue() -> dict:
    global _catalogue
    if _catalogue is None:
        _catalogue = json.loads(CATALOGUE_JSON.read_text(encoding="utf-8"))
    return _catalogue


def items_by_id() -> dict[str, dict]:
    return {item["id"]: item for item in catalogue()["items"]}


def load_base(obj_path: pathlib.Path = BODY_OBJ):
    """Parse the base mesh once into shared vertices and per-group triangles.

    The file holds the visible body plus ~170 helper and joint groups. The
    helpers are what the clothes are cut from, so keep all of them here and let
    callers ask for the group they want.
    """
    global _groups, _vertices
    if _groups is not None and _vertices is not None:
        return _vertices, _groups

    vertices: list[list[float]] = []
    groups: dict[str, list[tuple[int, int, int]]] = {}
    current = None
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            current = line[2:].strip()
            groups.setdefault(current, [])
        elif line.startswith("f ") and current is not None:
            idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
            for i in range(1, len(idx) - 1):  # the base mesh is quads
                groups[current].append((idx[0], idx[i], idx[i + 1]))

    _vertices = np.array(vertices, dtype=np.float64)
    _groups = {name: np.array(faces, dtype=np.int64) for name, faces in groups.items() if faces}
    return _vertices, _groups


def srgb_to_linear(colour) -> np.ndarray:
    """Convert a colour written the way a designer writes it into what glTF stores.

    Vertex colours in glTF are linear, and a viewer converts them to sRGB on the
    way out. Store 0x4C3A2E straight and it displays as a washed-out tan, which
    is how dark brown hair came out looking like a bald scalp. Catalogue colours
    are written in sRGB because that is what a colour picker gives you.
    """
    c = np.asarray(colour, dtype=np.float64) / 255.0
    return np.clip(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) * 255.0,
                   0, 255).astype(np.uint8)


def submesh(faces: np.ndarray) -> trimesh.Trimesh:
    """Build a mesh from base-mesh faces, remembering where each vertex came from.

    The rig's weights are indexed by base-mesh vertex, and every wearable is a
    slice of that mesh, so the mapping has to survive or the clothes cannot be
    skinned. `remove_unreferenced_vertices` throws it away, hence doing it here.
    """
    vertices, _ = load_base()
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = trimesh.Trimesh(vertices=vertices[used], faces=remap[faces], process=False)
    mesh.metadata["base_index"] = used
    return mesh


def _mesh_from(faces: np.ndarray, colour, offset: float) -> trimesh.Trimesh:
    mesh = submesh(faces)
    if offset:
        # Push the shell off the skin along its own normals, or it z-fights with
        # the body wherever the two surfaces touch.
        mesh.vertices = mesh.vertices + mesh.vertex_normals * offset
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=np.tile(np.append(srgb_to_linear(colour), 255),
                                    (len(mesh.vertices), 1))
    )
    return mesh


def build_item(item: dict, colour=None) -> trimesh.Trimesh:
    """One wearable, cut from the base mesh's helper shells by height."""
    vertices, groups = load_base()

    names = item["group"] if isinstance(item["group"], list) else [item["group"]]
    faces = np.vstack([groups[n] for n in names if n in groups])

    y_min, y_max = item.get("y_min"), item.get("y_max")
    if y_min is not None or y_max is not None:
        centroid_y = vertices[faces][:, :, 1].mean(axis=1)
        keep = np.ones(len(faces), dtype=bool)
        if y_min is not None:
            keep &= centroid_y >= y_min
        if y_max is not None:
            keep &= centroid_y <= y_max
        faces = faces[keep]

    return _mesh_from(faces, colour if colour is not None else item["colour"], item.get("offset", 0.0))


def nearest_distance(points: np.ndarray, targets: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance from each point to the closest target vertex.

    Brute force in numpy chunks. ~13k points against ~3k vertices is fast enough
    and keeps scipy out of the dependency list.
    ponytail: brute force, swap for a KD-tree if the catalogue gets big.
    """
    out = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        d = np.linalg.norm(block[:, None, :] - targets[None, :, :], axis=2)
        out[start:start + chunk] = d.min(axis=1)
    return out


def visible_faces(faces: np.ndarray, covers: list[trimesh.Trimesh],
                  threshold: float = 0.33) -> np.ndarray:
    """Which of these base-mesh faces are not hidden under something.

    This is the visibility-mask idea from the report: one body wears any outfit
    because the outfit hides the skin beneath it, instead of every outfit
    shipping its own body. Without it the bare mesh pokes through the clothes.

    Works on faces rather than on a built mesh so the base-vertex mapping the rig
    needs survives the cut.
    """
    if not covers:
        return np.ones(len(faces), dtype=bool)
    vertices, _ = load_base()
    covering = np.vstack([c.vertices for c in covers])
    centroids = vertices[faces].mean(axis=1)
    return nearest_distance(centroids, covering) > threshold


def hair_colour(photo, landmarks) -> np.ndarray:
    """Hair colour sampled just above the hairline of the photo.

    Landmark 10 sits at the top of the forehead, so stepping up from it lands in
    hair for most people. Falls back to a dark brown when that lands off-image or
    on a bright background, which is what happens with a bald or cropped photo.
    """
    rgb = photo.convert("RGB")
    w, h = rgb.size
    if len(landmarks) <= 10:
        return np.array([48, 36, 30], dtype=np.uint8)

    top = landmarks[10]
    chin = landmarks[152] if len(landmarks) > 152 else top
    face_height = max(abs(chin.y - top.y), 0.05)

    samples = []
    for up in (0.10, 0.16, 0.22):
        y = int((top.y - up * face_height) * h)
        if y < 0:
            continue
        for dx in (-0.04, 0.0, 0.04):
            x = int((top.x + dx) * w)
            if 0 <= x < w and 0 <= y < h:
                samples.append(rgb.getpixel((x, y)))
    if not samples:
        return np.array([48, 36, 30], dtype=np.uint8)

    median = np.median(np.array(samples, dtype=np.float64), axis=0)
    # A very bright sample is almost certainly backdrop, not hair.
    if median.mean() > 205:
        return np.array([48, 36, 30], dtype=np.uint8)
    return median.astype(np.uint8)
