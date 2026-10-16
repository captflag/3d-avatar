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
ASSET_DIR = HERE / "assets"
BODY_OBJ = ASSET_DIR / "body_base.obj"

_groups: dict[str, np.ndarray] | None = None
_vertices: np.ndarray | None = None
_uv: np.ndarray | None = None
_uv_groups: dict[str, np.ndarray] | None = None
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
    global _groups, _vertices, _uv, _uv_groups
    if _groups is not None and _vertices is not None:
        return _vertices, _groups

    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    groups: dict[str, list[tuple[int, int, int]]] = {}
    uv_groups: dict[str, list[tuple[int, int, int]]] = {}
    current = None
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            texcoords.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("g "):
            current = line[2:].strip()
            groups.setdefault(current, [])
            uv_groups.setdefault(current, [])
        elif line.startswith("f ") and current is not None:
            corners = line.split()[1:]
            idx = [int(c.split("/")[0]) - 1 for c in corners]
            # The face texture is baked through these, so keep the UV index of
            # every corner alongside its position index.
            tex = [int(c.split("/")[1]) - 1 if "/" in c and c.split("/")[1] else -1
                   for c in corners]
            for i in range(1, len(idx) - 1):  # the base mesh is quads
                groups[current].append((idx[0], idx[i], idx[i + 1]))
                uv_groups[current].append((tex[0], tex[i], tex[i + 1]))

    _vertices = np.array(vertices, dtype=np.float64)
    _uv = np.array(texcoords, dtype=np.float64) if texcoords else np.zeros((1, 2))
    _groups = {name: np.array(faces, dtype=np.int64) for name, faces in groups.items() if faces}
    _uv_groups = {name: np.array(faces, dtype=np.int64) for name, faces in uv_groups.items() if faces}
    return _vertices, _groups


def load_base_uv():
    """Texture coordinates and, per group, the UV index of every triangle corner."""
    load_base()
    return _uv, _uv_groups


def submesh_uv(faces: np.ndarray, uv_faces: np.ndarray, vertices: np.ndarray | None = None,
               uv: np.ndarray | None = None):
    """Like submesh, but carrying texture coordinates.

    A vertex sits at one position but can hold several texture coordinates where
    a UV seam runs through it, and glTF allows only one per vertex, so vertices
    are split per unique position/UV pair. The split keeps the base index, so the
    rig's weights still apply.
    """
    if vertices is None:
        vertices, _ = load_base()
    if uv is None:          # a remapped copy, when the island has its own texture
        uv, _ = load_base_uv()

    pairs = np.stack([faces.reshape(-1), uv_faces.reshape(-1)], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)

    mesh = trimesh.Trimesh(
        vertices=vertices[unique[:, 0]],
        faces=inverse.reshape(-1, 3),
        process=False,
    )
    mesh.metadata["base_index"] = unique[:, 0]
    return mesh, uv[unique[:, 1]]


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


def submesh(faces: np.ndarray, vertices: np.ndarray | None = None) -> trimesh.Trimesh:
    """Build a mesh from base-mesh faces, remembering where each vertex came from.

    The rig's weights are indexed by base-mesh vertex, and every wearable is a
    slice of that mesh, so the mapping has to survive or the clothes cannot be
    skinned. `remove_unreferenced_vertices` throws it away, hence doing it here.

    `vertices` is the base mesh after any shape fit; it defaults to the plain
    base, and every index still refers to the same vertex either way.
    """
    if vertices is None:
        vertices, _ = load_base()
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = trimesh.Trimesh(vertices=vertices[used], faces=remap[faces], process=False)
    mesh.metadata["base_index"] = used
    return mesh


def _mesh_from(faces: np.ndarray, colour, offset: float,
               y_min=None, y_max=None, vertices: np.ndarray | None = None,
               hairline=None) -> trimesh.Trimesh:
    mesh = submesh(faces, vertices)
    if hairline is not None:
        # A curved cut, snapped the same way as a hem: corners below the line
        # are lifted onto it, so the edge follows the curve instead of stepping.
        v = mesh.vertices.copy()
        v[:, 1] = np.maximum(v[:, 1], hairline(v))
        mesh.vertices = v
    if y_min is not None or y_max is not None:
        # Snap the cut edge to a clean line. Keeping or dropping whole faces
        # leaves a zig-zag hem, one tooth per face, which is the most obvious
        # sign the garment is a slice of something rather than a piece of cloth.
        # Boundary faces are kept and their outlying corners pulled onto the cut.
        y = mesh.vertices[:, 1].copy()
        if y_min is not None:
            y = np.maximum(y, y_min)
        if y_max is not None:
            y = np.minimum(y, y_max)
        mesh.vertices = np.column_stack([mesh.vertices[:, 0], y, mesh.vertices[:, 2]])
    if offset:
        # Push the shell off the skin along its own normals, or it z-fights with
        # the body wherever the two surfaces touch.
        mesh.vertices = mesh.vertices + mesh.vertex_normals * offset
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=np.tile(np.append(srgb_to_linear(colour), 255),
                                    (len(mesh.vertices), 1))
    )
    return mesh


def build_item(item: dict, colour=None, vertices: np.ndarray | None = None) -> trimesh.Trimesh:
    """One wearable: cut from the base mesh's helper shells, or loaded from a file.

    Every item must carry a licence. That is checked here, for all of them, so an
    asset of unknown provenance cannot slip into an avatar by any route.
    """
    if not str(item.get("licence") or "").strip():
        raise ValueError(f"catalogue item {item.get('id')!r} has no licence; refusing to use it")

    if "mhclo" in item:
        # A MakeHuman asset — garment, hair, brows or lashes: rebuilt on this
        # body from its reference vertices, so it follows the person's shape,
        # and rigged exactly from the same.
        from garments import load_mhclo, material_textures
        if vertices is None:
            vertices, _ = load_base()
        mesh = load_mhclo(ASSET_DIR / item["mhclo"], vertices)
        rgb = colour if colour is not None else item.get("colour")
        if isinstance(rgb, str):
            rgb = None                            # asked for a photo colour nobody supplied
        material = mesh.metadata["material"]
        texture = normal = None
        if mesh.metadata["uv"] is not None and material is not None and material.exists():
            texture, normal, transparent = material_textures(material, rgb)
            if texture is not None and item.get("recolour"):
                # One outfit file, any colours: dye the shirt and the trousers
                # separately, keeping the fabric's own shading.
                from garments import garment_regions, recolour
                top = garment_regions(mesh, vertices, load_base()[1])
                for part, dye in item["recolour"].items():
                    texture = recolour(texture, mesh, top if part == "top" else ~top, dye)
        if texture is not None:
            # Authored texture: fabric, stitching, folds, hair strands.
            mesh.metadata.update(texture=texture, normal_texture=normal,
                                 alpha_mask=texture.mode == "RGBA")
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=np.tile(np.append(srgb_to_linear(rgb if rgb is not None else (200, 200, 200)), 255),
                                        (len(mesh.vertices), 1))
        )
        return mesh

    if "file" in item:
        # A garment with its own topology. It has no base-mesh indices, so the
        # composer rigs it by weight transfer instead of looking weights up.
        from garments import load_garment
        mesh = load_garment(item)
        rgb = colour if colour is not None else item["colour"]
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=np.tile(np.append(srgb_to_linear(rgb), 255), (len(mesh.vertices), 1))
        )
        mesh.metadata["fit"] = item.get("fit", "tight")
        return mesh

    base, groups = load_base()
    if vertices is None:
        vertices = base

    names = item["group"] if isinstance(item["group"], list) else [item["group"]]
    faces = np.vstack([groups[n] for n in names if n in groups])

    hairline = item.get("hairline")
    if hairline is not None:
        # Keep any face reaching above the line; the boundary is snapped below.
        corners = vertices[faces]
        above = corners[:, :, 1] >= hairline(corners.reshape(-1, 3)).reshape(-1, 3)
        faces = faces[above.any(axis=1)]

    y_min, y_max = item.get("y_min"), item.get("y_max")
    if y_min is not None or y_max is not None:
        # Keep any face that reaches into the band, not just faces centred in
        # it, so the boundary faces survive to be snapped flat below.
        ys = vertices[faces][:, :, 1]
        keep = np.ones(len(faces), dtype=bool)
        if y_min is not None:
            keep &= ys.max(axis=1) >= y_min
        if y_max is not None:
            keep &= ys.min(axis=1) <= y_max
        faces = faces[keep]

    return _mesh_from(faces, colour if colour is not None else item["colour"],
                      item.get("offset", 0.0), y_min, y_max, vertices, hairline)


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


def coverage_alpha(points: np.ndarray, covers: list[trimesh.Trimesh],
                   reach: float = 0.33, margin: float = 0.25, feather: float = 0.1) -> np.ndarray:
    """Per body vertex, 1 where the skin shows and 0 where a garment hides it.

    Replaces deleting body faces. Deletion works in whole triangles and has to
    stop exactly at the garment's edge, so the moment an arm bends and the
    garment and body move a little differently, a gap opens at the cuff or
    collar with nothing behind it. Masking in the texture keeps every triangle,
    fades smoothly, and deliberately leaves a `margin` of skin under each edge,
    so a gap shows skin rather than a hole.
    """
    alpha = np.ones(len(points))
    if not covers:
        return alpha
    near = nearest_distance(points, np.vstack([c.vertices for c in covers]))

    edges = []
    for cover in covers:
        unique, counts = np.unique(cover.edges_sorted, axis=0, return_counts=True)
        rim = np.unique(unique[counts == 1])
        if len(rim):
            edges.append(cover.vertices[rim])
    from_edge = (nearest_distance(points, np.vstack(edges)) if edges
                 else np.full(len(points), np.inf))

    under = np.clip((reach - near) / feather, 0.0, 1.0)
    inside = np.clip((from_edge - margin) / feather, 0.0, 1.0)
    return 1.0 - under * inside


def hair_colour(photo, landmarks, cheek_pixels: np.ndarray | None = None) -> np.ndarray:
    """Hair colour, from the band of hair above and beside the forehead.

    Nine pixels straight above the forehead, as before, landed on the lit crown:
    auburn hair came out as pale peach (219, 171, 134), the colour of skin, and
    the hair read as a bald head. Now a band over the head is sampled, pixels
    that look like the backdrop or like the person's own cheek skin are thrown
    out, and the hair's base colour is read from the middle of what is left by
    lightness: below the highlights, above the deepest shadow.

    The backdrop is every pixel beside the head at the same height (plus the
    photo's top corners), and a band pixel is dropped if it is close to any of
    them. One median colour from the corners was not enough: in a portrait
    framed tight under a bright window the corners were red curtain, the window
    behind the head passed as hair, and short dark-grey hair came out
    (169, 169, 163), near white.

    Falls back to a dark brown when too little hair is found: a bald head, a
    cropped photo, or hair the colour of the backdrop.
    """
    import cv2
    from scipy.spatial import cKDTree

    fallback = np.array([48, 36, 30], dtype=np.uint8)
    rgb = np.asarray(photo.convert("RGB"))
    h, w = rgb.shape[:2]
    if len(landmarks) <= 454:
        return fallback

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    top, chin = landmarks[10], landmarks[152]
    face_h = max(abs(chin.y - top.y), 0.05) * h
    face_w = max(abs(landmarks[454].x - landmarks[234].x), 0.05) * w
    cx, ty = top.x * w, top.y * h

    rows = np.arange(int(ty - 0.45 * face_h), int(ty - 0.03 * face_h), 2)
    rows = rows[(rows >= 0) & (rows < h)]

    def columns(spans):
        xs = np.concatenate([np.arange(int(cx + a * face_w), int(cx + b * face_w), 2) for a, b in spans])
        return xs[(xs >= 0) & (xs < w)]

    band = lab[np.ix_(rows, columns([(-0.5, 0.5)]))].reshape(-1, 3)
    if len(band) < 50:
        return fallback

    def far_from(colour, distance):
        return np.linalg.norm(band - colour, axis=1) > distance

    corner = max(int(0.04 * min(h, w)), 2)
    beside = lab[np.ix_(rows, columns([(-1.0, -0.75), (0.75, 1.0)]))].reshape(-1, 3)
    backdrop = np.vstack([beside, lab[:corner, :corner].reshape(-1, 3),
                          lab[:corner, -corner:].reshape(-1, 3)])
    keep = cKDTree(backdrop).query(band)[0] > 10.0
    if cheek_pixels is not None and len(cheek_pixels):
        cy_ = np.clip(cheek_pixels[:, 1].astype(int), 0, h - 1)
        cx_ = np.clip(cheek_pixels[:, 0].astype(int), 0, w - 1)
        keep &= far_from(np.median(lab[cy_, cx_], axis=0), 14.0)
    hair = band[keep]
    if len(hair) < 50:
        return fallback

    lo, hi = np.percentile(hair[:, 0], [20, 60])
    base = np.median(hair[(hair[:, 0] >= lo) & (hair[:, 0] <= hi)], axis=0)
    return cv2.cvtColor(base.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB).reshape(3)
