"""Garments that do not come from the base mesh: weights, posing, import.

Every garment so far was a slice of the MakeHuman base mesh, so it inherited the
rig's weights for free — a slice keeps the base mesh's vertex indices, and the
weight table is indexed by them. A real garment, whether modelled by an artist,
bought, or generated, has its own vertices and no such indices. Without weights
it cannot follow the skeleton, and an avatar that cannot move is no use to a
game.

`transfer_weights` gives any garment weights by borrowing them from the body
underneath it: each garment vertex takes a distance-weighted blend of its
nearest body vertices' bone influences. This is the standard approach, and the
same thing MakeHuman's own clothes fitting does.

It is checked against ground truth we already own: the existing cut-shell
garments have true weights, so their weights can be thrown away, transferred as
if they were foreign, and the two posed side by side. See test_rig.py.

ponytail: nearest-neighbour inverse-distance blending, not a smooth RBF solve
or a learned skinning network. Good where a garment hugs the body; loose
garments such as skirts and long coats will want smoothing across the gap.
"""

from __future__ import annotations

import pathlib

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from rig import MAX_INFLUENCES, Skeleton

HERE = pathlib.Path(__file__).resolve().parent
GARMENT_DIR = HERE / "assets" / "garments"


# ------------------------------------------------------------- transfer ------

def transfer_weights(points: np.ndarray, body_vertices: np.ndarray,
                     body_joints: np.ndarray, body_weights: np.ndarray,
                     bone_count: int, neighbours: int = 8, power: float = 2.0,
                     faces: np.ndarray | None = None, loose_from: float = 0.15,
                     loose_to: float = 0.8, smoothing: int = 40):
    """Joints and weights for arbitrary `points`, borrowed from the body.

    Each point blends the bone influences of its `neighbours` nearest body
    vertices, weighted by inverse distance, then keeps the four strongest bones
    glTF can carry and renormalises.

    Pass the garment's `faces` and points that hang away from the body get their
    weights smoothed across the garment's own surface, in proportion to how far
    they hang. Without that, the part of a skirt between the legs borrows from
    whichever leg happens to be nearer, and tears down the middle when the legs
    move apart. Tight-fitting areas are left exact.
    """
    tree = cKDTree(body_vertices)
    distance, nearest = tree.query(points, k=neighbours)
    closeness = 1.0 / np.maximum(distance, 1e-6) ** power
    closeness /= closeness.sum(axis=1, keepdims=True)

    # Accumulate every neighbour's influences into a dense per-point bone row.
    accumulated = np.zeros((len(points), bone_count), dtype=np.float64)
    rows = np.repeat(np.arange(len(points)), neighbours * MAX_INFLUENCES)
    bones = body_joints[nearest].reshape(-1)
    amounts = (body_weights[nearest] * closeness[..., None]).reshape(-1)
    np.add.at(accumulated, (rows, bones), amounts)

    if faces is not None and smoothing > 0:
        accumulated = _smooth_loose(accumulated, faces, distance[:, 0],
                                    loose_from, loose_to, smoothing)

    top = np.argsort(-accumulated, axis=1)[:, :MAX_INFLUENCES]
    top_weights = np.take_along_axis(accumulated, top, axis=1)
    total = top_weights.sum(axis=1, keepdims=True)
    top_weights = np.where(total > 0, top_weights / np.maximum(total, 1e-12), 0.0)
    return top.astype(np.uint16), top_weights.astype(np.float32)


def _smooth_loose(weights: np.ndarray, faces: np.ndarray, gap: np.ndarray,
                  loose_from: float, loose_to: float, iterations: int) -> np.ndarray:
    """Diffuse bone weights across the garment where it hangs off the body.

    Each pass replaces a vertex's weights with a blend of its own and its
    neighbours' average; how much it blends grows with its gap from the body,
    from nothing at `loose_from` to full at `loose_to`.
    """
    from scipy.sparse import coo_matrix, diags

    n = len(weights)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    average = diags(1.0 / np.maximum(degree, 1.0)) @ adjacency

    looseness = np.clip((gap - loose_from) / max(loose_to - loose_from, 1e-6), 0.0, 1.0)[:, None]
    blended = weights.copy()
    for _ in range(iterations):
        blended = blended * (1.0 - looseness) + (average @ blended) * looseness
    return blended


def weights_for(mesh: trimesh.Trimesh, fit: str, vertices: np.ndarray, groups: dict,
                joints: np.ndarray, weights: np.ndarray, bone_count: int):
    """Rig a foreign garment, choosing where to borrow weights from by its fit.

    Measured against garments whose true weights are known, under a stride pose
    that swings parts of them by up to 30 cm:

        fitted shirt     mean 0.05 cm   p95 0.19 cm
        fitted trousers  mean 0.14 cm   p95 0.76 cm
        shoes            mean 0.00 cm
        skirt, from body mean 2.1 cm    p95 12.5 cm   <- tears between the legs

    So a `tight` garment borrows from the body, unsmoothed, because that is
    already near-exact. A `loose` one borrows from the base mesh's skirt helper
    as well — a shell MakeHuman authored with skirt weights, which move with both
    legs at once — and is smoothed across its own surface where it hangs free.

    ponytail: the loose path cannot be validated without a genuinely foreign
    skirt, since the only skirt we own is that helper. Check the first real one.
    """
    names = ["body"] + (["helper-skirt"] if fit == "loose" else [])
    source = np.unique(np.concatenate([np.unique(groups[n]) for n in names if n in groups]))
    return transfer_weights(
        mesh.vertices, vertices[source], joints[source], weights[source], bone_count,
        faces=mesh.faces if fit == "loose" else None,
    )


# --------------------------------------------------------------- posing ------

def _quat_to_matrix(q) -> np.ndarray:
    """glTF quaternion [x, y, z, w] to a 3x3 rotation."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def skinning_matrices(skeleton: Skeleton, pose: dict | None) -> np.ndarray:
    """Per-bone matrices that take a bind-pose vertex to its posed position.

    The same maths a viewer runs on the exported file, done in numpy so a test
    can pose the avatar without rendering it.
    """
    pose = pose or {}
    local_t = skeleton.local_translations()
    world = np.zeros((len(skeleton), 4, 4))
    for i, name in enumerate(skeleton.names):     # parents come first
        local = np.eye(4)
        local[:3, 3] = local_t[i]
        if name in pose:
            local[:3, :3] = _quat_to_matrix(pose[name])
        parent = skeleton.parents[i]
        world[i] = local if parent < 0 else world[parent] @ local
    return world @ skeleton.inverse_bind_matrices().astype(np.float64)


def deform(vertices: np.ndarray, joints: np.ndarray, weights: np.ndarray,
           matrices: np.ndarray) -> np.ndarray:
    """Linear blend skinning: where each vertex ends up in the pose."""
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    out = np.zeros_like(vertices, dtype=np.float64)
    for slot in range(joints.shape[1]):
        m = matrices[joints[:, slot]]
        moved = np.einsum("nij,nj->ni", m, homogeneous)[:, :3]
        out += weights[:, slot:slot + 1] * moved
    return out


# --------------------------------------------------------------- import ------

def load_garment(item: dict) -> trimesh.Trimesh:
    """An external garment mesh, in base-mesh space, ready to colour and rig.

    The catalogue entry names a file under assets/garments and may move it into
    place with `scale` and `translate` (not `offset`, which already means "push
    off the skin" for cut garments). The licence is checked before anything is
    loaded: an asset nobody can vouch for does not go into a product.
    """
    licence = str(item.get("licence") or "").strip()
    if not licence:
        raise ValueError(f"catalogue item {item.get('id')!r} has no licence; refusing to load it")

    path = GARMENT_DIR / item["file"]
    if not path.exists():
        raise FileNotFoundError(f"garment file missing: {path}")

    mesh = trimesh.load(path, force="mesh", process=False)
    if "scale" in item:
        mesh.apply_scale(float(item["scale"]))
    if "translate" in item:
        mesh.apply_translation(np.asarray(item["translate"], dtype=np.float64))
    if not len(mesh.faces):
        raise ValueError(f"garment file {path.name} has no faces")
    return mesh


# ---------------------------------------------------------------- mhclo ------

def load_mhclo(path: pathlib.Path, vertices: np.ndarray) -> trimesh.Trimesh:
    """A MakeHuman garment, fitted to this body.

    MHCLO is MakeHuman's own clothes format, and it suits us better than a plain
    mesh file: each garment vertex is stored as a blend of three body vertices
    plus an offset, so the garment is rebuilt on whatever body it is given —
    including one whose face and proportions were shaped to a photo. Offsets
    are scaled per axis by how far apart two reference vertices sit on this body
    compared with the base mesh, exactly as MakeHuman does.

    The returned mesh remembers its three reference vertices and blend weights,
    so it can be rigged exactly (`mhclo_weights`) rather than by approximation.
    It also carries the garment's texture coordinates (split at UV seams, since
    glTF allows one per vertex), its material file, and the body vertices its
    author marked as hidden underneath it.
    """
    obj_file, material, scales, refs, hidden = None, None, {}, [], []
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        numeric = parts[0].lstrip("-").isdigit()
        if section == "verts" and numeric:
            if len(parts) >= 9:
                refs.append([float(x) for x in parts[:9]])
            elif len(parts) == 1:                # pinned exactly to one vertex
                i = float(parts[0])
                refs.append([i, i, i, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        if section == "delete_verts" and numeric:
            hidden.extend(parts)                 # numbers and "a - b" ranges
            continue
        # Only another section header ends a section: some files put a
        # one-line setting such as `material` between `verts 0` and its data.
        if parts[0] in ("verts", "delete_verts"):
            section = parts[0]
        elif parts[0] == "obj_file":
            obj_file = parts[1]
        elif parts[0] == "material":
            material = parts[1]
        elif parts[0] in ("x_scale", "y_scale", "z_scale"):
            scales[parts[0][0]] = (int(parts[1]), int(parts[2]), float(parts[3]))

    if obj_file is None or not refs:
        raise ValueError(f"{path.name} is not a usable MHCLO file")
    refs = np.array(refs)
    idx, blend, offset = refs[:, :3].astype(np.int64), refs[:, 3:6], refs[:, 6:9]

    scale = np.ones(3)
    for axis, key in enumerate("xyz"):
        if key in scales:
            a, b, base_distance = scales[key]
            scale[axis] = abs(vertices[a, axis] - vertices[b, axis]) / max(base_distance, 1e-9)
    fitted = (blend[:, :, None] * vertices[idx]).sum(axis=1) + offset * scale

    texcoords, corners = [], []
    for raw in (path.parent / obj_file).read_text(encoding="utf-8").splitlines():
        if raw.startswith("vt "):
            texcoords.append([float(x) for x in raw.split()[1:3]])
        elif raw.startswith("f "):
            corner = [c.split("/") for c in raw.split()[1:]]
            pairs = [(int(c[0]) - 1, int(c[1]) - 1 if len(c) > 1 and c[1] else -1) for c in corner]
            for i in range(1, len(pairs) - 1):
                corners.append((pairs[0], pairs[i], pairs[i + 1]))
    corners = np.array(corners, dtype=np.int64)          # (faces, 3 corners, [v, vt])
    if len(fitted) != corners[..., 0].max() + 1:
        raise ValueError(f"{path.name}: {len(fitted)} fitted vertices but the mesh uses "
                         f"{corners[..., 0].max() + 1}")

    uv = None
    if texcoords and (corners[..., 1] >= 0).all():
        # One vertex per distinct position/UV pair, as glTF wants.
        unique, inverse = np.unique(corners.reshape(-1, 2), axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        source, uv = unique[:, 0], np.array(texcoords)[unique[:, 1]]
    else:
        faces, source = corners[..., 0], np.arange(len(fitted))

    mesh = trimesh.Trimesh(vertices=fitted[source], faces=faces, process=False)
    mesh.metadata["mhclo_idx"] = idx[source]
    mesh.metadata["mhclo_blend"] = blend[source]
    mesh.metadata["uv"] = uv
    mesh.metadata["material"] = path.parent / material if material else None
    mesh.metadata["hides"] = _ranges(hidden)
    return mesh


def _ranges(tokens: list[str]) -> np.ndarray:
    """MakeHuman's vertex lists: plain numbers and inclusive "a - b" ranges."""
    out, i = [], 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1] == "-":
            out.extend(range(int(tokens[i]), int(tokens[i + 2]) + 1))
            i += 3
        else:
            out.append(int(tokens[i]))
            i += 1
    return np.array(out, dtype=np.int64)


def load_mhmat(path: pathlib.Path) -> dict:
    """The few MakeHuman material settings a glTF material can use."""
    settings = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split(None, 1)
        if len(parts) == 2 and not parts[0].startswith(("#", "//")):
            settings[parts[0]] = parts[1].strip()
    return settings


def material_textures(mhmat: pathlib.Path, colour=None, size: int = 1024, derive_normal: bool = False):
    """Base colour and normal map for an MHCLO asset, ready for glTF.

    Ambient occlusion is multiplied into the base colour: it is where the
    folds, seams and collar shadows come from, and baking it in keeps it in
    every engine. `colour` recolours the texture — hair and brows take the
    person's own colour — keeping the strand-to-strand variation by scaling
    each texel's brightness relative to the texture's average.
    Textures are capped at `size` so a 4K normal map does not bloat the file.
    """
    from PIL import Image

    settings = load_mhmat(mhmat)
    transparent = settings.get("transparent", "False").lower() == "true"

    def image(key):
        name = settings.get(key)
        if not name or not (mhmat.parent / name).exists():
            return None
        im = Image.open(mhmat.parent / name)
        im.thumbnail((size, size))
        return im

    diffuse, normal, ao = image("diffuseTexture"), image("normalmapTexture"), image("aomapTexture")
    if normal is None and (mhmat.parent / f"{mhmat.stem}_normal.png").exists():
        # Shipped but not named in the material: MakeHuman's hair carries a
        # strand normal map its own .mhmat never refers to.
        normal = Image.open(mhmat.parent / f"{mhmat.stem}_normal.png")
        normal.thumbnail((size, size))
    if diffuse is None:
        return None, None, transparent

    rgba = np.asarray(diffuse.convert("RGBA")).astype(np.float64)
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    if normal is None and derive_normal:
        normal = strand_normals(rgb)
    if transparent and 0 < alpha.max() < 250:
        # Authored for blending: the thin brow's alpha tops out at 187, and cut
        # at half it kept a few clumps, a dotted line and a fleck above the
        # brow. Stretched to the full range, its strands survive the cut.
        alpha = alpha * (255.0 / alpha.max())
    if ao is not None:
        occlusion = np.asarray(ao.convert("L").resize(diffuse.size)).astype(np.float64) / 255.0
        rgb = rgb * occlusion[..., None]
    if colour is not None:
        brightness = rgb @ np.array([0.299, 0.587, 0.114])
        solid = alpha > 128
        mean = brightness[solid].mean() if solid.any() else brightness.mean()
        # Capped, or the few near-white texels a hair texture carries along its
        # edge come out at four times the hair's brightness: a white blotch.
        relative = np.clip(brightness / max(mean, 1.0), 0.0, 1.5)
        rgb = np.asarray(colour, dtype=np.float64)[None, None, :] * relative[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if transparent:
        out = Image.fromarray(np.dstack([rgb, alpha.astype(np.uint8)]), mode="RGBA")
        if colour is not None:
            # Recoloured to one hue, the texture is only brightness and
            # coverage: 256 palette entries hold it without visible loss, and
            # the hair's PNG shrinks to about a third.
            out = out.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    else:
        out = Image.fromarray(rgb, mode="RGB")
    return out, (normal.convert("RGB") if normal is not None else None), transparent


def mhclo_weights(idx: np.ndarray, blend: np.ndarray, joints: np.ndarray,
                  weights: np.ndarray, bone_count: int):
    """Exact skin weights for an MHCLO garment: the blend of its three body
    vertices' weights, using the same blend that places it."""
    blend = np.clip(blend, 0.0, None)            # small negative blends exist
    blend = blend / np.maximum(blend.sum(axis=1, keepdims=True), 1e-12)
    dense = np.zeros((len(idx), bone_count))
    rows = np.repeat(np.arange(len(idx)), 3 * MAX_INFLUENCES)
    bones = joints[idx].reshape(-1)
    amounts = (weights[idx] * blend[:, :, None]).reshape(-1)
    np.add.at(dense, (rows, bones), amounts)
    top = np.argsort(-dense, axis=1)[:, :MAX_INFLUENCES]
    top_w = np.take_along_axis(dense, top, axis=1)
    top_w /= np.maximum(top_w.sum(axis=1, keepdims=True), 1e-12)
    return top.astype(np.uint16), top_w.astype(np.float32)


def make_test_vest(out_path: pathlib.Path | None = None) -> pathlib.Path:
    """Build a garment with foreign topology to exercise the import path.

    No third-party asset is downloaded for testing. Instead the torso of the
    base mesh's tights shell is cut out and subdivided, which gives it vertices
    the base mesh has never seen — exactly the situation a bought or modelled
    garment is in — while staying CC0.
    """
    from catalogue import load_base, submesh

    out_path = out_path or GARMENT_DIR / "test_vest.obj"
    base, groups = load_base()
    faces = groups["helper-tights"]
    corners = base[faces]
    torso = ((corners[:, :, 1].min(axis=1) >= 1.0)
             & (corners[:, :, 1].max(axis=1) <= 5.6)
             & (np.abs(corners[:, :, 0]).max(axis=1) <= 1.55))
    mesh = submesh(faces[torso]).subdivide()
    mesh.vertices = mesh.vertices + mesh.vertex_normals * 0.11   # over a shirt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    return out_path


if __name__ == "__main__":
    print(make_test_vest())


def garment_regions(mesh: trimesh.Trimesh, vertices: np.ndarray, groups: dict) -> np.ndarray:
    """For each face of an outfit, True if it belongs to the top, False if to
    the bottom.

    An outfit file is one mesh with one texture for shirt and trousers alike.
    Each piece of cloth is its own UV island, so islands are classified whole:
    one that sits below the waist and near the body's centre line is trousers;
    anything else is the top. The centre-line test matters because the base
    mesh is modelled arms-out, and the cuffs hang down to hip height.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    waist = float(vertices[np.unique(groups["joint-spine-3"])][:, 1].mean())
    hip = 2.5 * abs(float(vertices[np.unique(groups["joint-l-upper-leg"])][:, 0].mean()))
    n = len(mesh.vertices)
    edges = np.vstack([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    centre = np.zeros((label.max() + 1, 3))
    np.add.at(centre, label, mesh.vertices)
    centre /= np.bincount(label, minlength=len(centre))[:, None]
    bottom = (centre[:, 1] < waist) & (np.abs(centre[:, 0]) < hip)
    return ~bottom[label[mesh.faces[:, 0]]]


# Of the stock teeth map's own shading, how much the whitened teeth keep.
TOOTH_SHADING = 0.25


def whiten_teeth(texture, colour) -> "Image.Image":
    """MakeHuman's teeth map with the teeth (not the gums) at `colour` (sRGB).

    Its teeth are mid grey (the brightest tenth about 166 of 255) with deep
    occlusion painted into every tooth, and rendered they came out grey with
    dark gaps, where a laughing photo's teeth read near white (photo-10's about
    230). The grey texels, told from the red gums by their lack of colour, take
    `colour`, keeping TOOTH_SHADING of their own variation about its mean.
    """
    import cv2
    from PIL import Image

    rgb = np.asarray(texture.convert("RGB")).astype(np.float64)
    grey = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    tooth = (grey > 40) & (chroma < 0.25 * grey)
    if not tooth.any():
        return texture
    relative = grey / max(float(np.median(grey[tooth])), 1.0)
    shade = 1.0 + TOOTH_SHADING * (np.clip(relative, 0.0, 1.5) - 1.0)
    # Feathered over a few texels so the gum line does not show a hard edge.
    weight = cv2.GaussianBlur(tooth.astype(np.float32), (0, 0), 1.0)[..., None]
    out = rgb * (1 - weight) + np.asarray(colour, np.float64)[None, None, :] * shade[..., None] * weight
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def recolour(texture, mesh: trimesh.Trimesh, faces_mask: np.ndarray, colour,
             plain: bool = False) -> "Image.Image":
    """`texture` with the cloth under `faces_mask` dyed `colour` (sRGB).

    The fabric keeps its own shading — weave, folds, seams, the baked
    occlusion — because lightness is scaled, not replaced; only the hue is
    moved, keeping a little of the original variation so it does not go flat.

    `plain` takes a printed pattern out first: the lightness is averaged over
    a few centimetres of cloth, inside the dyed pieces only, which keeps the
    folds and occlusion and drops stripes. A pinstriped blouse dyed light blue
    stayed pinstriped.
    """
    import cv2
    from PIL import Image
    from facetexture import rasterize

    size = texture.size[0]
    uv = np.column_stack([mesh.metadata["uv"][:, 0], mesh.metadata["uv"][:, 1]])
    faces = mesh.faces[faces_mask]
    region = rasterize(np.ones(len(mesh.vertices)), faces, uv, faces, size, fill=0.0) > 0.5
    region = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    rgb = np.asarray(texture.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    if plain:
        # A few texels past the faces too: a cuff's printed stripes lay just
        # outside the faces that map it, and filtering showed them.
        region = cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        sigma = size * 0.02
        inside = region.astype(np.float64)
        weight = cv2.GaussianBlur(inside, (0, 0), sigma)
        smooth = cv2.GaussianBlur(lab[..., 0] * inside, (0, 0), sigma) / np.maximum(weight, 1e-6)
        # Half the baked shading: the front panels carry a strong occlusion
        # gradient the back panel lacks, and they met at the side seams as pale
        # wedges. Folds still show.
        level = smooth[region].mean()
        lab[..., 0] = np.where(region, level + 0.5 * (smooth - level), lab[..., 0])
        lab[..., 1:] = np.where(region[..., None], lab[region][:, 1:].mean(axis=0), lab[..., 1:])
    target = cv2.cvtColor(np.uint8([[colour]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)
    mean = lab[region].mean(axis=0)
    dyed = lab.copy()
    dyed[..., 0] = lab[..., 0] * (target[0] / max(mean[0], 1.0))
    dyed[..., 1:] = target[1:] + (lab[..., 1:] - mean[1:]) * 0.35
    out = np.where(region[..., None], dyed, lab)
    # Unused texels round the pieces take the nearest dyed colour, so texture
    # filtering at a piece's edge reads the dye, not the old print.
    used = rasterize(np.ones(len(mesh.vertices)), mesh.faces, uv, mesh.faces, size, fill=0.0) > 0.5
    used = cv2.dilate(used.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    _, labels = cv2.distanceTransformWithLabels((~region).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(region)
    source = np.zeros((labels.max() + 1, 3))
    source[labels[ys, xs]] = out[ys, xs]
    near = cv2.dilate(region.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~used & ~region
    out[near] = source[labels[near]]
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    if texture.mode == "RGBA":
        return Image.fromarray(np.dstack([out, np.asarray(texture)[..., 3]]), "RGBA")
    return Image.fromarray(out, "RGB")


def drop_socks(shoes: trimesh.Trimesh, ankle_y: float) -> int:
    """Remove the sock from a MakeHuman shoe when trousers cover the ankle.

    MakeHuman's shoes carry a white sock as a separate piece of cloth reaching
    well up the shin. Under trousers it is hidden at rest, but as the legs pose
    it pokes through the fabric in white patches just above the shoe. The
    trouser hem already overlaps the shoe's collar, so nothing is lost by
    dropping it. Pieces are told apart as UV islands; the sock is the one that
    rises well above the ankle. Returns the number of faces removed.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(shoes.vertices)
    edges = np.vstack([shoes.faces[:, [0, 1]], shoes.faces[:, [1, 2]]])
    _, label = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                               shape=(n, n)), directed=False)
    top = np.full(label.max() + 1, -np.inf)
    np.maximum.at(top, label, shoes.vertices[:, 1])
    sock = top[label[shoes.faces[:, 0]]] > ankle_y + 0.5
    if sock.any() and not sock.all():
        shoes.update_faces(~sock)
    return int(sock.sum())


def short_sides(hair: trimesh.Trimesh, vertices: np.ndarray, groups: dict,
                cut: float = 0.5, taper: float = 0.3, close: float = 0.15,
                floor: float = 0.08, ear: float = 0.2, front: float | None = None) -> int:
    """Clip a haircut's back and sides short, keeping its top and front.

    A short back and sides is the commonest men's cut, and none of the stock
    haircuts has one: each carries the same volume all round the head. Heights
    are measured on the scalp under each hair vertex, 0 at the eyes and 1 at
    the crown, so the cut follows the person's own head. Round the sides and
    back the hair is laid down on the scalp, to `close` of its height below
    `cut` and back to full over `taper` above it, but never nearer than `floor`
    (the scalp layer under the hair sits at 0.055): the cards' own strands then
    read as clipped hair. Removing them instead left bare skin above the ears
    and a glossy scalp layer at the back. Hair hanging over the ears, below
    `ear`, is cut away. The front, temple to temple, keeps its volume; with
    `front` its cards end at that height, for a swept-up cut that leaves the
    forehead bare where the stock cut sweeps down over it. Returns the number
    of faces removed.

    The scalp, not the MHCLO anchors: hair hangs from a helper shell that
    already stands off the head, so drawing it to its anchors barely moved it.
    """
    from scipy.spatial import cKDTree

    eyes = vertices[np.unique(np.r_[groups["joint-l-eye"], groups["joint-r-eye"]])].mean(axis=0)
    crown = vertices[np.unique(groups["joint-head-2"])].mean(axis=0)
    centre = vertices[np.unique(groups["joint-head"])].mean(axis=0)
    skin = np.unique(groups["body"])
    skin = skin[vertices[skin, 1] > eyes[1] - 1.5]          # the head and neck
    scalp = vertices[skin][cKDTree(vertices[skin]).query(hair.vertices)[1]]

    height = (scalp[:, 1] - eyes[1]) / max(crown[1] - eyes[1], 1e-6)
    # 0 at the front, 1 from just behind the temples round the back.
    angle = np.degrees(np.abs(np.arctan2(scalp[:, 0] - centre[0], scalp[:, 2] - centre[2])))
    side = np.clip((angle - 35.0) / 30.0, 0.0, 1.0)
    side = side * side * (3.0 - 2.0 * side)

    keep = close + (1.0 - close) * np.clip((height - cut) / taper, 0.0, 1.0)
    keep = 1.0 - side * (1.0 - keep)
    offset = hair.vertices - scalp
    length = np.linalg.norm(offset, axis=1)
    target = np.maximum(length * keep, np.minimum(length, floor))
    hair.vertices = scalp + offset * (target / np.maximum(length, 1e-9))[:, None]

    over_ears = np.clip((angle - 50.0) / 15.0, 0.0, 1.0) * np.clip((130.0 - angle) / 15.0, 0.0, 1.0)
    below = (over_ears[hair.faces].mean(axis=1) > 0.5) & (height[hair.faces].max(axis=1) < ear)
    if front is not None:
        below |= (side[hair.faces].mean(axis=1) < 0.5) & (height[hair.faces].max(axis=1) < front)
    if below.any() and not below.all():
        hair.update_faces(~below)
    return int(below.sum())


# How far round the head the front of a haircut is swept back with it,
# degrees from straight ahead: all of it within the first, none past the second.
SWEEP_SPAN = (30.0, 75.0)


def sweep_back(hair: trimesh.Trimesh, vertices: np.ndarray, groups: dict, edge: np.ndarray,
               hairline_y, floor: float = 0.05, above: float | None = None) -> float:
    """Lift a haircut's front up and back over the head until it ends at
    `hairline_y` mid-forehead, without cutting it. Returns the angle, degrees.

    The stock cuts all bring a fringe down to a third of an eye spacing above
    the brows; the test photos' hair ends twice that high. The hair is turned
    about the cranium's centre round the ear-to-ear axis, so it moves along
    the head as combed-back hair does, by the angle that brings `edge` (the
    haircut's opaque points; those straight ahead set it) to the hairline, and less
    towards the sides (SWEEP_SPAN); the back does not move. Hair that would
    end up nearer the scalp than `floor` is pushed back out. Hair that already
    ends higher than the photo's is left alone.

    `hairline_y` may instead be the photo's hairline as a curve, (xs, ys)
    across the forehead in head units. Each part of the front is then turned by
    what brings the edge above it up to the curve, so the cut recedes at the
    temples where the photo's does; past the curve's ends nothing turns.
    Only the edge above `above` (the brow line; the curve's heights are read
    above the photo's brows) sets each part's turn, so hair hanging down the
    sides in front of the temples does not count as a low hairline.
    """
    from scipy.spatial import cKDTree

    skin = np.unique(groups["body"])
    eyes = vertices[np.unique(np.r_[groups["joint-l-eye"], groups["joint-r-eye"]])].mean(axis=0)
    cranium = vertices[skin[vertices[skin, 1] > eyes[1]]]
    centre = np.array([0.0, cranium[:, 1].mean(), (cranium[:, 2].max() + cranium[:, 2].min()) / 2.0])

    def needed(points, target):
        """The turn that brings the lowest of `points` (relative) up to
        `target`, and that lowest point's elevation now."""
        radius = float(np.median(np.hypot(points[:, 1], points[:, 2])))
        lift = np.arcsin(np.clip((target - centre[1]) / radius, -1.0, 1.0))
        now = np.arcsin(np.clip(np.percentile(points[:, 1], 3) / radius, -1.0, 1.0))
        return float(lift - now), float(now)

    rel = edge - centre
    round_edge = np.degrees(np.abs(np.arctan2(rel[:, 0], rel[:, 2])))
    p = hair.vertices - centre
    angle = np.degrees(np.abs(np.arctan2(p[:, 0], p[:, 2])))
    t = np.clip((SWEEP_SPAN[1] - angle) / (SWEEP_SPAN[1] - SWEEP_SPAN[0]), 0.0, 1.0)
    if np.ndim(hairline_y) == 0:
        # The hair's lowest visible points straight ahead (within 15 degrees).
        ahead = round_edge < 15.0
        if ahead.sum() < 10:
            return 0.0
        turn = needed(rel[ahead], hairline_y)[0]
        if turn <= 0.0:
            return 0.0
        a = turn * t * t * (3.0 - 2.0 * t)
    else:
        xs, ys = (np.asarray(v, dtype=np.float64) for v in hairline_y)
        step = float(np.median(np.diff(xs)))
        turns, nows = np.full(len(xs), np.nan), np.full(len(xs), np.nan)
        for i, (x, y) in enumerate(zip(xs, ys)):
            near = (np.abs(edge[:, 0] - x) < step) & (round_edge < SWEEP_SPAN[1])
            if above is not None:
                near &= edge[:, 1] > above
            if near.sum() >= 10:
                turns[i], nows[i] = needed(rel[near], y)
        known = ~np.isnan(turns)
        if not known.any():
            return 0.0
        turns = np.maximum(np.interp(xs, xs[known], turns[known]), 0.0)
        nows = np.interp(xs, xs[known], nows[known])
        turn = float(turns.max())
        if turn <= 0.0:
            return 0.0
        # Each part turned less the higher up the head it lies, the edge by
        # all of its turn and the crown not at all: turned whole, columns
        # needing 40-50 degrees at the temples stood the cut up in a block.
        # Fading to none a step past the curve's ends, and round the head
        # over the same span past SWEEP_SPAN that it fades over in front of it.
        ends = lambda values: np.interp(hair.vertices[:, 0], np.r_[xs[0] - 2 * step, xs, xs[-1] + 2 * step],
                                        np.r_[0.0, values, 0.0])
        now = np.interp(hair.vertices[:, 0], xs, nows)
        rise = np.arctan2(p[:, 1], p[:, 2])
        a = ends(turns) * np.clip((np.pi / 2 - rise) / (np.pi / 2 - now), 0.0, 1.0)
        t = np.clip((2 * SWEEP_SPAN[1] - SWEEP_SPAN[0] - angle) / (SWEEP_SPAN[1] - SWEEP_SPAN[0]), 0.0, 1.0)
        a = a * t * t * (3.0 - 2.0 * t)
    y, z = p[:, 1].copy(), p[:, 2].copy()
    p[:, 1] = y * np.cos(a) + z * np.sin(a)
    p[:, 2] = -y * np.sin(a) + z * np.cos(a)
    moved = p + centre

    # The forehead is flatter than the circle the hair turned on, so turned
    # hair stood off it, and from the side the hair's dark inside showed
    # through the gap. Each moved point is set back to the distance from the
    # scalp it had before (never nearer than `floor`).
    normals = trimesh.Trimesh(vertices, groups["body"], process=False).vertex_normals
    tree = cKDTree(vertices[skin])
    gap_of = lambda pts, near: ((pts - vertices[near]) * normals[near]).sum(axis=1)
    before = skin[tree.query(hair.vertices)[1]]
    keep = np.maximum(gap_of(hair.vertices, before), floor)
    near = skin[tree.query(moved)[1]]
    moved += normals[near] * ((keep - gap_of(moved, near)) * (a > 0))[:, None]
    hair.vertices = moved
    return float(np.degrees(turn))


# Where a lift spreads over a haircut: by height up the head (0 at the eyes,
# 1 at the crown), none below the first and all above the second; and round
# from the front, all within the first angle and none past the second.
VOLUME_SPAN = ((0.3, 0.8), (60.0, 150.0))
# How far up from the hair's front edge (mesh units) the lift takes to come
# in full. Lifted right at the edge, the hair stood off the forehead as a
# shell and its dark inside showed under the fringe.
VOLUME_RAMP = 0.6


# How many skin vertices each hair vertex's scalp normal is averaged over.
SCALP_NEIGHBOURS = 12


def scalp_normals(points: np.ndarray, vertices: np.ndarray, groups: dict) -> np.ndarray:
    """The scalp's normal under each point, averaged over the nearest skin
    vertices by inverse distance. Taken from the single nearest vertex, two
    neighbouring points on a card took two different normals, and the lifts
    (volumise, widen) bent the cards into creases that the hair's highlight
    drew as facets."""
    from scipy.spatial import cKDTree
    skin = np.unique(groups["body"])
    normals = trimesh.Trimesh(vertices, groups["body"], process=False).vertex_normals
    distance, index = cKDTree(vertices[skin]).query(points, k=SCALP_NEIGHBOURS)
    weight = 1.0 / np.maximum(distance, 1e-6)
    averaged = (normals[skin[index]] * weight[..., None]).sum(axis=1)
    return averaged / np.maximum(np.linalg.norm(averaged, axis=1, keepdims=True), 1e-9)


def volumise(hair: trimesh.Trimesh, vertices: np.ndarray, groups: dict, rise_y: float) -> float:
    """Lift a haircut's top and front off the scalp until it stands as high as
    `rise_y` over the middle of the forehead. Returns the lift at full weight.

    MakeHuman's cuts all lie close to the head: on the test photos the hair
    stands 1.2 to 1.5 eye spacings above the brows, every stock cut 0.9 to 1.2,
    the quiff of a swept-up cut being the commonest thing missing. The hair is
    pushed out along the scalp's normals, fully over the top and front and
    easing off towards the eyes, the sides and the back (VOLUME_SPAN), and
    from nothing at its front edge (VOLUME_RAMP), so it rises from the hairline
    as a quiff does and keeps its shape. A cut already that high is left alone.
    """
    from scipy.spatial import cKDTree

    smooth = lambda t: (lambda c: c * c * (3.0 - 2.0 * c))(np.clip(t, 0.0, 1.0))
    skin = np.unique(groups["body"])
    eyes = vertices[np.unique(np.r_[groups["joint-l-eye"], groups["joint-r-eye"]])].mean(axis=0)
    crown = vertices[np.unique(groups["joint-head-2"])].mean(axis=0)
    centre = vertices[np.unique(groups["joint-head"])].mean(axis=0)
    scalp = scalp_normals(hair.vertices, vertices, groups)

    (low, high), (full, none) = VOLUME_SPAN
    height = (hair.vertices[:, 1] - eyes[1]) / max(crown[1] - eyes[1], 1e-6)
    angle = np.degrees(np.abs(np.arctan2(hair.vertices[:, 0] - centre[0], hair.vertices[:, 2] - centre[2])))
    middle = np.abs(hair.vertices[:, 0] - centre[0]) < 0.3
    front = middle & (hair.vertices[:, 2] > centre[2])
    if not front.any():
        return 0.0
    edge = np.percentile(hair.vertices[front, 1], 3)
    weight = (smooth((height - low) / (high - low)) * smooth((none - angle) / (none - full))
              * smooth((hair.vertices[:, 1] - edge) / VOLUME_RAMP))

    top = np.percentile(hair.vertices[middle, 1], 98)
    need = rise_y - top
    if need <= 0.0:
        return 0.0
    # The top rises by the lift times how far its normal points up.
    up = float(np.median((weight * scalp[:, 1])[middle & (hair.vertices[:, 1] >= top - 0.05)]))
    lift = need / max(up, 0.3)
    hair.vertices = hair.vertices + scalp * (lift * weight)[:, None]
    return float(lift)


# How much wider than the cut the photo's outline may make it, at most (a
# share of the cut's own width at that height).
WIDEN_LIMIT = 0.2
# A cut's rendered outline against its vertices' (both in eye spacings, the
# render's read as the photo's are): 1.01-1.10 on the four test faces.
HAIR_RENDER_SPREAD = 1.06


def widen(hair: trimesh.Trimesh, vertices: np.ndarray, groups: dict, brow_y: float,
          spacing: float, widths: list) -> np.ndarray:
    """Push a haircut's sides out until its outline, seen from the front, has
    the photo's shape. Returns the push at each of appearance.HAIR_WIDTH_ROWS.

    `widths`: the photo's hair half-width at those heights above the brows
    (`brow_y`), in eye spacings (`spacing`), as read_hair reads them. The
    cut's own outline is measured the same way from its vertices, scaled by
    how much wider a cut renders than its vertices (HAIR_RENDER_SPREAD). On
    the test photos swept-back hair stands wider over the temples than any
    stock cut: up to a fifth wider at 0.75 eye spacings above the brows. The
    sides are pushed out along the scalp's normals, as far as each part faces
    sideways, so the cut keeps its shape from every side; never pulled in,
    which would sink cards into the scalp.

    Matching only the outline's shape (each height against the width at the
    brows) widened a cut whose photo was narrower at the brows than the cut
    all the way up, and on one photo the temples came out 14% too wide.
    """
    from scipy.spatial import cKDTree
    import appearance

    rows = np.array(appearance.HAIR_WIDTH_ROWS)
    if not widths or widths[0] is None or any(w is None for w in widths):
        return np.zeros(len(rows))
    centre = vertices[np.unique(groups["joint-head"])].mean(axis=0)
    v = hair.vertices
    own = []
    for above in rows:
        band = np.abs(v[:, 1] - (brow_y + above * spacing)) < 0.05 * spacing
        if not band.any():
            return np.zeros(len(rows))
        x = v[band, 0] - centre[0]
        own.append((x.max() - x.min()) / 2 / spacing)
    own = np.array(own)
    photo = np.array(widths, dtype=np.float64)
    push = np.clip(photo / HAIR_RENDER_SPREAD - own, 0.0, WIDEN_LIMIT * own) * spacing

    skin = np.unique(groups["body"])
    scalp = scalp_normals(v, vertices, groups)
    height = (v[:, 1] - brow_y) / spacing
    # From the top row up the push holds; below the brows it fades out, so the
    # sideburns and the hair over the ears stay as the cut has them.
    amount = np.interp(height, np.r_[rows[0] - 0.25, rows], np.r_[0.0, push])
    sideways = np.abs(scalp[:, 0])
    hair.vertices = v + scalp * (amount * sideways)[:, None]
    return push


# The hair's outline fitted to the photo's (fit_hair): rays from the cranium's
# centre every HAIR_FIT_STEP degrees, up to HAIR_FIT_SPAN either side of
# straight up; the gaps smoothed round the head over HAIR_FIT_SMOOTH degrees;
# how many rounds; and how close to the skull's outline hair may be pulled in,
# in eye spacings.
HAIR_FIT_SPAN = 100.0
HAIR_FIT_STEP = 4.0
HAIR_FIT_SMOOTH = 12.0
HAIR_FIT_ROUNDS = 3
HAIR_FIT_FLOOR = 0.04
# Pixels per eye spacing the outlines are drawn at.
HAIR_FIT_SCALE = 120.0
# How far the cards' opaque points, drawn, reach past the hair the render's
# segmentation finds, in eye spacings: 0.01-0.06 on the five test faces at
# their photos' poses, the same all round the head.
HAIR_DRAWN_MARGIN = 0.035


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 2x3 rotation, scale and shift taking `src` points onto `dst`."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - ms, dst - md
    u, sig, vt = np.linalg.svd(d.T @ s)
    flip = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ flip @ vt
    k = (sig * np.diag(flip)).sum() / (s ** 2).sum()
    return np.hstack([k * r, (md - k * r @ ms)[:, None]])


def _last_along(mask: np.ndarray, centre: np.ndarray, directions: np.ndarray, length: float) -> np.ndarray:
    """Per direction, how far from `centre` the last pixel of `mask` lies along
    it (0 where it has none)."""
    r = np.arange(0.0, length, 0.5)
    pts = np.round(centre[None, None] + r[None, :, None] * directions[:, None]).astype(int)
    ok = (pts >= 0).all(axis=2) & (pts[..., 0] < mask.shape[1]) & (pts[..., 1] < mask.shape[0])
    inside = np.zeros(ok.shape, bool)
    inside[ok] = mask[pts[ok][:, 1], pts[ok][:, 0]] > 0
    last = np.where(inside.any(axis=1), len(r) - 1 - np.argmax(inside[:, ::-1], axis=1), -1)
    return np.where(last >= 0, r[np.maximum(last, 0)], 0.0)


def _photo_view(person: np.ndarray, rotation: np.ndarray, pixels: np.ndarray):
    """The head seen as the photo saw it: `view` takes head points to a
    picture of the head turned by `rotation`, HAIR_FIT_SCALE pixels per eye
    spacing, and `to_view` (2x3) takes the photo's pixels into that picture by
    a similarity fit of its landmarks (`pixels`) onto the face's (`person`).
    Returns view, to_view, the picture's size, its scale and the eye spacing."""
    centre = person[:468].mean(axis=0)
    spacing = float(np.linalg.norm(person[33] - person[263]))
    scale = HAIR_FIT_SCALE / spacing
    size = int(7 * HAIR_FIT_SCALE)
    view = lambda q: np.stack([((q - centre) @ rotation.T)[:, 0] * scale + size / 2,
                               -((q - centre) @ rotation.T)[:, 1] * scale + size / 2], axis=1)
    to_view = _similarity(np.asarray(pixels[:468], dtype=np.float64), view(person[:468]))
    return view, to_view, size, scale, spacing


def fit_hair(hair: trimesh.Trimesh, vertices: np.ndarray, head_faces: np.ndarray, rotation: np.ndarray,
             person: np.ndarray, pixels: np.ndarray, photo_hair: np.ndarray, opaque) -> np.ndarray:
    """A haircut's outline moved until, seen as the photo saw the head, it runs
    along the photo's segmented hair: the crown's height, the width at the
    temples and over the ears.

    As morph.fit_outline does for the jaw: the head is turned by the photo's
    `rotation` (the render's: compose.photo_head_pose) and drawn, with the cut's opaque points (`opaque(hair)`) drawn
    over it; the photo's hair mask (`photo_hair`) is brought into the same
    picture by a similarity fit of its landmarks (`pixels`) onto the face's
    (`person`, placed on the head). Along rays from the cranium's centre over
    the top and down the sides, each outline's last pixel is found, and the
    hair between the skull's outline and its own is stretched in the picture's
    plane to end on the photo's: points on the skull stay, the outer edge goes
    the whole gap. Never in past the skull's outline plus HAIR_FIT_FLOOR, so
    no card is pulled into the scalp. Rays where the photo's hair runs off the
    picture, or where either has no hair, do not count. Returns the gaps of the
    last round, in eye spacings, per ray.
    """
    import cv2

    view, to_view, size, scale, spacing = _photo_view(person, rotation, pixels)
    # The photo's hair, one piece: the largest connected part, holes closed.
    hair_mask = cv2.morphologyEx(np.asarray(photo_hair, np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask)
    if count < 2:
        return np.zeros(0)
    hair_mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    photo_view = cv2.warpAffine(hair_mask, to_view, (size, size), flags=cv2.INTER_NEAREST)
    in_photo = cv2.warpAffine(np.ones_like(hair_mask), to_view, (size, size), flags=cv2.INTER_NEAREST,
                              borderValue=0)
    in_photo = cv2.erode(in_photo, np.ones((7, 7), np.uint8))

    eyes_y = float(person[[33, 263], 1].mean())
    skull = np.unique(head_faces)
    cranium = view(vertices[skull[vertices[skull, 1] > eyes_y]]).mean(axis=0)
    angles = np.arange(-HAIR_FIT_SPAN, HAIR_FIT_SPAN + 1e-6, HAIR_FIT_STEP)
    directions = np.stack([np.sin(np.radians(angles)), -np.cos(np.radians(angles))], axis=1)
    reach = float(size)
    head_mask = np.zeros((size, size), np.uint8)
    for tri in np.round(view(vertices)[head_faces]).astype(np.int32):
        cv2.fillConvexPoly(head_mask, tri, 1)
    r_head = _last_along(head_mask, cranium, directions, reach)
    r_photo = _last_along(photo_view, cranium, directions, reach)
    beyond = np.round(cranium + (r_photo[:, None] + 4) * directions).astype(int)
    seen = in_photo[np.clip(beyond[:, 1], 0, size - 1), np.clip(beyond[:, 0], 0, size - 1)] > 0
    floor = HAIR_FIT_FLOOR * HAIR_FIT_SCALE
    gaps = np.zeros(len(angles))
    # The scalp under each point, to keep pulled-in hair off it: in the
    # picture it stays outside the skull's outline, but a card over the
    # forehead drawn down in the picture's plane meets the brow in depth.
    normals = trimesh.Trimesh(vertices, head_faces, process=False).vertex_normals
    tree = cKDTree(vertices[skull])
    off_scalp = lambda pts: (lambda near: ((pts - vertices[near]) * normals[near]).sum(axis=1))(
        skull[tree.query(pts)[1]])
    before = off_scalp(hair.vertices)
    for _ in range(HAIR_FIT_ROUNDS):
        drawn = np.zeros((size, size), np.uint8)
        q = np.round(view(opaque(hair))).astype(int)
        q = q[(q >= 0).all(axis=1) & (q < size).all(axis=1)]
        drawn[q[:, 1], q[:, 0]] = 1
        drawn = cv2.morphologyEx(cv2.dilate(drawn, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8))
        r_hair = _last_along(drawn | head_mask, cranium, directions, reach)
        use = seen & (r_photo > 0) & (r_hair > r_head + 1)
        if use.sum() < 5:
            break
        gaps = np.where(use, np.maximum(r_photo + HAIR_DRAWN_MARGIN * HAIR_FIT_SCALE, r_head + floor) - r_hair, 0.0)
        # Smoothed round the head, over the rays that count.
        near = np.exp(-((angles[:, None] - angles[None, :]) / HAIR_FIT_SMOOTH) ** 2) * use[None, :]
        gaps = (near @ gaps) / np.maximum(near.sum(axis=1), 1e-9)
        gaps = np.where(near.sum(axis=1) > 0.5, gaps, 0.0)

        p = view(hair.vertices) - cranium
        radius = np.linalg.norm(p, axis=1)
        theta = np.degrees(np.arctan2(p[:, 0], -p[:, 1]))
        ends = np.r_[angles[0] - 2 * HAIR_FIT_STEP, angles, angles[-1] + 2 * HAIR_FIT_STEP]
        at = lambda values: np.interp(theta, ends, np.r_[values[0], values, values[-1]])
        gap = np.interp(theta, ends, np.r_[0.0, gaps, 0.0])
        inner, outer = at(r_head), at(r_hair)
        share = np.clip((radius - inner) / np.maximum(outer - inner, 1.0), 0.0, 1.0)
        move = (gap * share)[:, None] * p / np.maximum(radius, 1e-9)[:, None]
        hair.vertices = hair.vertices + np.stack([move[:, 0], -move[:, 1], np.zeros(len(move))],
                                                 axis=1) / scale @ rotation
    # Never nearer the scalp than before, nor nearer than the floor.
    keep = np.minimum(before, HAIR_FIT_FLOOR * spacing)
    now = off_scalp(hair.vertices)
    near = skull[tree.query(hair.vertices)[1]]
    hair.vertices = hair.vertices + normals[near] * np.maximum(keep - now, 0.0)[:, None]
    return gaps / HAIR_FIT_SCALE


# The hair combed the way the photo's runs (comb_hair). How far a direction
# read off the photo carries over the head, in eye spacings; how strongly the
# combed-back default holds where the photo says little (against a photo
# reading's full weight of 1); and a painted strand's reach either way along
# the cap's map, in texels of a 1024 map.
COMB_REACH = 0.8
COMB_PRIOR = 0.3
COMB_STRAND = 14
# Cards longer than this (eye spacings) are the cut's flaps at the front and
# temples, not tufts: turned about one root they swung off the head.
COMB_CARD_MAX = 0.6


def _strand_field(photo_rgb: np.ndarray, hair: np.ndarray, eye_px: float):
    """The photo's strand direction (radians, image axes) and how clearly it
    reads (coherence 0-1, zero off the hair and near its outline)."""
    import cv2
    grey = cv2.GaussianBlur(photo_rgb.astype(np.float32).mean(axis=2), (0, 0), max(0.01 * eye_px, 0.6))
    gx, gy = cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1)
    sigma = max(0.06 * eye_px, 2.0)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), sigma) for a in (gx * gx, gy * gy, gx * gy))
    strand = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2      # across the brightness gradient
    clear = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    # Kept off the hair's outline: its edge against background or forehead
    # reads as one long strand along it.
    inner = cv2.erode(hair.astype(np.uint8), np.ones((int(0.25 * eye_px) * 2 + 1,) * 2, np.uint8))
    return strand, clear * inner


def comb_hair(hair: trimesh.Trimesh, vertices: np.ndarray, head_faces: np.ndarray, rotation: np.ndarray,
              person: np.ndarray, pixels: np.ndarray, photo_rgb: np.ndarray, photo_hair: np.ndarray) -> dict:
    """A haircut combed the way the photo's hair runs.

    The stock cuts carry their own parting: short01's cap is painted with
    strands running from the middle of the crown out to the ears (on the
    crown 0.63 of their direction sideways, 0.52 front to back), and its 200
    tuft cards stand out from the scalp every which way. Rendered, the top of
    the head read sideways where the test photos' hair runs up from the
    forehead and over to one side: the strands the render shows lay 68-74
    degrees off the photos' on ref-man, photo-14 and photo-16 (random would
    be 45). Combed, 56, 38 and 66; the identity score at the photo's pose
    0.541 -> 0.562, every face up (level 0.529 -> 0.535, within noise).

    A direction is read at every hair point: where the photo sees it, the
    photo's strand direction there (`_strand_field`), lifted into the scalp's
    tangent plane through the photo's head pose (`_photo_view`); elsewhere,
    and blended in where the photo reads weakly, hair combed back from the
    hairline and down behind. Readings carry over COMB_REACH. Then each card
    is turned about its root, round the scalp's normal, to run that way (its
    height off the scalp kept), and the cap's strands are painted again along
    it, keeping the stock texture's shading at a larger scale and its
    see-through edge. Returns how far the cards turned and how much of the
    photo was read, for the log.

    ponytail: one direction per point, no curl; a cut whose largest piece is
    not a cap (bobs, long) gets its cards combed only.
    """
    import cv2
    from PIL import Image
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from facetexture import rasterize

    view, to_view, size, scale, spacing = _photo_view(person, rotation, pixels)
    eye_px = float(np.linalg.norm(pixels[33] - pixels[263]))
    strand, clear = _strand_field(photo_rgb, photo_hair, eye_px)
    ph, pw = clear.shape

    skull = np.unique(head_faces)
    normals = trimesh.Trimesh(vertices, head_faces, process=False).vertex_normals
    tree = cKDTree(vertices[skull])
    eyes_y = float(person[[33, 263], 1].mean())
    cranium = vertices[skull[vertices[skull, 1] > eyes_y]]
    centre = np.array([0.0, cranium[:, 1].mean(), (cranium[:, 2].max() + cranium[:, 2].min()) / 2.0])
    radius = float(np.median(np.linalg.norm(cranium - centre, axis=1)))

    def normal_at(points):
        return normals[skull[tree.query(points)[1]]]

    def tangent(d, n):
        d = d - (d * n).sum(axis=1, keepdims=True) * n
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    sweep = 0.0

    def combed_back(points, n):
        # Away from a point ahead of the face: up and back from the hairline,
        # back over the crown and above the ears; down behind. Over the top,
        # to the side the photo's hair goes (`sweep`, + towards the head's left).
        q = points - centre
        away = q - np.array([0.0, -0.3 * radius, 1.2 * radius])
        behind = np.clip(-q[:, 2] / radius, 0.0, 1.0)
        top = np.clip(q[:, 1] / radius, 0.0, 1.0)
        return tangent(away / radius - np.array([0.0, 2.0, 0.0]) * behind[:, None]
                       + np.array([sweep, 0.0, 0.0]) * top[:, None], n)

    # The photo's reading at each hair vertex.
    points = hair.vertices
    n = normal_at(points)
    in_view = view(points)
    inverse = np.linalg.inv(to_view[:, :2])
    photo_px = (in_view - to_view[:, 2]) @ inverse.T
    xs, ys = np.round(photo_px[:, 0]).astype(int), np.round(photo_px[:, 1]).astype(int)
    inside = (xs >= 0) & (xs < pw) & (ys >= 0) & (ys < ph)
    weight = np.zeros(len(points))
    weight[inside] = clear[ys[inside], xs[inside]]
    facing = np.clip((n @ rotation.T)[:, 2], 0.0, 1.0)
    weight *= facing ** 2
    angle = np.zeros(len(points))
    angle[inside] = strand[ys[inside], xs[inside]]
    seen = to_view[:, :2] @ np.stack([np.cos(angle), np.sin(angle)])          # (2, N) in the picture
    # Lifted to the tangent plane: the direction there whose picture runs along it.
    helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
    e1 = np.cross(n, helper); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    drawn = lambda e: np.stack([(e @ rotation.T)[:, 0], -(e @ rotation.T)[:, 1]], axis=1)
    cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    d1, d2 = drawn(e1), drawn(e2)
    lifted = cross(d2, seen.T)[:, None] * e1 - cross(d1, seen.T)[:, None] * e2
    lifted /= np.maximum(np.linalg.norm(lifted, axis=1, keepdims=True), 1e-9)
    # A strand read off the photo is a line, not an arrow. Those that clearly
    # rise point away from the hairline; the side they lean to is the sweep,
    # and strands lying across the head are turned to go with it (matched to
    # the default alone, they went either way and the sweep came out mirrored).
    default = combed_back(points, n)
    lifted *= np.sign((lifted * default).sum(axis=1) + 1e-9)[:, None]
    rising = weight * (np.abs((lifted * default).sum(axis=1)) > 0.5)
    sweep = float((rising * lifted[:, 0]).sum() / max(rising.sum(), 1e-9))
    default = combed_back(points, n)
    lifted *= np.sign((lifted * default).sum(axis=1) + 1e-9)[:, None]       # growing away from the hairline
    sources = np.nonzero(weight > 0.05)[0]
    source_tree = cKDTree(points[sources]) if len(sources) else None
    reach = COMB_REACH * spacing

    def direction(at):
        """The combed direction at head points `at`."""
        nn = normal_at(at)
        out = COMB_PRIOR * combed_back(at, nn)
        if source_tree is not None:
            for i, near in enumerate(source_tree.query_ball_point(at, 2.0 * reach)):
                if near:
                    k = sources[near]
                    g = weight[k] * np.exp(-((points[k] - at[i]) ** 2).sum(axis=1) / (2 * reach ** 2))
                    out[i] += (g[:, None] * lifted[k]).sum(axis=0)
        return tangent(out, nn)

    # The pieces: cards and, if there is one, the cap.
    edges = hair.edges
    count, label = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                                   shape=(len(points),) * 2), directed=False)
    sizes = np.bincount(label)
    # The cap: one piece many times any card (short01's is 1138 vertices, its
    # largest card 28); cuts made of cards alone have none.
    ordered = np.sort(sizes)[::-1]
    cap = int(np.argmax(sizes)) if len(ordered) > 1 and ordered[0] > 10 * ordered[1] else -1

    v = np.asarray(hair.vertices).copy()
    turns = []
    cards = [c for c in range(count) if c != cap]
    roots, axes, members = [], [], []
    for c in cards:
        idx = np.nonzero(label == c)[0]
        if np.linalg.norm(np.ptp(v[idx], axis=0)) > COMB_CARD_MAX * spacing:
            continue
        r = np.linalg.norm(v[idx] - centre, axis=1)
        roots.append(v[idx[r.argmin()]]); axes.append(v[idx[r.argmax()]] - v[idx[r.argmin()]])
        members.append(idx)
    roots, axes = np.array(roots).reshape(-1, 3), np.array(axes).reshape(-1, 3)
    if len(members):
        want = direction(roots)
        n_root = normal_at(roots)
        lying = tangent(axes, n_root)
        for j, idx in enumerate(members):
            k = n_root[j]
            phi = np.arctan2(np.dot(k, np.cross(lying[j], want[j])), np.dot(lying[j], want[j]))
            # Rodrigues, about the scalp's normal through the card's root.
            p = v[idx] - roots[j]
            v[idx] = roots[j] + (p * np.cos(phi) + np.cross(k, p) * np.sin(phi)
                                 + np.outer(p @ k, k) * (1 - np.cos(phi)))
            turns.append(abs(np.degrees(phi)))
    hair.vertices = v

    texture = hair.metadata.get("texture")
    uv = hair.metadata.get("uv")
    if cap >= 0 and texture is not None and uv is not None:
        cap_faces = hair.faces[label[hair.faces[:, 0]] == cap]
        hair.metadata["texture"], normal = _paint_cap(texture, hair.metadata.get("normal_texture"), v, cap_faces, uv,
                                                       direction, rasterize)
        if normal is not None:
            hair.metadata["normal_texture"] = normal
    return {"cards_combed": len(members), "cards_turned_deg": round(float(np.median(turns)), 1) if turns else 0.0,
            "photo_points": int(len(sources)), "sweep": round(sweep, 2)}


def _paint_cap(texture, normal_texture, v: np.ndarray, faces: np.ndarray, uv: np.ndarray, direction, rasterize):
    """The cap's strands painted again along `direction`: line integral
    convolution of fine noise along the direction carried into the map
    through each triangle's texture coordinates. The stock texture's shading
    at a few texels' scale stays, and its edge, where the alpha thins out into
    strands, is left as it was."""
    import cv2
    from PIL import Image

    rgba = np.asarray(texture.convert("RGBA")).astype(np.float32)
    size = rgba.shape[0]
    to_px = lambda t: np.stack([t[:, 0] * (size - 1), (1.0 - t[:, 1]) * (size - 1)], axis=1)
    # Per vertex, the direction on the map, as a doubled angle so opposite
    # directions agree when averaged across a vertex's triangles.
    want = direction(v[np.unique(faces)])
    at_vertex = np.zeros((len(v), 3)); at_vertex[np.unique(faces)] = want
    c2, s2, w = np.zeros(len(v)), np.zeros(len(v)), np.zeros(len(v))
    for f in faces:
        e3 = (v[f[1:]] - v[f[0]]).T                               # 3x2
        t = to_px(uv[f]); e2 = (t[1:] - t[0]).T                   # 2x2
        if abs(np.linalg.det(e2)) < 1e-9:
            continue
        d = at_vertex[f].mean(axis=0)
        a = np.linalg.lstsq(e3, d, rcond=None)[0]
        m = e2 @ a
        ang = np.arctan2(m[1], m[0])
        c2[f] += np.cos(2 * ang); s2[f] += np.sin(2 * ang); w[f] += 1
    c2, s2 = c2 / np.maximum(w, 1), s2 / np.maximum(w, 1)
    covered = rasterize(np.ones(len(v)), faces, uv, faces, size, fill=0.0) > 0.5
    fc = rasterize(c2, faces, uv, faces, size, fill=0.0).astype(np.float32)
    fs = rasterize(s2, faces, uv, faces, size, fill=0.0).astype(np.float32)

    # Fine noise streaked along the field, both ways from each texel.
    rng = np.random.default_rng(7)
    noise = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 0.6)
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float32)
    steps = max(int(round(COMB_STRAND * size / 1024)), 2)
    total = noise.copy()
    for sign in (1.0, -1.0):
        px, py = gx.copy(), gy.copy()
        prev = None
        for _ in range(steps):
            c = cv2.remap(fc, px, py, cv2.INTER_LINEAR); s = cv2.remap(fs, px, py, cv2.INTER_LINEAR)
            ang = 0.5 * np.arctan2(s, c)
            dx, dy = np.cos(ang), np.sin(ang)
            if prev is None:
                dx, dy = sign * dx, sign * dy
            else:
                flip = np.where(dx * prev[0] + dy * prev[1] < 0, -1.0, 1.0)
                dx, dy = dx * flip, dy * flip
            prev = (dx, dy)
            px, py = (px + dx).astype(np.float32), (py + dy).astype(np.float32)
            total += cv2.remap(noise, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    streaks = total - cv2.GaussianBlur(total, (0, 0), 3.0)

    luma = np.array([0.299, 0.587, 0.114], np.float32)
    bright = rgba[..., :3] @ luma
    core = covered & (rgba[..., 3] > 250)
    core = cv2.erode(core.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if core.sum() < 100:
        return texture, None
    colour = rgba[..., :3][core].sum(axis=0) / max(float(bright[core].sum()), 1.0)    # per unit brightness
    # Shading at a few texels' scale from the stock map, inside the cap only.
    inside = core.astype(np.float32)
    low = cv2.GaussianBlur(bright * inside, (0, 0), 4.0) / np.maximum(cv2.GaussianBlur(inside, (0, 0), 4.0), 1e-6)
    detail = (bright - low)[core].std()
    new_bright = np.clip(low + detail * streaks / max(float(streaks[core].std()), 1e-6), 0, 255)
    blend = cv2.GaussianBlur(inside, (0, 0), 2.0)[..., None]
    rgb = rgba[..., :3] * (1 - blend) + (new_bright[..., None] * colour[None, None, :]) * blend
    out = np.dstack([np.clip(rgb, 0, 255), rgba[..., 3]]).astype(np.uint8)
    painted = Image.fromarray(out, "RGBA")
    if texture.mode == "P":
        painted = painted.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)

    normal = None
    if normal_texture is not None:
        # The strands' relief again from the new strands, as deep as the stock
        # map's over the cap.
        old = np.asarray(normal_texture.convert("RGB").resize((size, size))).astype(np.float32) / 127.5 - 1.0
        slope = np.hypot(old[..., 0], old[..., 1])[core].std()
        height = cv2.GaussianBlur(new_bright / 255.0, (0, 0), 1.0)
        dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3); dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
        depth = slope / max(float(np.hypot(dx, dy)[core].std()), 1e-6)
        n = np.dstack([-dx * depth, dy * depth, np.ones_like(height)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        new = np.where(blend > 0.5, n, old)
        new /= np.linalg.norm(new, axis=2, keepdims=True)
        normal = Image.fromarray(((new * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")
        if normal.size != normal_texture.size:
            normal = normal.resize(normal_texture.size)
    return painted, normal


def enclosed_by_shoe(shoes: trimesh.Trimesh, points: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Which points a pair of shoes encloses: within their footprint and below
    the lowest point of the opening the foot goes in by, less `margin`.

    The opening is told from the sole's own open edges by height: it is the
    boundary in the shoe's upper part.
    """
    unique, counts = np.unique(shoes.edges_sorted, axis=0, return_counts=True)
    rim = shoes.vertices[np.unique(unique[counts == 1])]
    low, high = shoes.vertices[:, 1].min(), shoes.vertices[:, 1].max()
    collar = rim[rim[:, 1] > low + 0.4 * (high - low)]
    if not len(collar):
        return np.zeros(len(points), bool)
    lo, hi = shoes.vertices[:, [0, 2]].min(axis=0), shoes.vertices[:, [0, 2]].max(axis=0)
    within = np.all((points[:, [0, 2]] >= lo) & (points[:, [0, 2]] <= hi), axis=1)
    return within & (points[:, 1] < collar[:, 1].min() - margin)


def shorten_lower_lashes(lashes: trimesh.Trimesh, keep: float = 0.45) -> None:
    """Pull the lower lashes in towards the lid, to `keep` of their length.

    MakeHuman's lower lash cards hang about 5 mm, as long as the upper ones,
    and at render scale they read as separate spikes under the eye. Real lower
    lashes are short and sparse. Each piece below the eye's middle is drawn
    towards its root, the top of the piece where it meets the lid.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(lashes.vertices)
    edges = lashes.edges
    _, label = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                               shape=(n, n)), directed=False)
    v = np.asarray(lashes.vertices).copy()
    middle = v[:, 1].mean()
    for piece in np.unique(label):
        rows = label == piece
        p = v[rows]
        if p[:, 1].mean() >= middle:
            continue                                # an upper lash
        root = p[p[:, 1] >= np.percentile(p[:, 1], 70)]
        nearest = root[cKDTree(root).query(p)[1]]
        v[rows] = nearest + (p - nearest) * keep
    lashes.vertices = v


def strand_normals(rgb: np.ndarray, depth: float = 2.0):
    """A tangent-space normal map from a hair texture's own brightness.

    Only one of MakeHuman's haircuts ships a normal map. Light strands sit
    proud of dark ones, so the texture's lightness, lightly blurred, serves as
    a height; its slopes give the relief that catches the light per strand.
    """
    import cv2
    from PIL import Image
    height = cv2.GaussianBlur(rgb.mean(axis=2) / 255.0, (0, 0), 1.0)
    dx = cv2.Sobel(height, cv2.CV_64F, 1, 0, ksize=3) * depth
    dy = cv2.Sobel(height, cv2.CV_64F, 0, 1, ksize=3) * depth
    n = np.dstack([-dx, dy, np.ones_like(height)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")



def open_rim(mesh: trimesh.Trimesh, weld: float = 0.03) -> np.ndarray:
    """Vertices on a garment's real openings: neck, cuffs, waist, hem.

    An open edge of the mesh is not always an opening: a garment sewn from
    separate pieces has open edges along every seam, where another piece's
    edge runs alongside. Those are left out: an edge vertex counts only if no
    vertex of another piece lies within `weld`. Treated as openings, the
    blouse's armhole seams kept a band of skin showing through its sides.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    unique, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    rim = np.unique(unique[counts == 1])
    if not len(rim):
        return rim
    n = len(mesh.vertices)
    edges = mesh.edges
    _, piece = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)),
                                    directed=False)
    tree = cKDTree(mesh.vertices[rim])
    keep = np.ones(len(rim), bool)
    for i, v in enumerate(rim):
        others = [j for j in tree.query_ball_point(mesh.vertices[v], weld) if piece[rim[j]] != piece[v]]
        keep[i] = not others
    return rim[keep]
