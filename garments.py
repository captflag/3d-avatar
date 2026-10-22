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
               hairline_y: float, floor: float = 0.05) -> float:
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
    """
    from scipy.spatial import cKDTree

    skin = np.unique(groups["body"])
    eyes = vertices[np.unique(np.r_[groups["joint-l-eye"], groups["joint-r-eye"]])].mean(axis=0)
    cranium = vertices[skin[vertices[skin, 1] > eyes[1]]]
    centre = np.array([0.0, cranium[:, 1].mean(), (cranium[:, 2].max() + cranium[:, 2].min()) / 2.0])

    # The hair's lowest visible points straight ahead (within 15 degrees).
    rel = edge - centre
    ahead = np.degrees(np.abs(np.arctan2(rel[:, 0], rel[:, 2]))) < 15.0
    if ahead.sum() < 10:
        return 0.0
    edge, rel = edge[ahead], rel[ahead]
    radius = float(np.median(np.hypot(rel[:, 1], rel[:, 2])))
    lift = np.arcsin(np.clip((hairline_y - centre[1]) / radius, -1.0, 1.0))
    now = np.arcsin(np.clip((np.percentile(edge[:, 1], 3) - centre[1]) / radius, -1.0, 1.0))
    turn = float(lift - now)
    if turn <= 0.0:
        return 0.0

    p = hair.vertices - centre
    angle = np.degrees(np.abs(np.arctan2(p[:, 0], p[:, 2])))
    t = np.clip((SWEEP_SPAN[1] - angle) / (SWEEP_SPAN[1] - SWEEP_SPAN[0]), 0.0, 1.0)
    a = turn * t * t * (3.0 - 2.0 * t)
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
    normals = trimesh.Trimesh(vertices, groups["body"], process=False).vertex_normals
    near = skin[cKDTree(vertices[skin]).query(hair.vertices)[1]]

    (low, high), (full, none) = VOLUME_SPAN
    height = (vertices[near, 1] - eyes[1]) / max(crown[1] - eyes[1], 1e-6)
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
    up = float(np.median((weight * normals[near, 1])[middle & (hair.vertices[:, 1] >= top - 0.05)]))
    lift = need / max(up, 0.3)
    hair.vertices = hair.vertices + normals[near] * (lift * weight)[:, None]
    return float(lift)


# How much wider than the cut the photo's outline may make it, at most (a
# share of the cut's own width at that height).
WIDEN_LIMIT = 0.2


def widen(hair: trimesh.Trimesh, vertices: np.ndarray, groups: dict, brow_y: float,
          spacing: float, widths: list) -> np.ndarray:
    """Push a haircut's sides out until its outline, seen from the front, has
    the photo's shape. Returns the push at each of appearance.HAIR_WIDTH_ROWS.

    `widths`: the photo's hair half-width at those heights above the brows
    (`brow_y`), in eye spacings (`spacing`), as read_hair reads them. The
    cut's own outline is measured the same way from its vertices, and only
    its shape is matched (each height's width against the width at the brows),
    since alpha-faded strand tips make a rendered cut 0-10% narrower than its
    vertices, by an amount that varies with the cut. On the test photos
    swept-back hair stands wider over the temples than any stock cut: up to
    a third wider at 0.75 eye spacings above the brows. The sides are pushed
    out along the scalp's normals, as far as each part faces sideways, so the
    cut keeps its shape from every side; never pulled in, which would sink
    cards into the scalp.
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
    wider = np.clip((photo / photo[0]) / (own / own[0]) - 1.0, 0.0, WIDEN_LIMIT)
    push = wider * own * spacing

    skin = np.unique(groups["body"])
    normals = trimesh.Trimesh(vertices, groups["body"], process=False).vertex_normals
    near = skin[cKDTree(vertices[skin]).query(v)[1]]
    height = (v[:, 1] - brow_y) / spacing
    # From the top row up the push holds; below the brows it fades out, so the
    # sideburns and the hair over the ears stay as the cut has them.
    amount = np.interp(height, np.r_[rows[0] - 0.25, rows], np.r_[0.0, push])
    sideways = np.abs(normals[near, 0])
    hair.vertices = v + normals[near] * (amount * sideways)[:, None]
    return push


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
