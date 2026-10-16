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
    """
    obj_file, scales, refs = None, {}, []
    in_verts = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if in_verts:
            if parts[0].lstrip("-").isdigit():
                if len(parts) >= 9:
                    refs.append([float(x) for x in parts[:9]])
                elif len(parts) == 1:            # pinned exactly to one vertex
                    i = float(parts[0])
                    refs.append([i, i, i, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            in_verts = False                     # a new section begins
        if parts[0] == "verts":
            in_verts = True
        elif parts[0] == "obj_file":
            obj_file = parts[1]
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

    faces = []
    for raw in (path.parent / obj_file).read_text(encoding="utf-8").splitlines():
        if raw.startswith("f "):
            corner = [int(c.split("/")[0]) - 1 for c in raw.split()[1:]]
            for i in range(1, len(corner) - 1):
                faces.append((corner[0], corner[i], corner[i + 1]))
    faces = np.array(faces, dtype=np.int64)
    if len(fitted) != faces.max() + 1:
        raise ValueError(f"{path.name}: {len(fitted)} fitted vertices but the mesh uses {faces.max() + 1}")

    mesh = trimesh.Trimesh(vertices=fitted, faces=faces, process=False)
    mesh.metadata["mhclo_idx"] = idx
    mesh.metadata["mhclo_blend"] = blend
    return mesh


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
