"""The skeleton: bones, and which bone moves which vertex.

Without this an avatar is a statue. With it, it can be posed and animated, which
is what the Unity and Unreal SDKs need.

Both inputs are MakeHuman's, CC0 1.0, and authored against the exact base mesh we
dress:

  assets/default.mhskel        bone hierarchy; each bone's head and tail are named
                               joints, and a joint is a cluster of base-mesh
                               vertices whose centroid is the position
  assets/default_weights.mhw   per-bone lists of (vertex index, weight)

Because every wearable in the catalogue is cut from that same base mesh, its
vertices carry the same indices, so one weight table skins the body and the
clothes alike. The fitted face is the exception: it is our own geometry, so it
binds rigidly to the head bone.

ponytail: bones are translation-only, with no roll from the rotation planes. That
is enough to deform correctly and to retarget simple poses; authored animation
that assumes MakeHuman's bone axes would need the planes read too.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SKELETON_JSON = HERE / "assets" / "default.mhskel"
WEIGHTS_JSON = HERE / "assets" / "default_weights.mhw"

MAX_INFLUENCES = 4  # what glTF's JOINTS_0/WEIGHTS_0 pair carries

_weights = None


class Skeleton:
    def __init__(self, names, parents, heads, tails):
        self.names = names                 # bone name per index
        self.parents = parents             # parent index per bone, -1 for a root
        self.heads = heads                 # world-space head position per bone
        self.tails = tails
        self.index = {n: i for i, n in enumerate(names)}

    def __len__(self):
        return len(self.names)

    def local_translations(self) -> np.ndarray:
        """Each bone's offset from its parent, which is what a glTF node stores."""
        out = self.heads.copy()
        for i, parent in enumerate(self.parents):
            if parent >= 0:
                out[i] = self.heads[i] - self.heads[parent]
        return out

    def inverse_bind_matrices(self) -> np.ndarray:
        """One 4x4 per bone, undoing its bind-pose world transform.

        Bind pose has no rotation here, so the inverse is just a negative
        translation.
        """
        mats = np.tile(np.eye(4, dtype=np.float32), (len(self), 1, 1))
        mats[:, :3, 3] = -self.heads
        return mats


def load_skeleton(vertices: np.ndarray) -> Skeleton:
    """Bone hierarchy with joint positions resolved against the given mesh.

    Not cached: once the head is shaped to the person, the eye and jaw joints
    move with it, so joints must come from the shaped mesh every time. It is a
    few hundred small averages, so recomputing costs nothing.
    """

    data = json.loads(SKELETON_JSON.read_text(encoding="utf-8"))
    joints, bones = data["joints"], data["bones"]

    def position(joint_name):
        idx = joints.get(joint_name)
        if not idx:
            return np.zeros(3)
        return vertices[np.asarray(idx, dtype=np.int64)].mean(axis=0)

    # Parents must come before children so a node's local translation can be
    # worked out in one pass, and glTF readers like a clean tree.
    ordered, seen = [], set()

    def visit(name):
        if name in seen:
            return
        parent = bones[name].get("parent")
        if parent and parent in bones:
            visit(parent)
        seen.add(name)
        ordered.append(name)

    for name in bones:
        visit(name)

    index = {n: i for i, n in enumerate(ordered)}
    parents = np.array(
        [index.get(bones[n].get("parent") or "", -1) for n in ordered], dtype=np.int64
    )
    heads = np.array([position(bones[n]["head"]) for n in ordered], dtype=np.float64)
    tails = np.array([position(bones[n]["tail"]) for n in ordered], dtype=np.float64)

    return Skeleton(ordered, parents, heads, tails)


def load_weights() -> dict[str, list]:
    global _weights
    if _weights is None:
        _weights = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8"))["weights"]
    return _weights


def weight_table(skeleton: Skeleton, vertex_count: int):
    """Per base-mesh vertex, the bones that move it and by how much.

    Returns (joints, weights), each vertex_count x 4, already normalised. A
    vertex nobody claims falls back to the root bone so it still follows the
    figure instead of being left behind at the origin.
    """
    per_vertex: list[list[tuple[int, float]]] = [[] for _ in range(vertex_count)]
    for bone_name, entries in load_weights().items():
        bone = skeleton.index.get(bone_name)
        if bone is None:
            continue
        for vertex, weight in entries:
            if 0 <= vertex < vertex_count and weight > 0:
                per_vertex[vertex].append((bone, float(weight)))

    joints = np.zeros((vertex_count, MAX_INFLUENCES), dtype=np.uint16)
    weights = np.zeros((vertex_count, MAX_INFLUENCES), dtype=np.float32)
    root = int(np.where(skeleton.parents < 0)[0][0]) if (skeleton.parents < 0).any() else 0

    for v, influences in enumerate(per_vertex):
        if not influences:
            joints[v, 0], weights[v, 0] = root, 1.0
            continue
        influences.sort(key=lambda iw: -iw[1])
        influences = influences[:MAX_INFLUENCES]
        total = sum(w for _, w in influences)
        for slot, (bone, weight) in enumerate(influences):
            joints[v, slot] = bone
            weights[v, slot] = weight / total

    return joints, weights


def rigid_binding(count: int, bone: int):
    """Everything bound to one bone: what the fitted face needs."""
    joints = np.zeros((count, MAX_INFLUENCES), dtype=np.uint16)
    weights = np.zeros((count, MAX_INFLUENCES), dtype=np.float32)
    joints[:, 0] = bone
    weights[:, 0] = 1.0
    return joints, weights
