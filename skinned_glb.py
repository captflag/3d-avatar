"""Write a skinned GLB: meshes, a skeleton, and the weights that tie them together.

trimesh exports glTF but not skins, so the file is assembled here. Everything the
avatar is made of becomes one mesh with several primitives — body, face, each
garment — sharing one skin, because a single skin keeps the whole figure moving
together and is what engines expect to import.

ponytail: one buffer, no draco, no texture reuse across primitives. Fine at this
size; revisit if avatars grow past a few MB.
"""

from __future__ import annotations

import io
import pathlib
import struct

import numpy as np
import pygltflib as gl

COMPONENT = {
    np.dtype(np.float32): gl.FLOAT,
    np.dtype(np.uint32): gl.UNSIGNED_INT,
    np.dtype(np.uint16): gl.UNSIGNED_SHORT,
    np.dtype(np.uint8): gl.UNSIGNED_BYTE,
}
TYPE_FOR_WIDTH = {1: gl.SCALAR, 2: gl.VEC2, 3: gl.VEC3, 4: gl.VEC4, 16: gl.MAT4}


class Part:
    """One drawable piece: geometry, how it is painted, and how it is skinned."""

    def __init__(self, name, vertices, faces, joints, weights,
                 colour=None, uv=None, texture=None, normals=None):
        self.name = name
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)
        self.joints = np.asarray(joints, dtype=np.uint16)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.colour = None if colour is None else np.asarray(colour, dtype=np.float32)
        self.uv = None if uv is None else np.asarray(uv, dtype=np.float32)
        self.texture = texture          # a PIL image, or None
        self.normals = None if normals is None else np.asarray(normals, dtype=np.float32)


class _Buffer:
    """Accumulates binary data and hands back accessor indices."""

    def __init__(self):
        self.blob = bytearray()
        self.views: list[gl.BufferView] = []
        self.accessors: list[gl.Accessor] = []

    def _pad(self):
        while len(self.blob) % 4:
            self.blob.append(0)

    def add_view(self, data: bytes, target=None) -> int:
        self._pad()
        offset = len(self.blob)
        self.blob.extend(data)
        self.views.append(gl.BufferView(buffer=0, byteOffset=offset,
                                        byteLength=len(data), target=target))
        return len(self.views) - 1

    def add(self, array: np.ndarray, target=None, minmax=False) -> int:
        array = np.ascontiguousarray(array)
        width = 1 if array.ndim == 1 else array.shape[1]
        view = self.add_view(array.tobytes(), target)
        accessor = gl.Accessor(
            bufferView=view,
            componentType=COMPONENT[array.dtype],
            count=int(array.shape[0]),
            type=TYPE_FOR_WIDTH[width],
            normalized=False,
        )
        if minmax:
            accessor.min = array.min(axis=0).tolist()
            accessor.max = array.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def export(parts: list[Part], skeleton, out_path: pathlib.Path) -> None:
    """Write the avatar and its skeleton to `out_path` as a GLB."""
    buf = _Buffer()
    gltf = gl.GLTF2()
    gltf.asset = gl.Asset(version="2.0", generator="avatar-engine")

    # --- the skeleton: one node per bone, children hanging off their parent ---
    translations = skeleton.local_translations()
    bone_node_start = 0  # bone nodes are written first, so index == bone index
    for i, name in enumerate(skeleton.names):
        gltf.nodes.append(gl.Node(name=name,
                                  translation=[float(x) for x in translations[i]],
                                  children=[]))
    for i, parent in enumerate(skeleton.parents):
        if parent >= 0:
            gltf.nodes[int(parent)].children.append(i)
    roots = [i for i, p in enumerate(skeleton.parents) if p < 0]

    ibm = np.ascontiguousarray(
        skeleton.inverse_bind_matrices().transpose(0, 2, 1).reshape(-1, 16).astype(np.float32)
    )
    ibm_accessor = buf.add(ibm)
    gltf.skins.append(gl.Skin(inverseBindMatrices=ibm_accessor,
                              joints=list(range(len(skeleton.names))),
                              skeleton=roots[0] if roots else None))

    # --- the geometry ---
    mesh = gl.Mesh(name="avatar", primitives=[])
    for part in parts:
        attributes = {
            "POSITION": buf.add(part.vertices, gl.ARRAY_BUFFER, minmax=True),
            "JOINTS_0": buf.add(part.joints, gl.ARRAY_BUFFER),
            "WEIGHTS_0": buf.add(part.weights, gl.ARRAY_BUFFER),
        }
        if part.normals is not None:
            attributes["NORMAL"] = buf.add(part.normals, gl.ARRAY_BUFFER)
        if part.colour is not None:
            attributes["COLOR_0"] = buf.add(part.colour, gl.ARRAY_BUFFER)
        if part.uv is not None:
            attributes["TEXCOORD_0"] = buf.add(part.uv, gl.ARRAY_BUFFER)

        indices = buf.add(part.faces.reshape(-1), gl.ELEMENT_ARRAY_BUFFER)

        material = gl.Material(
            name=part.name,
            pbrMetallicRoughness=gl.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.85,
            ),
            doubleSided=True,
        )
        if part.texture is not None:
            png = io.BytesIO()
            part.texture.convert("RGB").save(png, format="PNG", optimize=True)
            view = buf.add_view(png.getvalue())
            gltf.images.append(gl.Image(mimeType="image/png", bufferView=view))
            gltf.samplers.append(gl.Sampler(magFilter=gl.LINEAR, minFilter=gl.LINEAR))
            gltf.textures.append(gl.Texture(source=len(gltf.images) - 1,
                                            sampler=len(gltf.samplers) - 1))
            material.pbrMetallicRoughness.baseColorTexture = gl.TextureInfo(
                index=len(gltf.textures) - 1
            )

        gltf.materials.append(material)
        mesh.primitives.append(gl.Primitive(attributes=gl.Attributes(**attributes),
                                            indices=indices,
                                            material=len(gltf.materials) - 1))

    gltf.meshes.append(mesh)
    mesh_node = len(gltf.nodes)
    gltf.nodes.append(gl.Node(name="avatar", mesh=0, skin=0))

    gltf.bufferViews = buf.views
    gltf.accessors = buf.accessors
    gltf.scenes.append(gl.Scene(nodes=roots + [mesh_node]))
    gltf.scene = 0

    blob = bytes(buf.blob)
    gltf.buffers.append(gl.Buffer(byteLength=len(blob)))
    gltf.set_binary_blob(blob)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(out_path))
