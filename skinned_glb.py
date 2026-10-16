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
                 colour=None, uv=None, texture=None, normals=None, normal_texture=None,
                 alpha_mask: bool = False, roughness: float = 0.85, normal_scale: float = 1.0):
        self.name = name
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)
        self.joints = np.asarray(joints, dtype=np.uint16)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.colour = None if colour is None else np.asarray(colour, dtype=np.float32)
        self.uv = None if uv is None else np.asarray(uv, dtype=np.float32)
        self.texture = texture          # a PIL image, or None
        self.normal_texture = normal_texture  # a PIL image in tangent space, or None
        self.alpha_mask = alpha_mask    # texture alpha below 0.5 is cut away
        self.roughness = roughness      # written into the file, so every engine agrees
        self.normal_scale = normal_scale
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

    def add(self, array: np.ndarray, target=None, minmax=False, normalized=False) -> int:
        array = np.ascontiguousarray(array)
        width = 1 if array.ndim == 1 else array.shape[1]
        view = self.add_view(array.tobytes(), target)
        accessor = gl.Accessor(
            bufferView=view,
            componentType=COMPONENT[array.dtype],
            count=int(array.shape[0]),
            type=TYPE_FOR_WIDTH[width],
            normalized=normalized,
        )
        if minmax:
            accessor.min = array.min(axis=0).tolist()
            accessor.max = array.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def axis_angle(axis, degrees: float):
    """A glTF rotation quaternion [x, y, z, w]."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = np.radians(degrees) / 2.0
    xyz = axis * np.sin(half)
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(np.cos(half))]


def _quantise_weights(weights: np.ndarray) -> np.ndarray:
    """Skin weights as normalised bytes that still sum to exactly 255.

    glTF allows weights as unsigned bytes, a quarter the size of floats. Plain
    rounding can leave a vertex summing to 253 or 257, and the spec wants a sum
    of one, so the rounding remainder goes onto each vertex's strongest bone.
    """
    q = np.rint(np.clip(weights, 0.0, 1.0) * 255.0).astype(np.int32)
    remainder = 255 - q.sum(axis=1)
    strongest = q.argmax(axis=1)
    q[np.arange(len(q)), strongest] += remainder
    return np.clip(q, 0, 255).astype(np.uint8)


def _add_texture(gltf, buf, image, quality: int = 90, _seen: dict | None = None) -> int:
    """Embed an image and return its texture index.

    The same image object passed twice is embedded once: head, body and face
    share one skin-grain map, and embedding it per part tripled its cost.

    JPEG, not PNG: skin and photo textures are noisy by nature and PNG barely
    compresses noise, so a PNG skin map alone ran to several megabytes.
    """
    if _seen is not None and id(image) in _seen:
        return _seen[id(image)]
    data = io.BytesIO()
    if image.mode == "RGBA":
        # JPEG has no alpha channel, and an alpha-masked texture needs one.
        image.save(data, format="PNG", optimize=True)
        mime = "image/png"
    else:
        image.convert("RGB").save(data, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    view = buf.add_view(data.getvalue())
    gltf.images.append(gl.Image(mimeType=mime, bufferView=view))
    gltf.samplers.append(gl.Sampler(magFilter=gl.LINEAR, minFilter=gl.LINEAR_MIPMAP_LINEAR))
    gltf.textures.append(gl.Texture(source=len(gltf.images) - 1, sampler=len(gltf.samplers) - 1))
    if _seen is not None:
        _seen[id(image)] = len(gltf.textures) - 1
    return len(gltf.textures) - 1


def export(parts: list[Part], skeleton, out_path: pathlib.Path,
           pose: dict[str, list[float]] | None = None) -> None:
    """Write the avatar and its skeleton to `out_path` as a GLB.

    `pose` rotates named bones in the rest pose. The base mesh is modelled in a
    mannequin stance, arms out and fingers splayed, which is the first thing that
    says 'not a person'.
    """
    buf = _Buffer()
    gltf = gl.GLTF2()
    gltf.asset = gl.Asset(version="2.0", generator="avatar-engine")

    # --- the skeleton: one node per bone, children hanging off their parent ---
    translations = skeleton.local_translations()
    bone_node_start = 0  # bone nodes are written first, so index == bone index
    pose = pose or {}
    for i, name in enumerate(skeleton.names):
        node = gl.Node(name=name,
                       translation=[float(x) for x in translations[i]],
                       children=[])
        if name in pose:
            node.rotation = pose[name]
        gltf.nodes.append(node)
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
    embedded: dict = {}
    for part in parts:
        # Compact encodings the spec allows: byte joints while there are fewer
        # than 256 bones, byte weights and colours, 16-bit indices while there
        # are fewer than 65,536 vertices. Together they roughly halve the file.
        joints = part.joints.astype(np.uint8 if len(skeleton.names) < 256 else np.uint16)
        attributes = {
            "POSITION": buf.add(part.vertices, gl.ARRAY_BUFFER, minmax=True),
            "JOINTS_0": buf.add(joints, gl.ARRAY_BUFFER),
            "WEIGHTS_0": buf.add(_quantise_weights(part.weights), gl.ARRAY_BUFFER, normalized=True),
        }
        if part.normals is not None:
            attributes["NORMAL"] = buf.add(part.normals, gl.ARRAY_BUFFER)
        if part.colour is not None:
            colour = np.rint(np.clip(part.colour, 0.0, 1.0) * 255.0).astype(np.uint8)
            attributes["COLOR_0"] = buf.add(colour, gl.ARRAY_BUFFER, normalized=True)
        if part.uv is not None:
            attributes["TEXCOORD_0"] = buf.add(part.uv, gl.ARRAY_BUFFER)

        index_type = np.uint16 if len(part.vertices) < 65536 else np.uint32
        indices = buf.add(part.faces.reshape(-1).astype(index_type), gl.ELEMENT_ARRAY_BUFFER)

        material = gl.Material(
            name=part.name,
            pbrMetallicRoughness=gl.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=float(part.roughness),
            ),
            doubleSided=True,
        )
        if part.texture is not None:
            material.pbrMetallicRoughness.baseColorTexture = gl.TextureInfo(
                index=_add_texture(gltf, buf, part.texture, _seen=embedded)
            )
        if part.alpha_mask:
            material.alphaMode = "MASK"
            material.alphaCutoff = 0.5
        if part.normal_texture is not None:
            # Pores and grain: breaks up the highlight so skin stops reading as
            # plastic under a key light.
            material.normalTexture = gl.NormalMaterialTexture(
                index=_add_texture(gltf, buf, part.normal_texture, quality=85, _seen=embedded),
                scale=float(part.normal_scale)
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
