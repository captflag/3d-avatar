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
from PIL import Image

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
        self.alpha_blend = False        # or blended by the texture's alpha (a soft edge)
        self.roughness = roughness      # written into the file, so every engine agrees
        self.normal_scale = normal_scale
        self.normals = None if normals is None else np.asarray(normals, dtype=np.float32)
        # Blendshapes: (targets, vertices, 3) offsets in the order export() is
        # given their names, or None where none of them moves this part.
        self.morphs = None
        # Optional extras: a roughness texture (PIL, roughness in green, as
        # glTF reads it) and a cloth sheen (KHR_materials_sheen) colour/roughness.
        self.roughness_texture = None
        self.sheen = None
        self.ior = None                 # KHR_materials_ior, where not glTF's 1.5
        self.anisotropy = None          # (strength, rotation in radians): KHR_materials_anisotropy
        self.clearcoat = None           # (strength, roughness): KHR_materials_clearcoat


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

    def add_quantised(self, array: np.ndarray, kind: str) -> int:
        """Normals as signed bytes or texture coordinates as unsigned shorts,
        both normalised (KHR_mesh_quantization): a quarter and a half the size
        of floats, with no visible loss. Byte normals are padded to four bytes
        a vertex, as vertex attributes must be."""
        array = np.asarray(array, dtype=np.float64)
        if kind == "normal":
            q = np.zeros((len(array), 4), np.int8)
            q[:, :3] = np.clip(np.rint(array * 127.0), -127, 127).astype(np.int8)
            view, component, accessor_type = self.add_view(q.tobytes(), gl.ARRAY_BUFFER), gl.BYTE, gl.VEC3
            self.views[view].byteStride = 4
        else:
            q = np.clip(np.rint(np.clip(array, 0.0, 1.0) * 65535.0), 0, 65535).astype(np.uint16)
            view, component, accessor_type = self.add_view(q.tobytes(), gl.ARRAY_BUFFER), gl.UNSIGNED_SHORT, gl.VEC2
        self.accessors.append(gl.Accessor(bufferView=view, componentType=component, count=int(len(array)),
                                          type=accessor_type, normalized=True))
        return len(self.accessors) - 1

    def add_offsets(self, offsets: np.ndarray) -> int:
        """A morph target's offsets, stored sparse: only the vertices it moves.

        An accessor with no buffer view reads as zeros, so a part a target does
        not touch costs nothing, and a face target moving a few hundred of the
        body's eleven thousand vertices stores only those.
        """
        offsets = np.asarray(offsets, dtype=np.float32)
        # Below 10 micrometres (1e-4 units) a move is the far tail of a smooth
        # falloff, not a movement: kept, the cheek puff alone stored the whole head.
        offsets = np.where(np.abs(offsets).max(axis=1, keepdims=True) > 1e-4, offsets, 0.0).astype(np.float32)
        moved = np.nonzero(np.abs(offsets).max(axis=1) > 0)[0]
        accessor = gl.Accessor(componentType=gl.FLOAT, count=int(len(offsets)), type=gl.VEC3,
                               min=offsets.min(axis=0).tolist() if len(offsets) else [0.0] * 3,
                               max=offsets.max(axis=0).tolist() if len(offsets) else [0.0] * 3)
        if len(moved):
            index_type = np.uint16 if len(offsets) < 65536 else np.uint32
            accessor.sparse = gl.Sparse(
                count=int(len(moved)),
                indices=gl.AccessorSparseIndices(bufferView=self.add_view(moved.astype(index_type).tobytes()),
                                                 componentType=COMPONENT[np.dtype(index_type)]),
                values=gl.AccessorSparseValues(bufferView=self.add_view(np.ascontiguousarray(offsets[moved]).tobytes())),
            )
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def tangents(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Per-vertex tangents (xyz, handedness) from the texture coordinates: the
    direction on the surface along which u increases, summed over each
    vertex's triangles, made perpendicular to the normal."""
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(uv, dtype=np.float64)
    f = np.asarray(faces)
    e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    d1, d2 = t[f[:, 1]] - t[f[:, 0]], t[f[:, 2]] - t[f[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-12, 1.0 / np.where(np.abs(det) > 1e-12, det, 1.0), 0.0)[:, None]
    face_t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    face_b = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    # Summed per position, not per vertex: a card mesh keeps a vertex per
    # texture island, and tangents summed per copy stepped at every seam.
    _, where = np.unique(np.round(v, 5), axis=0, return_inverse=True)
    where = where.reshape(-1)
    tangent = np.zeros_like(v)
    bitangent = np.zeros_like(v)
    for k in range(3):
        np.add.at(tangent, where[f[:, k]], face_t)
        np.add.at(bitangent, where[f[:, k]], face_b)
    tangent, bitangent = tangent[where], bitangent[where]
    n = np.asarray(normals, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    tangent = tangent - n * (tangent * n).sum(axis=1, keepdims=True)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    # A vertex with no usable texture gradient takes any direction across the normal.
    fallback = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]))
    tangent = np.where(length > 1e-9, tangent / np.maximum(length, 1e-9), fallback / np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-9))
    hand = np.where((np.cross(n, tangent) * bitangent).sum(axis=1) < 0.0, -1.0, 1.0)
    return np.column_stack([tangent, hand]).astype(np.float32)


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


# Of a cut-out texture's 255 palette entries, how many are spread over the
# colours it has rather than shared out by how much of the map each covers.
PALETTE_SPREAD = 64


def _add_texture(gltf, buf, image, quality: int = 90, _seen: dict | None = None,
                 smooth_alpha: bool = False) -> int:
    """Embed an image and return its texture index.

    The same image object passed twice is embedded once: head, body and face
    share one skin-grain map, and embedding it per part tripled its cost.

    JPEG, not PNG: skin and photo textures are noisy by nature and PNG barely
    compresses noise, so a PNG skin map alone ran to several megabytes.
    """
    if _seen is not None and id(image) in _seen:
        return _seen[id(image)]
    data = io.BytesIO()
    if image.mode == "RGBA" and smooth_alpha:
        # A blended part keeps its alpha as it is: the hair cap fades out
        # under the hairline, and cut at half alpha its edge was a sawtooth.
        image.save(data, format="PNG", optimize=True)
        mime = "image/png"
    elif image.mode == "RGBA":
        # Every RGBA texture here is a cut-out (alpha MASK at 0.5), so its alpha
        # is one transparent palette slot and its colour 255 entries, median
        # cut: the body's skin went from 947 to 432 KB at about 1.4 delta-E as
        # seen, below what shows. Median cut shares the entries out by how many
        # texels a colour covers, and the beard carried under the jaw is a small
        # part of the map: its fade from beard to skin came out as flat bands,
        # steps of up to 28 levels on photo-10's dark beard. So most entries
        # are median cut over every texel, refined by k-means, and
        # PALETTE_SPREAD are spread over the colours present, each counted
        # once; each texel then takes its nearest entry exactly (Pillow maps
        # onto a given palette through a coarse colour cache, which quadrupled
        # the skin's error). photo-10's worst step went 18 -> 6, the skin's
        # mean error 0.37 -> 0.42, 389 -> 379 KB. Dithering instead doubled
        # the map and banded more.
        from scipy.spatial import cKDTree
        rgba = np.asarray(image)
        shown = rgba[..., 3] >= 128
        rgb = np.ascontiguousarray(rgba[..., :3])
        popular = Image.fromarray(rgb).quantize(255 - PALETTE_SPREAD, method=Image.Quantize.MEDIANCUT, kmeans=3)
        packed, inverse = np.unique(rgb.reshape(-1, 3).astype(np.int32) @ np.array([65536, 256, 1]), return_inverse=True)
        colours = np.stack([packed >> 16, (packed >> 8) & 255, packed & 255], axis=1)
        coarse = np.unique(colours >> 3, axis=0) * 8 + 4
        side = int(np.ceil(np.sqrt(len(coarse))))
        grid = np.vstack([coarse, np.repeat(coarse[:1], side * side - len(coarse), axis=0)]).reshape(side, side, 3)
        spread = Image.fromarray(grid.astype(np.uint8)).quantize(PALETTE_SPREAD, method=Image.Quantize.MEDIANCUT)
        entries = np.vstack([np.array(popular.getpalette()[:(255 - PALETTE_SPREAD) * 3]).reshape(-1, 3),
                             np.array(spread.getpalette()[:PALETTE_SPREAD * 3]).reshape(-1, 3)])
        index = cKDTree(entries).query(colours.astype(np.float64))[1][inverse.reshape(-1)].reshape(rgb.shape[:2])
        index[~shown] = 255
        out = Image.fromarray(index.astype(np.uint8), "P")
        out.putpalette([int(c) for c in entries.reshape(-1)] + [0, 0, 0])
        out.save(data, format="PNG", optimize=True, transparency=255)
        mime = "image/png"
    elif image.mode == "P":
        # A palette image (alpha kept per palette entry) is a recoloured
        # single-hue texture such as hair, a quarter the size of full RGBA.
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
           pose: dict[str, list[float]] | None = None, morph_names: list[str] | None = None,
           expression: dict[str, float] | None = None,
           head_pose: list[float] | None = None) -> None:
    """Write the avatar and its skeleton to `out_path` as a GLB.

    `pose` rotates named bones in the rest pose. The base mesh is modelled in a
    mannequin stance, arms out and fingers splayed, which is the first thing that
    says 'not a person'.

    `morph_names` names the blendshapes each part's `morphs` holds. glTF wants
    every primitive of a mesh to carry the same targets, so a part without
    them gets empty ones.

    `expression` ({blendshape: weight}) is the face the photo was making,
    stored in the mesh's extras as `photoExpression` for a viewer to show. The
    file's own weights stay at rest, so a rig driven by face tracking or
    animation starts from a neutral face.

    `head_pose` (a quaternion, x y z w, in the scene's axes) is how the head
    was turned in the photo, stored in the mesh's extras as `photoHeadPose`
    for a viewer that wants to render the head as photographed. The file's
    own pose stays level.
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
            unit = part.normals / np.maximum(np.linalg.norm(part.normals, axis=1, keepdims=True), 1e-9)
            attributes["NORMAL"] = buf.add_quantised(unit, "normal")
        if part.colour is not None:
            colour = np.rint(np.clip(part.colour, 0.0, 1.0) * 255.0).astype(np.uint8)
            attributes["COLOR_0"] = buf.add(colour, gl.ARRAY_BUFFER, normalized=True)
        if part.uv is not None:
            attributes["TEXCOORD_0"] = buf.add_quantised(part.uv, "uv")
        if part.anisotropy is not None and part.uv is not None and part.normals is not None:
            # An anisotropic highlight runs along the tangent. Without one in
            # the file a viewer derives tangents from each triangle's texture
            # coordinates, and the hair's sheen broke into grey facets, a
            # triangle at a time.
            attributes["TANGENT"] = buf.add(tangents(part.vertices, part.faces, part.uv, part.normals), gl.ARRAY_BUFFER)

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
                index=_add_texture(gltf, buf, part.texture, _seen=embedded, smooth_alpha=part.alpha_blend)
            )
        if part.alpha_mask:
            material.alphaMode = "MASK"
            material.alphaCutoff = 0.5
        if part.alpha_blend:
            material.alphaMode = "BLEND"
        if part.normal_texture is not None:
            # Pores and grain: breaks up the highlight so skin stops reading as
            # plastic under a key light.
            material.normalTexture = gl.NormalMaterialTexture(
                index=_add_texture(gltf, buf, part.normal_texture, quality=85, _seen=embedded),
                scale=float(part.normal_scale)
            )
        if part.roughness_texture is not None:
            # glTF multiplies the factor by the texture's green channel.
            material.pbrMetallicRoughness.roughnessFactor = 1.0
            material.pbrMetallicRoughness.metallicRoughnessTexture = gl.TextureInfo(
                index=_add_texture(gltf, buf, part.roughness_texture, quality=90, _seen=embedded))
        if part.ior is not None:
            material.extensions["KHR_materials_ior"] = {"ior": float(part.ior)}
            if "KHR_materials_ior" not in gltf.extensionsUsed:
                gltf.extensionsUsed.append("KHR_materials_ior")
        if part.anisotropy is not None:
            strength, rotation = part.anisotropy
            material.extensions["KHR_materials_anisotropy"] = {
                "anisotropyStrength": float(strength), "anisotropyRotation": float(rotation)}
            if "KHR_materials_anisotropy" not in gltf.extensionsUsed:
                gltf.extensionsUsed.append("KHR_materials_anisotropy")
        if part.clearcoat is not None:
            coat, coat_roughness = part.clearcoat
            material.extensions["KHR_materials_clearcoat"] = {
                "clearcoatFactor": float(coat), "clearcoatRoughnessFactor": float(coat_roughness)}
            if "KHR_materials_clearcoat" not in gltf.extensionsUsed:
                gltf.extensionsUsed.append("KHR_materials_clearcoat")
        if part.sheen is not None:
            colour, sheen_roughness = part.sheen
            material.extensions["KHR_materials_sheen"] = {
                "sheenColorFactor": [float(c) for c in colour], "sheenRoughnessFactor": float(sheen_roughness)}
            if "KHR_materials_sheen" not in gltf.extensionsUsed:
                gltf.extensionsUsed.append("KHR_materials_sheen")

        targets = []
        if morph_names:
            count = len(part.vertices)
            targets = [gl.Attributes(POSITION=buf.add_offsets(part.morphs[t] if part.morphs is not None
                                                             else np.zeros((count, 3), np.float32)))
                       for t in range(len(morph_names))]

        gltf.materials.append(material)
        mesh.primitives.append(gl.Primitive(attributes=gl.Attributes(**attributes),
                                            indices=indices, targets=targets,
                                            material=len(gltf.materials) - 1))

    if morph_names:
        mesh.weights = [0.0] * len(morph_names)
        mesh.extras = {"targetNames": list(morph_names)}
        if expression:
            mesh.extras["photoExpression"] = {k: round(float(v), 3) for k, v in expression.items()
                                              if k in morph_names}
    if head_pose is not None:
        mesh.extras = {**(mesh.extras or {}), "photoHeadPose": [round(float(c), 5) for c in head_pose]}
    gltf.meshes.append(mesh)
    gltf.extensionsUsed.append("KHR_mesh_quantization")
    gltf.extensionsRequired.append("KHR_mesh_quantization")
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


def dequantised(glb: bytes) -> bytes:
    """The same GLB with its quantised normals and texture coordinates written
    back as floats, and KHR_mesh_quantization dropped, for viewers that do not
    know the extension (Windows' 3D Viewer refused every avatar: "Couldn't load
    3D model"). About a third larger. Skin weights and vertex colours stay
    normalised integers, which core glTF allows."""
    import json
    json_len = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20:20 + json_len])
    bin_start = 20 + json_len + 8
    blob = bytearray(glb[bin_start:bin_start + struct.unpack_from("<I", glb, 20 + json_len)[0]])
    scale = {gl.BYTE: (np.int8, 127.0), gl.UNSIGNED_BYTE: (np.uint8, 255.0),
             gl.SHORT: (np.int16, 32767.0), gl.UNSIGNED_SHORT: (np.uint16, 65535.0)}
    width = {"VEC2": 2, "VEC3": 3, "VEC4": 4}
    done = set()
    for mesh in doc.get("meshes", []):
        for primitive in mesh["primitives"]:
            for name, index in primitive["attributes"].items():
                accessor = doc["accessors"][index]
                if index in done or not (name == "NORMAL" or name.startswith("TEXCOORD")) \
                        or accessor["componentType"] == gl.FLOAT:
                    continue
                done.add(index)
                kind, top = scale[accessor["componentType"]]
                view = doc["bufferViews"][accessor["bufferView"]]
                n, k = accessor["count"], width[accessor["type"]]
                size = np.dtype(kind).itemsize
                stride = view.get("byteStride", k * size)
                start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
                rows = np.frombuffer(bytes(blob[start:start + stride * n]).ljust(stride * n, b"\0"),
                                     dtype=np.uint8).reshape(n, stride)[:, :k * size]
                values = np.ascontiguousarray(rows).view(kind).reshape(n, k).astype(np.float32) / top
                values = np.maximum(values, -1.0) if np.dtype(kind).kind == "i" else values
                blob.extend(b"\0" * (-len(blob) % 4))
                doc["bufferViews"].append({"buffer": 0, "byteOffset": len(blob),
                                           "byteLength": values.nbytes, "target": gl.ARRAY_BUFFER})
                blob.extend(values.tobytes())
                accessor.update(bufferView=len(doc["bufferViews"]) - 1, componentType=gl.FLOAT, normalized=False)
                accessor.pop("byteOffset", None)
                if name == "NORMAL":
                    accessor["min"], accessor["max"] = values.min(axis=0).tolist(), values.max(axis=0).tolist()
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in doc:
            doc[key] = [e for e in doc[key] if e != "KHR_mesh_quantization"]
            if not doc[key]:
                del doc[key]
    blob.extend(b"\0" * (-len(blob) % 4))
    doc["buffers"][0]["byteLength"] = len(blob)
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body
