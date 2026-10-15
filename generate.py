"""Photo -> textured 3D face mesh (GLB). Version 1 of the avatar engine.

One photo in, one GLB out, plus a sidecar JSON with the detected expression.
Usable from the command line or imported by the local studio app (`app.py`).

How it works: MediaPipe FaceLandmarker fits 468 landmarks to the photo. Those
landmarks become the vertex positions; the triangle topology comes from the
canonical face model; the texture is the photo itself, projected by using each
landmark's position in the image as its UV. So the output is the real geometry
of that face wearing that photo.

Licensing: MediaPipe and the canonical face model are Apache-2.0, commercial use
permitted. Nothing here is research-only.

ponytail: face patch only, no scalp/ears/back of head, no hair, no body, front
projection so anything the camera could not see is unpainted. Upgrade path is a
full head/body model behind the same function, once its licence is cleared.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading

import numpy as np
import trimesh
from PIL import Image

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

HERE = pathlib.Path(__file__).resolve().parent
LANDMARKER_TASK = HERE / "assets" / "face_landmarker.task"
CANONICAL_OBJ = HERE / "assets" / "canonical_face_model.obj"


class NoFaceFound(Exception):
    """The photo had no detectable face."""


# Loading the model costs about a second, so keep one around. MediaPipe makes no
# thread-safety promise, so one request detects at a time.
# ponytail: global lock, split into a worker pool if the team ever queues up.
_landmarker = None
_landmarker_lock = threading.Lock()

_canonical_faces: np.ndarray | None = None


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        options = vision.FaceLandmarkerOptions(
            base_options=base_options_module.BaseOptions(
                model_asset_path=str(LANDMARKER_TASK)
            ),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def load_canonical_faces(obj_path: pathlib.Path = CANONICAL_OBJ) -> np.ndarray:
    """Triangle indices from the canonical face model. Only the `f` lines matter:
    we keep the topology and throw away its vertices, because our vertices come
    from the photo."""
    global _canonical_faces
    if _canonical_faces is not None:
        return _canonical_faces

    faces = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("f "):
            continue
        # `f v/vt/vn ...` - the vertex index is the part before the first slash,
        # and OBJ indices are 1-based.
        idx = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
        for i in range(1, len(idx) - 1):  # fan-triangulate, though these are already tris
            faces.append((idx[0], idx[i], idx[i + 1]))
    _canonical_faces = np.array(faces, dtype=np.int64)
    return _canonical_faces


def upright(vertices: np.ndarray, matrix) -> np.ndarray:
    """Rotate a fitted face into a canonical upright, forward-facing pose.

    Without this the avatar inherits however the head was tilted in the photo,
    so a sideways snapshot yields a sideways avatar — useless to rig or to put
    on a body. MediaPipe hands us the canonical->camera transform; undoing its
    rotation puts the head back on its axis.
    """
    if matrix is None:
        return vertices
    rotation = np.array(matrix, dtype=np.float64)[:3, :3]
    # Drop any scale so this is a pure rotation, then invert it (orthonormal, so
    # the transpose is the inverse). Row-vector convention: v @ R undoes R.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    centre = vertices.mean(axis=0)
    return (vertices - centre) @ rotation + centre


def build_mesh(landmarks, faces: np.ndarray, photo: Image.Image, matrix=None) -> trimesh.Trimesh:
    w, h = photo.size
    aspect = w / h

    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float64)

    # Image space -> a centred, Y-up mesh about one unit tall. MediaPipe's z is
    # roughly in x-units and points away from the camera, so negate it to make
    # the nose stick towards the viewer.
    vertices = np.column_stack([
        (pts[:, 0] - 0.5) * aspect,
        -(pts[:, 1] - 0.5),
        -pts[:, 2] * aspect,
    ])
    vertices = upright(vertices, matrix)

    # Each landmark is painted with the pixel it was found on. Image coordinates
    # run top-down; trimesh carries UVs bottom-up and flips them on glTF export,
    # so hand it the bottom-up form or every face samples the floor of the photo.
    uv = np.column_stack([pts[:, 0], 1.0 - pts[:, 1]])

    # Spell the material out. Left to itself trimesh exports a 40% grey
    # baseColorFactor, which multiplies the photo down to a muddy brown in any
    # correct viewer; our own page only looked right because its ambient light
    # was compensating. Skin is not metal and not glossy.
    material = trimesh.visual.material.PBRMaterial(
        name="face",
        baseColorTexture=photo.convert("RGB"),
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.85,
    )

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,  # keep the 1:1 landmark->vertex mapping
    )


def fit_face(photo_path: pathlib.Path):
    """Fit the face in `photo_path` and return (mesh, landmarks, result).
    Raises NoFaceFound if there is no face. Used by generate_avatar and by the
    full-body composer."""
    with _landmarker_lock:
        image = mp.Image.create_from_file(str(photo_path))
        result = get_landmarker().detect(image)

    if not result.face_landmarks:
        raise NoFaceFound(photo_path.name)

    photo = Image.open(photo_path)
    faces = load_canonical_faces()
    landmarks = result.face_landmarks[0]
    matrix = (
        result.facial_transformation_matrixes[0]
        if result.facial_transformation_matrixes
        else None
    )
    mesh = build_mesh(landmarks, faces, photo, matrix)

    assert len(mesh.vertices) == len(landmarks), "a vertex per landmark"
    assert faces.max() < len(landmarks), "topology refers to a landmark that exists"
    assert len(mesh.visual.uv) == len(mesh.vertices), "a UV per vertex"
    return mesh, landmarks, result


def expression_of(result) -> dict:
    """The blendshape scores worth showing: what the face in the photo was doing."""
    if not result.face_blendshapes:
        return {}
    scores = {
        c.category_name: round(c.score, 4)
        for c in result.face_blendshapes[0]
        if c.score > 0.02
    }
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def generate_avatar(photo_path: pathlib.Path, out_path: pathlib.Path) -> dict:
    """Fit a face from `photo_path`, write `out_path` (.glb) and its sidecar
    JSON, and return the sidecar contents. Raises NoFaceFound if there is no
    face in the photo."""
    mesh, landmarks, result = fit_face(photo_path)
    photo = Image.open(photo_path)
    faces = load_canonical_faces()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)

    info = {
        "source_photo": photo_path.name,
        "photo_size": list(photo.size),
        "landmarks": len(landmarks),
        "triangles": int(len(faces)),
        "size_kb": round(out_path.stat().st_size / 1024),
        "expression": expression_of(result),
    }
    out_path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 3D face from one photo.")
    parser.add_argument("photo", type=pathlib.Path)
    parser.add_argument("-o", "--out", type=pathlib.Path, required=True, help="output .glb")
    args = parser.parse_args()

    if not args.photo.exists():
        print(f"no such photo: {args.photo}", file=sys.stderr)
        return 2

    try:
        info = generate_avatar(args.photo, args.out)
    except NoFaceFound:
        print("no face found in the photo", file=sys.stderr)
        return 1

    print(f"{args.out}  {info['landmarks']} verts  {info['triangles']} tris  {info['size_kb']} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
