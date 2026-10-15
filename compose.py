"""Assemble a dressed avatar: fitted face, body, clothes, hair and eyes.

The body is not generated from the photo and never will be. It is a catalogue
asset — a base mesh we dress — exactly as the product we are competing with does
it (../avatarsdk-re/REPORT.md section 5: their export is a body, a head and a
pile of separately authored wearables, fitted with visibility masks).

Base mesh and every wearable cut from it: MakeHuman, CC0 1.0, so commercial use
is unrestricted.

ponytail: the face is placed by bounding box and feathered at the edge, not
wrapped onto the skull. Flat garment colours, no fabric. Both are v1 cuts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import trimesh
from PIL import Image

import skinned_glb
from catalogue import (BODY_OBJ, build_item, catalogue, hair_colour, items_by_id,
                       load_base, srgb_to_linear, submesh, visible_faces)
from generate import NoFaceFound, expression_of, fit_face
from rig import load_skeleton, rigid_binding, weight_table

HERE = pathlib.Path(__file__).resolve().parent

# Landmarks that sit on cheek skin: away from eyes, lips, brows and hairline, so
# the median of them is a fair read of skin tone whatever the expression.
SKIN_LANDMARKS = [50, 280, 205, 425, 123, 352, 101, 330, 36, 266]

# Where the fitted face sits on the skull, tuned by eye against the base mesh.
# Scaled by height, not width: the landmark mesh runs chin to hairline, which is
# roughly two thirds of chin-to-crown on a real head. Matching widths instead
# made the patch tall enough to swallow the scalp, leaving nowhere for hair.
FACE_HEIGHT_OF_HEAD = 0.62
FACE_CENTRE_UP_HEAD = 0.42
FACE_FORWARD_BIAS = 0.06


def body_mesh() -> trimesh.Trimesh:
    """The visible body, without the helper and joint groups."""
    vertices, groups = load_base()
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=groups["body"], process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def skin_tone(photo: Image.Image, landmarks) -> np.ndarray:
    """Median skin colour of the face, as 0-255 RGB."""
    rgb = photo.convert("RGB")
    w, h = rgb.size
    samples = []
    for i in SKIN_LANDMARKS:
        if i >= len(landmarks):
            continue
        x = min(max(int(landmarks[i].x * w), 0), w - 1)
        y = min(max(int(landmarks[i].y * h), 0), h - 1)
        samples.append(rgb.getpixel((x, y)))
    if not samples:
        return np.array([200, 170, 150], dtype=np.uint8)
    return np.median(np.array(samples, dtype=np.float64), axis=0).astype(np.uint8)


def trim_face_border(face: trimesh.Trimesh, rings: int = 2) -> trimesh.Trimesh:
    """Peel the outermost ring of triangles off the fitted face.

    The rim of the patch is where the photo's background smears onto the side of
    the head, so dropping it loses nothing and lets the skull's own skin tone
    carry the jaw and hairline.

    A transparent falloff was tried first and was worse: a double-sided
    transparent mesh sorts badly and the face came out ghostly and streaked.
    ponytail: a hard edge one ring in, not a real wrap-and-blend.
    """
    mesh = face.copy()
    for _ in range(max(rings, 0)):
        edges = mesh.edges_sorted
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        border = set(np.unique(unique[counts == 1]).tolist())
        if not border:
            break
        keep = [not (set(tri.tolist()) & border) for tri in mesh.faces]
        if not any(keep):
            break
        mesh.update_faces(np.array(keep, dtype=bool))
    mesh.remove_unreferenced_vertices()
    return mesh


def place_face(face: trimesh.Trimesh, body: trimesh.Trimesh) -> trimesh.Trimesh:
    """Scale and move the face onto the body's head."""
    bmin, bmax = body.bounds
    body_height = bmax[1] - bmin[1]

    # The head is the top eighth or so of a human figure; take that slab and use
    # its box rather than hand-written coordinates, so a different base mesh or a
    # reshaped body still lands the face in the right place.
    head = body.vertices[body.vertices[:, 1] > bmax[1] - 0.13 * body_height]
    head_min, head_max = head.min(axis=0), head.max(axis=0)
    head_width = head_max[0] - head_min[0]
    head_height = head_max[1] - head_min[1]

    face = face.copy()
    fmin, fmax = face.bounds
    face.apply_scale((head_height * FACE_HEIGHT_OF_HEAD) / (fmax[1] - fmin[1]))

    fmin, fmax = face.bounds
    centre = (fmin + fmax) / 2.0
    target = np.array([
        (head_min[0] + head_max[0]) / 2.0,
        head_min[1] + FACE_CENTRE_UP_HEAD * head_height,
        # Sit the face just proud of the front of the skull so it does not
        # z-fight with the base mesh's own face underneath.
        head_max[2] - (fmax[2] - fmin[2]) / 2.0 + FACE_FORWARD_BIAS * head_height,
    ])
    face.apply_translation(target - centre)
    return face


def outfit_for(gender: str, chosen: list[str] | None) -> list[dict]:
    """The wearables to put on, by id, falling back to the catalogue default."""
    index = items_by_id()
    ids = chosen or catalogue()["defaults"].get(gender, catalogue()["defaults"]["male"])
    missing = [i for i in ids if i not in index]
    if missing:
        raise KeyError(f"not in the catalogue: {', '.join(missing)}")
    return [index[i] for i in ids]


def compose(photo_path: pathlib.Path, out_path: pathlib.Path,
            gender: str = "male", wear: list[str] | None = None) -> dict:
    if not BODY_OBJ.exists():
        raise FileNotFoundError(
            f"{BODY_OBJ} is missing. See README.md 'Set up' for the body mesh download."
        )

    vertices, groups = load_base()
    face, landmarks, result = fit_face(photo_path)
    photo = Image.open(photo_path)

    tone = skin_tone(photo, landmarks)
    placed = trim_face_border(place_face(face, body_mesh()))

    worn, hair_rgb = [], None
    for item in outfit_for(gender, wear):
        colour = None
        if item["colour"] == "from_photo":
            hair_rgb = hair_colour(photo, landmarks)
            colour = hair_rgb
        if item["category"] == "hair":
            # Start the hair where the fitted face stops. A fixed height put the
            # cap across the forehead and, being offset outwards, it covered the
            # face it was supposed to sit above.
            item = {**item, "y_min": float(placed.bounds[1][1]) - 0.08}
        worn.append((item, build_item(item, colour)))

    # Only clothing hides skin; hair sits on top of it. The fitted face is a
    # wearable too: the base mesh has a face of its own underneath whose nose,
    # lips and brows protrude past the flatter patch and poke through it, so the
    # skull's front is masked away as well.
    covering = [m for item, m in worn if item["category"] in ("top", "bottom", "shoes")]
    body_faces = groups["body"][
        visible_faces(groups["body"], covering)
        & visible_faces(groups["body"], [placed], threshold=0.42)
    ]
    body = submesh(body_faces)
    body.visual = trimesh.visual.ColorVisuals(
        body, vertex_colors=np.tile(np.append(srgb_to_linear(tone), 255), (len(body.vertices), 1))
    )

    # --- rig it ---
    skeleton = load_skeleton(vertices)
    joints, weights = weight_table(skeleton, len(vertices))

    def skinned_part(name, mesh):
        """A catalogue piece, skinned by the base-mesh vertices it was cut from."""
        base = mesh.metadata["base_index"]
        return skinned_glb.Part(
            name, mesh.vertices, mesh.faces, joints[base], weights[base],
            colour=mesh.visual.vertex_colors.astype(np.float32) / 255.0,
            normals=mesh.vertex_normals,
        )

    parts = [skinned_part("body", body)]
    for item, mesh in worn:
        parts.append(skinned_part(item["id"], mesh))

    # The face is our own geometry, not a slice of the base mesh, so it has no
    # weights to inherit. Bind it rigidly to the head bone: a face does not flex
    # independently of the skull.
    head_bone = skeleton.index.get("head", 0)
    face_joints, face_weights = rigid_binding(len(placed.vertices), head_bone)
    uv = np.asarray(placed.visual.uv, dtype=np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # trimesh carries UVs bottom-up; glTF wants top-down
    parts.append(skinned_glb.Part(
        "face", placed.vertices, placed.faces, face_joints, face_weights,
        uv=uv, texture=placed.visual.material.baseColorTexture,
        normals=placed.vertex_normals,
    ))

    skinned_glb.export(parts, skeleton, out_path)

    info = {
        "source_photo": photo_path.name,
        "photo_size": list(photo.size),
        "kind": "full-body",
        "rigged": True,
        "bones": len(skeleton),
        "gender": gender,
        "wearing": [item["id"] for item, _ in worn],
        "skin_tone": [int(c) for c in tone],
        "hair_colour": None if hair_rgb is None else [int(c) for c in hair_rgb],
        "body_triangles": int(len(body.faces)),
        "face_triangles": int(len(placed.faces)),
        "triangles": int(len(body.faces) + len(placed.faces) + sum(len(m.faces) for _, m in worn)),
        "landmarks": len(landmarks),
        "size_kb": round(out_path.stat().st_size / 1024),
        "expression": expression_of(result),
    }
    # Same sidecar shape as a face-only run, so the studio gallery and the
    # review page do not care which kind of avatar this is.
    out_path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="A dressed avatar from one photo.")
    parser.add_argument("photo", type=pathlib.Path)
    parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
    parser.add_argument("--gender", choices=["male", "female"], default="male")
    parser.add_argument("--wear", help="comma-separated catalogue ids, overriding the default outfit")
    args = parser.parse_args()

    try:
        info = compose(args.photo, args.out, args.gender,
                       args.wear.split(",") if args.wear else None)
    except NoFaceFound:
        print("no face found in the photo", file=sys.stderr)
        return 1
    except (FileNotFoundError, KeyError) as err:
        print(err, file=sys.stderr)
        return 2

    print(f"{args.out}  {info['triangles']} tris  wearing {', '.join(info['wearing'])}"
          f"  {info['size_kb']} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
