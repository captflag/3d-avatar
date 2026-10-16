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
import facetexture
import garments
import morph
import skin
from catalogue import (BODY_OBJ, build_item, catalogue, hair_colour, items_by_id,
                       load_base, load_base_uv, srgb_to_linear, submesh, submesh_uv,
                       coverage_alpha)
from generate import NoFaceFound, expression_of, fit_face
from rig import load_skeleton, rigid_binding, weight_table

HERE = pathlib.Path(__file__).resolve().parent

# Landmarks that sit on cheek skin: away from eyes, lips, brows and hairline, so
# the median of them is a fair read of skin tone whatever the expression.
SKIN_LANDMARKS = [50, 280, 205, 425, 123, 352, 101, 330, 36, 266]

# How far the fitted face patch sits proud of the skull, in mesh units (about
# 10 cm each), so the two surfaces do not z-fight.
FACE_FORWARD_BIAS = 0.05

# How far the eyeballs sit behind the eyelid surface, in mesh units (~10 cm each).
EYE_RECESS = 0.06

# Surface finish, written into the file itself so Blender, Unity and Unreal show
# what the studio shows. Everything used to go out at 0.85, which is right for
# cotton and wrong for everything else: skin read as dry plaster and the eyes
# had no wet highlight. Our own viewer hid it by overriding the values.
# A catalogue item can set its own `roughness`; these are the defaults.
ROUGHNESS = {"skin": 0.5, "eyes": 0.05, "hair": 0.45,
             "top": 0.85, "bottom": 0.85, "shoes": 0.4}


def body_mesh(vertices: np.ndarray | None = None) -> trimesh.Trimesh:
    """The visible body, without the helper and joint groups."""
    base, groups = load_base()
    if vertices is None:
        vertices = base
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


def face_anchor_points(face: trimesh.Trimesh) -> np.ndarray:
    """Eyes, nose tip, mouth and chin on a fitted (or canonical) face mesh.

    Eye-contour midpoints rather than iris landmarks, so MediaPipe's canonical
    face, which has no iris points, is anchored the same way as a real one. +X
    is the subject's left on both meshes, so the eyes are paired by x.
    """
    v = face.vertices
    a, b = (v[33] + v[133]) / 2.0, (v[362] + v[263]) / 2.0
    left, right = (a, b) if a[0] > b[0] else (b, a)
    return np.array([left, right, v[1], (v[13] + v[14]) / 2.0, v[152]])


def place_face(face: trimesh.Trimesh, vertices: np.ndarray, groups: dict) -> trimesh.Trimesh:
    """Put the fitted face on the skull by anatomy.

    Its eyes, nose tip, mouth and chin are lined up with where the skull keeps
    its own: the eyeball and teeth helpers, and the skull's furthest-forward and
    lowest facial points. A uniform scale and an offset are solved by least
    squares across the face plane, and depth is set by the nose tip.

    This replaces placement by bounding box with hand-tuned ratios, which sat
    the face 3 to 5 cm too high: the photo's mouth landed on the skull's nose and
    its chin on the skull's mouth. It also matters beyond the patch, because the
    shape fit finds its landmark correspondences from where the average face is
    placed.
    """
    skull = facetexture.head_features(vertices, groups)
    target = np.array([skull["left_eye"], skull["right_eye"], skull["nose"],
                       skull["mouth"], skull["chin"]])
    source = face_anchor_points(face)

    src, dst = source[:, :2], target[:, :2]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    scale = float(((src - src_mean) * (dst - dst_mean)).sum() / ((src - src_mean) ** 2).sum())

    placed = face.copy()
    placed.apply_scale(scale)
    shift = np.zeros(3)
    shift[:2] = dst_mean - scale * src_mean
    # Sit the nose tip just proud of the skull's, so the patch does not z-fight
    # with the skull's own face underneath.
    shift[2] = skull["nose"][2] - scale * source[2, 2] + FACE_FORWARD_BIAS
    placed.apply_translation(shift)
    return placed


def rest_pose() -> dict:
    """A relaxed stance instead of the base mesh's modelling pose.

    Arms out and fingers splayed is how a mesh is modelled, not how a person
    stands, and it is the first thing that reads as mannequin. Bones here are
    translation-only with world-aligned axes, so these are plain world-axis
    rotations.
    """
    pose = {}
    # Bring the arms down towards the body. Mirrored, hence the sign flip.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        pose[f"upperarm01.{side}"] = skinned_glb.axis_angle([0, 0, 1], -22.0 * sign)
        pose[f"lowerarm01.{side}"] = skinned_glb.axis_angle([0, 0, 1], -10.0 * sign)
        # A hand at rest curls; splayed fingers look like a shop dummy.
        for finger in range(1, 6):
            for joint in range(1, 4):
                pose[f"finger{finger}-{joint}.{side}"] = skinned_glb.axis_angle(
                    [0, 1, 0], 12.0 * sign
                )
    return pose


def hairline_for(face: trimesh.Trimesh, vertices: np.ndarray, groups: dict):
    """Where the hair stops, as a height that varies around the head.

    A straight horizontal cut sat the hair on the crown like a swim cap and left
    the temples and the back of the skull bare. Real hair starts above the
    forehead, comes down at the temples to just above the ears, and drops
    towards the nape at the back.

    Driven by the angle around the head's vertical axis — front, temples, ears,
    back — rather than by depth: measuring "how far back" from the nose tip
    counted the curve of the forehead itself as the back of the head, and the
    first version brought hair down over the brows.

    ponytail: one hairline shape for everyone. Receding lines, fringes and long
    hair want authored pieces from the catalogue.
    """
    body = vertices[np.unique(groups["body"])]
    top = float(face.bounds[1][1])                       # upper forehead landmark
    head = body[body[:, 1] > float(face.bounds[0][1])]   # everything above the chin
    centre_x = float(np.median(face.vertices[:, 0]))
    centre_z = float((head[:, 2].max() + head[:, 2].min()) / 2.0)

    forehead = top + 0.10      # a little above the highest landmark
    above_ears = top - 0.30    # temples and sides
    nape = top - 1.05          # back of the head

    def line(points: np.ndarray) -> np.ndarray:
        angle = np.degrees(np.abs(np.arctan2(points[:, 0] - centre_x, points[:, 2] - centre_z)))
        # 0 deg is straight ahead, 90 the ears, 180 the back of the head.
        to_temple = np.clip((angle - 30.0) / 45.0, 0.0, 1.0)
        to_back = np.clip((angle - 105.0) / 75.0, 0.0, 1.0)
        smooth = lambda t: t * t * (3.0 - 2.0 * t)
        height = forehead + (above_ears - forehead) * smooth(to_temple)
        return height + (nape - above_ears) * smooth(to_back)
    return line


def outfit_for(gender: str, chosen: list[str] | None) -> list[dict]:
    """The wearables to put on, by id, falling back to the catalogue default."""
    index = items_by_id()
    ids = chosen or catalogue()["defaults"].get(gender, catalogue()["defaults"]["male"])
    missing = [i for i in ids if i not in index]
    if missing:
        raise KeyError(f"not in the catalogue: {', '.join(missing)}")
    return [index[i] for i in ids]


def compose(photo_path: pathlib.Path, out_path: pathlib.Path,
            gender: str = "male", wear: list[str] | None = None,
            face_patch: bool = True) -> dict:
    if not BODY_OBJ.exists():
        raise FileNotFoundError(
            f"{BODY_OBJ} is missing. See README.md 'Set up' for the body mesh download."
        )

    base, groups = load_base()
    face, landmarks, result = fit_face(photo_path)
    photo = Image.open(photo_path)

    # Shape the skull to this person before anything is built from it. The
    # person's face and MediaPipe's average face are placed on the head the same
    # way, and the targets are fitted to the difference, so placement error
    # cancels instead of being baked into the shape.
    average = place_face(morph.canonical_face(), base, groups).vertices
    person = place_face(face, base, groups).vertices
    corr = morph.correspondences(base, groups["body"], average[:468])
    shape_weights, shape_names = morph.fit(base, corr, person, average)
    vertices = morph.apply(base, shape_weights)
    shape_fit = morph.explained(base, corr, person, average, shape_weights)

    pw, ph = photo.size
    photo_rgb = np.asarray(photo.convert("RGB"))
    cheeks = np.array([[landmarks[i].x * pw, landmarks[i].y * ph]
                       for i in SKIN_LANDMARKS if i < len(landmarks)])
    tone = facetexture.skin_tone_lab(photo_rgb, cheeks)
    placed = trim_face_border(place_face(face, vertices, groups))

    worn, hair_rgb = [], None
    for item in outfit_for(gender, wear):
        colour = None
        if item["colour"] == "from_photo":
            hair_rgb = hair_colour(photo, landmarks)
            colour = hair_rgb
        if item["category"] == "hair":
            item = {**item, "hairline": hairline_for(placed, vertices, groups)}
            item.pop("y_min", None)
        worn.append((item, build_item(item, colour, vertices)))

    # Only clothing hides skin; hair sits on top of it. The fitted face is a
    # wearable too: the base mesh has a face of its own underneath whose nose,
    # lips and brows protrude past the flatter patch and poke through it, so the
    # skull's front is masked away as well.
    covering = [m for item, m in worn if item["category"] in ("top", "bottom", "shoes")]
    # Every body triangle is kept; skin under clothes is hidden by the texture's
    # alpha instead, with a band left under each garment edge (coverage_alpha).
    body_faces = groups["body"]
    skin_alpha = np.ones(len(vertices))
    body_vertices_used = np.unique(body_faces)
    skin_alpha[body_vertices_used] = coverage_alpha(vertices[body_vertices_used], covering)
    uv_all, uv_groups = load_base_uv()
    uv_faces = uv_groups["body"]

    # Dense landmark sampler: every point on the head is looked up through the
    # anatomically placed landmark sheet, so the photo's mouth lands on the
    # skull's mouth and its chin on the skull's chin.
    full_face = place_face(face, vertices, groups)
    landmark_pixels = np.array([[lm.x * pw, lm.y * ph] for lm in landmarks])
    rim_edges, rim_counts = np.unique(full_face.edges_sorted, axis=0, return_counts=True)
    rim = np.unique(rim_edges[rim_counts == 1])
    sampler = facetexture.FaceSampler(full_face.vertices, landmark_pixels, rim)
    face_centre = full_face.vertices.mean(axis=0)
    face_radius = float(np.linalg.norm(full_face.vertices - face_centre, axis=1).max())

    # The head gets a texture of its own. In the shared body atlas the whole face
    # is about 130 pixels across, which is why fine features came out blurred
    # however carefully they were placed. Its UV island is found by connectivity
    # and stretched to fill a 1024 map, around eight times the texels.
    used_vertices = np.unique(body_faces)
    nose_vertex = used_vertices[np.linalg.norm(
        vertices[used_vertices] - facetexture.head_features(vertices, groups)["nose"], axis=1).argmin()]
    nose_uv = uv_faces.reshape(-1)[np.where(body_faces.reshape(-1) == nose_vertex)[0][0]]
    in_head = facetexture.uv_island(uv_faces, nose_uv)

    def island_uv(faces_uv):
        """Stretch an island's texture coordinates to fill the unit square."""
        idx = np.unique(faces_uv)
        lo, hi = uv_all[idx].min(axis=0), uv_all[idx].max(axis=0)
        remapped = uv_all.copy()
        remapped[idx] = (uv_all[idx] - lo) / np.maximum(hi - lo, 1e-9)
        return remapped

    head_faces, head_uv_faces = body_faces[in_head], uv_faces[in_head]
    rest_faces, rest_uv_faces = body_faces[~in_head], uv_faces[~in_head]
    head_uv_all = island_uv(head_uv_faces)

    # A skin texture built to the person's own cheek statistics, so skin is
    # mottled rather than one flat colour.
    skin_mean, skin_spread = skin.cheek_stats(photo_rgb, cheeks)
    skin_normals = Image.fromarray(skin.pore_normals(1024))

    def finish(texture_rgb, base, faces_, uv_all_, uv_faces_):
        """Cut skin under clothes away with alpha, flattening what is hidden.

        A part nothing covers keeps a plain RGB texture, which goes out as JPEG
        at a fraction of the size of the PNG an alpha channel forces.
        """
        alpha = facetexture.rasterize(skin_alpha, faces_, uv_all_, uv_faces_, 1024)
        if alpha.min() >= 0.5:
            return texture_rgb.convert("RGB")
        rgb = np.asarray(texture_rgb.convert("RGB")).copy()
        rgb[alpha < 0.5] = np.asarray(base).reshape(-1, 3).mean(axis=0).astype(np.uint8)
        out = Image.fromarray(rgb).convert("RGBA")
        out.putalpha(Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)))
        return out

    # Face feathered in rather than Poisson-blended: Poisson solves the face's
    # colours to match the generated skin around it, which dragged the whole
    # face towards that tone and washed the likeness out.
    head_base = skin.albedo(1024, skin_mean, skin_spread, seed=3)
    head_texture = finish(
        facetexture.bake(vertices, head_faces, head_uv_all, head_uv_faces,
                         sampler, face_centre, face_radius, photo_rgb, head_base, blend="none"),
        head_base, head_faces, head_uv_all, head_uv_faces)
    rest_base = skin.albedo(1024, skin_mean, skin_spread, seed=7)
    rest_texture = finish(Image.fromarray(rest_base), rest_base, rest_faces, uv_all, rest_uv_faces)

    head, head_uv = submesh_uv(head_faces, head_uv_faces, vertices, uv=head_uv_all)
    body, body_uv = submesh_uv(rest_faces, rest_uv_faces, vertices)

    # Eyes: generated, with only the iris colour taken from the photo. Painting
    # the photo's eye region onto the eyeball put eyelid and brow on a sphere,
    # which reads as skin, not an eye.
    eye_faces = np.vstack([groups["helper-l-eye"], groups["helper-r-eye"]])
    eye_uv_faces = np.vstack([uv_groups["helper-l-eye"], uv_groups["helper-r-eye"]])
    eye_uv_all = island_uv(eye_uv_faces)
    eye_centres = np.array([vertices[np.unique(groups["helper-l-eye"])].mean(axis=0),
                            vertices[np.unique(groups["helper-r-eye"])].mean(axis=0)])
    iris = facetexture.iris_colour(photo_rgb, landmarks)
    eye_texture = facetexture.bake_eyes(vertices, eye_faces, eye_uv_all, eye_uv_faces,
                                        eye_centres, iris)
    eyes, eye_uv = submesh_uv(eye_faces, eye_uv_faces, vertices, uv=eye_uv_all)
    # Seat the eyeballs a few millimetres behind the lids. Where they sat level
    # with the lid surface they drew over it, the whole iris showed ringed by
    # white, and the avatar stared.
    eyes.vertices = eyes.vertices - np.array([0.0, 0.0, EYE_RECESS])

    # --- rig it ---
    skeleton = load_skeleton(vertices)
    joints, weights = weight_table(skeleton, len(vertices))

    def skinned_part(name, mesh, roughness=0.8):
        """A catalogue piece, skinned to the rig.

        Cut garments look their weights up by the base-mesh vertex they came
        from. Imported ones have no such vertices and borrow weights from the
        body underneath them instead.
        """
        if "base_index" in mesh.metadata:
            source = mesh.metadata["base_index"]
            part_joints, part_weights = joints[source], weights[source]
        elif "mhclo_idx" in mesh.metadata:
            part_joints, part_weights = garments.mhclo_weights(
                mesh.metadata["mhclo_idx"], mesh.metadata["mhclo_blend"],
                joints, weights, len(skeleton))
        else:
            part_joints, part_weights = garments.weights_for(
                mesh, mesh.metadata.get("fit", "tight"), vertices, groups,
                joints, weights, len(skeleton),
            )
        return skinned_glb.Part(
            name, mesh.vertices, mesh.faces, part_joints, part_weights,
            colour=mesh.visual.vertex_colors.astype(np.float32) / 255.0,
            normals=mesh.vertex_normals, roughness=roughness,
        )

    def gltf_uv(uv):
        return np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float32)

    parts = []
    for name, mesh, uv, texture in (("head", head, head_uv, head_texture),
                                    ("body", body, body_uv, rest_texture)):
        source = mesh.metadata["base_index"]
        parts.append(skinned_glb.Part(
            name, mesh.vertices, mesh.faces, joints[source], weights[source],
            uv=gltf_uv(uv), texture=texture, normals=mesh.vertex_normals,
            normal_texture=skin_normals, alpha_mask=texture.mode == "RGBA",
            roughness=ROUGHNESS["skin"],
        ))
    for item, mesh in worn:
        finish = item.get("roughness", ROUGHNESS.get(item["category"], 0.8))
        parts.append(skinned_part(item["id"], mesh, finish))

    eye_base = eyes.metadata["base_index"]
    parts.append(skinned_glb.Part(
        "eyes", eyes.vertices, eyes.faces, joints[eye_base], weights[eye_base],
        uv=gltf_uv(eye_uv),
        texture=eye_texture, normals=eyes.vertex_normals,
        roughness=ROUGHNESS["eyes"],                 # wet, so the highlight shows
    ))

    if face_patch:
        # The fitted landmark patch laid over the head. Sharper features, but a
        # visible edge, because the patch and the skull are separate surfaces.
        head_bone = skeleton.index.get("head", 0)
        face_joints, face_weights = rigid_binding(len(placed.vertices), head_bone)
        face_uv = np.asarray(placed.visual.uv, dtype=np.float32).copy()
        face_uv[:, 1] = 1.0 - face_uv[:, 1]  # trimesh is bottom-up; glTF is top-down
        parts.append(skinned_glb.Part(
            "face", placed.vertices, placed.faces, face_joints, face_weights,
            uv=face_uv, texture=placed.visual.material.baseColorTexture,
            normals=placed.vertex_normals, roughness=ROUGHNESS["skin"],
            # Same grain as the skin around it, a little lighter: the photo
            # already carries its own surface detail.
            normal_texture=skin_normals, normal_scale=0.6,
        ))

    skinned_glb.export(parts, skeleton, out_path, pose=rest_pose())

    info = {
        "source_photo": photo_path.name,
        "photo_size": list(photo.size),
        "kind": "full-body",
        "shape": {
            "explained": round(shape_fit, 3),
            "active_targets": int((shape_weights > 0.05).sum()),
            "strongest": [
                [shape_names[i], round(float(shape_weights[i]), 2)]
                for i in np.argsort(-shape_weights)[:6] if shape_weights[i] > 0.05
            ],
        },
        "rigged": True,
        "bones": len(skeleton),
        "gender": gender,
        "wearing": [item["id"] for item, _ in worn],
        "skin_tone": [int(c) for c in tone],
        "hair_colour": None if hair_rgb is None else [int(c) for c in hair_rgb],
        "body_triangles": int(len(body.faces) + len(head.faces)),
        "face": "dense landmark bake + patch" if face_patch else "dense landmark bake",
        "iris_colour": [int(c) for c in iris],
        "triangles": int(len(body.faces) + len(head.faces) + sum(len(m.faces) for _, m in worn)),
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
    parser.add_argument("--no-face-patch", dest="face_patch", action="store_false",
                        help="seamless head only: no visible edge, weaker likeness")
    args = parser.parse_args()

    try:
        info = compose(args.photo, args.out, args.gender,
                       args.wear.split(",") if args.wear else None, args.face_patch)
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
