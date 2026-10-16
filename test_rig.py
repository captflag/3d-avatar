"""One runnable check that the rig is sound.

Run: avatar-engine/.venv/Scripts/python.exe avatar-engine/test_rig.py

Skinning fails quietly: a bad weight table or a missing attribute still exports,
still opens, and only shows up as a figure that collapses to the origin the
moment something poses it. These assertions catch that before an engine does.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import pygltflib as gl

from catalogue import build_item, items_by_id, load_base
from compose import compose, rest_pose
from garments import GARMENT_DIR, deform, make_test_vest, skinning_matrices, transfer_weights
from rig import MAX_INFLUENCES, load_skeleton, weight_table
from skinned_glb import axis_angle

HERE = pathlib.Path(__file__).resolve().parent


def check_weights():
    vertices, _ = load_base()
    skeleton = load_skeleton(vertices)
    joints, weights = weight_table(skeleton, len(vertices))

    assert len(skeleton) > 50, f"only {len(skeleton)} bones, the skeleton did not load"
    assert joints.shape == (len(vertices), MAX_INFLUENCES)
    assert np.allclose(weights.sum(axis=1), 1.0), "weights must sum to 1 per vertex"
    assert joints.max() < len(skeleton), "a weight points at a bone that does not exist"
    assert (weights >= 0).all(), "negative weight"

    # Bones should sit where the body part is: a head near the top, feet at the
    # bottom. Catches joints resolved against the wrong vertex indices.
    top, bottom = vertices[:, 1].max(), vertices[:, 1].min()
    head = skeleton.heads[skeleton.index["head"]][1]
    foot = skeleton.heads[skeleton.index["foot.L"]][1]
    assert head > top - 0.25 * (top - bottom), f"head bone too low ({head:.2f})"
    assert foot < bottom + 0.25 * (top - bottom), f"foot bone too high ({foot:.2f})"
    print(f"ok  weights: {len(skeleton)} bones, {len(vertices)} vertices, all normalised")


def check_export():
    photo = HERE / "samples" / "portrait.jpg"
    if not photo.exists():
        print("skip export check: samples/portrait.jpg is missing")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "rigged.glb"
        compose(photo, out)
        gltf = gl.GLTF2().load(str(out))

        assert len(gltf.skins) == 1, "no skin in the exported file"
        skin = gltf.skins[0]
        assert skin.inverseBindMatrices is not None, "skin has no inverse bind matrices"
        assert len(skin.joints) > 50, "skin has too few joints"

        skinned = [n for n in gltf.nodes if n.mesh is not None and n.skin is not None]
        assert skinned, "the mesh node is not attached to the skin"

        for i, primitive in enumerate(gltf.meshes[0].primitives):
            attrs = primitive.attributes
            assert attrs.JOINTS_0 is not None, f"primitive {i} has no JOINTS_0"
            assert attrs.WEIGHTS_0 is not None, f"primitive {i} has no WEIGHTS_0"
            assert attrs.POSITION is not None, f"primitive {i} has no POSITION"

        print(f"ok  export: {len(skin.joints)} joints, "
              f"{len(gltf.meshes[0].primitives)} primitives, all skinned")


def check_transfer():
    """Weights borrowed from the body must pose a garment like its true weights.

    The cut garments have true weights. Throw them away, transfer as if the
    garment were foreign, pose both with a stride that swings parts of them by
    up to 30 cm, and compare where every vertex lands. One mesh unit is about
    10 cm on this body.
    """
    vertices, groups = load_base()
    skeleton = load_skeleton(vertices)
    joints, weights = weight_table(skeleton, len(vertices))
    body = np.unique(groups["body"])

    pose = rest_pose()
    pose["upperleg01.L"] = axis_angle([1, 0, 0], -35)
    pose["lowerleg01.L"] = axis_angle([1, 0, 0], 50)
    pose["upperleg01.R"] = axis_angle([1, 0, 0], 20)
    pose["spine03"] = axis_angle([0, 1, 0], 15)
    matrices = skinning_matrices(skeleton, pose)

    for gid in ("shirt_green", "trousers_navy", "shoes_dark"):
        mesh = build_item(items_by_id()[gid], None, vertices)
        source = mesh.metadata["base_index"]
        tj, tw = transfer_weights(mesh.vertices, vertices[body], joints[body], weights[body],
                                  len(skeleton))
        truth = deform(mesh.vertices, joints[source], weights[source], matrices)
        error = np.linalg.norm(truth - deform(mesh.vertices, tj, tw, matrices), axis=1)
        p95 = float(np.percentile(error, 95))
        assert p95 < 0.15, f"{gid}: transferred weights miss by {p95 * 10:.1f} cm at p95"
        print(f"ok  transfer {gid}: mean {error.mean() * 10:.2f} cm, p95 {p95 * 10:.2f} cm")


def check_alignment():
    """The face must sit on the skull's own features, not near them.

    Placement by bounding box once sat the face 3 to 5 cm too high: the photo's
    mouth on the skull's nose, its chin on the skull's mouth. It looked
    plausible at a glance and was only caught by measuring.
    """
    from compose import place_face
    from facetexture import head_features
    from generate import fit_face
    import morph

    photo = HERE / "samples" / "portrait.jpg"
    if not photo.exists():
        print("skip alignment check: samples/portrait.jpg is missing")
        return
    vertices, groups = load_base()
    face, _, _ = fit_face(photo)
    average = place_face(morph.canonical_face(), vertices, groups).vertices
    person = place_face(face, vertices, groups).vertices
    corr = morph.correspondences(vertices, groups["body"], average[:468])
    weights, _, _ = morph.fit(vertices, corr, person, average)
    shaped = morph.apply(vertices, weights)
    placed = place_face(face, shaped, groups).vertices
    skull = head_features(shaped, groups)

    for name, point, anatomy in (("nose tip", placed[1], skull["nose"]),
                                 ("mouth", (placed[13] + placed[14]) / 2, skull["mouth"]),
                                 ("chin", placed[152], skull["chin"])):
        off = float(np.linalg.norm((point - anatomy)[:2]))
        assert off < 0.1, f"face {name} is {off * 10:.1f} cm off the skull's {name}"
    print("ok  alignment: nose, mouth and chin within 1 cm of the skull's own")


def check_expression():
    """A smile is fitted as expression, not identity, and comes off the face.

    MediaPipe's average face, smiling and squinting by MakeHuman's own units,
    must fit back to that expression with next to no change to who it is, and
    taking the fitted expression off its landmarks must give the average back.
    """
    import morph
    from compose import place_face

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    front = morph.front_of(vertices, groups["body"], corr)
    smile = {"mouth-corner-puller": 0.6, "eye-left-slit": 0.4, "eye-right-slit": 0.4}
    person = average + morph.expression_offsets(vertices, front, smile)
    scores = {"mouthSmileLeft": 0.7, "mouthSmileRight": 0.7,
              "eyeSquintLeft": 0.5, "eyeSquintRight": 0.5}

    weights, _, expression = morph.fit(vertices, corr, person, average, scores, front)
    moved = np.linalg.norm(person - average)
    identity = np.linalg.norm(morph.apply(vertices, weights)[corr] - vertices[corr])
    left = np.linalg.norm(person - morph.expression_offsets(vertices, front, expression) - average)
    assert identity < 0.15 * moved, f"the smile was taken as identity ({identity / moved:.0%})"
    assert left < 0.15 * moved, f"neutralising left {left / moved:.0%} of the smile"
    print(f"ok  expression: a known smile fits back as expression, "
          f"{identity / moved:.0%} taken as identity, {left / moved:.0%} left after neutralising")


def check_mhclo():
    """A MakeHuman garment fits the body it is built on and is rigged exactly."""
    from catalogue import nearest_distance
    from garments import load_mhclo, mhclo_weights

    path = GARMENT_DIR / "catsuit" / "catsuit.mhclo"
    if not path.exists():
        print("skip mhclo check: assets/garments/catsuit is missing (see README)")
        return
    vertices, groups = load_base()
    suit = load_mhclo(path, vertices)
    gap = nearest_distance(suit.vertices, vertices[np.unique(groups["body"])])
    assert np.median(gap) < 0.2, f"catsuit floats {np.median(gap) * 10:.1f} cm off the body"

    skeleton = load_skeleton(vertices)
    joints, weights = weight_table(skeleton, len(vertices))
    sj, sw = mhclo_weights(suit.metadata["mhclo_idx"], suit.metadata["mhclo_blend"],
                           joints, weights, len(skeleton))
    assert np.allclose(sw.sum(axis=1), 1.0), "catsuit weights do not sum to one"
    print(f"ok  mhclo: catsuit fits {np.median(gap) * 10:.2f} cm off the skin, rigged exactly")


def check_import():
    """An external garment loads, refuses without a licence, and is rigged."""
    try:
        build_item({"id": "no_licence", "group": "body", "colour": [1, 1, 1]})
    except ValueError:
        pass
    else:
        raise AssertionError("an item with no licence was accepted")

    if not (GARMENT_DIR / "test_vest.obj").exists():
        make_test_vest()

    photo = HERE / "samples" / "portrait.jpg"
    if not photo.exists():
        print("skip import check: samples/portrait.jpg is missing")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "vest.glb"
        compose(photo, out, wear=["shirt_white", "vest_test", "trousers_navy", "shoes_dark"])
        gltf = gl.GLTF2().load(str(out))
        names = [m.name for m in gltf.materials]
        assert "vest_test" in names, "the imported vest is not in the avatar"
        vest = gltf.meshes[0].primitives[names.index("vest_test")]
        assert vest.attributes.JOINTS_0 is not None and vest.attributes.WEIGHTS_0 is not None,             "the imported vest was not rigged"
        body = gltf.materials[names.index("body")]
        assert body.alphaMode == "MASK", "skin under clothes is not alpha-masked"
    print("ok  import: licence enforced, vest rigged by transfer, body alpha-masked")


def _landmarks(points: dict):
    """A stand-in for MediaPipe's 478 landmarks, with a few placed by hand."""
    from types import SimpleNamespace
    marks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for i, (x, y) in points.items():
        marks[i] = SimpleNamespace(x=x, y=y)
    return marks


def check_photo_reading():
    """A beard survives the bake's fade; hair colour comes from hair, not skin."""
    from PIL import Image
    from catalogue import hair_colour
    from facetexture import facial_hair

    skin, beard, hair = (230, 190, 170), (120, 70, 50), (110, 75, 55)
    face = np.full((200, 200, 3), skin, dtype=np.uint8)
    face[150:190] = beard                       # below the nose
    face[40:55] = beard                         # brows: dark, but above the nose
    marks = _landmarks({1: (0.5, 0.6), 152: (0.5, 0.95)})
    keep = facial_hair(face, marks, np.array([[100.0, 110.0]]))
    assert keep[170, 100] > 0.9, f"beard not kept ({keep[170, 100]:.2f})"
    assert keep[130, 100] < 0.1, "plain skin below the nose was kept as beard"
    assert keep[47, 100] < 0.05, "brows above the nose were taken for a beard"

    photo = np.full((200, 200, 3), 255, dtype=np.uint8)   # white backdrop
    photo[10:60, 40:160] = hair
    photo[12:60:6, 40:160] = (190, 150, 120)              # highlight streaks
    photo[60:190, 60:140] = skin
    marks = _landmarks({10: (0.5, 0.32), 152: (0.5, 0.95), 234: (0.3, 0.6), 454: (0.7, 0.6)})
    got = hair_colour(Image.fromarray(photo), marks, np.array([[100.0, 120.0]])).astype(int)
    assert np.abs(got - hair).max() < 30, f"hair read as {got.tolist()}, expected about {hair}"
    print(f"ok  photo: beard kept through the fade, hair read as {got.tolist()}")


def check_bodies_and_assets():
    """Male and female bodies differ; MakeHuman assets load with UVs; the pose
    names real bones and brings the arms in."""
    import morph
    from compose import relaxed_pose
    from garments import _ranges, load_mhclo

    base, groups = load_base()
    body = np.unique(groups["body"])
    heights = {g: np.ptp(morph.body_for(base, g)[body][:, 1]) for g in ("male", "female")}
    assert heights["female"] < heights["male"] * 0.97, f"bodies barely differ: {heights}"

    assert _ranges(["3", "-", "5", "9"]).tolist() == [3, 4, 5, 9]
    system = HERE / "assets" / "mh_system"
    files = sorted(system.glob("*/*/*.mhclo"))
    for path in files:
        mesh = load_mhclo(path, base)
        assert mesh.metadata["uv"] is not None and len(mesh.metadata["uv"]) == len(mesh.vertices), \
            f"{path.name}: texture coordinates did not survive the load"
        if path.parts[-3] == "clothes":
            assert len(mesh.metadata["hides"]), f"{path.name}: its hidden-vertex list was lost"

    skeleton = load_skeleton(base)
    pose = relaxed_pose()
    missing = [name for name in pose if name not in skeleton.index]
    assert not missing, f"the pose names bones the rig does not have: {missing}"
    assert all(abs(np.linalg.norm(q) - 1.0) < 1e-6 for q in pose.values())

    def hand_out(p):
        m = skinning_matrices(skeleton, p)
        i = skeleton.index["wrist.L"]
        return float((m[i] @ np.append(skeleton.heads[i], 1.0))[0])
    assert hand_out(pose) < hand_out(rest_pose()), "the relaxed pose did not bring the arms in"
    print(f"ok  bodies: female {heights['female']:.1f} vs male {heights['male']:.1f} units; "
          f"{len(files)} MakeHuman assets load with UVs; pose sound")


def check_delight():
    """The lighting fit flattens a surface lit from one side; exposure is capped
    only for skin brighter than skin can be; nothing outside the face moves."""
    import delight
    from compose import SKIN_LANDMARKS
    from generate import fit_face, load_canonical_faces

    # A hemisphere facing the camera, one albedo, lit from the upper right.
    rng = np.random.default_rng(1)
    n = rng.normal(size=(4000, 3))
    n[:, 2] = -np.abs(n[:, 2])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    light = np.array([0.6, -0.6, -0.5]) / np.linalg.norm([0.6, -0.6, -0.5])
    shading = 0.55 + 0.35 * (n @ light)
    colour = np.outer(shading, [0.5, 0.35, 0.3])
    coeffs = delight.fit_lighting(n, colour)
    fitted = delight.sh_basis(n) @ coeffs
    flat = colour / fitted * (delight.sh_basis(np.array([[0, 0, -1.0]])) @ coeffs)
    spread = float((flat.std(axis=0) / flat.mean(axis=0)).max())
    assert spread < 0.01, f"one-sided lighting left {spread:.1%} variation"

    photo = HERE / "samples" / "portrait.jpg"
    if not photo.exists():
        print(f"ok  delight: synthetic lighting removed ({spread:.2%} left); portrait check skipped")
        return
    from PIL import Image
    _, marks, _ = fit_face(photo)
    rgb = np.asarray(Image.open(photo).convert("RGB"))
    h, w = rgb.shape[:2]
    cheeks = np.array([[marks[i].x * w, marks[i].y * h] for i in SKIN_LANDMARKS])
    out, report, mask = delight.delight(rgb, marks, load_canonical_faces(), cheeks)
    assert report["exposure"] == 1.0, "skin within the natural range was darkened"
    import cv2
    far = cv2.dilate(mask.astype(np.uint8), np.ones((51, 51), np.uint8)) == 0
    assert np.abs(out[far].astype(int) - rgb[far]).max() <= 1, "pixels away from the face changed"

    bright = delight.to_srgb(delight.to_linear(rgb) * 1.8)        # an overexposed copy
    out, report, _ = delight.delight(bright, marks, load_canonical_faces(), cheeks)
    lum = delight.to_linear(out[cheeks[:, 1].astype(int), cheeks[:, 0].astype(int)]) @ [0.2126, 0.7152, 0.0722]
    assert report["exposure"] < 1.0 and abs(np.median(lum) - delight.SKIN_CEILING) < 0.05, \
        f"overexposed skin not capped: cheeks at {np.median(lum):.2f}"
    print(f"ok  delight: synthetic lighting removed ({spread:.2%} left), "
          f"overexposed skin capped, dark skin and background untouched")


def check_appearance():
    """Haircut, hair and eye colour are read from the photo; outfits dye by part."""
    import appearance
    from catalogue import ASSET_DIR

    pick = appearance.hair_style
    reading = {"amount": 0.5, "top": 0.5, "bottom": -0.6, "fringe": 0.0, "colour": None}
    assert pick(reading, "x") == "hair_short02"
    assert pick({**reading, "top": 0.3}, "x") == "hair_short01"
    assert pick({**reading, "fringe": 0.6}, "x") == "hair_short03"
    assert pick({**reading, "bottom": 0.0}, "x") == "hair_bob01"
    assert pick({**reading, "bottom": 0.4}, "x") == "hair_long01"
    assert pick({**reading, "amount": 0.02}, "x") is None, "a bald head was given hair"
    assert pick(None, "hair_short02") == "hair_short02", "no segmenter should keep the default"

    photo = HERE / "samples" / "portrait.jpg"
    if photo.exists() and appearance.SEGMENTER_TASK.exists():
        from PIL import Image
        from generate import fit_face
        _, marks, _ = fit_face(photo)
        rgb = np.asarray(Image.open(photo).convert("RGB"))
        hair = appearance.read_hair(rgb, marks, appearance.segment(rgb))
        assert pick(hair, "x") == "hair_short01", f"close-cropped hair read as {pick(hair, 'x')}"
        assert max(hair["colour"]) < 140, f"dark grey hair read as {hair['colour'].tolist()}"
        _, family = appearance.iris_colour(rgb, marks)
        assert family in ("dark brown", "brown"), f"brown eyes read as {family}"

    outfit = ASSET_DIR / "mh_system" / "clothes" / "male_casualsuit01" / "male_casualsuit01.mhclo"
    if outfit.exists():
        import cv2
        from garments import garment_regions, load_mhclo, material_textures, recolour
        base, groups = load_base()
        mesh = load_mhclo(outfit, base)
        top = garment_regions(mesh, base, groups)
        heights = mesh.vertices[mesh.faces].mean(axis=1)[:, 1]
        assert top[heights > 3.5].mean() > 0.95, "chest-high cloth was not counted as the shirt"
        assert (~top[heights < -3.0]).mean() > 0.95, "shin-high cloth was not counted as trousers"
        texture, _, _ = material_textures(mesh.metadata["material"])
        green = (64, 102, 84)
        dyed = np.asarray(recolour(texture, mesh, top, green).convert("RGB"))
        changed = np.abs(dyed.astype(int) - np.asarray(texture.convert("RGB")).astype(int)).sum(axis=2) > 30
        lab = cv2.cvtColor(dyed[changed].reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        want = cv2.cvtColor(np.uint8([[green]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        assert np.linalg.norm(np.median(lab[:, 1:], axis=0) - want[1:]) < 6, "the shirt was not dyed green"
    print("ok  appearance: haircut rules, hair and eye colour from the photo, shirt and trousers dyed apart")


if __name__ == "__main__":
    try:
        check_weights()
        check_export()
        check_transfer()
        check_alignment()
        check_expression()
        check_mhclo()
        check_import()
        check_photo_reading()
        check_bodies_and_assets()
        check_delight()
        check_appearance()
    except AssertionError as err:
        print(f"FAIL  {err}", file=sys.stderr)
        raise SystemExit(1)
    print("rig ok")
