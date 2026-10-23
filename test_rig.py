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
from PIL import Image
import pygltflib as gl

from catalogue import build_item, items_by_id, load_base, load_base_uv
from compose import compose, rest_pose
from generate import load_canonical_faces
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
    face, *_ = fit_face(photo)
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


def check_proportions():
    """A face with a longer lower face and a wider face than the skull, and
    landmark noise no target can follow, fits back to the distances a face is
    recognised by. Fitted through the landmarks alone, those few landmarks were
    outvoted and every avatar came out short-chinned."""
    import morph
    from compose import place_face

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    person = morph.reference(vertices, corr, average)
    nose = person[2, 1]
    below = person[:, 1] < nose
    person[below, 1] = nose + (person[below, 1] - nose) * 1.06
    person[:, 0] *= 1.03
    # What no target can follow, as on a real photo: among 468 landmarks the
    # few that set these distances were outvoted by it.
    spacing = np.linalg.norm(person[33] - person[263])
    person[:, :2] += np.random.default_rng(1).normal(0, 0.04 * spacing, (len(person), 2))
    want = {k: np.linalg.norm((person[j] - person[i])[:2]) for k, (i, j) in enumerate(morph.PROPORTIONS)}

    weights, _, _ = morph.fit(vertices, corr, person, average)
    shaped = morph.apply(vertices, weights)[corr]
    for k, (name, (i, j)) in enumerate(zip(("eye spacing", "face width", "jaw width", "face height",
                                            "nose to chin", "cheekbone width", "cheek width"), morph.PROPORTIONS)):
        got = np.linalg.norm((shaped[j] - shaped[i])[:2])
        assert abs(got / want[k] - 1) < 0.015, f"{name} fitted {100 * (got / want[k] - 1):+.1f}% off"
    print("ok  proportions: eye spacing, widths, height and chin fit within 1.5%")


def check_side_views():
    """Turned views of a known face fuse back to its depth, though each view's
    own depth is a flattened guess and its turn is read a sixth short, as
    MediaPipe reads them; and a fit given measured depth leaves a face alone
    that differs from the skull only by where it sits in depth."""
    import morph
    import views
    from compose import place_face

    truth = np.asarray(morph.canonical_face().vertices, dtype=np.float64)
    truth = truth - truth.mean(axis=0)
    rng = np.random.default_rng(2)
    spacing = np.linalg.norm(truth[33] - truth[263])

    def turn(degrees):
        a = np.radians(degrees)
        return np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])

    def view(degrees, scale, shift):
        points = scale * truth @ turn(degrees).T + shift
        points[:, 2] *= 0.6                                   # a flattened depth guess
        points[:, :2] += rng.normal(0, 0.005 * spacing * scale, (len(points), 2))
        return points, turn(degrees * 29 / 35)                # the turn read short

    front = view(0, 1.0, np.zeros(3))
    fused, measured, _ = views.fuse([front, view(35, 1.3, np.array([0.2, -0.1, 0.0])),
                                  view(-35, 0.8, np.array([-0.1, 0.05, 0.0]))])
    fused = fused - fused.mean(axis=0)
    error = np.sqrt(((fused[:, 2] - truth[:, 2]) ** 2).mean()) / spacing
    guessed = np.sqrt(((front[0][:, 2] - truth[:, 2]) ** 2).mean()) / spacing
    assert error < 0.25 * guessed, f"fused depth off by {100 * error:.1f}% of eye spacing, guessed {100 * guessed:.1f}%"
    assert measured[1] > 0.5, f"nose tip depth counted as measured only {measured[1]:.2f}"

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    person = morph.reference(vertices, corr, average) + [0.0, 0.0, 0.3 * spacing]
    weights, _, _ = morph.fit(vertices, corr, person, average, depth=np.ones(len(corr)))
    assert weights.max() < 0.05, f"a face only sitting further forward bent target {weights.argmax()} to {weights.max():.2f}"
    print(f"ok  side views: depth fused to {100 * error:.1f}% of eye spacing (one view guessed {100 * guessed:.1f}%); "
          f"a depth offset bends no target")


def check_brow_shading():
    """Under the brows the skin's normals lose the eye socket's hollow, so the
    3D light stops shading again what the photo already shows there; the rest
    of the face keeps its own."""
    import morph
    import appearance
    import trimesh
    from compose import PHOTO_NORMAL_ROUNDS, place_face

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    faces = groups["body"]
    skin = np.zeros(len(vertices), bool)
    skin[np.unique(faces)] = True
    normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
    weight = appearance.under_brow(vertices, corr)[:, None]
    blended = normals * (1 - weight) + appearance.broad_normals(faces, normals, PHOTO_NORMAL_ROUNDS) * weight
    blended /= np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-12)
    band, away = (weight[:, 0] > 0.5) & skin, (weight[:, 0] < 0.01) & skin
    before, after = normals[band, 1].mean(), blended[band, 1].mean()
    assert after > before + 0.03, f"under-brow skin still faces down ({before:.3f} -> {after:.3f})"
    assert np.abs(blended - normals)[away].max() < 0.01, "normals changed away from the brows"
    print(f"ok  brow shading: under-brow normals lifted {before:.2f} -> {after:.2f}, rest of the face kept")


def check_arkit():
    """ARKit's 52 blendshapes, each moving the face the way ARKit names it:
    sided shapes on their own side (the subject's left is +x), the eyes turning
    about their centres, the jaw carrying the lower teeth, the tongue coming
    out, and the split mouth units adding back up to the unit they came from."""
    import morph
    from compose import place_face

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    shapes = morph.arkit_shapes(vertices, corr, groups)
    get = dict(zip(morph.ARKIT_NAMES, shapes))
    assert len(morph.ARKIT_NAMES) == 52 == len(set(morph.ARKIT_NAMES))
    still = [n for n, s in get.items() if np.abs(s).max() < 1e-3]
    assert not still, f"blendshapes that move nothing: {still}"
    for name, s in get.items():
        if name.endswith(("Left", "Right")) and not name.startswith(("eyeLook", "jaw", "mouthLeft", "mouthRight")):
            moved = np.linalg.norm(s, axis=1)
            x = (vertices[:, 0] * moved).sum() / moved.sum()
            assert (x > 0) == name.endswith("Left"), f"{name} moves the wrong side (x {x:+.3f})"
    for side, group in (("Left", "helper-l-eye"), ("Right", "helper-r-eye")):
        ball = np.unique(groups[group])
        front = ball[np.argsort(vertices[ball, 2])[-8:]]
        nose = -np.sign(vertices[ball, 0].mean())
        move = {look: get[f"eyeLook{look}{side}"][front].mean(axis=0) for look in ("Up", "Down", "In", "Out")}
        assert move["Up"][1] > 0.02 and move["Down"][1] < -0.02, f"{side} eye does not look up and down"
        assert move["In"][0] * nose > 0.02 and move["Out"][0] * nose < -0.02, f"{side} eye does not look in and out"
    teeth = np.unique(groups["helper-lower-teeth"])
    assert get["jawOpen"][teeth, 1].mean() < -0.05, "jawOpen leaves the lower teeth behind"
    tongue = np.unique(groups["helper-tongue"])
    assert get["tongueOut"][tongue, 2].max() > 0.2, "tongueOut does not put the tongue out"
    units = {n: (i, o) for n, i, o in morph.load_expressions()}
    whole = np.zeros_like(shapes[0])
    for name, k in dict(morph.SPLIT_UNITS)["mouthSmile"].items():
        idx, off = units[name]
        whole[idx] += k * off
    assert np.allclose(get["mouthSmileLeft"] + get["mouthSmileRight"], whole, atol=1e-5), "smile halves do not add up"
    # A photo's scores, read back through MediaPipe's measured response: a
    # smile scored what half weight scores is half weight, one at rest is nothing, the eyes
    # are left to the shape fit, and a shape MediaPipe hardly reads on an
    # avatar takes its score as it is.
    half = morph.ARKIT_RESPONSE["mouthSmileLeft"][2]
    face = morph.photo_expression({"mouthSmileLeft": half, "mouthSmileRight": 0.0, "eyeBlinkLeft": 0.9,
                                   "eyeSquintLeft": 0.5, "browDownLeft": 0.4})
    assert abs(face.get("mouthSmileLeft", 0) - 0.5) < 0.01, f"smile not read back through the response: {face}"
    assert not {"mouthSmileRight", "eyeBlinkLeft", "eyeSquintLeft"} & set(face), f"rest or eyes taken from the photo: {face}"
    assert abs(face.get("browDownLeft", 0) - 0.4) < 1e-6, f"unreadable shape not taken as scored: {face}"
    print("ok  arkit: 52 blendshapes, sided right, eyes turn, jaw carries the teeth, tongue out, halves add up, "
          "photo expression read back")


def check_mouth_bake():
    """A laughing photo's mouth, wider than the avatar's closed one, is moved
    onto the avatar's mouth before the bake, and the forehead stays put."""
    import morph
    from compose import mouth_onto_mesh, place_face

    vertices, groups = load_base()
    sheet = place_face(morph.canonical_face(), vertices, groups).vertices
    corr = morph.correspondences(vertices, groups["body"], sheet[:468])
    grin = sheet.copy()
    middle = (grin[61, 0] + grin[291, 0]) / 2
    lips = [61, 291] + morph.OUTER_UPPER + morph.OUTER_LOWER
    grin[lips, 0] = middle + (grin[lips, 0] - middle) * 1.15
    moved = mouth_onto_mesh(grin, vertices, corr)
    width = lambda v: abs(v[61, 0] - v[291, 0])
    mesh = abs(vertices[corr[61], 0] - vertices[corr[291], 0])
    left = abs(width(moved) - mesh) / abs(width(grin) - mesh)
    assert left < 0.3, f"the laugh's mouth stayed {left:.0%} of the way too wide"
    assert np.linalg.norm(moved[10] - grin[10]) < 1e-3, "the forehead moved with the mouth"
    print(f"ok  mouth bake: a laughing mouth moved onto the avatar's ({left:.0%} of the gap left), forehead put")


def check_hairline_sweep():
    """A stock fringe swept back to a higher hairline: its front edge rises most
    of the way there, the back of the hair stays put, and nothing sinks into
    the scalp."""
    import appearance
    import garments
    import trimesh
    from scipy.spatial import cKDTree

    vertices, groups = load_base()
    hair = build_item(items_by_id()["hair_short02"], None, vertices)
    opaque = appearance.opaque_hair(hair)
    front = lambda pts: pts[(np.abs(pts[:, 0]) < 0.2) & (pts[:, 2] > np.median(pts[:, 2]))]
    edge = np.percentile(front(opaque)[:, 1], 3)
    original = hair.vertices.copy()
    turned = garments.sweep_back(hair, vertices, groups, opaque, edge + 0.3)
    assert turned > 0, "the fringe was not swept back"
    risen = np.percentile(front(appearance.opaque_hair(hair))[:, 1], 3) - edge
    assert risen > 0.2, f"the hairline rose {risen:.2f} of the 0.3 asked"
    back = original[:, 2] < np.percentile(original[:, 2], 20)
    assert np.abs(hair.vertices[back] - original[back]).max() < 1e-6, "the back of the hair moved"
    skin = np.unique(groups["body"])
    normals = trimesh.Trimesh(vertices, groups["body"], process=False).vertex_normals
    near = skin[cKDTree(vertices[skin]).query(hair.vertices)[1]]
    moved = np.linalg.norm(hair.vertices - original, axis=1) > 1e-6
    gap = ((hair.vertices - vertices[near]) * normals[near]).sum(axis=1)[moved]
    assert gap.min() > 0.04, f"swept hair sank to {gap.min():.3f} from the scalp"
    print(f"ok  hairline sweep: fringe turned back {turned:.0f} deg, edge rose {risen:.2f} of 0.30, back kept, off the scalp")


def check_hair_volume():
    """A flat stock cut given a photo's height: its top rises to the height
    asked, its front edge stays where it was (the lift ramps in behind it, so
    no shell stands off the forehead), and the back of the hair stays put."""
    import appearance
    import garments

    vertices, groups = load_base()
    hair = build_item(items_by_id()["hair_short02"], None, vertices)
    # Swept back first, as compose does: that is what puts the front edge
    # high enough up the head for the lift to reach it.
    opaque = appearance.opaque_hair(hair)
    ahead = opaque[(np.abs(opaque[:, 0]) < 0.2) & (opaque[:, 2] > np.median(opaque[:, 2]))]
    garments.sweep_back(hair, vertices, groups, opaque, np.percentile(ahead[:, 1], 3) + 0.3)
    original = hair.vertices.copy()
    middle = np.abs(original[:, 0]) < 0.3
    top = np.percentile(original[middle, 1], 98)
    front = middle & (original[:, 2] > np.median(original[:, 2]))
    edge = np.percentile(original[front, 1], 3)
    lift = garments.volumise(hair, vertices, groups, top + 0.2)
    assert lift > 0, "the cut was not lifted"
    risen = np.percentile(hair.vertices[middle, 1], 98) - top
    assert 0.12 < risen < 0.3, f"the top rose {risen:.2f} for 0.2 asked"
    at_edge = front & (original[:, 1] < edge + 0.05)
    assert np.linalg.norm(hair.vertices[at_edge] - original[at_edge], axis=1).max() < 0.02, "the front edge was lifted off"
    back = original[:, 2] < np.percentile(original[:, 2], 15)
    assert np.linalg.norm(hair.vertices[back] - original[back], axis=1).max() < 0.02, "the back of the hair moved"
    print(f"ok  hair volume: top rose {risen:.2f} for 0.20 asked, front edge and back kept")


def check_hair_tangents():
    """A card mesh with an anisotropic material goes out with per-vertex
    tangents: unit length, across the normal, running along u on the
    texture, with a handedness in the last place. Without them a viewer
    derives one tangent per triangle and the hair's sheen broke into facets.
    And the scalp normal a lift follows is smooth across a card."""
    import garments
    from skinned_glb import tangents

    # A flat square with u along x: tangents along +x, w = +1.
    v = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]])
    n = np.tile([0.0, 0, 1], (4, 1))
    t = tangents(v, f, uv, n)
    assert np.allclose(t[:, :3], [1, 0, 0], atol=1e-6) and np.all(t[:, 3] == 1.0), f"tangents {t.tolist()}"
    # v flipped (as glTF texture coordinates are): the direction holds, the handedness flips.
    t2 = tangents(v, f, uv * [1, -1] + [0, 1], n)
    assert np.allclose(t2[:, :3], [1, 0, 0], atol=1e-6) and np.all(t2[:, 3] == -1.0), f"flipped tangents {t2.tolist()}"

    vertices, groups = load_base()
    hair = build_item(items_by_id()["hair_short02"], None, vertices)
    # Across a card seam, where one position is kept as two vertices, the
    # normals and tangents written are the same.
    from compose import welded_normals
    normals = welded_normals(hair)
    tang = tangents(hair.vertices, hair.faces, hair.metadata["uv"], normals)
    _, inverse = np.unique(np.round(hair.vertices, 5), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    copies = np.nonzero(np.bincount(inverse) > 1)[0]
    assert len(copies) > 100, "no card seams found to check"
    for u in copies:
        ids = np.nonzero(inverse == u)[0]
        assert np.allclose(normals[ids], normals[ids[0]], atol=1e-6), "normals differ across a seam"
        assert np.allclose(tang[ids, :3], tang[ids[0], :3], atol=1e-6), "tangents differ across a seam"
    scalp = garments.scalp_normals(hair.vertices, vertices, groups)
    assert np.allclose(np.linalg.norm(scalp, axis=1), 1.0, atol=1e-6), "scalp normals not unit"
    # Along each card edge the normal turns slowly: no two joined vertices differ by more than 30 degrees.
    e = hair.edges_unique
    turn = np.degrees(np.arccos(np.clip((scalp[e[:, 0]] * scalp[e[:, 1]]).sum(axis=1), -1, 1)))
    assert np.percentile(turn, 99) < 30, f"scalp normal turns {np.percentile(turn, 99):.0f} degrees along an edge"
    print(f"ok  hair tangents: along u, handedness follows the texture's v, one per position across {len(copies)} seams; scalp normals turn at most "
          f"{np.percentile(turn, 99):.0f} degrees along a card edge")


def check_hair_cap():
    """A haircut's avatar carries a blended cap under the cards: in the
    hair's root shade, above the brows only, its alpha smooth (not a
    cut-out), opaque under the cards and gone a little below their edge."""
    import io
    import pygltflib as gl2
    from PIL import Image

    photo = HERE / "samples" / "portrait.jpg"
    if not photo.exists():
        print("skip hair cap check: samples/portrait.jpg is missing")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "cap.glb"
        compose(photo, out, "male")
        g = gl2.GLTF2().load(str(out)); blob = g.binary_blob()
        names = [m.name for m in g.materials]
        assert "scalp_cap" in names, f"no scalp cap among {names}"
        m = g.materials[names.index("scalp_cap")]
        assert m.alphaMode == "BLEND", f"cap alpha mode {m.alphaMode}"
        img = g.images[g.textures[m.pbrMetallicRoughness.baseColorTexture.index].source]
        bv = g.bufferViews[img.bufferView]
        alpha = np.asarray(Image.open(io.BytesIO(blob[bv.byteOffset:bv.byteOffset + bv.byteLength])).convert("RGBA"))[..., 3]
        partial = float(((alpha > 0) & (alpha < 255)).mean())
        assert partial > 0.02, f"cap alpha is a cut-out ({partial:.3f} partial)"
        assert (alpha == 255).mean() > 0.05, "cap is nowhere opaque"
        prim = next(p for p in g.meshes[0].primitives if names[p.material] == "scalp_cap")
        a = g.accessors[prim.attributes.POSITION]
        assert a.min[1] > 7.3, f"cap reaches down to y {a.min[1]:.2f}, below the brows"
    print(f"ok  hair cap: blended, {partial:.0%} of its texels fading, opaque under the cards, above the brows")


def check_hair_width():
    """A cut whose outline is narrower over the temples than the photo's is
    pushed out there to the photo's shape, and no further than WIDEN_LIMIT;
    below the brows and at the back of the head it stays put; and a photo
    whose outline the cut already has moves nothing."""
    import appearance
    import garments

    vertices, groups = load_base()
    hair = build_item(items_by_id()["hair_short02"], None, vertices)
    original = hair.vertices.copy()
    brow_y, spacing = float(np.percentile(original[:, 1], 30)), 0.6
    own = []
    for above in appearance.HAIR_WIDTH_ROWS:
        band = np.abs(original[:, 1] - (brow_y + above * spacing)) < 0.05 * spacing
        own.append(np.ptp(original[band, 0]) / 2 / spacing)
    own = np.array(own)
    assert not garments.widen(hair, vertices, groups, brow_y, spacing, list(own)).any(), \
        "a cut with the photo's own outline was widened"
    asked = own * garments.HAIR_RENDER_SPREAD * np.array([1.0, 1.05, 1.1, 1.5])
    push = garments.widen(hair, vertices, groups, brow_y, spacing, list(asked))
    assert push[0] == 0 and push[1] > 0 and push[2] > push[1], f"pushes {push.round(3)}"
    assert push[3] <= garments.WIDEN_LIMIT * own[3] * spacing + 1e-9, "widened past the limit"
    low = original[:, 1] < brow_y - 0.3 * spacing
    assert np.abs(hair.vertices[low] - original[low]).max() < 1e-6, "hair below the brows moved"
    print(f"ok  hair width: pushed out {np.round(push / spacing, 2).tolist()} eye spacings up the temples, "
          f"capped, nothing below the brows")


def check_eye_carry():
    """A sheet whose eyes sit above the mesh's is moved down onto them, its
    brows with it, and its mouth and chin are left where they were."""
    import morph
    from compose import eyes_onto_mesh, place_face
    import appearance

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    sheet = vertices[corr].copy()
    spacing = np.linalg.norm(sheet[33, :2] - sheet[263, :2])
    sheet[:, 1] += 0.05 * spacing                          # the whole sheet a little high
    moved = eyes_onto_mesh(sheet, vertices, corr)
    eyes = list(appearance.EYE_CONTOURS[0]) + list(appearance.EYE_CONTOURS[1])
    share = lambda i: float((sheet[i, 1] - moved[i, 1]).mean() / (0.05 * spacing))
    assert share(eyes) > 0.6, f"the eyes moved {share(eyes):.2f} of the way"
    assert share([105, 334]) > 0.4, f"the brows moved {share([105, 334]):.2f} of the way"
    assert share([152, 17, 0]) < 0.1, f"the mouth and chin moved {share([152, 17, 0]):.2f} of the way"
    print(f"ok  eye carry: eyes moved {share(eyes):.2f} of the way, brows {share([105, 334]):.2f}, "
          f"mouth and chin {share([152, 17, 0]):.2f}")


def check_likeness_details():
    """The small things that each cost a likeness on one photo: skin written
    darker by the renderer's lift and no brighter than the photo; a closed
    mouth left closed; thin lips thinned at the volume targets; and the
    hairline tint solid under the hair's edge, not spared for the photo's
    forehead there."""
    import cv2
    import appearance
    import delight
    import morph
    from PIL import Image
    from compose import place_face

    # Skin: a flat mid-tan face under even light comes back a third darker in
    # linear luminance, and a bright one no brighter than the ceiling.
    canon = load_canonical_faces()
    lab = lambda rgb: cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0, 0]
    class Mark:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z
    face = morph.canonical_face().vertices
    xy = face[:, :2] - face[:, :2].min(axis=0)
    xy = xy / xy.max() * 0.6 + 0.2
    marks = [Mark(float(x), float(1.0 - y)) for x, y in xy]
    photo = np.full((300, 300, 3), (190, 150, 125), np.uint8)
    cheeks = np.array([[m.x * 300, m.y * 300] for m in (marks[50], marks[280], marks[205], marks[425])])
    out, report, mask = delight.delight(photo, marks, canon, cheeks)
    shown = delight.to_linear(photo[0, 0].astype(float)) @ [0.2126, 0.7152, 0.0722]
    got = delight.to_linear(out[mask].astype(float)) @ [0.2126, 0.7152, 0.0722]
    ratio = float(np.median(got)) / shown
    assert abs(ratio * delight.RENDER_LIFT - 1.0) < 0.12, f"skin written at {ratio:.2f} of the photo's, lift {delight.RENDER_LIFT}"

    # Mouth: lips 0.7% of the mouth's width apart stay as the photo has them.
    rgb = np.full((300, 300, 3), (200, 160, 140), np.uint8)
    closed = [Mark(m.x, m.y) for m in marks]
    closed[14] = Mark(closed[13].x, closed[13].y)          # inner lips touching
    filled, was_open = appearance.fill_mouth(rgb, closed, cheeks)
    assert not was_open, "a closed mouth was taken for open"

    # Lips: a photo whose lips are three quarters the skull's height sets the
    # thinning targets, and clears the thickening ones.
    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    names = [n for n, _, _ in morph.load_targets()]
    person = vertices[corr].copy()
    mid = (person[0] + person[17]) / 2
    person[0] = mid + 0.75 * (person[0] - mid)
    person[17] = mid + 0.75 * (person[17] - mid)
    weights = np.zeros(len(names))
    weights[names.index("mouth/mouth-lowerlip-volume-incr")] = 0.5
    weights = morph.lip_volume(weights, names, vertices, corr, person)
    assert weights[names.index("mouth/mouth-lowerlip-volume-decr")] > 0.5, "thin lips did not thin the lower lip"
    assert weights[names.index("mouth/mouth-lowerlip-volume-incr")] == 0.0, "the thickening target was left on"

    # Hairline: on a head with hair points down to a line, the texels just
    # below that line are tinted though the photo shows forehead there.
    uv_all, uv_groups = load_base_uv()
    from compose import head_island, island_uv
    body_faces = groups["body"]
    in_head = head_island(vertices, groups, body_faces, uv_groups["body"])
    faces, uv_faces = body_faces[in_head], uv_groups["body"][in_head]
    uv = island_uv(uv_all, uv_faces)
    brow = vertices[corr[[105, 334]], 1].mean()
    spacing = np.linalg.norm(vertices[corr[33], :2] - vertices[corr[263], :2])
    line_y = brow + 0.35 * spacing
    hair = vertices[np.unique(faces)]
    hair = hair[hair[:, 1] > line_y] + [0.0, 0.0, 0.05]
    plain = Image.new("RGB", (512, 512), (200, 160, 140))
    spare = np.ones(len(vertices))
    tinted = np.asarray(appearance.hairline_tint(plain, vertices, faces, uv, uv_faces, hair, (30, 25, 24), spare=spare))
    used = np.unique(faces)
    ahead = vertices[used, 2] > np.median(vertices[used, 2])
    front = used[ahead & (np.abs(vertices[used, 0]) < 0.25 * spacing) & (vertices[used, 1] < line_y)
                 & (vertices[used, 1] > line_y - 0.9 * appearance.HAIRLINE_SOLID)]
    assert len(front), "no forehead vertex just under the line"
    rows = [np.nonzero(faces == v) for v in front]
    dark = []
    for v, hit in zip(front, rows):
        i = uv_faces[hit[0][0], hit[1][0]]
        x, y = int(uv[i, 0] * 511), int((1 - uv[i, 1]) * 511)
        dark.append(lab(tinted[y, x]))
    assert np.mean(dark) < lab((200, 160, 140)) - 40, f"under the edge the skin stayed L {np.mean(dark):.0f}"
    print(f"ok  likeness details: skin at {ratio:.2f} of the photo for lift {delight.RENDER_LIFT}, closed mouth kept, "
          f"thin lips thinned, hairline tint solid under the edge (L {np.mean(dark):.0f})")


def check_mouth_set():
    """On a face that is not smiling, the mouth's press and draw-in are not
    taken off as expression; on a smiling face they still are."""
    import morph
    from compose import place_face

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    person = morph.reference(vertices, corr, average)
    set_scores = {"mouthPressLeft": 0.5, "mouthPressRight": 0.5, "mouthDimpleLeft": 0.4, "mouthSmileLeft": 0.02}
    _, _, at_rest = morph.fit(vertices, corr, person, average, set_scores)
    assert at_rest["mouth-compression"] == 0.0 and at_rest["mouth-retraction"] == 0.0,         f"a set mouth was fitted as expression: {at_rest}"
    smiling = dict(set_scores, mouthSmileLeft=0.6, mouthSmileRight=0.6)
    _, _, grin = morph.fit(vertices, corr, person, average, smiling)
    assert "mouth-corner-puller" in grin, "the smile unit is missing"
    print("ok  mouth set: press and draw-in kept as shape on a face at rest, fitted as expression with a smile")


def check_lashes_and_brows():
    """After the eye opening is painted over, the lash line is drawn back as
    dark as the photo's along the upper lid, and a brow blurred by a small
    photo is pulled back to its density; skin away from both is untouched."""
    import cv2
    import appearance
    from generate import fit_face

    photo = HERE / "samples" / "portrait.jpg"
    if not photo.exists():
        print("skip lashes and brows check: samples/portrait.jpg is missing")
        return
    _, marks, _, upright = fit_face(photo)
    rgb = np.asarray(upright)
    h, w = rgb.shape[:2]
    L = lambda img: cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB)[..., 0].astype(float)
    contour = appearance.EYE_CONTOURS[1]
    line = np.array([[marks[i].x * w, marks[i].y * h] for i in contour[8:] + [contour[0]]], np.int32)
    band = np.zeros((h, w), np.uint8)
    cv2.polylines(band, [line], False, 1, 2)
    band = band.astype(bool)
    filled = appearance.fill_eyes(rgb, marks)
    lashed = appearance.paint_lashes(filled, rgb, marks)
    before, after, photo_line = L(filled)[band].mean(), L(lashed)[band].mean(), L(rgb)[band].mean()
    assert after < before - 15 and abs(after - photo_line) < 20, f"lash line L {photo_line:.0f} in the photo, {before:.0f} filled, {after:.0f} drawn"
    cheek = (slice(int(marks[50].y * h) - 4, int(marks[50].y * h) + 4), slice(int(marks[50].x * w) - 4, int(marks[50].x * w) + 4))
    assert np.array_equal(lashed[cheek], filled[cheek]), "the cheek changed with the lashes"
    brows = appearance.tighten_brows(lashed, marks)
    from facetexture import BROW_LANDMARKS
    pts = np.array([[marks[i].x * w, marks[i].y * h] for i in BROW_LANDMARKS[:10]], np.int32)
    region = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(pts), 1)
    region = region.astype(bool)
    dark_before = np.percentile(L(lashed)[region], 20)
    dark_after = np.percentile(L(brows)[region], 20)
    assert dark_after < dark_before - 5, f"brow's darkest fifth {dark_before:.0f} -> {dark_after:.0f}"
    assert np.array_equal(brows[cheek], lashed[cheek]), "the cheek changed with the brows"
    print(f"ok  lashes and brows: lash line L {before:.0f} -> {after:.0f} (photo {photo_line:.0f}), brow's darkest fifth {dark_before:.0f} -> {dark_after:.0f}, cheek kept")


def check_lip_correction():
    """Lips the targets cannot reach are moved there free-form, smoothly: the
    lip landmarks most of the way, the eyes and forehead not at all."""
    import morph
    from compose import place_face

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    lips = morph.INNER_UPPER + morph.INNER_LOWER + morph.OUTER_UPPER + morph.OUTER_LOWER + [61, 291, 78, 308]
    person = morph.reference(vertices, corr, average)
    spacing = np.linalg.norm(vertices[corr[33], :2] - vertices[corr[263], :2])
    # Fuller lips: each lip point 4% of the eye spacing further from the mouth's middle.
    mouth = person[lips].mean(axis=0)
    away = person[lips] - mouth
    person[lips, :2] += 0.04 * spacing * (away / np.linalg.norm(away, axis=1, keepdims=True))[:, :2]
    marks = np.zeros(len(corr))
    marks[lips] = 1.0
    moved = morph.correct(vertices, vertices, corr, person, average, np.zeros(1), reliable=marks)
    want = person[lips] - morph.reference(vertices, corr, average)[lips]
    got = moved[corr[lips]] - vertices[corr[lips]]
    share = float((got * want).sum() / (want * want).sum())
    assert 0.5 < share < 1.1, f"the lips moved {share:.2f} of the way"
    still = np.linalg.norm(moved[corr[[33, 263, 10, 151]]] - vertices[corr[[33, 263, 10, 151]]], axis=1).max()
    assert still < 0.002 * spacing, f"eyes and forehead moved {still / spacing:.4f} eye spacings"
    print(f"ok  lip correction: lips moved {share:.2f} of the way, eyes and forehead kept")


def check_side_paint():
    """A surface the front photo never saw, facing a turned photo's camera,
    takes that photo's colour; the texture round it keeps its own."""
    import trimesh
    import appearance
    import views

    # A square facing +x, its texture in the middle of a grey map, and a
    # turned photo, all red, whose camera looks along -x at it.
    square = np.array([[1.0, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    if trimesh.Trimesh(square, faces, process=False).face_normals[0, 0] < 0:
        faces = faces[:, ::-1]
    uv = 0.3 + 0.4 * (square[:, 1:] + 1) / 2
    turn = np.array([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]])     # its third row: towards +x
    red = np.zeros((100, 100, 3), np.uint8)
    red[..., 0] = 220
    real = appearance.segment
    appearance.segment = lambda photo: np.full(photo.shape[:2], appearance.FACE_SKIN)
    try:
        texture, strength = views.paint_sides(Image.new("RGB", (64, 64), (128, 128, 128)), square, faces, uv,
                                              faces, np.zeros(4), (1.0, np.zeros(3)),
                                              [(red, (turn, 0.2, np.zeros(3)))])
    finally:
        appearance.segment = real
    rgb = np.asarray(texture)
    assert tuple(rgb[32, 32]) == (220, 0, 0), f"the middle took {rgb[32, 32]}"
    assert tuple(rgb[2, 2]) == (128, 128, 128), f"the corner took {rgb[2, 2]}"
    print("ok  side paint: a surface facing the turned photo took its colour, the rest kept its own")


def check_expression():
    """A smile is fitted as expression, not identity, and comes off the face.

    The skull's own face, smiling and squinting by MakeHuman's own units, must
    fit back to that expression with next to no change to who it is, and
    taking the fitted expression off its landmarks must give that face back.
    """
    import morph
    from compose import place_face

    vertices, groups = load_base()
    average = place_face(morph.canonical_face(), vertices, groups).vertices[:468]
    corr = morph.correspondences(vertices, groups["body"], average)
    front = morph.front_of(vertices, groups["body"], corr)
    smile = {"mouth-corner-puller": 0.6, "eye-left-slit": 0.4, "eye-right-slit": 0.4}
    own = morph.reference(vertices, corr, average)
    person = own + morph.expression_offsets(vertices, front, smile)
    scores = {"mouthSmileLeft": 0.7, "mouthSmileRight": 0.7,
              "eyeSquintLeft": 0.5, "eyeSquintRight": 0.5}

    weights, _, expression = morph.fit(vertices, corr, person, average, scores, front)
    moved = np.linalg.norm(person - own)
    identity = np.linalg.norm(morph.apply(vertices, weights)[corr] - vertices[corr])
    left = np.linalg.norm(person - morph.expression_offsets(vertices, front, expression) - own)
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
    _, marks, _, upright_photo = fit_face(photo)
    rgb = np.asarray(upright_photo)
    h, w = rgb.shape[:2]
    cheeks = np.array([[marks[i].x * w, marks[i].y * h] for i in SKIN_LANDMARKS])
    out, report, mask = delight.delight(rgb, marks, load_canonical_faces(), cheeks)
    # Skin within the natural range keeps the photo's own brightness, less
    # the renderer's lift (delight.RENDER_LIFT), so it renders as the photo shows it.
    luma = np.array([0.2126, 0.7152, 0.0722])
    at = (cheeks[:, 1].astype(int), cheeks[:, 0].astype(int))
    shown = np.median(delight.to_linear(rgb[at]) @ luma)
    written = np.median(delight.to_linear(out[at]) @ luma)
    assert abs(written * delight.RENDER_LIFT / shown - 1.0) < 0.12,         f"cheeks written at {written / shown:.2f} of the photo's for a lift of {delight.RENDER_LIFT}"
    import cv2
    far = cv2.dilate(mask.astype(np.uint8), np.ones((51, 51), np.uint8)) == 0
    # Away from the face only the exposure factor applies, the same to every pixel.
    scaled = delight.to_srgb(delight.to_linear(rgb) * report["exposure"])
    assert np.abs(out[far].astype(int) - scaled[far].astype(int)).max() <= 1, "pixels away from the face changed"

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
    assert pick({**reading, "sides": 0.08}, "x") == "hair_short_sides", "tight sides not read as a short back and sides"
    assert pick({**reading, "sides": 0.15}, "x") == "hair_short02", "full sides given a short back and sides"

    # The top is the cloth under the chin, not someone else's behind.
    from types import SimpleNamespace as P
    marks = [P(x=0.5, y=0.5)] * 455
    marks[152], marks[234], marks[454] = P(x=0.5, y=0.3), P(x=0.4, y=0.2), P(x=0.6, y=0.2)
    parts = np.zeros((100, 100), np.uint8)
    picture = np.zeros((100, 100, 3), np.uint8)
    parts[40:100, 30:70], picture[40:100, 30:70] = appearance.CLOTHES, (150, 30, 30)
    parts[0:30, 80:100], picture[0:30, 80:100] = appearance.CLOTHES, (30, 30, 150)
    top = appearance.clothing_colour(picture, marks, parts)
    assert top is not None and top[0] > 120 and top[2] < 60, f"read the wrong cloth: {top}"

    photo = HERE / "samples" / "portrait.jpg"
    if photo.exists() and appearance.SEGMENTER_TASK.exists():
        from PIL import Image
        from generate import fit_face
        _, marks, _, upright_photo = fit_face(photo)
        rgb = np.asarray(upright_photo)
        hair = appearance.read_hair(rgb, marks, appearance.segment(rgb))
        assert pick(hair, "x") == "hair_short01", f"close-cropped hair read as {pick(hair, 'x')}"
        assert max(hair["colour"]) < 140, f"dark grey hair read as {hair['colour'].tolist()}"
        _, family = appearance.iris_colour(rgb, marks)
        assert family in ("dark brown", "brown"), f"brown eyes read as {family}"

        # A photo on its side is stood up before anything reads it: a sideways
        # head once read as bald.
        with tempfile.TemporaryDirectory() as tmp:
            sideways = pathlib.Path(tmp) / "sideways.jpg"
            Image.open(photo).rotate(-90, expand=True).save(sideways, quality=95)
            _, marks, _, upright_photo = fit_face(sideways)
            rgb = np.asarray(upright_photo)
            turned = appearance.read_hair(rgb, marks, appearance.segment(rgb))
        assert pick(turned, "x") == "hair_short01", f"a sideways photo's hair read as {pick(turned, 'x')}"
        assert abs(turned["amount"] - hair["amount"]) < 0.05, f"sideways hair amount {turned['amount']} vs {hair['amount']}"

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

    if (ASSET_DIR / "mh_system" / "hair" / "short02" / "short02.mhclo").exists():
        # The short back and sides lies closer to the scalp at the sides than
        # the cut it is made from, and clears the ears.
        from scipy.spatial import cKDTree
        from catalogue import build_item, items_by_id
        base, groups = load_base()
        scalp = cKDTree(base[np.unique(groups["body"])])

        def sides(cut):
            hair = build_item(items_by_id()[cut], np.array([40, 40, 40], np.uint8), base)
            v = hair.vertices[np.unique(hair.faces)]
            v = v[(np.abs(v[:, 0]) > 0.75) & (v[:, 1] < 7.9)]
            return scalp.query(v)[0].mean(), len(v)
        (full, n_full), (clipped, n_clipped) = sides("hair_short02"), sides("hair_short_sides")
        assert clipped < 0.8 * full and n_clipped < n_full, \
            f"sides not clipped: {clipped:.3f} vs {full:.3f} off the scalp, {n_clipped} vs {n_full} vertices"
    print("ok  appearance: haircut rules, hair and eye colour from the photo, shirt and trousers dyed apart, short sides")


def check_finishing():
    """Teeth painted out of an open mouth, impossible skin casts pulled back,
    crevices lifted only where they belong, socks gone under trousers,
    trousers taken in, recoloured hair stored as a palette."""
    import cv2
    import appearance
    import delight
    from catalogue import ASSET_DIR

    photo = HERE / "samples" / "portrait.jpg"
    if photo.exists():
        from generate import fit_face
        _, marks, _, upright_photo = fit_face(photo)
        rgb = np.asarray(upright_photo)
        h, w = rgb.shape[:2]
        mouth = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mouth, [np.array([[marks[i].x * w, marks[i].y * h] for i in delight.INNER_LIPS], np.int32)], 1)
        filled, did = appearance.fill_mouth(rgb, marks)
        bright = lambda img: int((cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0][mouth > 0] > 200).sum())
        assert did and bright(rgb) > 50 and bright(filled) == 0, "teeth were left in the mouth"

        mask = np.zeros((h, w), bool)
        mask[int(marks[10].y * h):int(marks[152].y * h), int(marks[234].x * w):int(marks[454].x * w)] = True
        assert appearance.correct_cast(rgb, mask)[1] == (0.0, 0.0), "natural skin was colour-corrected"
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
        lab[..., 1] -= 16                                            # a green cast
        green = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        _, shift = appearance.correct_cast(green, mask)
        assert shift[0] > 4, f"a green cast was not pulled back (shift {shift})"

        lifted, amount = delight.lift_crevices(rgb, marks, (w, h))
        corner = (slice(0, h // 10), slice(0, w // 10))
        assert amount > 0 and np.array_equal(lifted[corner], rgb[corner]), "lift reached outside the face"

    shoes = ASSET_DIR / "mh_system" / "clothes" / "shoes01" / "shoes01.mhclo"
    if shoes.exists():
        from garments import drop_socks, load_mhclo, material_textures
        base, groups = load_base()
        ankle = float(base[np.unique(groups["joint-l-ankle"])][:, 1].mean())
        mesh = load_mhclo(shoes, base)
        removed = drop_socks(mesh, ankle)
        assert removed > 0 and mesh.vertices[np.unique(mesh.faces)][:, 1].max() < ankle + 0.5, "sock kept"
        hair = ASSET_DIR / "mh_system" / "hair" / "short02" / "short02.mhmat"
        texture, _, _ = material_textures(hair, (100, 70, 50))
        import io
        from PIL import Image
        png = io.BytesIO()
        texture.save(png, format="PNG")
        alpha = np.asarray(Image.open(png).convert("RGBA"))[..., 3]
        assert texture.mode == "P" and alpha.min() == 0 and alpha.max() == 255, \
            "recoloured hair lost its palette or its alpha"

        item = dict(items_by_id()["outfit_light_blue_shirt_navy_trousers"])
        loose = build_item({**item, "slim": None}, None, base)
        slim = build_item(item, None, base)
        gap = lambda m: np.linalg.norm(m.vertices - (m.metadata["mhclo_blend"][:, :, None]
                                                     * base[m.metadata["mhclo_idx"]]).sum(axis=1), axis=1)
        legs = loose.vertices[:, 1] < -3.0
        assert gap(slim)[legs].mean() < 0.75 * gap(loose)[legs].mean(), "the trousers were not taken in"
    print("ok  finishing: teeth painted out, casts pulled back, crevices lifted, socks dropped, "
          "trousers slimmed, hair as palette")


def check_review_fixes():
    """Forearms hang instead of reaching forward; a broad smile is not taken as
    a wide mouth; the neck is never cut away under a collar."""
    from compose import relaxed_pose

    base, groups = load_base()
    skeleton = load_skeleton(base)
    m = skinning_matrices(skeleton, relaxed_pose())
    at = lambda name: (m[skeleton.index[name]] @ np.append(skeleton.heads[skeleton.index[name]], 1.0))[:3]
    forearm = at("wrist.L") - at("lowerarm01.L")
    # Forward lean only (the arm's sideways splay is separate). The base mesh
    # is modelled with forearms reaching forward ~47 degrees; relaxed is ~24.
    off_vertical = np.degrees(np.arctan2(forearm[2], -forearm[1]))
    assert off_vertical < 30, f"forearm leans {off_vertical:.0f} degrees forward: reaching out"
    # The palm (thumb side of the knuckle plane) faces the thigh, not forward
    # and up, and the fingers curl towards it.
    head = lambda name: skeleton.heads[skeleton.index[name]]
    knuckles = (head("finger2-1.L"), head("finger5-1.L"))
    palm = np.cross(knuckles[1] - knuckles[0], skeleton.tails[skeleton.index["finger3-3.L"]] - head("finger3-1.L"))
    palm *= np.sign(palm @ (skeleton.tails[skeleton.index["finger1-3.L"]] - sum(knuckles) / 2))
    facing = m[skeleton.index["wrist.L"]][:3, :3] @ (palm / np.linalg.norm(palm))
    assert facing[0] < -0.8, f"left palm faces {facing.round(2)}, not the thigh"
    tip = lambda pose: (skinning_matrices(skeleton, pose)[skeleton.index["finger3-3.L"]]
                        @ np.append(skeleton.tails[skeleton.index["finger3-3.L"]], 1.0))[:3]
    straight = {k: v for k, v in relaxed_pose().items() if not k.startswith("finger")}
    reach = lambda pose: np.linalg.norm(tip(pose) - at("finger3-1.L"))
    assert reach(relaxed_pose()) < 0.97 * reach(straight), "the fingers do not curl"
    # The thumb hangs down beside the index finger, not across the thigh.
    thumb = (m[skeleton.index["finger1-3.L"]] @ np.append(skeleton.tails[skeleton.index["finger1-3.L"]], 1.0))[:3] \
        - at("finger1-1.L")
    drop = np.degrees(np.arcsin(-thumb[1] / np.linalg.norm(thumb)))
    assert drop > 40, f"the thumb points {drop:.0f} degrees below horizontal: across the thigh"

    photo = HERE / "samples" / "portrait.jpg"
    if not photo.exists():
        print(f"ok  review fixes: forearm leans {off_vertical:.0f} deg forward; portrait checks skipped")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "portrait.glb"
        compose(photo, out)
        import json
        info = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        mouth = [w for name, w in info["shape"]["strongest"] if name.startswith("mouth/")]
        assert not any(w > 0.6 for w in mouth), f"the smile was taken as mouth shape: {info['shape']['strongest']}"
        # No scalp layer under a real haircut: standing 0.055 off the skin, it
        # covered the cards at the back of the head as a flat bald patch.
        assert "hair_underlayer" not in info["wearing"], "a scalp layer was worn under the haircut"

        import io
        from PIL import Image
        gltf = gl.GLTF2().load(str(out))
        blob = gltf.binary_blob()
        names = [mat.name for mat in gltf.materials]
        prim = gltf.meshes[0].primitives[names.index("body")]

        def read(index, width):
            acc = gltf.accessors[index]
            view = gltf.bufferViews[acc.bufferView]
            # Texture coordinates are stored as normalised shorts (KHR_mesh_quantization).
            if acc.componentType == gl.UNSIGNED_SHORT:
                raw = np.frombuffer(blob[view.byteOffset:view.byteOffset + view.byteLength], np.uint16)
                return raw.reshape(-1, width).astype(np.float64) / 65535.0
            return np.frombuffer(blob[view.byteOffset:view.byteOffset + view.byteLength], np.float32).reshape(-1, width)
        pos, uv = read(prim.attributes.POSITION, 3), read(prim.attributes.TEXCOORD_0, 2)
        tex = gltf.materials[names.index("body")].pbrMetallicRoughness.baseColorTexture.index
        view = gltf.bufferViews[gltf.images[gltf.textures[tex].source].bufferView]
        alpha = np.asarray(Image.open(io.BytesIO(blob[view.byteOffset:view.byteOffset + view.byteLength])).convert("RGBA"))[..., 3]
        collarbone = float(base[np.unique(groups["joint-l-clavicle"])][:, 1].mean())
        neck = (pos[:, 1] > collarbone + 0.15) & (np.abs(pos[:, 0]) < 0.6)
        px = (uv[neck] * (np.array(alpha.shape[::-1]) - 1)).astype(int)
        assert neck.any() and (alpha[px[:, 1], px[:, 0]] > 127).mean() > 0.98, "neck skin was cut away"

        # Severity-3 detail: blendshapes on every part, teeth and tongue worn,
        # skin with its own finish, cloth with sheen, hair with anisotropy.
        import morph
        assert gltf.meshes[0].extras.get("targetNames") == morph.ARKIT_NAMES, "ARKit blendshapes missing"
        assert all(len(p.targets) == 52 for p in gltf.meshes[0].primitives), "a part lacks blendshapes"
        assert {"teeth", "tongue"} <= set(info["wearing"]), "no teeth or tongue in the mouth"
        skin_mat = gltf.materials[names.index("head")]
        assert skin_mat.pbrMetallicRoughness.metallicRoughnessTexture is not None, "skin has one roughness"
        assert "KHR_materials_ior" in skin_mat.extensions, "skin reflects like plastic"
        from PIL import Image as _Image
        for part in ("head", "body"):
            view = gltf.bufferViews[gltf.images[gltf.textures[
                gltf.materials[names.index(part)].pbrMetallicRoughness.baseColorTexture.index].source].bufferView]
            size = _Image.open(io.BytesIO(blob[view.byteOffset:view.byteOffset + view.byteLength])).size
            assert size == (2048, 2048), f"{part} skin texture is {size}, not 2048"
        cloth = [m for m in gltf.materials if m.name.startswith("outfit")]
        assert cloth and all("KHR_materials_sheen" in m.extensions for m in cloth), "cloth has no sheen"
        hair = [m for m in gltf.materials if m.name.startswith("hair_")]
        assert hair and all("KHR_materials_anisotropy" in m.extensions and m.normalTexture for m in hair), \
            "hair has no strand highlight or relief"
    print(f"ok  review fixes: forearm leans {off_vertical:.0f} deg forward, smile not taken as mouth shape, neck kept")


def check_forehead():
    """Past the landmarks the bake reaches the hairline the segmenter saw, only
    on surface facing the camera."""
    from facetexture import FaceSampler

    # A flat 1 x 1 sheet whose pixels are its coordinates x 100, and a photo
    # that is skin everywhere.
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11)), -1).reshape(-1, 2)
    sheet = np.c_[grid, np.zeros(len(grid))]
    rim = np.nonzero((grid == 0).any(axis=1) | (grid == 1).any(axis=1))[0]
    sampler = FaceSampler(sheet, grid * 100, rim, beyond=np.ones((200, 200)))
    above = np.array([[0.5, 1.1, 0.0], [0.5, 1.1, 0.0], [0.5, 2.0, 0.0]])
    pixels, strength = sampler.sample(above, np.array([1.0, 0.3, 1.0]))
    assert np.allclose(pixels[0], [50, 110], atol=1), f"forehead mapped to {pixels[0]}"
    assert strength[0] > 0.9, "the forehead above the landmarks was not baked"
    assert strength[1] == 0, "a surface turned away from the camera was baked"
    assert strength[2] == 0, "the bake reached far past the landmarks"
    print("ok  forehead: baked to the hairline, only on surface facing the camera")


def check_details():
    """The detail audit's faults: feet hidden inside shoes, hands clear of a
    woman's hips, irises with fibres rather than a flat disc."""
    import cv2
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    import compose
    import garments
    import morph
    from catalogue import ASSET_DIR, build_item, items_by_id, submesh

    base, groups = load_base()

    # Feet: below the shoe's opening the foot is enclosed and its skin hidden;
    # a toe fuller than the shoe came through the sole.
    shoes = build_item(items_by_id()["shoes_black"], None, base)
    enclosed = garments.enclosed_by_shoe(shoes, base)
    body = np.unique(groups["body"])
    soles = body[base[body, 1] < base[body, 1].min() + 0.5]
    assert enclosed[soles].all(), "foot skin under the sole is still drawn"
    assert not enclosed[base[:, 1] > shoes.vertices[:, 1].max()].any(), "skin above the shoes was hidden"

    # Hands: the arm pose is set on the male body; on the female body the
    # hands sat 0.27 inside the thighs. Swung out, they clear the hips, and
    # neither body's arms turn further than they need.
    for gender, most in (("female", 10.0), ("male", 4.0)):
        v = morph.body_for(base, gender)
        skeleton = load_skeleton(v)
        joints, weights = weight_table(skeleton, len(v))
        mesh = submesh(groups["body"], v)
        source = mesh.metadata["base_index"]
        part = SimpleNamespace(vertices=np.asarray(mesh.vertices, np.float32), joints=joints[source],
                               weights=weights[source], normals=np.asarray(mesh.vertex_normals, np.float32))
        pose = compose.relaxed_pose()
        posed = compose.arms_clear(pose, skeleton, part)
        clear = compose.hand_clearance(posed, skeleton, part)
        turned = max(np.degrees((Rotation.from_quat(posed[f"upperarm01.{s}"])
                                 * Rotation.from_quat(pose[f"upperarm01.{s}"]).inv()).magnitude()) for s in "LR")
        assert min(clear.values()) >= 0.12, f"{gender} hands inside the hips: {clear}"
        assert turned <= most, f"{gender} arms swung out {turned:.0f} degrees"
        if gender == "female":
            assert min(compose.hand_clearance(pose, skeleton, part).values()) < 0, "the fault no longer reproduces"

    # Eyes: fibres across the iris, not a flat disc shaded only by radius.
    _, _, texture = compose.build_eyes(base, groups, np.array([90.0, 120.0, 150.0]))
    lab = cv2.cvtColor(np.asarray(texture), cv2.COLOR_RGB2LAB).astype(np.float32)
    iris = cv2.erode((lab[..., 2] < 118).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    detail = lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), 3)
    assert iris.sum() > 1000 and detail[iris].std() > 3, f"the iris is flat (detail {detail[iris].std():.1f})"

    # Brows: a card authored for blending (alpha up to 187) keeps its strands
    # through the 0.5 cut, not a few clumps.
    brow = ASSET_DIR / "mh_system" / "eyebrows" / "eyebrow006"
    if (brow / "eyebrow006.mhmat").exists():
        from PIL import Image
        texture, _, _ = garments.material_textures(brow / "eyebrow006.mhmat", (60, 40, 30))
        authored = np.asarray(Image.open(brow / "eyebrow006.png").convert("RGBA"))[..., 3]
        kept = np.asarray(texture.convert("RGBA"))[..., 3]
        assert (kept > 127).sum() > 2 * (authored > 127).sum(), "the thin brow is still cut to clumps"

    # Seams: a kept beard fades out at the face sheet's edge instead of
    # stopping on it.
    from facetexture import FaceSampler
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11)), -1).reshape(-1, 2)
    rim = np.nonzero((grid == 0).any(axis=1) | (grid == 1).any(axis=1))[0]
    sampler = FaceSampler(np.c_[grid, np.zeros(len(grid))], grid * 100, rim, keep=np.ones((200, 200)))
    _, strength = sampler.sample(np.array([[0.5, 0.5, 0.0], [0.5, 0.003, 0.0]]))
    assert strength[0] > 0.99 and strength[1] < 0.2, f"beard kept to the sheet's edge: {strength.round(2)}"
    # Plain dye: a pinstripe dyed plain loses its stripes, keeps broad shading.
    import trimesh
    from PIL import Image
    # A sheet of small triangles: the rasteriser skips any spanning half the
    # texture as a UV seam.
    cells = 8
    ij = np.stack(np.meshgrid(np.arange(cells + 1), np.arange(cells + 1), indexing="ij"), -1).reshape(-1, 2)
    corner = lambda i, j: i * (cells + 1) + j
    tris = [[corner(i, j), corner(i + 1, j), corner(i + 1, j + 1)] for i in range(cells) for j in range(cells)] +            [[corner(i, j), corner(i + 1, j + 1), corner(i, j + 1)] for i in range(cells) for j in range(cells)]
    quad = trimesh.Trimesh(np.c_[ij / cells, np.zeros(len(ij))], tris, process=False)
    quad.metadata["uv"] = ij / cells
    stripes = np.full((256, 256, 3), 40, np.uint8)
    stripes[:, ::12] = (200, 30, 60)
    stripes[:128] = (stripes[:128] * 0.6).astype(np.uint8)          # a broad fold: top half darker
    dyed = np.asarray(garments.recolour(Image.fromarray(stripes), quad, np.ones(len(tris), bool), (168, 192, 222), plain=True))
    lightness = dyed.astype(float).mean(axis=2)
    assert lightness[200, 20:120].std() < 3, f"stripes survived a plain dye (spread {lightness[200, 20:120].std():.1f})"
    assert lightness[40, 128] < lightness[200, 128] - 8, "a plain dye flattened the fold too"

    # Nails: on the back of each finger's last joint, a patch of tens of vertices.
    import trimesh as _t
    import appearance
    joints, weights = weight_table(load_skeleton(base), len(base))
    skeleton = load_skeleton(base)
    nails = appearance.nail_weight(base, _t.Trimesh(base, groups["body"], process=False).vertex_normals,
                                   skeleton, joints, weights)
    dominant = joints[np.arange(len(joints)), weights.argmax(axis=1)]
    for finger in range(2, 6):
        bone = f"finger{finger}-3.L"
        rows = dominant == skeleton.index[bone]
        assert 15 <= (nails[rows] > 0.5).sum() <= 80, f"{bone}: {(nails[rows] > 0.5).sum()} nail vertices"
    # Seams are not openings: two pieces sewn edge to edge keep no skin band.
    piece = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0.0, 0], [2, 0, 0], [1, 1, 0]],
                            [[0, 1, 2], [3, 4, 5]], process=False)
    rim = set(garments.open_rim(piece).tolist())
    assert 1 not in rim and 3 not in rim and {0, 2, 4, 5} <= rim, f"seam taken as an opening: {sorted(rim)}"

    # Advice: a small face and a broad smile are both told to the user.
    import compose as _c
    from types import SimpleNamespace as P
    marks = [P(x=0.5, y=0.5)] * 455
    marks[234], marks[454] = P(x=0.45, y=0.5), P(x=0.55, y=0.5)
    advice = _c.photo_advice(marks, (600, 800), {"mouthSmileLeft": 0.8})
    assert len(advice) == 2 and "pixels" in advice[0] and "smiling" in advice[1], advice
    assert _c.photo_advice(marks, (6000, 8000), {}) == [], "a good photo was given advice"

    # Ears: the mask from the ear targets touches the sides of the head only.
    ear = appearance.ear_weight(len(base))
    on = ear > 0.5
    assert on.sum() > 200 and np.abs(base[on, 0]).min() > 0.4, "ear mask strays off the ears"
    print("ok  details: feet hidden in shoes, hands clear the hips, irises have fibres, "
          "thin brows kept, beard fades at the sheet's edge, plain dye, nails, ears")


def check_brows():
    """The brow cards lie on the brows the bake paints, not under them.

    MakeHuman hangs its cards 4 to 9 mm below the middle of MediaPipe's brow,
    and the bake paints the photo's brows where MediaPipe puts them: a second,
    thinner brow under the first. MediaPipe's brow also runs high, so the
    cards go where the photo's brow is darkest. Checked on the average face,
    where the gap is the template's alone, on both bodies.
    """
    import trimesh
    import morph
    from compose import body_mesh, brows_to_photo, place_face
    from facetexture import BROW_LANDMARKS, brow_depth

    # A dark band 0.9 of the way down from the upper contour to the lower.
    photo = np.full((200, 200, 3), (225, 185, 165), dtype=np.uint8)
    photo[88:91, 30:170] = (70, 50, 40)
    contours = np.array(BROW_LANDMARKS).reshape(2, 2, 5)   # side, upper/lower, outer to inner
    marks = _landmarks({int(i): (x + 0.05 * k, 0.40 + 0.05 * row) for x, side in zip((0.2, 0.6), contours)
                        for row, ids in enumerate(side) for k, i in enumerate(ids)})
    depth = brow_depth(photo, marks)
    assert np.allclose(depth, 0.9, atol=0.1), f"a brow 0.9 of the way down read as {depth}"

    raw, groups = load_base()
    for gender, item in (("male", "eyebrows"), ("female", "eyebrows_thin")):
        vertices = morph.body_for(raw, gender)
        sheet = place_face(morph.canonical_face(), vertices, groups).vertices
        card = build_item(items_by_id()[item], None, vertices)
        skin = body_mesh(vertices)
        band = skin.triangles[np.abs(skin.triangles_center[:, 1] - card.vertices[:, 1].mean()) < 0.3]

        def off_skin(p):
            """Median distance to the skin's surface, brute force over the brow band."""
            q = trimesh.triangles.closest_point(np.tile(band, (len(p), 1, 1)), np.repeat(p, len(band), 0))
            return np.median(np.linalg.norm(q - np.repeat(p, len(band), 0), axis=1).reshape(len(p), -1).min(axis=1))

        before = off_skin(card.vertices)
        v = brows_to_photo(card, sheet, depth, vertices, groups).vertices
        upper, lower = np.moveaxis(sheet[contours], 1, 0)
        for line, own in zip(upper + 0.9 * (lower - upper), (v[:, 0] < 0, v[:, 0] > 0)):
            line = line[np.argsort(line[:, 0])]
            span = own & (v[:, 0] > line[0, 0]) & (v[:, 0] < line[-1, 0])
            off = float((v[span, 1] - np.interp(v[span, 0], line[:, 0], line[:, 1])).mean())
            assert abs(off) < 0.01, f"{item} sits {off * 100:+.1f} mm off the painted brow"
        after = off_skin(v)
        assert after < before + 0.003, f"{item} lifted {(after - before) * 100:.1f} mm off the skin"
    print(f"ok  brows: read {depth.round(2).tolist()} down the contours; cards on that line "
          f"within 1 mm, still on the skin")

if __name__ == "__main__":
    try:
        check_weights()
        check_export()
        check_transfer()
        check_alignment()
        check_proportions()
        check_side_views()
        check_brow_shading()
        check_arkit()
        check_mouth_bake()
        check_hairline_sweep()
        check_hair_volume()
        check_hair_width()
        check_hair_tangents()
        check_hair_cap()
        check_lip_correction()
        check_lashes_and_brows()
        check_mouth_set()
        check_eye_carry()
        check_likeness_details()
        check_side_paint()
        check_expression()
        check_mhclo()
        check_import()
        check_photo_reading()
        check_bodies_and_assets()
        check_delight()
        check_appearance()
        check_finishing()
        check_review_fixes()
        check_forehead()
        check_brows()
        check_details()
    except AssertionError as err:
        print(f"FAIL  {err}", file=sys.stderr)
        raise SystemExit(1)
    print("rig ok")
