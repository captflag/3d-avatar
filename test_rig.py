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
    weights, _ = morph.fit(vertices, corr, person, average)
    shaped = morph.apply(vertices, weights)
    placed = place_face(face, shaped, groups).vertices
    skull = head_features(shaped, groups)

    for name, point, anatomy in (("nose tip", placed[1], skull["nose"]),
                                 ("mouth", (placed[13] + placed[14]) / 2, skull["mouth"]),
                                 ("chin", placed[152], skull["chin"])):
        off = float(np.linalg.norm((point - anatomy)[:2]))
        assert off < 0.1, f"face {name} is {off * 10:.1f} cm off the skull's {name}"
    print("ok  alignment: nose, mouth and chin within 1 cm of the skull's own")


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


if __name__ == "__main__":
    try:
        check_weights()
        check_export()
        check_transfer()
        check_alignment()
        check_mhclo()
        check_import()
    except AssertionError as err:
        print(f"FAIL  {err}", file=sys.stderr)
        raise SystemExit(1)
    print("rig ok")
