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

from catalogue import load_base
from compose import compose
from rig import MAX_INFLUENCES, load_skeleton, weight_table

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


if __name__ == "__main__":
    try:
        check_weights()
        check_export()
    except AssertionError as err:
        print(f"FAIL  {err}", file=sys.stderr)
        raise SystemExit(1)
    print("rig ok")
