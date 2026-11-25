"""A fitted garment hung under gravity on the body standing, in Blender's cloth simulation, at higher
detail, and bound back to the body: a new OBJ and MHCLO (`pose relaxed`) in OUT_DIR, beside the
original's material and textures.

    python drape.py IN.mhclo OUT_DIR --gender male [--levels 1] [--slack 0.08] [--frames 80]

Fitted by projection, a garment hugs the body: sleeves and trouser legs follow every muscle, its
surface is crinkled all over by the fit, and its only folds are those painted into its texture.
Here its seams are sewn, the crinkle smoothed out, its surface subdivided (`--levels`, each about
four times the points, the texture coordinates carried along), given `--slack` off the skin, then
dropped onto the standing body (a collision object) with its supports pinned: the shoulders and
neck of a top, the waistband of trousers or a skirt, where they lie within PIN_NEAR of the skin.
It settles into the folds cloth makes. Each point of the result is then bound to the nearest body
triangle (its three corners, barycentric weights and the offset off it), as fit_garment binds.
Blender (GPL) is only run here, as a tool; nothing of it ships.

Units: the mesh is in decimetres, Y up; Blender is handed metres, Y up, with gravity along -Y.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import trimesh

import catalogue
import compose
import garments
import morph
import rig

BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe"
# How far apart seam vertices may lie and still be sewn (m, in Blender); how close to the skin a
# support point lies to be pinned (dm).
WELD_M, PIN_NEAR = 0.004, 0.35
# A simulation whose points move further than this (dm, 99.9th percentile) blew up and is rejected.
BLOWUP = 1.0
# How far off the skin the cloth starts, at least (dm).
START_CLEAR = 0.06

SIM = r'''
import bpy, bmesh, numpy as np, sys, json
from mathutils import kdtree
a = json.loads(sys.argv[sys.argv.index("--") + 1])
d = np.load(a["npz"])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.gravity = (0.0, -9.81, 0.0)
scene.frame_start, scene.frame_end = 1, a["frames"]

def link(name, me):
    ob = bpy.data.objects.new(name, me); scene.collection.objects.link(ob); return ob

me = bpy.data.meshes.new("body"); me.from_pydata(d["body_v"].tolist(), [], d["body_f"].tolist()); me.update()
body = link("body", me)
body.modifiers.new("collision", "COLLISION")
body.collision.thickness_outer = a["thickness"]; body.collision.cloth_friction = 10.0

# The cloth with its texture coordinates per corner, its seams sewn.
me = bpy.data.meshes.new("cloth"); me.from_pydata(d["cloth_v"].tolist(), [], d["cloth_f"].tolist()); me.update()
uv = me.uv_layers.new(name="UV"); uv.data.foreach_set("uv", d["cloth_uv"].reshape(-1))
bm = bmesh.new(); bm.from_mesh(me)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=a["weld"])
bm.to_mesh(me); bm.free()
cloth = link("cloth", me)
low = [v.co.copy() for v in me.vertices]
pin_low = d["cloth_pin"]                    # per input vertex; mapped by nearest after welding
tree_in = kdtree.KDTree(len(d["cloth_v"]))
for i, co in enumerate(d["cloth_v"]): tree_in.insert(co, i)
tree_in.balance()
if a["smooth"] > 0:
    # Not the parts held as fitted (a hood, a collar): smoothed, a hood flattened onto the back.
    soft = cloth.vertex_groups.new(name="soft")
    for v in me.vertices:
        co, i, dist = tree_in.find(v.co)
        soft.add([v.index], float(d["cloth_soft"][i]), "REPLACE")
    m = cloth.modifiers.new("smooth", "LAPLACIANSMOOTH"); m.iterations = a["smooth"]; m.lambda_factor = 0.5
    m.use_volume_preserve = True; m.use_normalized = True; m.vertex_group = "soft"
if a["levels"] > 0:
    m = cloth.modifiers.new("subsurf", "SUBSURF"); m.levels = m.render_levels = a["levels"]
    m.uv_smooth = "PRESERVE_BOUNDARIES"; m.boundary_smooth = "PRESERVE_CORNERS"
deps = bpy.context.evaluated_depsgraph_get()
fine = bpy.data.meshes.new_from_object(cloth.evaluated_get(deps))
cloth.modifiers.clear(); cloth.data = fine
# Slack: each point lifted along its normal.
for v in fine.vertices: v.co += v.normal * a["slack"]
group = cloth.vertex_groups.new(name="pin")
for v in fine.vertices:
    co, i, dist = tree_in.find(v.co - v.normal * a["slack"])
    if pin_low[i] > 0: group.add([v.index], float(pin_low[i]), "REPLACE")
mod = cloth.modifiers.new("cloth", "CLOTH"); s = mod.settings
s.quality = a["quality"]; s.mass = a["mass"]; s.air_damping = 2.0
s.tension_stiffness = s.compression_stiffness = 20.0; s.shear_stiffness = 8.0
s.bending_stiffness = a["bending"]; s.bending_model = "ANGULAR"
s.vertex_group_mass = "pin"; s.pin_stiffness = 1.0
c = mod.collision_settings
c.use_collision = True; c.distance_min = a["thickness"]; c.collision_quality = 3
c.use_self_collision = a["self"]; c.self_distance_min = 0.0015; c.self_friction = 5.0
mod.point_cache.frame_start, mod.point_cache.frame_end = 1, a["frames"]
if a["after"] > 0:
    m = cloth.modifiers.new("after", "SMOOTH"); m.factor = 0.5; m.iterations = a["after"]
for f in range(1, a["frames"] + 1): scene.frame_set(f)
deps = bpy.context.evaluated_depsgraph_get()
res = cloth.evaluated_get(deps).to_mesh()
res.calc_loop_triangles()
V = np.empty(len(res.vertices) * 3); res.vertices.foreach_get("co", V)
T = np.empty(len(res.loop_triangles) * 3, dtype=np.int64); res.loop_triangles.foreach_get("vertices", T)
L = np.empty(len(res.loop_triangles) * 3, dtype=np.int64); res.loop_triangles.foreach_get("loops", L)
UV = np.empty(len(res.loops) * 2); res.uv_layers.active.data.foreach_get("uv", UV)
P0 = np.empty(len(fine.vertices) * 3); fine.vertices.foreach_get("co", P0)
np.savez(a["out"], v=V.reshape(-1, 3), f=T.reshape(-1, 3), uv=UV.reshape(-1, 2)[L].reshape(-1, 3, 2), start=P0.reshape(-1, 3))
'''


def closest_on_triangle(p, a, b, c):
    """The closest point of each triangle (a, b, c) to each point p, as barycentric weights (n, 3)
    (Ericson, Real-Time Collision Detection 5.1.5, vectorised)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v, w = vb / denom, vc / denom
        out = np.c_[1 - v - w, v, w]
        t_ab, t_ac = d1 / (d1 - d3), d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    zero = np.zeros(len(p))
    cases = [((d1 <= 0) & (d2 <= 0), np.c_[zero + 1, zero, zero]),
             ((d3 >= 0) & (d4 <= d3), np.c_[zero, zero + 1, zero]),
             ((d6 >= 0) & (d5 <= d6), np.c_[zero, zero, zero + 1]),
             ((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.c_[1 - t_ab, t_ab, zero]),
             ((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.c_[1 - t_ac, zero, t_ac]),
             ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), np.c_[zero, 1 - t_bc, t_bc])]
    done = np.zeros(len(p), bool)
    for mask, val in cases:
        m = mask & ~done
        out[m] = val[m]; done |= m
    return np.nan_to_num(out)


def nearest_triangle(points, vertices, faces, k: int = 16):
    """Each point's nearest triangle of a mesh (its corner indices) and barycentric weights of the
    closest point on it, among the `k` triangles whose centres are nearest."""
    from scipy.spatial import cKDTree
    tris = vertices[faces]
    _, cand = cKDTree(tris.mean(axis=1)).query(points, k=k)
    best_d, best_t, best_b = np.full(len(points), np.inf), np.zeros(len(points), int), np.zeros((len(points), 3))
    for j in range(k):
        t = cand[:, j]
        b = closest_on_triangle(points, tris[t, 0], tris[t, 1], tris[t, 2])
        q = np.einsum("nk,nkd->nd", b, tris[t])
        d = np.linalg.norm(points - q, axis=1)
        better = d < best_d
        best_d[better], best_t[better], best_b[better] = d[better], t[better], b[better]
    return faces[best_t], best_b


def parse_mhclo(path: pathlib.Path):
    """The file's lines, where its vertex rows start and stop, the rows (idx, weights, offsets), its
    scales and its pose."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("verts")) + 1
    rows, k = [], start
    while k < len(lines) and lines[k].split() and lines[k].split()[0].lstrip("-").isdigit():
        p = lines[k].split()
        rows.append([float(x) for x in p] if len(p) >= 9 else [float(p[0])] * 3 + [1.0, 0, 0, 0, 0, 0])
        k += 1
    scales = {l.split()[0][0]: (int(l.split()[1]), int(l.split()[2]), float(l.split()[3]))
              for l in lines if l.split() and l.split()[0] in ("x_scale", "y_scale", "z_scale")}
    pose = next((l.split()[1] for l in lines if l.split()[:1] == ["pose"]), None)
    return lines, start, k, np.array(rows), scales, pose


def obj_faces(path: pathlib.Path, with_uv: bool = False):
    faces, uvf, vt = [], [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("vt "):
            vt.append([float(x) for x in raw.split()[1:3]])
        elif raw.startswith("f "):
            c = [t.split("/") for t in raw.split()[1:]]
            v = [int(t[0]) - 1 for t in c]
            t = [int(x[1]) - 1 if len(x) > 1 and x[1] else 0 for x in c]
            for i in range(1, len(c) - 1):
                faces.append([v[0], v[i], v[i + 1]]); uvf.append([t[0], t[i], t[i + 1]])
    faces = np.array(faces, dtype=np.int64)
    if not with_uv:
        return faces
    return faces, np.array(uvf, dtype=np.int64), np.array(vt if vt else [[0.0, 0.0]])


def standing_body(gender: str):
    base, G = catalogue.load_base()
    T, back = morph.t_pose_body(base, gender)
    skel = rig.load_skeleton(T)
    joints, weights = rig.weight_table(skel, len(T))
    M = garments.skinning_matrices(skel, compose.standing(T, G["body"], skel, joints, weights, back))
    return T, G, skel, joints, weights, M, garments.deform(T, joints, weights, M)


def standing_positions(rows, scales, pose, T, P, joints, weights, skel, M):
    idx, w, off = rows[:, :3].astype(np.int64), rows[:, 3:6], rows[:, 6:9]
    if pose == "relaxed":
        return np.einsum("nk,nkd->nd", w, P[idx]) + off
    scale = np.ones(3)
    for axis, key in enumerate("xyz"):
        if key in scales:
            i, j, dist = scales[key]
            scale[axis] = abs(T[i, axis] - T[j, axis]) / max(dist, 1e-9)
    XT = np.einsum("nk,nkd->nd", w, T[idx]) + off * scale
    gj, gw = garments.mhclo_weights(idx, w, joints, weights, len(skel.names))
    return garments.deform(XT, gj.astype(int), gw, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out_dir")
    ap.add_argument("--gender", required=True)
    ap.add_argument("--levels", type=int, default=1)
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--slack", type=float, default=0.08)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--bending", type=float, default=0.3)
    ap.add_argument("--mass", type=float, default=0.03)
    ap.add_argument("--quality", type=int, default=8)
    ap.add_argument("--after", type=int, default=3)
    ap.add_argument("--no-self", action="store_true")
    ap.add_argument("--blender", default=BLENDER)
    a = ap.parse_args()
    src, out_dir = pathlib.Path(a.src), pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T, G, skel, joints, weights, M, P = standing_body(a.gender)
    lines, start, stop, rows, scales, pose = parse_mhclo(src)
    obj_name = next(l.split()[1] for l in lines if l.split()[:1] == ["obj_file"])
    faces, uvf, vt = obj_faces(src.parent / obj_name, with_uv=True)
    X = standing_positions(rows, scales, pose, T, P, joints, weights, skel, M)

    # Supports: the shoulders and neck of a top, the waistband of anything reaching below the hips.
    from scipy.spatial import cKDTree
    used = np.unique(G["body"])
    dist, near = cKDTree(P[used]).query(X)
    near = used[near]
    head = lambda name: garments.deform(skel.heads[[skel.names.index(name)]],
                                        np.array([[skel.names.index(name)]]), np.ones((1, 1)), M)[0]
    shoulder, hip = head("upperarm01.L"), head("pelvis.L")
    on = dist < PIN_NEAR
    y = P[near, 1]
    shoulders = on & (y > shoulder[1] - 0.5) & (np.abs(P[near, 0]) < abs(shoulder[0]) - 0.1)
    collar = X[:, 1] > shoulder[1] + 0.3                     # a hood or collar standing above the shoulders
    # Each piece reaching down the legs (trousers, a skirt; not a coat hanging from the shoulders) held
    # at its own waistband, its top band near the skin: held at a fixed height, jeans sitting lower slid
    # down to the knees.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = cKDTree(X).query_pairs(WELD_M * 10, output_type="ndarray")
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], pairs])
    _, piece = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(X), len(X))), directed=False)
    waist, hems = np.zeros(len(X), bool), np.zeros(len(X), bool)
    for k in np.unique(piece):
        m = piece == k
        top, low = X[m, 1].max(), X[m, 1].min()
        if low < hip[1] - 1.0 and top < shoulder[1] - 1.5:
            waist |= m & (X[:, 1] > top - 0.8) & (dist < PIN_NEAR)
        if low < hip[1] - 5.0:
            # A trouser leg's hem held as fitted, over the shoe: with the feet out of the collider the
            # jeans' hems flopped 14 cm, and loose, trouser legs rode up the shins.
            hems |= m & (X[:, 1] < low + 0.8)
    # And, whatever piece it is part of, cloth hugging the hips: a suit's trousers are one piece with
    # its shirt, and their legs fell off the feet.
    if X[:, 1].min() < hip[1] - 1.0:
        waist |= (dist < 0.25) & (np.abs(P[near, 1] - hip[1]) < 0.8)
    # The lower half of each forearm held: free, sleeves rode up the arm and bunched at the shoulder.
    cuffs = np.zeros(len(X), bool)
    for side in "LR":
        el, wr = head(f"lowerarm01.{side}"), head(f"wrist.{side}")
        t = ((X - el) @ (wr - el)) / max(float((wr - el) @ (wr - el)), 1e-9)
        radial = np.linalg.norm(X - el - np.outer(t, wr - el), axis=1)
        cuffs |= (t > 0.55) & (t < 1.3) & (radial < 0.9)
    # Inner layers held where they are (a shirt under a jacket): only the outermost cloth hangs.
    normal = trimesh.Trimesh(P, G["body"], process=False).vertex_normals[near]
    # Off the skin before it starts: cloth fitted into a calf or a thigh was thrown out of it by the
    # collision, and a suit's trouser legs flew half a metre.
    depth = ((X - P[near]) * normal).sum(axis=1)
    X = X + np.clip(START_CLEAR - depth, 0.0, None)[:, None] * normal
    tris = X[faces]
    inner = garments._ray_hits(X + 0.04 * normal, normal, tris, 0.4) < np.inf
    # Small separate pieces (a zip pull, buttons, a tag, drawstrings) held as they are: free, a zip pull
    # fell 27 cm and the simulation was thrown out.
    sizes = np.bincount(piece)
    trims = sizes[piece] < 0.03 * len(X)
    pin = (shoulders | waist | collar | cuffs | inner | trims | hems).astype(np.float64)
    print(f"{len(X)} vertices; pinned {int(shoulders.sum())} at the shoulders, {int(waist.sum())} at the waist, "
          f"{int(collar.sum())} above them, {int(cuffs.sum())} on the forearms, {int(inner.sum())} under other cloth, {int(trims.sum())} on small pieces, {int(hems.sum())} at the hems", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        # The body without its hands and feet: a cuff round the wrist, a hem round the ankle, shoved up
        # the slope of a hand or a foot, ratcheted up the limb (sleeves to the shoulder, trousers to the knee).
        ends = [i for i, n in enumerate(skel.names) if any(k in n for k in ("wrist", "finger", "metacarpal", "foot", "toe"))]
        end_share = np.where(np.isin(joints, ends), weights, 0.0).sum(axis=1)
        collider = G["body"][(end_share[G["body"]] < 0.5).all(axis=1)]
        np.savez(tmp / "in.npz", body_v=P * 0.1, body_f=collider, cloth_v=X * 0.1, cloth_f=faces,
                 cloth_uv=vt[uvf], cloth_pin=pin, cloth_soft=(~collar).astype(np.float64))
        (tmp / "sim.py").write_text(SIM, encoding="utf-8")
        args = {"npz": str(tmp / "in.npz"), "out": str(tmp / "out.npz"), "frames": a.frames, "thickness": 0.003,
                "bending": a.bending, "mass": a.mass, "quality": a.quality, "self": not a.no_self,
                "levels": a.levels, "smooth": a.smooth, "after": a.after, "slack": a.slack * 0.1, "weld": WELD_M}
        run = subprocess.run([a.blender, "--background", "--factory-startup", "--python", str(tmp / "sim.py"),
                              "--", json.dumps(args)], capture_output=True, text=True)
        if not (tmp / "out.npz").exists():
            sys.exit("Blender failed:\n" + run.stdout[-4000:] + run.stderr[-3000:])
        with np.load(tmp / "out.npz") as r:
            V, F, UV, start_v = r["v"] * 10.0, r["f"], r["uv"], r["start"] * 10.0
    moved = np.linalg.norm(V - start_v, axis=1)
    if np.percentile(moved, 99.9) > BLOWUP:
        far = moved > BLOWUP
        st = start_v[far]
        print(f"flying: {int(far.sum())} points; started at y {np.percentile(st[:, 1], [5, 50, 95]).round(2)}, "
              f"|x| {np.percentile(np.abs(st[:, 0]), [5, 50, 95]).round(2)}, z {np.percentile(st[:, 2], [5, 50, 95]).round(2)} "
              f"(shoulder y {shoulder[1]:.2f}, x {abs(shoulder[0]):.2f}; hip y {hip[1]:.2f}); "
              f"moved on average {(V - start_v)[far].mean(axis=0).round(2)} dm", flush=True)
        if not os.environ.get("DRAPE_KEEP"):
            sys.exit(f"rejected: points flew {np.percentile(moved, 99.9) * 100:.0f} mm (99.9%), the simulation blew up; the fit stays")
    print(f"{len(V)} points after subdividing; settled {np.median(moved) * 100:.1f} mm (median), "
          f"{np.percentile(moved, 95) * 100:.1f} mm (95%) from where they started", flush=True)

    # Bound to the body standing: the nearest body triangle, barycentric weights, the offset off it.
    corners, bary = nearest_triangle(V, P, G["body"])
    offset = V - np.einsum("nk,nkd->nd", bary, P[corners])

    # The new mesh: one texture coordinate per corner, as OBJ keeps them.
    uv_flat = UV.reshape(-1, 2)
    uniq, uv_index = np.unique(np.round(uv_flat, 6), axis=0, return_inverse=True)
    uv_index = uv_index.reshape(-1, 3)
    name = src.stem
    obj = [f"# {name}, draped by drape.py ({a.levels} subdivision, slack {a.slack} dm, {a.frames} frames)"]
    obj += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in V]
    obj += [f"vt {u:.6f} {v:.6f}" for u, v in uniq]
    obj += ["f " + " ".join(f"{F[k, j] + 1}/{uv_index[k, j] + 1}" for j in range(3)) for k in range(len(F))]
    (out_dir / f"{name}.obj").write_text("\n".join(obj) + "\n", encoding="utf-8")
    keep = [l for l in lines[:start - 1] if l.split()[:1] not in (["x_scale"], ["y_scale"], ["z_scale"], ["pose"], ["obj_file"])]
    head_lines = [f"# draped by drape.py from {src.name}"] + keep + [f"obj_file {name}.obj", "pose relaxed", "verts 0"]
    rows_out = [" " + " ".join([str(int(i)) for i in corners[k]] + [f"{x:.5f}" for x in bary[k]] + [f"{x:.5f}" for x in offset[k]])
                for k in range(len(V))]
    (out_dir / f"{name}.mhclo").write_text("\n".join(head_lines + rows_out + lines[stop:]) + "\n", encoding="utf-8")
    # The material and its textures, as they are.
    material = next((l.split()[1] for l in lines if l.split()[:1] == ["material"]), None)
    if material:
        shutil.copy(src.parent / material, out_dir / material)
        for l in (src.parent / material).read_text(encoding="utf-8").splitlines():
            p = l.split()
            if len(p) == 2 and p[0].endswith("Texture") and (src.parent / p[1]).exists():
                shutil.copy(src.parent / p[1], out_dir / p[1])
    print("wrote", out_dir / f"{name}.mhclo")


if __name__ == "__main__":
    main()
