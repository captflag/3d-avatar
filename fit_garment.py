"""A garment made for another body and pose, fitted to ours in its T-pose and written as a MakeHuman
clothes asset (OBJ + MHCLO + MHMAT + textures), so it reshapes to every person like the catalogue's.

    python fit_garment.py IN(.glb|.gltf|.zip|.obj) OUT_DIR NAME --kind top|bottom|skirt|shoes|full
                          [--gender male|female] [--turn DEG] [--material SUBSTRING] [--clear DM]
                          [--arms DEG] [--scale F] [--subdivide N] [--legs] [--hem Y]
    Environment switches: FIT_NOSEG=1 short or no sleeves (no sleeve/torso split); FIT_TORSO_BIND=1
    bind to the torso's skin alone (sleeveless); FIT_TEAR=999 no cutting of stretched triangles;
    FIT_GRID=1 print the best arm fits; FIT_DEBUG=path save the intermediate arrays.
    Each downloaded pack keeps its garments' settings beside them (assets/community_packs/<pack>/RECIPES.md,
    local). Check a fit in the relaxed stance with garment_views.py.

Steps: our body is posed to the garment's own arm angle; the garment is sized and placed onto it
(scale + shift, trimmed nearest-surface fit over the body part it covers); it borrows the posed body's
skin weights; it is unposed into our T-pose by inverting each vertex's blended bone matrix; lifted
clear of the skin; and bound to the T-pose body by the nearest triangle (three body vertices,
barycentric, and an offset), which is what an MHCLO is."""
import os, argparse, io, json, pathlib, shutil, sys, tempfile, zipfile
import numpy as np, trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import catalogue, morph, rig, garments

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("name")
ap.add_argument("--kind", required=True, choices=["top", "bottom", "skirt", "shoes", "full"])
ap.add_argument("--gender", default="male"); ap.add_argument("--turn", type=float, default=0.0)
ap.add_argument("--material", default=None)
ap.add_argument("--subdivide", type=int, default=0, help="split every triangle in four this many times first (long flat triangles cut through a curved arm)")
ap.add_argument("--scale", type=float, default=None, help="size as a fraction of the height-matched size, instead of searching it (a narrow-shouldered fit on a broad body)")
ap.add_argument("--hem", type=float, default=None, help="cut straight across at this height on our T-pose body, keeping what is above")
ap.add_argument("--legs", action="store_true", help="trousers: also fit our legs' spread to the garment's legs")
ap.add_argument("--arms", type=float, default=None, help="the garment's arm angle below horizontal (elbows still searched) instead of searching it (a skirt wider than the chest misleads the search)")
ap.add_argument("--clear", type=float, default=0.02, help="clearance off the skin, decimetres")
args = ap.parse_args()
out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

# --- the garment: one mesh, its texture coordinates and its material's maps ---
src = pathlib.Path(args.src)
if src.suffix == ".zip":
    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z:
        for m in z.infolist():
            target = (tmp / m.filename).resolve()
            if not str(target).startswith(str(tmp.resolve())): raise SystemExit(f"unsafe path in zip: {m.filename}")
        z.extractall(tmp)
    src = next(p for p in tmp.rglob("*") if p.suffix.lower() in (".glb", ".gltf"))
scene = trimesh.load(src, force="scene", process=False)
pieces = []
for node in scene.graph.nodes_geometry:
    tf, gname = scene.graph[node]
    g = scene.geometry[gname]
    mat = getattr(g.visual, "material", None)
    mname = getattr(mat, "name", "") or gname
    if args.material and args.material.lower() not in mname.lower() and args.material.lower() not in gname.lower():
        continue
    g = g.copy(); g.apply_transform(tf)
    if not hasattr(g.visual, "uv") or g.visual.uv is None:
        g.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(g.vertices), 2)), material=mat)
    pieces.append((mname, g))
if not pieces: raise SystemExit("no mesh found (check --material)")
names = sorted({n for n, _ in pieces}); print("materials:", names)
# All parts fitted together (one placement, one arm pose), written one asset per material.
counts = np.cumsum([0] + [len(g.vertices) for _, g in pieces])
part_of = np.concatenate([np.full(len(g.vertices), k) for k, (_, g) in enumerate(pieces)])
UV = np.vstack([np.asarray(g.visual.uv, np.float64) for _, g in pieces])
F = np.vstack([np.asarray(g.faces) + counts[k] for k, (_, g) in enumerate(pieces)])
V = np.vstack([np.asarray(g.vertices, np.float64) for _, g in pieces])
mesh = trimesh.Trimesh(V, F, process=False)
print(f"garment: {len(V)} vertices, {len(F)} triangles, size {np.ptp(V, 0).round(3)}")
# glTF is y-up; Sketchfab models usually face +z. --turn turns it about y if not.
if args.turn: V = V @ Rotation.from_euler("y", args.turn, degrees=True).as_matrix().T

# --- our body: T-pose, rig, and the part this garment covers ---
base, groups = catalogue.load_base()
T, back = morph.t_pose_body(base, args.gender)
body_faces = groups["body"]; body_idx = np.unique(body_faces)
skel = rig.load_skeleton(T); joints, weights = rig.weight_table(skel, len(T))
y = T[body_idx, 1]; top, low = y.max(), y.min(); H = top - low
hip_y = float(T[np.unique(groups["joint-pelvis"]) if "joint-pelvis" in groups else body_idx, 1].mean())
# A top's body is matched from the hips to the collarbone, not up the neck: matched to the chin, a crop
# top (little cloth below the chest) slid up until its neckline sat on the chin. Collars still rise past it.
clav_y = float(T[np.unique(groups["joint-l-clavicle"]), 1].mean())
part = {"top": (hip_y - 0.05 * H, clav_y + 0.02 * H), "bottom": (low + 0.05 * H, hip_y + 0.12 * H),
        "shoes": (low - 0.01, low + 0.08 * H), "full": (low + 0.05 * H, clav_y + 0.02 * H),
        # A skirt matched from the hips to the upper thigh: against the legs to the ankle, a mini came out
        # 55 cm long, past the knee.
        "skirt": (hip_y - 0.15 * H, hip_y + 0.08 * H)}[args.kind]

# --- the garment's arm angle, and our body posed to it ---
def arm_pose(angle_deg):
    """Our T-pose rig's arms lowered by `angle_deg` from horizontal (about z, at the shoulder)."""
    pose = {}
    for side, s in (("L", 1.0), ("R", -1.0)):
        pose[f"upperarm01.{side}"] = Rotation.from_euler("z", -angle_deg * s, degrees=True).as_quat().tolist()
    return pose

c0 = V.mean(0)
Vn = V - c0
dominant = np.array(skel.names)[joints[np.arange(len(T)), weights.argmax(1)]]
on_arm = np.array([n.startswith(("upperarm", "lowerarm", "wrist", "finger", "metacarpal", "shoulder")) for n in dominant])
on_leg = np.array([n.startswith(("upperleg", "lowerleg", "foot", "toe")) for n in dominant])
in_part = (T[:, 1] > part[0]) & (T[:, 1] < part[1])
core = body_idx[(in_part[body_idx]) & ~on_arm[body_idx] & (~on_leg[body_idx] if args.kind == "top" else True)]
if args.kind in ("bottom", "skirt", "shoes"):
    core = body_idx[in_part[body_idx] & ~on_arm[body_idx]]
arm = body_idx[on_arm[body_idx]]
s_height = np.ptp(T[body_idx[in_part[body_idx]], 1]) / max(np.ptp(Vn[:, 1]), 1e-9)
body_n = np.asarray(trimesh.Trimesh(T, body_faces, process=False).vertex_normals)
# The garment's torso: points within its own middle band of width (not the sleeves).
mid_y = np.percentile(Vn[:, 1], 70)
half = np.percentile(np.abs(Vn[np.abs(Vn[:, 1] - mid_y) < 0.03 * np.ptp(Vn[:, 1]), 0]), 50)
g_core = (np.abs(Vn[:, 0]) < 1.05 * half) if args.kind in ("top", "full") else np.ones(len(Vn), bool)


x_mid = 0.5 * (Vn[:, 0].min() + Vn[:, 0].max())
tree_core = cKDTree(T[core])


def refine(scale, t_):
    """The garment's torso shifted onto our body's from a start, side to side held at the centre (the
    body and the garment are symmetric); scored by the trimmed distance plus a heavy charge for any of
    it inside the body. From the shapes' average positions it settled 8 cm aside and 13 cm low."""
    t_ = np.array(t_, float)
    for _ in range(10):
        d, j = tree_core.query(scale * Vn[g_core] + t_)
        keep = d <= np.percentile(d, 60)
        step = (T[core][j[keep]] - (scale * Vn[g_core][keep] + t_)).mean(0); step[0] = 0.0
        t_ = t_ + step
    X_ = scale * Vn + t_
    d, j = tree_core.query(X_[g_core]); k = core[j]
    inside = ((X_[g_core] - T[k]) * body_n[k]).sum(1) < -0.005
    return np.percentile(d, 60) + 2.0 * inside.mean(), X_


centre = T[core].mean(0)
starts = []
for f in ([args.scale] if args.scale else np.geomspace(0.45, 1.15, 12)):    # crop tops and minis cover less than the part's height
    sc = s_height * f
    for dy in np.arange(-3.0, 3.01, 0.5):
        for dz in (-0.6, 0.0, 0.6):
            t0 = np.array([-sc * x_mid, centre[1] - sc * Vn[g_core][:, 1].mean() + dy, centre[2] - sc * Vn[g_core][:, 2].mean() + dz])
            starts.append((refine(sc, t0), sc))
(score, X), s = min(starts, key=lambda r: r[0][0])
# Finer in scale round the best start, from where it ended.
t_best = X[0] - s * Vn[0]
(score, X), s = min([((score, X), s)] + [(refine(s * f, t_best), s * f) for f in (np.linspace(0.95, 1.05, 11) if not args.scale else [])],
                    key=lambda r: r[0][0])
print(f"scale {s:.3f} (from height {s_height:.3f}); torso score {score:.3f}")

angle, bend, posed_m = 0.0, 0.0, garments.skinning_matrices(skel, {})
if args.kind in ("top", "full") and (~g_core).sum() > 50:
    sleeve = X[~g_core & (X[:, 1] > np.percentile(X[:, 1], 30))]
    def arm_fit(a, b, fwd):
        """Arms `a` below horizontal, elbows bent `b` further down and `fwd` forward (in the upper
        arm's frame: MakeHuman's own arms-down rest reaches its forearms ~46 degrees forward)."""
        pose = arm_pose(a)
        for side, sg in (("L", 1.0), ("R", -1.0)):
            pose[f"lowerarm01.{side}"] = (Rotation.from_euler("y", -fwd * sg, degrees=True)
                                          * Rotation.from_euler("z", -b * sg, degrees=True)).as_quat().tolist()
        m_ = garments.skinning_matrices(skel, pose)
        P_ = garments.deform(T[arm], joints[arm], weights[arm], m_)
        return float(np.median(cKDTree(P_).query(sleeve)[0])), m_
    if args.arms is not None:
        grid = [(arm_fit(args.arms, b, f), args.arms, b, f) for b in range(-10, 41, 6) for f in range(-20, 61, 10)]
    else:
      grid = [(arm_fit(a, b, f), a, b, f) for a in range(0, 81, 6) for b in (0, 15, 30) for f in (0, 20, 40, 60)]
      _, a0, b0, f0 = min(grid, key=lambda r: r[0][0])
      grid += [(arm_fit(a, b, f), a, b, f) for a in (a0 - 3, a0, a0 + 3) for b in (b0 - 7, b0, b0 + 7)
             for f in (f0 - 10, f0, f0 + 10)]
    if os.environ.get("FIT_GRID"): print("best arm fits:", [(round(r[0][0], 3), r[1], r[2], r[3]) for r in sorted(grid, key=lambda r: r[0][0])[:8]])
    (fit_d, posed_m), angle, bend, fwd = min(grid, key=lambda r: r[0][0])
    print(f"garment arms {angle} degrees below horizontal, elbows {bend} down and {fwd} forward; sleeve to arm {fit_d * 100:.1f} mm")
if args.legs:
    # Trousers modelled with the legs straight down; our T-pose stands with them apart. Each upper leg
    # turned in about z until our legs sit in the trouser legs, along with the arms found above.
    pose0 = arm_pose(angle)
    if args.kind in ("top", "full") and (~g_core).sum() > 50:
        for side, sg in (("L", 1.0), ("R", -1.0)):
            pose0[f"lowerarm01.{side}"] = (Rotation.from_euler("y", -fwd * sg, degrees=True)
                                           * Rotation.from_euler("z", -bend * sg, degrees=True)).as_quat().tolist()
    leg = body_idx[on_leg[body_idx]]
    trousers = X[X[:, 1] < hip_y - 0.08 * H]
    def leg_fit(g_):
        pose = dict(pose0)
        for side, sg in (("L", 1.0), ("R", -1.0)):
            pose[f"upperleg01.{side}"] = Rotation.from_euler("z", g_ * sg, degrees=True).as_quat().tolist()
        m_ = garments.skinning_matrices(skel, pose)
        return float(np.median(cKDTree(garments.deform(T[leg], joints[leg], weights[leg], m_)).query(trousers)[0])), m_
    fits = [(leg_fit(g_), g_) for g_ in np.arange(-15, 15.1, 1.5)]
    (leg_d, posed_m), leg_angle = min(fits, key=lambda r: r[0][0])
    print(f"legs turned {leg_angle:+.1f} degrees about z; trousers to legs {leg_d * 100:.1f} mm (straight {fits[10][0][0] * 100:.1f})")
P = garments.deform(T, joints, weights, posed_m)
print(f"placed: mean gap to the body {np.mean(cKDTree(P[body_idx]).query(X)[0]) * 100:.1f} mm")

# --hem: cut straight across at that height on our body, keeping what is above (a skirt modelled longer
# than its name: cut by whole triangles, the hem came out in steps with straps' scraps hanging off it).
if args.hem is not None:
    assert len(set(part_of.tolist())) == 1, "--hem cuts one-material garments only"
    X, F, UV = trimesh.intersections.slice_faces_plane(X, F, np.array([0.0, 1.0, 0.0]), np.array([0.0, args.hem, 0.0]), uv=UV)
    used = np.unique(F); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
    X, UV, F = X[used], UV[used], remap[F]
    part_of = np.zeros(len(X), int)
    print(f"hem cut at y {args.hem}: {len(F)} faces left")
# Split every triangle in four (after placing, which is fitted on the garment's own vertices).
for _ in range(args.subdivide):
    X, F, at = trimesh.remesh.subdivide(X, F, vertex_attributes={"uv": UV, "part": part_of.astype(float)[:, None]})
    UV, part_of = at["uv"], np.rint(at["part"][:, 0]).astype(int)
mesh = trimesh.Trimesh(X, F, process=False)
# --- weights from the posed body, then unposed into our T-pose ---
# Garment normals turned outward per connected piece (downloads are often wound either way).
gn = np.asarray(trimesh.Trimesh(X, F, process=False).vertex_normals).copy()
_, kb = cKDTree(P[body_idx]).query(X)
out_dir = X - P[body_idx][kb]
labels = trimesh.graph.connected_component_labels(trimesh.Trimesh(X, F, process=False).face_adjacency, len(F))
vlab = np.zeros(len(X), int); vlab[F.ravel()] = np.repeat(labels, 3)
for c in np.unique(vlab):
    m = vlab == c
    if (gn[m] * out_dir[m]).sum() < 0: gn[m] = -gn[m]
pn = np.asarray(trimesh.Trimesh(P, body_faces, process=False).vertex_normals)
# The garment welded by position (texture seams split its vertices), for walking over its surface.
_, wid = np.unique(np.round(X / 1e-3).astype(np.int64), axis=0, return_inverse=True); wid = wid.ravel()
nw = wid.max() + 1
E = np.unique(np.sort(np.vstack([wid[F[:, [0, 1]]], wid[F[:, [1, 2]]], wid[F[:, [2, 0]]]]), 1), axis=0)
E = E[E[:, 0] != E[:, 1]]
deg = np.maximum(np.bincount(E.ravel(), minlength=nw), 1)
def spread(v):
    """Each welded vertex's value averaged with its neighbours' (one ring)."""
    t = v.copy(); np.add.at(t, E[:, 0], v[E[:, 1]]); np.add.at(t, E[:, 1], v[E[:, 0]]); return t / (deg + 1)[:, None] if v.ndim == 2 else t / (deg + 1)
moving = np.array([n.startswith(("upperarm", "lowerarm", "wrist", "finger", "metacarpal")) for n in skel.names])
arm_share = (weights * moving[joints]).sum(1)                      # how much each body vertex follows the arm
sleeve_a = np.zeros(len(X))
if angle > 30 and args.kind in ("top", "full") and not os.environ.get("FIT_NOSEG"):   # FIT_NOSEG: short or no sleeves     # arms nearly out already: a wide sleeve needs no split
    # Sleeve or torso, over the garment's own surface: walking from the cuff (the garment nearest the
    # posed wrist), the ground covered per step stays the sleeve's girth until the walk spills out
    # through the armhole onto the torso. Everything walked before that spill is sleeve.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    hp = lambda b: (posed_m[b] @ np.r_[skel.heads[b], 1.0])[:3]
    Xw = np.zeros((nw, 3)); Xw[wid] = X
    G = coo_matrix((np.linalg.norm(Xw[E[:, 0]] - Xw[E[:, 1]], axis=1), (E[:, 0], E[:, 1])), shape=(nw, nw)).tocsr()
    area_w = np.zeros(nw); np.add.at(area_w, wid[F].ravel(), np.repeat(trimesh.Trimesh(X, F, process=False).area_faces / 3, 3))
    from scipy.sparse.csgraph import connected_components
    _, comp = connected_components(G, directed=False)
    big = np.bincount(comp, area_w)[comp] > 0.03 * area_w.sum()       # not the cuff's buttons
    gnw = np.zeros((nw, 3)); gnw[wid] = gn
    lw = np.zeros(nw)
    for side in ("L", "R"):
        wrist = hp(skel.names.index(f"wrist.{side}")); sh = hp(skel.names.index(f"upperarm01.{side}"))
        L = np.linalg.norm(wrist - sh)
        same = (np.sign(Xw[:, 0]) == np.sign(wrist[0])) & big
        cand = np.flatnonzero(same); seeds = cand[np.argsort(np.linalg.norm(Xw[cand] - wrist, axis=1))[:12]]
        dist = dijkstra(G, directed=False, indices=seeds, min_only=True)
        h = L / 30; nb = int(2 * L / h); A = np.bincount(np.clip((dist[np.isfinite(dist)] / h).astype(int), 0, nb), area_w[np.isfinite(dist)], nb + 1)
        base = np.median(A[int(0.2 * L / h):int(0.6 * L / h)])
        As = np.convolve(A, np.ones(3) / 3, "same")
        over = np.flatnonzero((np.arange(nb + 1) * h > 0.5 * L) & (As > float(os.environ.get("FIT_SPILL", 1.6)) * base))
        cut = (over[0] if len(over) else int(L / h)) * h
        print(f"sleeve {side}: arm {L:.2f}, cut at {cut:.2f} ({cut / L:.2f} of the arm); girth profile", (A[:int(1.3 * L / h)] / base).round(1).tolist())
        lw = np.maximum(lw, (dist < cut).astype(float))
        # The walk spills at the armpit, the short way round; the sleeve's cap over the top of the arm
        # is further on. Taken too: cloth a little past the spill, over the upper arm (from the shoulder
        # joint down), facing away from the arm's axis (the torso's side next to the arm faces it).
        el = hp(skel.names.index(f"lowerarm01.{side}")); ab = el - sh
        t = ((Xw - sh) @ ab) / (ab @ ab); r = Xw - (sh + np.clip(t, 0, 1)[:, None] * ab)
        radial = (gnw * r).sum(1) / np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        ext = float(os.environ.get("FIT_CAP", 0.35))
        rr = np.linalg.norm(r, axis=1); on = (dist < cut) & (t > 0.3) & (t < 0.9)
        reach = 1.3 * np.median(rr[on]) if on.sum() > 10 else 0.0       # the sleeve's own radius round the upper arm
        cap = same & (dist < cut + ext * L) & (dist >= cut) & (t > -0.05) & (radial > 0.3) & (rr < reach)
        print(f"  cap over the {side} shoulder: {cap.sum()} more vertices")
        lw = np.maximum(lw, cap.astype(float))
    lab = lw[wid]
    if os.environ.get("FIT_DEBUG"): np.savez(os.environ["FIT_DEBUG"] + ".seg.npz", X=X, lab=lab, radial=lab)
    # Softened across the armhole over a set distance on the cloth (not a count of rings, which on a
    # coarse sleeve reaches halfway to the elbow): half way at the cut, all sleeve or all torso `w` off it.
    w = float(os.environ.get("FIT_BAND", 0.08)) * L
    on_s, off_s = np.flatnonzero((lw > 0.5) & big), np.flatnonzero((lw <= 0.5) & big)
    d_s = dijkstra(G, directed=False, indices=on_s, min_only=True); d_t = dijkstra(G, directed=False, indices=off_s, min_only=True)
    lw = np.where(big, np.clip(0.5 + (np.minimum(d_t, 9) - np.minimum(d_s, 9)) / (2 * w), 0, 1), lw)
    # The centring shift below fades out further into the torso than the weights do, so a shoulder
    # yoke narrower than ours is drawn out to our shoulder with its sleeve instead of left inside it.
    ws = float(os.environ.get("FIT_YOKE", 0.25)) * L
    sw = np.where(big, np.clip(0.75 + (np.minimum(d_t, 9) - np.minimum(d_s, 9)) / (2 * ws), 0, 1), 0.0)
    if (~big).any():                                               # buttons and the like go with the cloth under them
        lw[~big] = lw[big][cKDTree(Xw[big]).query(Xw[~big])[1]]
    sleeve_a = lw[wid]
    shift_a = np.maximum(sw[wid], sleeve_a)
    # Each sleeve centred on our posed arm before it is unposed (the garment's arms hang closer in, or
    # further out, than ours): ring by ring along the arm, the gap between the arm's axis and the
    # ring's centre is closed, fading out across the armhole with the sleeve's share.
    for side in ("L", "R"):
        pts = [hp(skel.names.index(f"{b}.{side}")) for b in ("upperarm01", "lowerarm01", "wrist")]
        m_ = (shift_a > 0.01) & (np.sign(X[:, 0]) == np.sign(pts[2][0]))
        best = np.full(len(X), np.inf); u = np.zeros(len(X)); q = np.zeros_like(X)
        for s_, (a_, b_) in enumerate(zip(pts[:-1], pts[1:])):
            ab = b_ - a_; t = np.clip(((X - a_) @ ab) / (ab @ ab), 0, 1); qq = a_ + t[:, None] * ab
            d = np.linalg.norm(X - qq, axis=1); c_ = d < best; best[c_], u[c_], q[c_] = d[c_], (s_ + t)[c_], qq[c_]
        edges_u = np.linspace(0.1, 1.9, 13); mids, shifts = [], []
        for lo, hi in zip(edges_u[:-1], edges_u[1:]):
            sel = m_ & (sleeve_a > 0.5) & (u >= lo) & (u < hi)
            if sel.sum() >= 8: mids.append(0.5 * (lo + hi)); shifts.append((q[sel] - X[sel]).mean(0))
        if not mids: continue
        shifts = np.array(shifts); sh_u = np.stack([np.interp(u[m_], mids, shifts[:, c]) for c in range(3)], 1)
        print(f"sleeve {side} centred on the arm: shift {np.linalg.norm(shifts, axis=1).round(2).tolist()}")
        X = X.copy(); X[m_] += shift_a[m_, None] * sh_u
on_arm_b = arm_share > 0.5
ja, wa = garments.transfer_weights(X, P[body_idx[on_arm_b[body_idx]]], joints[body_idx[on_arm_b[body_idx]]],
                                   weights[body_idx[on_arm_b[body_idx]]], len(skel), normals=(gn, pn[body_idx[on_arm_b[body_idx]]]))
jt, wt = garments.transfer_weights(X, P[body_idx[~on_arm_b[body_idx]]], joints[body_idx[~on_arm_b[body_idx]]],
                                   weights[body_idx[~on_arm_b[body_idx]]], len(skel), normals=(gn, pn[body_idx[~on_arm_b[body_idx]]]))
gj, gw = garments.transfer_weights(X, P[body_idx], joints[body_idx], weights[body_idx], len(skel), normals=(gn, pn[body_idx]))
def qmul(a, b):
    w1, v1, w2, v2 = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate([w1 * w2 - (v1 * v2).sum(-1, keepdims=True), w1 * v2 + w2 * v1 + np.cross(v1, v2)], -1)
def rigid_blend(j_, w_):
    """Each vertex's bones blended as dual quaternions: a rigid turn part-way between them, where
    the matrices' linear blend shrinks towards the joint (and its inverse throws points far out)."""
    q = Rotation.from_matrix(posed_m[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    d = 0.5 * qmul(np.c_[np.zeros(len(q)), posed_m[:, :3, 3]], q)
    qs, ds = q[j_], d[j_]; sgn = np.sign((qs * qs[:, :1]).sum(-1, keepdims=True)); sgn[sgn == 0] = 1
    Q = (w_[..., None] * sgn * qs).sum(1); D = (w_[..., None] * sgn * ds).sum(1)
    n = np.linalg.norm(Q, axis=1, keepdims=True); Q, D = Q / n, D / n
    t = 2 * qmul(D, Q * np.array([1, -1, -1, -1]))[:, 1:]
    M_ = np.tile(np.eye(4), (len(Q), 1, 1)); M_[:, :3, :3] = Rotation.from_quat(Q[:, [1, 2, 3, 0]]).as_matrix(); M_[:, :3, 3] = t
    return M_
assert np.allclose(rigid_blend(np.arange(len(posed_m))[:, None], np.ones((len(posed_m), 1))), posed_m, atol=1e-6)
if sleeve_a.any():
    # Off the sleeve, the body's own weights (the shoulder cap follows the deltoid), less the forearm
    # and hand: jacket hems beside the lowered hands would be flung up with them.
    hand = np.array([n.startswith(("lowerarm", "wrist", "finger", "metacarpal")) for n in skel.names])
    rest_j, rest_w = gj, np.where(hand[gj], 0.0, gw)
    lost = rest_w.sum(1) < 0.5
    rest_j, rest_w = np.where(lost[:, None], jt, rest_j), np.where(lost[:, None], wt, rest_w / np.maximum(rest_w.sum(1, keepdims=True), 1e-9))
    M = rigid_blend(np.c_[ja, rest_j], np.c_[sleeve_a[:, None] * wa, (1 - sleeve_a[:, None]) * rest_w])
else:
    M = rigid_blend(gj, gw)                                         # each vertex's blended bone transform
Xh = np.c_[X, np.ones(len(X))]
XT = np.einsum("nij,nj->ni", np.linalg.inv(M), Xh)[:, :3]

if os.environ.get('FIT_DEBUG'): np.savez(os.environ['FIT_DEBUG'], X=X, XT=XT, gj=gj, gw=gw, gn=gn, sa=sleeve_a, names=np.array(skel.names), P=P, T=T)

# --- sleeves narrower than our arm: each ring widened evenly about the T-pose arm's axis until it
# clears the skin all round (pushing each point to its nearest skin slides the cloth under the arm) ---
if sleeve_a.any():
    arm_v = body_idx[arm_share[body_idx] > 0.5]
    for side in ("L", "R"):
        a_, b_ = (skel.heads[skel.names.index(f"{n}.{side}")] for n in ("upperarm01", "wrist"))
        ab = b_ - a_; La = np.linalg.norm(ab); e = ab / La
        def polar(Q):
            t = (Q - a_) @ e; r = Q - a_ - t[:, None] * e
            return t, np.linalg.norm(r, axis=1), np.arctan2(r @ np.cross(e, [0, 0, 1.0]), r[:, 2])
        g = np.flatnonzero((sleeve_a > 0.01) & (np.sign(XT[:, 0]) == np.sign(b_[0])))
        tg, rg, ag = polar(XT[g]); tb, rb_, ab_ = polar(T[arm_v])
        okb = (tb > -0.05 * La) & (tb < 1.02 * La) & (rb_ < 0.35 * La)
        tb, rb_, ab_ = tb[okb], rb_[okb], ab_[okb]
        nbins, nang = 24, 16
        ib = lambda t: np.clip((t / La * nbins).astype(int), 0, nbins - 1)
        ia = lambda a: ((a + np.pi) / (2 * np.pi) * nang).astype(int) % nang
        R = np.zeros((nbins, nang)); np.maximum.at(R, (ib(tb), ia(ab_)), rb_)
        # Rb smoothed round the arm (the bins' maxima are ragged), then each point under the skin is
        # lifted straight out from the axis to it, so the cloth stays spread all round.
        Rb = np.maximum(R, 0.5 * (np.roll(R, 1, 1) + np.roll(R, -1, 1)))
        target = Rb[ib(tg), ia(ag)] + args.clear
        f = np.where((Rb[ib(tg), ia(ag)] > 0) & (rg < target), target / np.maximum(rg, 1e-6), 1.0)
        f = 1 + (np.minimum(f, 4.0) - 1) * np.clip(2 * sleeve_a[g] - 0.5, 0, 1)
        axis_pt = a_ + tg[:, None] * e
        XT[g] = axis_pt + (XT[g] - axis_pt) * f[:, None]
        print(f"sleeve {side}: {np.mean(f > 1.001):.0%} of it lifted out from the arm's axis")

# --- off the skin: any point under it or within `clear` lifted along the skin's normal, smoothed ---
body = trimesh.Trimesh(T, body_faces, process=False); bn = np.asarray(body.vertex_normals)
for _ in range(3):
    dist, k = cKDTree(T[body_idx]).query(XT); k = body_idx[k]
    over = ((XT - T[k]) * bn[k]).sum(1)
    need = np.where((dist < 0.5) & (over < args.clear), args.clear - over, 0.0)
    # Layers move together: each point takes the biggest lift among the cloth within `layer` of it, so
    # a shirt pushed out from under the skin carries its jacket out with it and stays beneath.
    nd, nj = cKDTree(XT).query(XT, k=48, distance_upper_bound=float(os.environ.get("FIT_LAYER", 0.3)))
    need = np.max(np.where(np.isfinite(nd), np.r_[need, 0.0][nj], 0.0), axis=1)
    edges = mesh.edges_unique
    for _ in range(4):
        tot = np.zeros_like(need); np.add.at(tot, edges[:, 0], need[edges[:, 1]]); np.add.at(tot, edges[:, 1], need[edges[:, 0]])
        cnt = np.bincount(edges.ravel(), minlength=len(need))
        need = np.maximum(need, 0.5 * need + 0.5 * tot / np.maximum(cnt, 1))
    XT = XT + need[:, None] * bn[k]

# --- the armhole band laid onto the skin: where sleeve meets torso, cloth the unposing stretched
# out into a web is drawn in to the clearance over the skin (most at the band's middle) ---
if sleeve_a.any():
    pull = np.clip(4 * sleeve_a * (1 - sleeve_a), 0, 1) * float(os.environ.get("FIT_PULL", 1.0))
    for _ in range(3):
        dist, k = cKDTree(T[body_idx]).query(XT); k = body_idx[k]
        over = ((XT - T[k]) * bn[k]).sum(1)
        target = T[k] + 2 * args.clear * bn[k]
        XT = XT + np.where(over > 2 * args.clear, pull, 0.0)[:, None] * (target - XT)
        dist, k = cKDTree(T[body_idx]).query(XT); k = body_idx[k]
        over = ((XT - T[k]) * bn[k]).sum(1)
        XT = XT + np.where(over < args.clear, args.clear - over, 0.0)[:, None] * bn[k]

# --- bound to the T-pose body: nearest triangle's three vertices, barycentric, and the offset ---
# FIT_TORSO_BIND: a sleeveless top bound to the torso's skin alone. Bound to the nearest skin anywhere,
# its armholes' edges took the upper arm's and stood up off the shoulders once the arms came down.
bind_faces = body_faces[(arm_share[body_faces] < 0.2).all(1)] if os.environ.get("FIT_TORSO_BIND") else body_faces
tri_c = T[bind_faces].mean(1); _, fi = cKDTree(tri_c).query(XT)
refs = bind_faces[fi]
A, B, C = T[refs[:, 0]], T[refs[:, 1]], T[refs[:, 2]]
v0, v1 = B - A, C - A; v2 = XT - A
d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1); d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
den = np.maximum(d00 * d11 - d01 * d01, 1e-12)
bw = np.clip(np.c_[1 - (d11 * d20 - d01 * d21) / den - (d00 * d21 - d01 * d20) / den, (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den], 0, 1)
bw /= bw.sum(1, keepdims=True)
offset = XT - (bw[:, :, None] * T[refs]).sum(1)

# --- torn: triangles the unposing stretched (sleeves modelled fused to the torso's sides) are cut ---
def longest(V_): return np.stack([np.linalg.norm(V_[F[:, i]] - V_[F[:, (i + 1) % 3]], axis=1) for i in range(3)], 1).max(1)
torn = (longest(XT) > float(os.environ.get("FIT_TEAR", 1.8)) * longest(X)) & (longest(XT) > 0.25)
# Each face's piece before the cut, and its piece after: a piece under 1% of the garment that was
# part of a bigger one is a shred the cut left (buttons and zip pulls were small pieces already).
orig = trimesh.graph.connected_component_labels(trimesh.Trimesh(X, F, process=False).face_adjacency, len(F))
kept = F[~torn]
now = trimesh.graph.connected_component_labels(trimesh.Trimesh(X, kept, process=False).face_adjacency, len(kept))
shred = (np.bincount(now)[now] < 0.01 * len(F)) & (np.bincount(orig)[orig[~torn]] >= 0.01 * len(F))
print(f"cut {torn.sum()} triangles stretched by the T-pose, {shred.sum()} more in shreds")
F = kept[~shred]

# --- write the assets: one per material ---
def save_map(img, name, size=2048):
    if img is None: return None
    im = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB"); im.thumbnail((size, size)); im.save(out / name); return name
written = []
for mname in names:
    ks = [k for k, (n, _) in enumerate(pieces) if n == mname]
    keep_v = np.isin(part_of, ks); keep_f = keep_v[F].all(1)
    idx = np.unique(F[keep_f]); remap = -np.ones(len(XT), int); remap[idx] = np.arange(len(idx))   # only vertices still in a triangle
    Fm = remap[F[keep_f]]
    material = pieces[ks[0]][1].visual.material
    tag = args.name if len(names) == 1 else f"{args.name}_{''.join(c if c.isalnum() else '_' for c in mname).strip('_').lower()}"
    diff = save_map(getattr(material, "baseColorTexture", None) or getattr(material, "image", None), f"{tag}_diffuse.png")
    nrm = save_map(getattr(material, "normalTexture", None), f"{tag}_normal.png")
    with open(out / f"{tag}.obj", "w", encoding="utf-8") as f:
        f.write("# a garment fitted to the base body's T-pose; see the .mhclo\n")
        for q in XT[idx]: f.write(f"v {q[0]:.5f} {q[1]:.5f} {q[2]:.5f}\n")
        for u, w_ in UV[idx]: f.write(f"vt {u:.5f} {w_:.5f}\n")
        for tri in Fm: f.write("f " + " ".join(f"{i + 1}/{i + 1}" for i in tri) + "\n")
    with open(out / f"{tag}.mhclo", "w", encoding="utf-8") as f:
        f.write(f"# fitted by fit_garment.py to the T-pose base body\nname {tag}\nobj_file {tag}.obj\nmaterial {tag}.mhmat\nverts 0\n")
        for (i, j, k), (wa, wb, wc), d in zip(refs[idx], bw[idx], offset[idx]):
            f.write(f" {i} {j} {k} {wa:.5f} {wb:.5f} {wc:.5f} {d[0]:.5f} {d[1]:.5f} {d[2]:.5f}\n")
    factor = getattr(material, "baseColorFactor", None)
    colour = (np.asarray(factor[:3], float) / (255.0 if np.max(factor[:3]) > 1 else 1.0)) if factor is not None else np.ones(3)
    mat = [f"name {tag}", f"diffuseColor {colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f}", "shininess 0.2", "opacity 1.0", "backfaceCull False"]
    if diff: mat.append(f"diffuseTexture {diff}")
    if nrm: mat.append(f"normalmapTexture {nrm}")
    if diff and Image.open(out / diff).mode == "RGBA" and np.asarray(Image.open(out / diff))[..., 3].min() < 250:
        mat.append("transparent True")
    (out / f"{tag}.mhmat").write_text("\n".join(mat) + "\n", encoding="utf-8")
    written.append(tag)
print(f"wrote {out}: {written}; gap to the T-pose skin p5/p50 {np.percentile(over, 5) * 100:.1f}/{np.percentile(over, 50) * 100:.1f} mm")
