"""A pair of glasses from any file, placed as the catalogue's own pair sits on the T-pose head, written as
an MHCLO asset. The parts' colours (textures averaged, or their factors) go into one small palette map.
    python fit_glasses.py IN.glb OUT_DIR NAME [TURN_DEGREES]     (run from the repo)"""
import pathlib, sys, io
import numpy as np, trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import catalogue, morph, garments

src, out, name = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
base, groups = catalogue.load_base()
T, _ = morph.t_pose_body(base, "male")
eyes = np.array([T[np.unique(groups[g])].mean(0) for g in ("helper-l-eye", "helper-r-eye")])
spacing = float(np.linalg.norm(eyes[0] - eyes[1])); eye_mid = eyes.mean(0)

# The reference: the catalogue's first pair on the same head.
ref_item = next(i for i in catalogue.items_by_id().values() if i["category"] == "glasses")
ref = garments.load_mhclo(catalogue.ASSET_DIR / ref_item["mhclo"], T).vertices
front_ref = ref[ref[:, 2] > np.percentile(ref[:, 2], 70)]
want_width = np.ptp(front_ref[:, 0]) / spacing
want_front = front_ref[:, 2].max() - eye_mid[2]
want_y = (front_ref[:, 1].max() + front_ref[:, 1].min()) / 2 - eye_mid[1]

scene = trimesh.load(src, force="scene", process=False)
parts = []
for node in scene.graph.nodes_geometry:
    tf, gn = scene.graph[node]; g = scene.geometry[gn].copy(); g.apply_transform(tf)
    mat = getattr(g.visual, "material", None)
    f0 = getattr(mat, "baseColorFactor", None)
    if getattr(mat, "alphaMode", None) == "BLEND" or (f0 is not None and len(f0) > 3 and f0[3] < 128):
        continue                  # a lens: left out (the build removes lenses, and one welded to its rim took the rim)
    tex = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if tex is not None:
        colour = np.asarray(tex.convert("RGB"), float).reshape(-1, 3).mean(0)
    else:
        f = getattr(mat, "baseColorFactor", None)
        colour = np.asarray(f[:3], float) if f is not None else np.array([60, 60, 60.0])   # trimesh gives glTF factors as 0-255
    parts.append((g, colour))
V = np.vstack([np.asarray(g.vertices, float) for g, _ in parts])
if len(sys.argv) > 4:                 # a file facing sideways: turned about y first
    turn0 = Rotation.from_euler("y", float(sys.argv[4]), degrees=True).as_matrix()
    for g, _ in parts: g.vertices = g.vertices @ turn0.T
    V = V @ turn0.T
# Facing: the bridge (the narrow middle) is at the front.
width = np.ptp(V[:, 0]); mid = np.abs(V[:, 0] - (V[:, 0].min() + width / 2)) < 0.08 * width
if np.median(V[mid, 2]) < V[:, 2].mean():
    turn = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    for g, _ in parts: g.vertices = g.vertices @ turn.T
    V = V @ turn.T
front = V[V[:, 2] > np.percentile(V[:, 2], 70)]
# Hinge to hinge 93% of the head's width at the eyes (real frames sit just inside the temples; the
# catalogue's pairs, at 156 mm on a 144 mm head, read too big).
b = T[np.unique(groups["body"])]
head_w = np.ptp(b[(np.abs(b[:, 1] - eye_mid[1]) < 0.1) & (b[:, 2] > eye_mid[2] - 0.6), 0])
s = 0.93 * head_w / np.ptp(V[:, 0])
c = np.array([(front[:, 0].min() + front[:, 0].max()) / 2, (front[:, 1].min() + front[:, 1].max()) / 2, front[:, 2].max()])
shift = np.array([eye_mid[0], eye_mid[1] + want_y, eye_mid[2] + want_front]) - s * c

# One mesh, each part's texture coordinates on its own patch of an 8-patch-wide palette map.
n = len(parts); cols = 8; rows = int(np.ceil(n / cols)); P = 16
pal = np.zeros((rows * P, cols * P, 3), np.uint8)
Vs, Fs, UVs, k0 = [], [], [], 0
for k, (g, colour) in enumerate(parts):
    r, cc = divmod(k, cols)
    pal[r * P:(r + 1) * P, cc * P:(cc + 1) * P] = np.clip(colour, 0, 255)
    uv = np.tile([(cc + 0.5) / cols, 1 - (r + 0.5) / rows], (len(g.vertices), 1))
    Vs.append(s * np.asarray(g.vertices, float) + shift); Fs.append(np.asarray(g.faces) + k0); UVs.append(uv); k0 += len(g.vertices)
X, F, UV = np.vstack(Vs), np.vstack(Fs), np.vstack(UVs)
Image.fromarray(pal).resize((cols * 32, rows * 32), Image.NEAREST).save(out / f"{name}_diffuse.png")

# Bound to the head: nearest triangle of the body, barycentric, plus the offset.
body_faces = groups["body"]
head = np.unique(body_faces)[T[np.unique(body_faces), 1] > T[np.unique(body_faces), 1].max() - 2.5]
hf = body_faces[np.isin(body_faces, head).all(1)]
_, fi = cKDTree(T[hf].mean(1)).query(X); refs = hf[fi]
A, B, C = T[refs[:, 0]], T[refs[:, 1]], T[refs[:, 2]]
v0, v1, v2 = B - A, C - A, X - A
d00, d01, d11, d20, d21 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1), (v2 * v0).sum(1), (v2 * v1).sum(1)
den = np.maximum(d00 * d11 - d01 * d01, 1e-12)
bv, bwc = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
bw = np.clip(np.c_[1 - bv - bwc, bv, bwc], 0, 1); bw /= bw.sum(1, keepdims=True)
offset = X - (bw[:, :, None] * T[refs]).sum(1)
with open(out / f"{name}.obj", "w", encoding="utf-8") as f:
    f.write("# glasses placed on the T-pose base head; see the .mhclo\n")
    for q in X: f.write(f"v {q[0]:.5f} {q[1]:.5f} {q[2]:.5f}\n")
    for u, w in UV: f.write(f"vt {u:.5f} {w:.5f}\n")
    for t in F: f.write("f " + " ".join(f"{i + 1}/{i + 1}" for i in t) + "\n")
with open(out / f"{name}.mhclo", "w", encoding="utf-8") as f:
    f.write(f"# placed by fit_glasses.py\nname {name}\nobj_file {name}.obj\nmaterial {name}.mhmat\nverts 0\n")
    for (i, j, k), (a, b, c_), d in zip(refs, bw, offset):
        f.write(f" {i} {j} {k} {a:.5f} {b:.5f} {c_:.5f} {d[0]:.5f} {d[1]:.5f} {d[2]:.5f}\n")
(out / f"{name}.mhmat").write_text(f"name {name}\ndiffuseColor 1.0 1.0 1.0\nshininess 0.5\nopacity 1.0\nbackfaceCull False\ndiffuseTexture {name}_diffuse.png\n", encoding="utf-8")
print(f"wrote {out}: {len(X)} vertices, {n} parts, scale {s:.4f}, width {np.ptp(X[:, 0]) * 100:.0f} mm")
