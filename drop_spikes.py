"""A fitted top (OBJ + MHCLO) without what stands off the shoulders once the arms are down: each vertex
placed by its MHCLO rows on the body in the app's relaxed stance; faces whose vertices all sit more than
GAP off that skin, above the armpits and further out than XMIN, dropped.   python drop_spikes.py DIR NAME XMIN GAP"""
import sys, pathlib, numpy as np
from scipy.spatial import cKDTree
import catalogue, garments, morph, rig, compose
d, name, xmin, gap = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
base, G = catalogue.load_base(); T, back = morph.t_pose_body(base, "female"); body = np.unique(G["body"])
skel = rig.load_skeleton(T); joints, weights = rig.weight_table(skel, len(T))
P = garments.deform(T, joints, weights, garments.skinning_matrices(skel, compose.carried(compose.relaxed_pose(), skel, back)))
clav = P[np.unique(G["joint-l-clavicle"]), 1].mean()
obj, clo = d / f"{name}.obj", d / f"{name}.mhclo"
lines = obj.read_text(encoding="utf-8").splitlines()
v = [l for l in lines if l.startswith("v ")]; vt = [l for l in lines if l.startswith("vt ")]; f = [l for l in lines if l.startswith("f ")]
c = clo.read_text(encoding="utf-8").splitlines(); start = next(i for i, l in enumerate(c) if l.startswith("verts")) + 1
rows = np.array([[float(x) for x in l.split()] for l in c[start:start + len(v)]])
refs, bw, off = rows[:, :3].astype(int), rows[:, 3:6], rows[:, 6:9]
Q = (bw[:, :, None] * P[refs]).sum(1) + off                      # where each vertex sits, posed
far = cKDTree(P[body]).query(Q)[0] > gap
spike = far & (Q[:, 1] > clav - 0.6) & (np.abs(Q[:, 0]) > xmin)
faces = [[int(t.split("/")[0]) - 1 for t in l.split()[1:]] for l in f]
keep = [k for k, fc in enumerate(faces) if not spike[fc].all()]
used = sorted({i for k in keep for i in faces[k]}); new = {o: n for n, o in enumerate(used)}
head = [l for l in lines if not l.startswith(("v ", "vt ", "f "))]
obj.write_text("\n".join(head + [v[i] for i in used] + [vt[i] for i in used] +
                         ["f " + " ".join(f"{new[i] + 1}/{new[i] + 1}" for i in faces[k]) for k in keep]) + "\n", encoding="utf-8")
clo.write_text("\n".join(c[:start] + [c[start + i] for i in used] + c[start + len(v):]) + "\n", encoding="utf-8")
print(f"{name}: {spike.sum()} vertices stand off the shoulders posed; {len(f)} -> {len(keep)} faces")
