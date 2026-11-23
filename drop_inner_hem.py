"""A fitted top (OBJ + MHCLO) without the inner layer of its hem: in the hem's lowest SHARE of the
garment's height, the faces that face the body (their normal against the direction out from the
body's axis by more than FACING) are dropped, with the vertices no face uses.
    python drop_inner_hem.py DIR NAME [SHARE FACING]       (defaults 0.1 and 0.2)

A folded hem has its facing inside; fitted, the off-skin push moved the facing down off the
underbust (the skin faces down there) until it hung 3 cm below the outer edge, a white band with
dark gaps above it. On the graphic crop top the lowest tenth faced inward on 40% of its faces (the
facing) and the cloth above on 5%. Telling the layers apart by distance from the body cut holes in
the front instead: below the bust the outer cloth falls inward."""
import sys, pathlib, numpy as np
import catalogue, morph

d, name = pathlib.Path(sys.argv[1]), sys.argv[2]
share, facing_in = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.1, 0.2)
base, G = catalogue.load_base(); T = morph.t_pose_body(base, "female")[0]; body = T[np.unique(G["body"])]
obj, clo = d / f"{name}.obj", d / f"{name}.mhclo"
lines = obj.read_text(encoding="utf-8").splitlines()
v = [l for l in lines if l.startswith("v ")]; vt = [l for l in lines if l.startswith("vt ")]; f = [l for l in lines if l.startswith("f ")]
P = np.array([[float(x) for x in l.split()[1:4]] for l in v])
faces = np.array([[int(t.split("/")[0]) - 1 for t in l.split()[1:]] for l in f])
c = P[faces].mean(1)
n = np.cross(P[faces[:, 1]] - P[faces[:, 0]], P[faces[:, 2]] - P[faces[:, 0]])
n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
# The body's axis at the hem's height: the middle of its torso there.
band = body[(np.abs(body[:, 1] - c[:, 1].min() - 0.3) < 0.4) & (np.abs(body[:, 0]) < 1.5)]
cx, cz = band[:, 0].mean(), band[:, 2].mean()
out = np.c_[c[:, 0] - cx, np.zeros(len(c)), c[:, 2] - cz]
out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
inner = (c[:, 1] < P[:, 1].min() + share * np.ptp(P[:, 1])) & ((n * out).sum(1) < -facing_in)
keep = np.flatnonzero(~inner)
used = np.unique(faces[keep]); new = -np.ones(len(P), int); new[used] = np.arange(len(used))
head = [l for l in lines if not l.startswith(("v ", "vt ", "f "))]
obj.write_text("\n".join(head + [v[i] for i in used] + [vt[i] for i in used] +
                         ["f " + " ".join(f"{new[i] + 1}/{new[i] + 1}" for i in faces[k]) for k in keep]) + "\n", encoding="utf-8")
cl = clo.read_text(encoding="utf-8").splitlines(); start = next(i for i, l in enumerate(cl) if l.startswith("verts")) + 1
clo.write_text("\n".join(cl[:start] + [cl[start + i] for i in used] + cl[start + len(v):]) + "\n", encoding="utf-8")
print(f"{name}: {inner.sum()} faces of the hem's inner layer dropped; {len(f)} -> {len(keep)} faces")
