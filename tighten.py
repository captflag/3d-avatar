"""A fitted garment (MHCLO) pulled in toward the skin where it sits outside the torso: every vertex bound
beyond XMIN to the side of the male T-pose body's centre (the shoulders and arms) keeps SHARE of its offset
off the skin, but no less than CLEAR.
    python tighten.py DIR NAME SHARE [XMIN CLEAR]       (defaults 1.2 and 0.05, decimetres)

The suit's jacket is cut with padded shoulders and roomy sleeves; fitted to our arms, which hang down
in the app, it stood out from them like a second pair of shoulders. Its offsets off the skin (a binding's
offset is the step from the nearest point on its triangle, so about along the skin's normal) were in
line with the other jackets, so this narrows the cloth instead of moving it."""
import sys, pathlib, numpy as np
import catalogue, morph

d, name, share = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
xmin = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
clear = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
base, _ = catalogue.load_base(); T = morph.t_pose_body(base, "male")[0]
clo = d / f"{name}.mhclo"; lines = clo.read_text(encoding="utf-8").splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("verts")) + 1
n = 0
for i in range(start, len(lines)):
    p = lines[i].split()
    if len(p) != 9: break
    idx, w, off = [int(x) for x in p[:3]], np.array(p[3:6], float), np.array(p[6:], float)
    if abs(w @ T[idx, 0]) <= xmin: continue
    size = np.linalg.norm(off)
    if size <= clear: continue
    off *= max(share, clear / size); n += 1
    lines[i] = " ".join(p[:6] + [f"{x:.5f}" for x in off])
clo.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{name}: {n} vertices pulled in to {share:.0%} of their offset (not under {clear * 100:.0f} mm)")
