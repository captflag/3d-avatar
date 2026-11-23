"""Fitted garments (MHCLO folders) on the female body in the app's relaxed stance (compose.carried relaxed_pose on the T-pose rig),
front and 40 degrees.   python garment_views.py OUT.png "label=DIR,DIR" ..."""
import sys, pathlib, numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
import catalogue, garments, morph, render, rig, compose
base, G = catalogue.load_base(); T, back = morph.t_pose_body(base, "female")
skel = rig.load_skeleton(T); joints, weights = rig.weight_table(skel, len(T))
P = garments.deform(T, joints, weights, garments.skinning_matrices(skel, compose.carried(compose.relaxed_pose(), skel, back)))
out_png, looks = sys.argv[1], [a.split("=") for a in sys.argv[2:]]
YAWS = [int(a) for a in __import__("os").environ.get("VIEWS", "0,40").split(",")]   # degrees turned, e.g. VIEWS=0,90
jobs = []
for k, (label, dirs) in enumerate(looks):
    sc = trimesh.Scene(); b = trimesh.Trimesh(P, G["body"], process=False)
    b.visual = trimesh.visual.ColorVisuals(b, vertex_colors=np.tile([200, 160, 135, 255], (len(P), 1))); sc.add_geometry(b, node_name="body")
    for d in dirs.split(","):
        for p in sorted(pathlib.Path(d).glob("*.mhclo")):
            m = garments.load_mhclo(p, P); tex, nrm, tr = garments.material_textures(m.metadata["material"], None, size=1024)
            m.visual = trimesh.visual.TextureVisuals(uv=m.metadata["uv"], material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=tex, normalTexture=nrm, alphaMode="MASK" if tr else "OPAQUE", doubleSided=True, roughnessFactor=0.8))
            sc.add_geometry(m, node_name=p.stem)
    for yaw in YAWS:
        s2 = sc.copy(); s2.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 1, 0]))
        g = pathlib.Path(f"out/picked/relax-{k}-{yaw}.glb"); s2.export(g); jobs.append((g, pathlib.Path(f"out/renders/relax-{k}-{yaw}.png"), False))
render.render(jobs, size=1000, body=True)
font = ImageFont.truetype("arial.ttf", 14); tiles = []
for g, png, _ in jobs:
    im = Image.open(png).convert("RGBA"); bg = Image.new("RGBA", im.size, (215, 215, 215, 255)); bg.alpha_composite(im)
    x0, y0, x1, y1 = im.getbbox(); w, h = x1 - x0, y1 - y0
    t = bg.convert("RGB").crop((x0, y0, x1, y0 + int(0.62 * h))); t.thumbnail((330, 420)); tiles.append(t)
sheet = Image.new("RGB", (len(YAWS) * 335, len(looks) * 445), "white"); dr = ImageDraw.Draw(sheet)
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % len(YAWS)) * 335, (k // len(YAWS)) * 445))
    if k % len(YAWS) == 0: dr.text((4, (k // len(YAWS)) * 445 + 424), looks[k // len(YAWS)][0], fill="black", font=font)
sheet.save(out_png); print(out_png)
