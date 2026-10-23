# Skin and eyes shaded closer to a photograph

Branch `claude/funny-benz-98b565`, on top of `side-views` (c420a1c). Changes are in
`lib/scene-setup.js` only: `applyMaterials` now also calls `shadeSkinAndEyes`. The GLB
files and the lights are unchanged. Every page that draws through `applyMaterials` gets it:
studio, review and synth. synth.html renders the synthetic training heads, so synthetic
data rendered from now on has the new skin and eyes.

## What is in

| piece | how | cost |
|---|---|---|
| Skin, red light scattering further | Direct diffuse per channel `pow(N.L, k)`, k = (0.75, 0.9, 1.0). Red fades later towards the shadow side than blue. It is still zero at N.L = 0, so it never lights what the key's shadow map darkens (a wrap term would draw a line where the shadow cuts it). | 3 pow per light |
| Skin, pores blurred by scattering | Each channel's diffuse (direct and environment) sees a different blend of the pore normal map and the surface without it: red 0.3, green 0.7, blue 1.0. Specular keeps the full pores. | 3 environment irradiance lookups instead of 1 |
| Wet lower lid | Per fragment: within 0.06 eyeball radii of the eye opening, below the eye's centre, fading towards the corners. There the roughness is 0.06 and the specular x4. The opening is the head mesh's open boundary round each eye (vertices welded by position), stored as the distance from the eye's axis in 72 directions (uniforms). | 2-iteration loop on head fragments |
| Cornea | The eyes become MeshPhysicalMaterial with a clear coat (roughness 0.03) over the file's eye roughness (0.18, kept). The iris is read from where the refracted view ray (IOR 1.376) meets it, with the anterior chamber 0.25 eyeball radii deep at the pupil, thinning to nothing at the iris edge (facetexture.bake_eyes' 28.5 degrees). The sideways travel is capped at 0.8 x the local depth, so it never reads the white from beyond the edge. | derivatives + one refract on eye fragments |

The calibration values are in `SHADING` at the top of the new code. `out/lab.html` overrides
them per render.

## Tried and left out

- **Specular x 1.3 on the T-zone, x 1.5 on the lips** (scaled from the file's roughness map).
  It barely shows at portrait scale. It made the foreheads of photo-14 and photo-16 3-4 L*
  brighter, and they already render 18-20 too bright at the photo's pose. Region-varying shine
  stays as the file's roughness map (T-zone 0.45, cheeks 0.6, lips 0.35).
- **Mesh-scale soft normals for red** (vertex normals averaged with their neighbours 3, 6 or 12
  times, skinned, used per channel for direct and environment diffuse). The head mesh has a
  vertex every few millimetres, so even 3 rounds tilt whole regions. Chins turn down towards
  the neck and cheeks sideways. The faces came out cooler and flatter, the opposite of
  scattering, and the chins darker (chin |dL*| sum 74 -> 91 at 6 rounds, 105 at 12).
- **Cornea depth 0.5-0.6**: the iris edge doubled in a 30-degree turn.
- **Wet line gain 2 / roughness 0.1**: invisible. A 0.1-radius band glared.
- **Per-vertex wet attribute**: the head's lid has 4-5 vertices along the lower lid, so a
  per-vertex value smeared into blobs under the eye. Per-sector "nearest vertex" rims put
  patches on the cheek.

## Measured

Five test heads built from this branch (compose.py, ref-woman female). Renders follow
studio.html's `drawAvatar(focusHead)`: 900x900, the same lights, GTAO and tone mapping.

- **lm**: whole-face landmark error at the photo's pose, % of iris spacing, after a
  similarity fit on the 468 MediaPipe points.
- **fh / ch / cn**: render minus photo L* (OpenCV 0-255), median in disks of 0.07 eye
  spacings at forehead 151, cheeks 50+280 and chin 199, at the photo's pose.

Rendering is not deterministic (GTAO noise and alpha-to-coverage), so "off" (all of this
disabled, identical to the old shader) was rendered twice to show the noise:

| face | lm off / off2 | lm final / final2 | fh off -> final | ch off -> final | cn off -> final |
|---|---|---|---|---|---|
| ref-man | 2.04 / 2.06 | 2.02 / 2.03 | -2 -> -2 | -12 -> -11 | +28 -> +27 |
| ref-woman | 1.91 / 1.91 | 1.91 / 1.93 | -3 -> -3 | -5 -> -4 | +1 -> +2 |
| photo-10 | 1.79 / 1.80 | 1.75 / 1.73 | -6 -> -6 | -4.5 -> -5 | -2/-8 -> -5/-2 |
| photo-14 | 1.47 / 1.44 | 1.42 / 1.36 | +18 -> +18 | -24.5 -> -22.5/-20.5 | -19 -> -20 |
| photo-16 | 1.38 / 1.38 | 1.37 / 1.37 | +20 -> +20 | -21 -> -20 | -18 -> -18 |
| sum | 8.59 / 8.58 | 8.47 / 8.41 | 49 -> 49 | 66.5 -> 61.5 | 71 -> 70.5 |

Landmarks hold. Every face is within 0.06 of "off" or better, and the sums are 0.1-0.2
lower, which is within the noise. Brightness: cheeks come about 5 L* closer to the photos in
sum (the red falloff brightens surfaces that face partly away), with forehead and chin
unchanged. Level renders (lmL): sums 12.11/12.15 off against 12.13/12.13 final.

Each piece was also measured alone (pose lm sum, against 8.45-8.59 for "off" across runs):
red falloff + pore blend 8.57, wet line 8.61, cornea 8.47. All are inside the noise. The
wet line and cornea were at their earlier settings then: gain 2, and no cap on the travel.

## Look

The images stay outside the repo: they are renders of real people, and the repo is public. They are in the shared
notes folder of this group of sessions, `photo-shading/`:

- `pose.png`: photo | off | final at the photo's pose. At this scale the change is small:
  slightly warmer cheeks.
- `turn.png`, `turn-full.png`: off | final with the head turned 30 degrees. The warm, softer
  transition into the shadow side is where the skin change shows most.
- `eyes.png`, `eye-detail.png`: eye close-ups (2.6x zoom), level and turned. Before, the
  catchlight was a soft grey blob. Now it is a crisp white reflection of the room's lights,
  and a thin bright waterline runs along the lower lid. Turned, the iris sits a little
  behind the surface.

Judged side by side with the photos, the eyes are the clear win. The skin change is real but
subtle. What still makes these heads read as CG is not shading: geometry (coarse head mesh,
lids), the de-lit texture (eye-socket darkening, pasted brows) and the sclera texture (too
white and grey-blue, with a hard edge to the lid).

## Tools

These are outside the repo, in the same `photo-shading/` folder. Copy `lab.html` to the
worktree's `out/` first, then run from the worktree root:
`python serve.py` (port 8795; saves PUT renders to out/renders/),
`node shoot.mjs '[[tag, glbDir, shadingOverrides, faces, views]]'` (headless Chrome),
`measure.py tag...` and `sheet.py out.png suffix tag...`.
The Python is the Microsoft Store build, which cannot see files under AppData\Roaming, so
copy the scripts into the worktree (out/tools) before running them.
