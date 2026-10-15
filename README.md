# Avatar engine — version 1

One photo in, one textured 3D face out, plus the expression it detected. This is
the generation step the rest of the avatar product hangs off, kept deliberately
small so we can judge one thing: does the avatar look like the person in the photo.

## Run it for the team

Double-click **`start-studio.cmd`**, or run it from a terminal. Then open:

```
http://localhost:8777
```

Drop a photo on the page and the avatar appears next to it, orbitable. Everything
generated stays in the gallery at the bottom, so the team can compare runs.

**Full body** is on by default: the fitted face goes on a base body, dressed from
the catalogue, with skin tone and hair colour sampled from the photo. Pick the male
or female outfit beside it. Untick it for the face mesh on its own.

Add `&focus=head` to a review URL to frame the head instead of the whole figure,
which is the only way to judge likeness on a full-body avatar.

Teammates on the same network use this PC's address instead of localhost — the
server prints it on startup, e.g. `http://192.168.3.42:8777`. Windows will likely
ask once whether to allow Python through the firewall; allow it for the private
network only. There is no login, so keep this on the office network and off the
public internet.

Generation takes a couple of seconds on CPU. One photo is processed at a time.

## Run one photo from the command line

```bash
avatar-engine/.venv/Scripts/python.exe avatar-engine/generate.py path/to/photo.jpg -o avatar-engine/out/name.glb
```

For the full dressed figure instead of the face alone:

```bash
avatar-engine/.venv/Scripts/python.exe avatar-engine/compose.py path/to/photo.jpg -o avatar-engine/out/name.glb --gender female
```

`--wear shirt_white,trousers_navy,shoes_dark,hair_short` overrides the default outfit.

## The catalogue

Wearables live in `catalogue.json`: id, category, which part of the base mesh the
item is cut from, the heights to cut it at, its colour and its licence. Adding an
item is a data change; the pipeline reads the file and does not need touching.

This follows what the competitor does, which the reverse engineering showed in
detail (`../avatarsdk-re/REPORT.md` section 5): one body, a pile of separately
authored wearables, and **visibility masks** so the garment hides the skin beneath
it rather than every outfit shipping its own body. `hide_covered()` is that mask.
Without it the bare mesh pokes through the clothes.

Hair colour is sampled from just above the hairline in the photo, skin tone from
the cheeks, so both follow the person rather than a preset.

## The rig

Every avatar comes out skinned to a 163-bone skeleton, so it can be posed and
animated in Unity, Unreal, Blender or three.js.

Both inputs are MakeHuman's and CC0: `default.mhskel` gives the bone hierarchy,
where each bone's head and tail is a named cluster of base-mesh vertices whose
centroid is the position, and `default_weights.mhw` gives per-bone vertex
weights for that same mesh. Because every wearable is cut from the base mesh, one
weight table skins the body and the clothes together — rotate an arm and the
sleeve follows. The fitted face is our own geometry, so it binds rigidly to the
head bone.

To check it by hand, open a review page and use the console:

```js
window.__avatar.getObjectByName('upperarm01L').rotation.z = -1.1
```

Bone names lose their dots in three.js, so `upperarm01.L` becomes `upperarm01L`.

Bones are translation-only: the rest pose is whatever stance the base mesh is in,
and the roll angles in the file's rotation planes are not read. That deforms
correctly and takes simple retargeted poses; animation authored against
MakeHuman's own bone axes would need the planes too.

Writes `name.glb` (mesh plus embedded texture) and `name.json` (source photo,
landmark and triangle counts, detected expression). Anything made this way also
shows up in the studio gallery.

## Set up

The virtualenv, the two model files and anything generated are gitignored, so a
fresh checkout needs this once:

```bash
cd avatar-engine && uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe "mediapipe<1" numpy pillow trimesh flask
```

```bash
cd avatar-engine && mkdir -p assets && curl -sSL -o assets/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task && curl -sSL -o assets/canonical_face_model.obj https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj && curl -sSL -o assets/body_base.obj https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/3dobjs/base.obj && curl -sSL -o assets/default.mhskel https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/rigs/default.mhskel && curl -sSL -o assets/default_weights.mhw https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/rigs/default_weights.mhw
```

Pinned to the 0.10 line for reproducibility; the code only uses the tasks API,
which 1.x also has. Neither package ships the face topology any more, which is
why the canonical model is downloaded separately rather than read out of the
installed package.

## How it works

1. MediaPipe FaceLandmarker fits 468 landmarks (plus 10 iris points) to the photo.
2. Those landmarks become vertex positions; the triangles come from the canonical
   face model, so the topology is identical for every avatar we produce.
3. The photo is the texture. Each landmark's position in the image is its UV, so
   every vertex is painted with the pixel it was detected on.
4. The head is rotated into an upright, forward-facing pose using the facial
   transform MediaPipe returns. Without that step a tilted photo produces a
   tilted avatar, which nothing downstream can rig.

Consistent topology is the property worth protecting: it is what later lets a
haircut, a pair of glasses or a blendshape set made once fit every avatar.

## Files

| File | What it is |
|---|---|
| `generate.py` | the engine: photo to a face GLB, importable or run from the command line |
| `compose.py` | assembles the avatar: face on body, dressed, skin and hair coloured |
| `catalogue.json` | the wearables, as data: what to cut, where, what colour, what licence |
| `catalogue.py` | cuts items out of the base mesh and applies the visibility masks |
| `rig.py` | the skeleton: bone hierarchy, joint positions, per-vertex weights |
| `skinned_glb.py` | writes the GLB, because trimesh cannot export a skin |
| `test_rig.py` | the check that the rig is sound; run it after touching any of the above |
| `app.py` | the studio server: upload, validate, generate, gallery |
| `studio.html` | the upload page the team uses |
| `review.html` | the photo-versus-avatar viewer, embedded by the studio and usable on its own |
| `start-studio.cmd` | starts the studio |

## What it is not, yet

- **No bone roll, and no authored poses or animations.** The skeleton deforms
  correctly but carries no rest pose of its own beyond the base mesh's stance.
- **Clothes are cut from the body, not tailored.** Items are slices of the base
  mesh's own garment shells, so they fit perfectly but have no collars, cuffs,
  buttons or fabric texture. Real authored garments drop in at `build_item()`.
- **The face is placed, not wrapped.** It is scaled onto the skull by bounding
  box and its border ring trimmed off, so there is still a visible seam at the
  jaw. A proper wrap-and-blend is the upgrade.
- **Hair is a skull cap**, not strands, and only one style.
- **One body shape.** Nothing reads height, build or age from the photo; male and
  female differ only by default outfit.
- **Front projection only.** Anything the camera could not see is unpainted, and
  the texture carries the photo's own lighting rather than a neutral albedo.
- **A testing tool, not the product.** No login, no accounts, no queue, no
  database, and the uploads sit on disk. The product path is the Fastify API in
  `apps/api`.

`generate_avatar()`, `compose()` and `build_item()` are the seams: a fuller head
model, a real body or authored garments drop in behind them without changing the
studio, the viewer or anything above.

## Licensing

MediaPipe and the canonical face model are Apache-2.0. The body base mesh is
MakeHuman's, released under CC0 1.0 along with the rest of their assets, meshes,
targets, textures and clothes, so it is public domain and carries no attribution
requirement. The skeleton and skin weights (`default.mhskel`, `default_weights.mhw`) are CC0
on the same terms. Their application code is AGPL, but we use none of it.

Both were chosen over the better-known research face and body models, which are
free for research only and need a paid licence to ship. Keep that constraint when
swapping the engine: see `../avatarsdk-re/CLONE-PLAN.md`.

Sample photos under `samples/` are Google's public MediaPipe test assets, used
here as fixtures only. Photos the team uploads stay on this PC under `uploads/`;
delete them when done, since they are faces.
