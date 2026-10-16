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
generated stays in the gallery at the bottom, so the team can compare runs. Each
card is a render of the avatar itself, badged `full body` or `face only`, so you
can see at a glance what each run produced. Opening the review page with no model
named shows the newest avatar.

Pick **3D model** or **2D image** beside the photo picker, then use **Download**
on the result. **Full body** is on by default: the fitted face goes on a base body, dressed from
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

## What you can download

| Choice | File | Where it opens |
|---|---|---|
| 3D model | `.glb` (glTF 2.0 binary), rigged | Unity, Unreal, Blender, three.js, Babylon.js, Godot, most viewers |
| 2D image | `.png`, transparent background, 900x1200 full body or 900x900 head | anything |

GLB is the default because it is the most widely supported 3D format and the only
common one that carries the mesh, textures, skeleton and skin weights in a single
file. The 2D image is that same avatar rendered in the browser, so it always
matches what you see.

FBX is the other format game pipelines ask for. It is not here because nothing in
this stack can write it: exporting FBX means driving Blender or Adobe's FBX SDK
from the pipeline. Worth adding only if an engine team actually needs it, since
Unity and Unreal both import GLB.

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

**Garments from files.** An item can name a mesh under `assets/garments/`
instead of a base-mesh group (`"file": "jacket.obj"`), positioned with `scale`
and `translate`, and marked `"fit": "tight"` or `"loose"`. It has no base-mesh
vertices, so it is rigged by borrowing weights from the body underneath it.
Every item must carry a `licence`; anything without one is refused.

Measured against garments whose true weights are known, posed in a stride that
moves parts of them by up to 30 cm, transferred weights land within 0.2 cm (shirt)
and 0.8 cm (trousers) at the 95th percentile. Loose garments are the weak case:
a skirt borrowing from the body tears between the legs, so `loose` items borrow
from MakeHuman's authored skirt shell instead. That path has no foreign skirt to
test against yet; check the first real one.

**MakeHuman garments (MHCLO).** An item can name a MakeHuman clothes file
(`"mhclo": "catsuit/catsuit.mhclo"`). These are the best route for real clothes:
each garment vertex is stored as a blend of three body vertices plus an offset,
so the garment is rebuilt on each person's shaped body, and its skin weights are
exact rather than borrowed. Two CC0 garments from MakeHuman's community
repository are wired in to prove the path (catsuit, tights); fetch them with:

```bash
cd avatar-engine && for g in tightsv2 catsuit; do mkdir -p assets/garments/$g && for e in mhclo obj mhmat; do curl -sSL -o assets/garments/$g/$g.$e https://raw.githubusercontent.com/makehumancommunity/communityassets-wip/master/clothes/$g/$g.$e; done; done
```

Hoodies, jeans and shoes in MHCLO exist on MakeHuman's asset portal; licences
there are per item, so check each before it goes into the catalogue.

**Surface finish lives in the file.** Each part is exported with its own
roughness (skin 0.5, eyes 0.05, hair 0.45, cloth 0.85, shoes 0.4; a catalogue
item may set `roughness`). The viewer no longer overrides it, so the studio
shows what Blender, Unity or Unreal will.

**Skin under clothes is alpha-masked, not deleted.** Every body triangle is
kept and the texture's alpha hides what a garment covers, leaving a band of skin
under each edge. Deleting triangles left nothing behind a cuff or collar when
the figure moved.

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
cd avatar-engine && uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe "mediapipe<1" numpy pillow trimesh flask scipy pygltflib opencv-python-headless
```

```bash
cd avatar-engine && mkdir -p assets && curl -sSL -o assets/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task && curl -sSL -o assets/canonical_face_model.obj https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj && curl -sSL -o assets/body_base.obj https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/3dobjs/base.obj && curl -sSL -o assets/default.mhskel https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/rigs/default.mhskel && curl -sSL -o assets/default_weights.mhw https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/rigs/default_weights.mhw
```

The head-shape fit needs MakeHuman's face targets (CC0, about 3 MB), fetched by
sparse checkout so the rest of their repository is not downloaded:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/makehumancommunity/makehuman.git mh && cd mh && git sparse-checkout set makehuman/data/targets/nose makehuman/data/targets/mouth makehuman/data/targets/eyes makehuman/data/targets/chin makehuman/data/targets/cheek makehuman/data/targets/forehead makehuman/data/targets/head makehuman/data/targets/eyebrows makehuman/data/targets/ears && mkdir -p ../avatar-engine/assets/targets && cp -r makehuman/data/targets/* ../avatar-engine/assets/targets/
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
| `catalogue.py` | builds catalogue items, cut from the base mesh or loaded from files, and masks skin under them |
| `garments.py` | rigs garments with their own topology by weight transfer, poses meshes in numpy, imports garment files |
| `morph.py` | fits the head's shape to the person from MakeHuman's CC0 face targets |
| `facetexture.py` | warps the photo onto the head's features, bakes it into UV, Poisson-blends the seam |
| `skin.py` | generates a body skin texture and fine grain matched to the person's cheek colour |
| `rig.py` | the skeleton: bone hierarchy, joint positions, per-vertex weights |
| `skinned_glb.py` | writes the GLB, because trimesh cannot export a skin |
| `test_rig.py` | the check that the rig is sound; run it after touching any of the above |
| `ARCHITECTURE.md` | target architecture, the licence audit behind it, and what is built versus next |
| `adr/` | architecture decisions, starting with why the head model is MakeHuman and not FLAME |
| `REALISM-GAP.md` | every reason an avatar looks fake, and what each costs to fix |
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
on the same terms, so both can be edited and shipped as our own: repaint weights,
drop bones we do not need, rename them to a convention an engine expects. Weights
are indexed by base-mesh vertex, so the one rule is that changing mesh topology
means remapping them, and `test_rig.py` is what catches it when that goes wrong. Their application code is AGPL, but we use none of it.

Both were chosen over the better-known research face and body models, which are
free for research only and need a paid licence to ship. Keep that constraint when
swapping the engine: see `../avatarsdk-re/CLONE-PLAN.md`.

Sample photos under `samples/` are Google's public MediaPipe test assets, used
here as fixtures only. Photos the team uploads stay on this PC under `uploads/`;
delete them when done, since they are faces.
