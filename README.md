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

There are two ways in. **Choose from gallery** takes any photo. **Guided
capture** takes three, each checked before it is taken against the limits the
pipeline was measured to need: the face straight on, then the head turned to the
person's left and to their right, 30-45 degrees. The turned photos measure how
far the nose, lips and chin stand out, which one photo only guesses, and paint
the cheeks' sides and the jaw towards the ears, which the front photo never sees
(`views.py`; full body only). The checks:
- the face 500+ pixels across (400 turned), twice what the photo advice asks;
- facing the camera, or turned the asked way by 30-45 degrees (MediaPipe reads
  that as 25-40), chin level and head upright;
- mouth closed and relaxed, since a broad smile is never fully taken off the
  resting face; eyes open;
- sharp: how much of the face's edge contrast a further blur takes away, so a
  soft or shaken photo fails whatever its texture;
- front only: even light on both cheeks, and the hair in frame.

With a camera, once every line is ticked for half a second the photo takes
itself, and **Take photo anyway** takes it regardless. The checks use the same
MediaPipe face model as the pipeline, served at `/models/face_landmarker.task`.
Browsers open the camera only on `localhost` or https; without one (a teammate's
phone on the office network, or no camera fitted) each step takes a photo file
instead (**Use a photo instead**; a phone opens its own camera for it), checked
the same way, with **Use this photo anyway** to override. **Skip** leaves a
turned photo out and **Done** stops at any point. From the command line,
`compose.py front.jpg -o out/x.glb --side left.jpg --side right.jpg`.

**How alike is it?** `score.py` asks AWS Rekognition's CompareFaces how alike
the photo and the avatar's head render are, 0-100 (a recognition model, paid
per call, commercial use allowed). Get the render from the studio (**2D
image**, **Full body** unticked, **Download**), then:
`.venv/Scripts/python.exe score.py photo.jpg render.png`. It uses the usual AWS
credentials (`aws configure`); it needs `boto3` (`uv pip install --python
.venv/Scripts/python.exe boto3`).

Free and local: `identity_score.py photo.jpg render.png` gives SFace's cosine
similarity (OpenCV's same-person threshold is 0.363; two real photos of one
person score about 0.5-0.7, different people about 0.15). It is the keep/drop
measure for likeness work. It is an internal yardstick only: the weights are
Apache-2.0 but were trained on research face datasets, so it must never ship
or be imported by the product. Its two model files go in `assets/`; the URLs
are in the script.

**Scoring a change in one go.** `score_run.py TAG --against BASE` builds every
test face, renders it at the photo's head pose and level with no browser window
(`render.py`, the studio's own page in headless Chrome or Edge), and prints
identity and the hair's strand angle against the photo beside the other tag's
(about 90 s for five faces). Builds and renders are repeatable, so a rerun gives
the same numbers and any difference is the change. The test faces are listed in
`uploads/test-set.json` (git-ignored, beside the photos): each with a gender,
`others` (other photos of the same person, scored against, never built from)
and `sides` (turned photos built from, as guided capture sends them).
`render.py a.glb b.glb` renders GLBs on their own.

**Is the photo good enough?** With `ANTHROPIC_API_KEY` set in the studio's
environment, a chosen photo is first sent to the Claude API (`photo_check.py`,
claude-sonnet-5-5) and the studio lists what would cost likeness ("Take off the
sunglasses", "Face the camera straight on") with the hair, facial hair and
glasses it saw; where the photo's segmentation finds no hair to measure, its
haircut picks the catalogue cut. Without the key nothing is sent and nothing
changes. It needs `anthropic` (`uv pip install --python .venv/Scripts/python.exe
anthropic`); from the command line, `.venv/Scripts/python.exe photo_check.py photo.jpg`.

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

`--wear outfit_white_tee_jeans,shoes_black,hair_underlayer,hair_short01,eyebrows,eyelashes`
overrides the default outfit. `--pose rest` opens the file in the plain rest stance
instead of the relaxed one (same skeleton either way). `--face-patch` also lays
the fitted landmark patch over the head; it is off by default because, with the
skull shaped to the face, it only ever showed as a hard edge on the forehead.

## The catalogue

Wearables live in `catalogue.json`: id, category, which part of the base mesh the
item is cut from, the heights to cut it at, its colour and its licence. Adding an
item is a data change; the pipeline reads the file and does not need touching.

This follows what the competitor does, which the reverse engineering showed in
detail (`../avatarsdk-re/REPORT.md` section 5): one body, a pile of separately
authored wearables, and **visibility masks** so the garment hides the skin beneath
it rather than every outfit shipping its own body. `hide_covered()` is that mask.
Without it the bare mesh pokes through the clothes.

Hair colour is read from a band of hair above and beside the forehead, with
backdrop and skin pixels thrown out; brow colour from the brows; skin tone from
the cheeks. All three follow the person rather than a preset. A beard or
moustache is kept from the photo: the head bake fades the photo into generated
skin towards the edge of the face, except where the photo is clearly darker than
the person's own skin below the nose (`facetexture.facial_hair`).

**MakeHuman's own hair, clothes, brows and lashes.** The defaults are
MakeHuman's system assets: shirt-and-jeans outfits with fabric, normal and
ambient-occlusion maps, leather shoes, six short haircuts as alpha-cut hair
cards, eyebrows and eyelashes. All are CC0 — the pack's manifest marks every
item so. Hair and brows are recoloured to the person's own, keeping the
strand-to-strand variation; a darker skull cap (`hair_underlayer`) sits under
the haircut so no pale scalp shows between strands. Outfits with a MakeHuman
logo print are deliberately left out. `python fetch_assets.py` fetches exactly
the folders the catalogue names.

Seven more haircuts come from the MakeHuman community's asset packs (Hair 01,
CC0; Hair 02 and 03, CC-BY), fetched the same way and checked against each
pack's own manifest. The short cut every photo with short hair and no fringe
gets is `hair_culturalibre_hair_05` (CC0), which outscored all fourteen short
cuts on the test faces (`appearance.hair_style` has the numbers); the rest are
there for the photo check and `--wear`.

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
(`"mhclo": "garments/catsuit/catsuit.mhclo"`, relative to `assets/`). These are the best route for real clothes:
each garment vertex is stored as a blend of three body vertices plus an offset,
so the garment is rebuilt on each person's shaped body, and its skin weights are
exact rather than borrowed. Two CC0 garments from MakeHuman's community
repository are wired in to prove the path (catsuit, tights); fetch them with:

```bash
cd avatar-engine && for g in tightsv2 catsuit; do mkdir -p assets/garments/$g && for e in mhclo obj mhmat; do curl -sSL -o assets/garments/$g/$g.$e https://raw.githubusercontent.com/makehumancommunity/communityassets-wip/master/clothes/$g/$g.$e; done; done
```

More MHCLO clothes exist on MakeHuman's asset portal; licences there are per
item, unlike the system pack, so check each before it goes into the catalogue.

**Surface finish lives in the file.** Each part is exported with its own
roughness (skin 0.5, eyes 0.05, hair cards 0.8 so the crown does not flare
white under the key light, cloth 0.85, shoes 0.4; a catalogue item may set
`roughness`). The viewer no longer overrides it, so the studio
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

The face carries Apple ARKit's 52 blendshapes under ARKit's own names
(`eyeBlinkLeft`, `jawOpen`, `mouthSmileLeft` … `tongueOut`), on every part
they move: skin, eyes, teeth, tongue, lashes. iPhone face tracking, MediaPipe's
FaceLandmarker and audio-to-expression models all emit those names, so their
output drives the avatar as it is. "Left" is the subject's own left. They are
built from MakeHuman's CC0 expression units (`morph.arkit_shapes`), and the
shapes MakeHuman has no unit for are made there too.

The file's own weights rest at zero, so a driven rig starts from a neutral
face. The face the photo was making is kept beside them, in each mesh's extras
as `photoExpression` (three.js: `userData.photoExpression`), and the studio
shows it (`applyPhotoExpression` in `lib/scene-setup.js`). Those weights are
MediaPipe's scores for the photo, turned back through MediaPipe's measured
response to our own shapes (`morph.photo_expression`). The eyes are left to
the shape fit.

```js
// Every part the shape moves: skin, teeth and tongue open together.
window.__avatar.traverse((o) => {
  const i = o.morphTargetDictionary?.jawOpen;
  if (i !== undefined) o.morphTargetInfluences[i] = 1;
});
```

Bones are translation-only: the rest pose is whatever stance the base mesh is in,
and the roll angles in the file's rotation planes are not read. That deforms
correctly and takes simple retargeted poses; animation authored against
MakeHuman's own bone axes would need the planes too.

Writes `name.glb` (mesh plus embedded texture) and `name.json` (source photo,
landmark and triangle counts, detected expression). Anything made this way also
shows up in the studio gallery.

## Synthetic training data

`synth.py` makes the training set for the learned head-shape correction that
`adr/ADR-001-head-geometry-model.md` plans: random heads built from the same
MakeHuman CC0 face targets the fit uses, rendered as photos, each with the exact
shape behind it.

```bash
cd avatar-engine && .venv/Scripts/python.exe synth.py -n 200 --check
```

This writes `data/synth/NNNNNN.png` (512x512, `--size` to change) with a
`NNNNNN.json` beside each one. `data/` is gitignored, and a second run adds to the
folder instead of overwriting it. The JSON holds every non-zero face-target
weight (names as in `morph.load_targets()`; a missing name means 0), the
male/female macro blend, and the camera: yaw, pitch, focal length, position and
look-at in model units. It also records the lighting, exposure, skin, lips, hair,
iris and outfit that were drawn. `synth.vertices_for(female, weights)` rebuilds
the exact mesh. `--check` runs MediaPipe over the folder and fails if it finds a
face in fewer than 95% of the images.

Per head: about 40% of the face sliders move, each towards one end only, with
left and right nearly symmetric. Skin runs from very dark to very light, lips
are tinted where MakeHuman's lip-volume targets act, and hair is one of the
catalogue's seven cuts (or bald) in a natural colour. Yaw is within ±20° and
pitch within ±10°, at 24–85 mm equivalent. Key, fill, rim and environment light
and exposure are all randomised, over a gradient backdrop. Worker processes build
the heads (`--workers`, half the cores by default). `synth.html` renders them in
headless Chrome or Edge, using the studio's own lighting code, so three.js is
fetched from unpkg as in the studio and needs the network. No new dependencies.

On this PC (GTX 1660, 12 threads, 6 workers), 200 samples took 2 min 16 s, or
0.68 s each (about 5,300 an hour). Building heads on the CPU is the limit, not
the GPU render. MediaPipe found a face in 198 of 200. Both misses were the
`bob01` cut, whose fringe falls across a turned face.

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

Then MakeHuman's hair, clothes, brows and lashes, the body targets that make
a man's or a woman's figure, and the expression units the face fit uses to take
a smile off (CC0, about 60 MB, only the parts the catalogue uses):

```bash
cd avatar-engine && .venv/Scripts/python.exe fetch_assets.py
```

And MediaPipe's multiclass selfie segmenter (Apache-2.0, about 16 MB), which
reads the hair for the haircut and hair colour. Without it the default haircut
is used and hair colour falls back to sampling above the forehead:

```bash
cd avatar-engine && curl -sSL -o assets/selfie_multiclass_256x256.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

MediaPipe is pinned to the 0.10 line for reproducibility; the code only uses the tasks API,
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
| `delight.py` | takes the photo's lighting, shine and over-exposure out of the skin before it is baked |
| `appearance.py` | reads the photo for haircut, hair colour and eye colour (segmenter, whites of the eyes as reference) |
| `facetexture.py` | maps the photo onto the head through every landmark, bakes it into UV, fades it into generated skin, keeps a beard |
| `skin.py` | generates a body skin texture and fine grain matched to the person's cheek colour |
| `rig.py` | the skeleton: bone hierarchy, joint positions, per-vertex weights |
| `skinned_glb.py` | writes the GLB, because trimesh cannot export a skin |
| `test_rig.py` | the check that the rig is sound; run it after touching any of the above |
| `synth.py` | synthetic training data: random heads with known shape, rendered as photos, plus a MediaPipe check |
| `synth.html` | the headless render page `synth.py` drives, using the studio's lighting |
| `fetch_assets.py` | fetches the MakeHuman assets the catalogue names, without the whole 280 MB pack |
| `ARCHITECTURE.md` | target architecture, the licence audit behind it, and what is built versus next |
| `adr/` | architecture decisions, starting with why the head model is MakeHuman and not FLAME |
| `REALISM-GAP.md` | every reason an avatar looks fake, and what each costs to fix |
| `app.py` | the studio server: upload, validate, generate, gallery |
| `studio.html` | the upload page the team uses |
| `review.html` | the photo-versus-avatar viewer, embedded by the studio and usable on its own |
| `start-studio.cmd` | starts the studio |

## What it is not, yet

- **No bone roll and no animations.** Two stances (relaxed, rest), set by hand.
- **A small wardrobe.** Four MakeHuman outfits and two pairs of shoes; any
  outfit can be dyed per part (`"recolour": {"top": [r, g, b], "bottom": ...}`
  in `catalogue.json`) and taken in (`"slim": {"bottom": 0.6}`). Jeans come in
  a men's cut only, slimmed for the women's default; the one women's cut is a
  blouse and skirt.
- **Seven haircuts, chosen by four measurements.** The photo's hair mask picks
  close crop, short, fringe, bob or long, or none when bald. Checked on real
  photos for short and close-cropped hair, and on rendered avatars for long hair,
  bob and fringe; not yet on a real long-haired photo.
- **The face is textured onto the skull, not wrapped.** Its edge fades into
  generated skin built to the face's own colour. The photo's broad lighting,
  shine, over-exposure and crevice shadows are taken out first (`delight.py`),
  and a cast that puts skin outside the natural range is pulled back; a milder
  cast stays, and a dark photo is not brightened.
- **Two body shapes.** Male and female are MakeHuman's own macro targets; nothing
  reads height, build or age from the photo.
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

Four of the community haircuts are CC-BY: they can be shipped and sold, but
their authors must be credited wherever an avatar wearing one is shown or
distributed. The catalogue marks each with a `credit`, and an avatar's info
file (`credits`) lists the authors of what it wears:

| item | author | source |
|---|---|---|
| `hair_elvs_grump_hair`, `hair_elvs_short_side_do`, `hair_elvs_maxwell_hair` | Elvaerwyn | MakeHuman community asset repository, Hair 02 pack |
| `hair_culturalibre_hair_14` | culturalibre | MakeHuman community asset repository, Hair 03 pack |

The default cut and the other community cuts are CC0 and need no credit.

Both were chosen over the better-known research face and body models, which are
free for research only and need a paid licence to ship. Keep that constraint when
swapping the engine: see `../avatarsdk-re/CLONE-PLAN.md`.

Sample photos under `samples/` are Google's public MediaPipe test assets, used
here as fixtures only. Photos the team uploads stay on this PC under `uploads/`;
delete them when done, since they are faces.
