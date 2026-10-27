# Haircut picked by likeness; the skin's double shading

Branch `claude/hair-pick-lighting`, on side-views 9744847. Two changes kept:

- `appearance.hair_style`: short hair without a fringe always gets `hair_short01` (the close crop).
- `lib/scene-setup.js`: only half the skin's diffuse takes the studio's lights (`SHADING.skin`).

Stronger de-lighting was tried and left out. The grown cut (`hair_from_photo`, refs/heads/claude/hairgrow) was measured but not merged.

| five test faces, identity | photo's pose | level |
|---|---|---|
| side-views as it was (two builds) | 0.443 | 0.412 |
| + close crop | 0.471 | 0.454 |
| + skin half lit (shipped) | **0.495** | **0.474** |

On photo-16's two other photos (sk-21, sk-22 built as faces of their own, scored against
themselves): 0.328 / 0.301 -> 0.379 / 0.329 -> 0.414 / 0.353.

## Method

- **Score.** `identity_score.py`: SFace cosine, photo against render.
- **Faces.** The five test faces: ref-man, ref-woman (female), photo-10, photo-14 (the jfif) and
  photo-16. sk-21 and sk-22 (photo-16's man) are held out: they were not used to choose anything.
- **Renders.** A copy of studio.html's `drawAvatar` (scene-setup's lights, tone curve, occlusion,
  900x900 head framing), in headless Chrome. "Pose" is at the photo's head pose, "level" is square
  to the camera. Every build is rendered both ways.
- **Haircuts.** Every build forces its cut through a `--hair` override (the hairgrow branch's flag,
  merged only for the experiment).
- **Noise.**
  - The same build rendered twice: up to 0.008 per face.
  - Two builds of the same cut: up to 0.017 per face, 0.003-0.006 on the five-face mean.
  - Every kept change was built twice.
- **Tools** (not in the repo): the shared notes folder, `hair-pick-and-lighting/tools/`. Sheets are
  in `hair-pick-and-lighting/` there. They hold the test photos, so they are not committed.

## 1. Haircut by score

Every hair item in the catalogue was built on each face, plus the grown cut and bald. Rules = what
`hair_style` picked: short02 on four faces, short_sides on photo-10.

| cut | man | woman | p10 | p14 | p16 | **pose mean** | **level mean** |
|---|---|---|---|---|---|---|---|
| rules | 0.617 | 0.359 | 0.373 | 0.446 | 0.419 | 0.443 | 0.411 |
| short (skull cap) | 0.545 | 0.308 | 0.321 | 0.374 | 0.380 | 0.385 | 0.388 |
| **short01** | 0.622 | 0.409 | 0.455 | 0.429 | 0.423 | **0.468** | **0.455** |
| short02 | 0.615 | 0.373 | 0.432 | 0.444 | 0.427 | 0.458 | 0.429 |
| short03 (fringe) | 0.612 | 0.344 | 0.408 | 0.411 | 0.400 | 0.435 | 0.418 |
| short04 | 0.571 | 0.409 | 0.414 | 0.449 | 0.447 | 0.458 | 0.432 |
| short_sides | 0.649 | 0.384 | 0.381 | 0.447 | 0.419 | 0.456 | 0.419 |
| bob01 | 0.462 | 0.303 | 0.467 | 0.379 | 0.369 | 0.396 | 0.382 |
| bob02 | 0.434 | 0.284 | 0.428 | 0.376 | 0.366 | 0.378 | 0.360 |
| long01 | 0.502 | 0.402 | 0.519 | 0.467 | 0.461 | 0.470 | 0.448 |
| hair_from_photo | 0.629 | 0.319 | 0.399 | 0.455 | 0.470 | 0.454 | 0.441 |
| bald | 0.577 | 0.339 | 0.409 | 0.437 | 0.439 | 0.440 | 0.416 |

(Per-face columns are at the photo's pose. Sheet: `cuts-all.png`.)

### Picking by score is not safe

**long01 scores near the top** on all four men, all of whom have short hair, and it looks plainly
wrong. SFace's 112 px crop reaches the temples, and hair hanging over the sides of the face seems
to hide where the head's outline is wrong. A score-picked cut would choose shoulder-length hair for
short-haired men.

The best plausible short cut per face beats the rules by ~0.045 at the pose. That figure is
inflated, though, because it takes the maximum of a dozen noisy scores.

So scoring cuts is not a build-time choice. What it shows is that **one rule** wins.

### short01 wins on every view

|  | build 1 | build 2 | held out (sk-21, sk-22) |
|---|---|---|---|
| rules, pose / level | 0.445 / 0.411 | 0.442 / 0.414 | 0.328 / 0.301 |
| short01, pose / level | 0.474 / 0.452 | 0.468 / 0.456 | 0.379 / 0.329 |

Per face, mean of the two builds, pose / level:

| face | change |
|---|---|
| ref-man | +0.020 / +0.010 |
| ref-woman | +0.042 / +0.067 |
| photo-10 (rules gave short_sides) | +0.081 / +0.082 |
| photo-14 | -0.008 / +0.024 |
| photo-16 | +0.002 / +0.026 |

photo-14 at its pose is the only loss, and it is within build noise.

### What the rule reads

All seven photos read as short hair with no fringe: top 0.47-0.63, sides 0.08-0.21. The old rule
split them by `top` and `sides` (0.4 and 0.11). That split no longer earns its place: the cut is
raised and widened to the photo afterwards (`garments.volumise`, `widen`, `fit_hair`). So the stock
cut only decides the hairline and the strands, and there short01 is better:

- short02 leaves a see-through, skin-coloured band at the front of its hairline (ref-man,
  ref-woman, photo-14 in `rule-vs-short01.png`).
- short01 gives photo-10 the swept-up top its photo has.

The rule is now: short with no fringe -> short01. short03 (fringe), the bobs, long and bald are
unchanged. test_rig's two `sides` assertions are gone, and its fuller-top case now expects short01.

Untested, because no test photo has them: genuinely full, voluminous short hair, and an actual
short back and sides. short02 and short_sides stay in the catalogue for `--wear`, and for the photo
check's hint where the segmentation reads no hair.

### Not kept

- **hair_from_photo**: 0.454 / 0.441 on the five faces, below short01's 0.471 / 0.454.
  - It is the best cut on the two held-out photos: 0.396 / 0.363 against short01's 0.379 / 0.329.
  - Mixed results, so it stays opt-in on its own branch. It merges into side-views with one
    conflict in compose.py; resolve it by keeping side-views' `render_turn`, which hairgrow can
    reuse.
- **Explaining scores by hair overlap.** The idea was to compare photo and render hair masks inside
  SFace's crop, with the multiclass segmenter run on the renders. It failed because the segmenter
  barely sees rendered hair: short04 on ref-man read as 2% hair. The correlation with identity
  per face was only +0.34 to +0.56. Dropped.

## 2. Lighting against the shading the photo baked in

### Two routes

- **(a) De-light the photo more strongly** (build time). `delight.STRENGTH` 0.7 -> 0.85 or 1.0, and
  1.0 with the gain clamp widened from [0.75, 1.4] to [0.5, 2.0].
- **(b) Flatten the studio's light on the skin only** (render time). The `head` and `body`
  materials:
  - take the lights at `k` of their diffuse (`color` x k);
  - add their own texture unshaded, at 2.1 x (1 - k). That is emissive, with emissiveMap = map.
    2.1 is `delight.RENDER_LIGHT`, the light the cheeks get, so the cheeks keep their brightness
    and the baked skin stays right.
  - Hair, eyes, brows and clothes keep the full studio, so the scene still reads as lit.

### Measuring flatness

"Shading kept" is how lit the face still looks, the studio being 100%. It is computed over the
inner face box of the level render. For each pixel, take log(render at k / render at k = 0); the
spread between the 10th and 90th percentiles of that, as a share of the same spread at k = 1, is
the shading kept.

### (b) results

On the short01 builds (the new rule), mean of two builds, with the held-out photos:

| k | five, pose | five, level | sk-21/22 pose / level | shading kept | look |
|---|---|---|---|---|---|
| 1 (studio) | 0.471 | 0.454 | 0.379 / 0.329 | 100% | |
| 0.85 | 0.479 | 0.457 | 0.387 / 0.336 | ~80% | |
| 0.7 | 0.482 | 0.462 | 0.409 / 0.341 | ~62% | clearly lit |
| **0.5 (shipped)** | **0.495** | **0.474** | **0.414 / 0.353** | **~42%** | the edge: ref-woman's nose loses most of its modelling |
| 0.3 | 0.492 | 0.480 | 0.410 / 0.358 | ~24% | flat: nose and cheeks lose form |
| 0 | 0.487 | 0.471 | 0.412 / 0.356 | 0 | flat-shaded |

At k = 0.5, per face (mean of two builds, pose / level):

| face | change |
|---|---|
| ref-man | -0.011 / -0.012 (within noise; on the old cut it was -0.02) |
| ref-woman | +0.048 / +0.013 |
| photo-10 | -0.005 / +0.027 |
| photo-14 | +0.042 / +0.035 |
| photo-16 | +0.046 / +0.035 |

**Brightness is not the cause.** Holding the face darker (the unshaded part at 1.6 or 1.3 x (1-k)
instead of 2.1) scored lower, not higher, so what helps is the flattening itself.

The audit's "unlit beats lit" was partly a darker-skin effect: unlit renders the skin ~40 levels
darker, without the tone curve.

**The shipped code was checked against the lab knob.** `SHADING.skin` in scene-setup, rendered
through `applyMaterials`, scores 0.498 / 0.472 against the lab knob's 0.494 / 0.473. It differs per
pixel only as much as two renders of the same file do.

Sheets:

- `light-tradeoff-level.png`: all five faces at k = 0.85, 0.7, 0.5, 0.3 and 0.
- `light-closeup.png`: ref-woman and photo-16, larger, at studio, 0.7, 0.5, 0.3 and 0.

The trade-off, plainly:

- **At 0.5:** photo-16 still has nose, cheek and jaw form, while ref-woman's nose shadow is mostly
  gone. Her photo is lit almost flat, so it reads like the photo, but it is the flattest setting I
  would ship.
- **At 0.7:** about half the gain with no visible loss. Take that if 0.5 looks too flat in the
  studio.
- **Below 0.5:** no further gain.

### (a) results: not kept

Rules' cut became short01 mid-run, so all of these are on short01 builds, with the studio's light
and with k = 0.5:

| delight | five pose / level, studio | five pose / level, k = 0.5 | sk-21/22 pose, studio / k = 0.5 |
|---|---|---|---|
| 0.7 (as is) | 0.471 / 0.454 | 0.495 / 0.474 | 0.379 / 0.414 |
| 0.85 | 0.474 / 0.452 | 0.493 / 0.475 | 0.386 / 0.414 |
| 1.0 | 0.464 / 0.447 | 0.490 / 0.470 | 0.380 / 0.401 |
| 1.0, gain [0.5, 2.0] | 0.473 / 0.445 | 0.489 / 0.467 | 0.367 / 0.371 |

All within noise or worse. The first-order lighting fit is smoothed to the broad scale, so turning
it up removes broad shading the studio would add anyway. The shading the photo leaves in the skin
is finer: round the nose, the eye sockets and the cheeks' form. Only taking light off the skin at
render time reaches it.

## Left

- **Full body unchecked.** The renders are head-framed. The body material is flattened as well
  (neck, hands), which was not looked at in a full-body render.
- **sk-22's dark hair renders grey** under every cut. That is the hair colour reading, outside this
  task.
- **Per-face k?** ref-man (an evenly lit studio photo) was the one face that did not gain from
  flattening. A k taken from how much shading delight's fit finds in the photo might win, but five
  faces cannot show it.
