# The studio's light direction against the photos'

Branch `claude/light-direction-c5e391`. Changes in lib/scene-setup.js and delight.py (constants only).

## Verdict on "the studio is too top-down"

Refuted as stated. The studio is not lit from too high. It is lit **too directionally**:
light falls off too fast away from the front in every direction, worst towards the
sides, where the cheeks (landmarks 50/280 face partly sideways) and the jaw are.

- The photos' own light, from delight's first-order SH fit (skin only, beard excluded,
  camera frame): elevation 44 (ref-man), 75 (ref-woman), 21 (photo-10, a weak fit:
  its ambient term came out ~0), 47 (photo-14 and photo-16) degrees above level.
- The studio's light on the rendered heads, fitted the same way to per-pixel E (lit
  against unlit render, through the tone curve): 15-20 degrees up, az +7.
  So the photos are lit *from higher* than the studio.
- What matters is what the light does to the baked albedo (delight takes 70% of the
  photo's shading out). Per landmark, the E that would render the photo's brightness
  from the baked albedo, fitted with SH1, pooled over the five faces: direction/ambient
  0.19 (nearly flat), up45/front 1.00, down45/front 0.91, left45 0.93. The studio gave
  2.95, 0.92, 0.64, 0.69. Every face showed the same: needed 0.14-0.44, given 1.0-5.4.
- By region, relative to the cheeks (level): forehead needed 1.04-1.82 (mean 1.36),
  got 1.38-1.67 (mean ~1.53); jaw needed 0.46-1.0, got 0.48-0.98, too dark on
  ref-woman, photo-14 and photo-16. The room environment gives about 70% of the face's
  light, and its own pattern is forehead 1.60x cheeks, chin 1.33x, jaw 0.55x. That
  front-heavy pattern is what read as "top-down" from the forehead (+16) and jaw (-37)
  of photo-16.

## Method

Five test heads built from this branch (compose.py, ref-woman female). Renders: a copy of
studio.html's `drawAvatar` (same lights, tone mapping, occlusion, 900x900 head framing),
level and at the photo's head pose (`applyPhotoPose`), lit and unlit (MeshBasicMaterial, no
tone mapping: the albedo as the camera sees it). L* = OpenCV Lab L (0-255), median in disks
of 0.07 eye spacings at forehead 151, cheeks 50+280, chin 199; jaw = 8 disks of 0.15 eye
spacings 20% in from the jaw outline (172, 397, 136, 365, 150, 379, 149, 378). Landmarks by
MediaPipe on the photo and on each render. "split" = (cheek 50 - cheek 280) on the render
minus the same on the photo, to watch the earlier cheek-split fix.

Lights add before the tone curve, so E = sum of each light's E. Each light was rendered
alone on all ten views (key, fill, hemisphere, rim, environment, and candidates: the
environment tipped 0/15/37/45 degrees, the key at 15/45 degrees up and at 45 degrees round,
side pairs at 60/80/100 degrees, from 30 degrees up, from below, from the front, an ambient).
A mix then predicts render minus photo for any weights; the base mix predicted the measured
base within ~1 L*. Every mix was scaled so the cheeks keep the base's E, so delight's
exposure model stays right (where it would not, delight's exposure was re-solved per face
for the new cheek E). Each chosen change was then rendered for real and scored.

Tools (scripts, not in the repo): the shared notes folder `light-direction-tools/`.

## Changes, one at a time (real renders, sum of |render - photo| over the five faces)

| step | level fh+ch+chin | pose fh+ch+chin | jaw level+pose | split level/pose | kept |
|---|---|---|---|---|---|
| base | 139.0 | 199.5 | 321 | 65 / 65 | |
| + side lights at +-100 deg, level, 2.0 (others x0.94) | 97.5 | 169.0 | 185 | 60 / 73 | yes |
| (same at +-80 deg, 10 up) | 113.0 | 177.5 | 183 | | no, worse |
| + environment tipped 37 (was 30) | 97.0 | 133.5 | 163 | 59 / 66 | yes |
| + environment x0.8 against the lamps | 92.0 | 132.5 | 159 | 59 / 62 | no (below) |
| **committed** (exposure 1.0 instead of scaling the lamps) | **98.0** | **137.5** | **161** | 62 / 66 | |

Model-only candidates that lost or barely won on top of the side lights: fill x2-4, hemisphere
x2-4, key 15 or 45 degrees up or 45 round, a light from below, an ambient (better jaw, no
better forehead/cheeks/chin), stronger side lights (jaw better, level worse), environment 45
(pose better, level and chin worse), environment 15 or 0 (much worse).

Environment x0.8 won by ~6 L* in all, inside the measurement's step (medians move in whole L*
steps). In three.js r162 the environment can only be weakened by scaling exposure and every
other light, so it was not worth the change.

### Committed, per face (render - photo L*)

| view | forehead | cheeks | chin | jaw |
|---|---|---|---|---|
| ref-man | -3 -> -4 | -10 -> -4.5 | +28 -> +30 | -17 -> +15 |
| ref-woman | -3 -> -5 | -4 -> +1 | +3 -> +11 | -45 -> -14 |
| photo-10 | -6 -> -9 | -4 -> +3 | -1 -> -5 | -7.5 -> +17.5 |
| photo-14 | +15 -> +7 | -20 -> -1.5 | -4 -> +2 | -40 -> -14 |
| photo-16 | +17 -> +10 | -18 -> 0 | -3 -> +5 | -39 -> -13 |
| ref-man, pose | -2 -> -4 | -12 -> -5 | +28 -> +30 | -18 -> +14 |
| ref-woman, pose | -3 -> -5 | -4 -> +1 | +6 -> +8 | -48 -> -17 |
| photo-10, pose | -5 -> -7 | -7 -> +2 | -10 -> -14 | -5.5 -> +13.5 |
| photo-14, pose | +19 -> +15 | -23.5 -> -5.5 | -20 -> -11 | -51 -> -22 |
| photo-16, pose | +20 -> +17 | -21 -> -3 | -19 -> -10 | -50 -> -21 |

Sums: forehead 44 -> 35 level, 49 -> 48 pose; cheeks 56 -> 10, 67.5 -> 16.5; chin 39 -> 53,
83 -> 73; jaw 148.5 -> 73.5, 172.5 -> 87.5.

Costs, plainly: level chin sums 39 -> 53 (ref-woman +3 -> +11, photo-16 -3 -> +5); the
foreheads of ref-woman and photo-10 go 2-3 darker; photo-10's jaw (a full beard) goes from
-7.5 to +17.5. ref-man's chin (+28 to +30) is not lighting: it moved by less than 6 under every light
tried. Its baked chin is pale where the photo has beard (texture, not this task).

## delight.rendered kept consistent

The side lights were paid for with exposure 1.05 -> 1.0, so the cheeks get the same light
times exposure as before. E at delight's cheek landmarks (SKIN_LANDMARKS, disks of 0.03 eye
spacings, lit against unlit): base 1.96 at exposure 1.05; committed 2.00 at 1.05, which is
2.10 at 1.0 (per face 1.96-2.21 level). delight: `RENDER_LIGHT` 2.0 -> 2.1,
`RENDER_EXPOSURE` 1.05 -> 1.0. The product is unchanged (2.1), so builds write the same skin;
checked by rebuilding all five heads.

## Look

Before/after sheets (photo, base level, committed level, base pose, committed pose; and two
full bodies) are in the worktree's `out/renders/before-after.png` and
`out/renders/full-before-after.png` (git-ignored). The heads read as a soft-lit studio
portrait: sides of the face lighter, nose and lips still modelled, hair and eyes
unchanged to the eye. Full body: clothes lose a little side shading (folds flatter).

Untouched: the environment's 14-degree turn (the cheek split, which stayed at 59-66 against
65), eye roughness, key, fill, hemisphere, rim.

## Left

- photo-14/16's forehead at the photo pose is still +15/+17: the head pitched 12 degrees down
  faces its forehead into the environment's upper light. Nothing tried fixed it without
  costing the level render.
- Pose chins on photo-10/14/16 read -10 to -14.
- The eye texture and hairline root still divide by `RENDER_LIFT` 1.35 (not changed here).
