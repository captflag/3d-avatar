# Guided capture: front, left, right, each checked before it is taken

**Guided capture** in the studio (was **Use camera**) takes three photos in order:
straight on, the head turned to the person's left, then to their right, and sends
them to `/generate` as the front photo and two `side` photos, which compose fuses
(`views.fuse`, full body only). `app.py` needed no change: it already took `side`.

Each step is checked live with MediaPipe's in-browser face landmarker (the
pipeline's own model, `/models/face_landmarker.task`):

| check | limit | steps |
|---|---|---|
| face size, cheek to cheek (as `compose.photo_advice` measures) | 500 px (400 turned) | all |
| head turn, from the face transform | front within 10°; sides 25-40° as MediaPipe reads it, the asked way | all |
| chin level / head upright | pitch 10°, roll 8° | all |
| neutral, mouth closed | smile and jawOpen blendshapes ≤ 0.3 | all |
| eyes open | blink ≤ 0.4 | all |
| sharp | re-blur measure ≥ 0.4 (below) | all |
| even light | cheek brightness split ≤ 25% | front |
| hair in the picture | | front |

With a camera the shot takes itself after half a second of all ticks; **Take
photo anyway** overrides. Without one (no device, or not localhost/https, as on a
phone over the LAN) every step takes a photo file instead, checked the same way:
a passing photo is kept, a failing one is shown with what it misses and **Use
this photo anyway** keeps it. Overridden shots are marked in the strip and the
status says which.

## Calibration

**Turn.** MediaPipe under-reads turns. Rendering a studio avatar's head at known
camera angles and reading it back (front read +3.9°, the avatar's own pose):

| true turn | 20 | 30 | 35 | 45 | 55 |
|---|---|---|---|---|---|
| read, to its left | 19.9 | 27.7 | 32.2 | 39.5 | 52.1 |
| read, to its right | -14.5 | -22.3 | -26.6 | -32.7 | -40.7 |

Mean |read - front| ≈ 18, 26, 30, 39, 49: the views.py figure (35 reads 29).
So 30-45° true is about 25-40 read. Turning to one's own left reads positive yaw
(`LEFT = 1`). Results: front accepted only at 0; left accepted at 30, 35, 45;
right at 35, 45 (30 read 22, short of 25 by the avatar's 4° offset); 20 and 55
rejected both ways; every wrong-way turn rejected as "Wrong way".

**Sharpness.** Plain Laplacian variance failed: it measures texture as much as
focus (the smooth AI reference faces read 20 unblurred, the real portrait 1000).
The check is Crete-Roffet's re-blur measure instead: the share of the face's
edge contrast that a 9-pixel box blur removes, on the face scaled to 320 px wide,
the blurrier axis. 0 is flat, sharp faces read 0.5-0.66.

| photo | sharp |
|---|---|
| portrait, its 2x, 2x + blur σ2, 2x + blur σ4 | 0.63, 0.66, 0.35, 0.20 |
| avatar render, + CSS blur 1 px, 2 px, 3 px | 0.52, 0.45, 0.33, 0.25 |
| photo-16 (214 px face) | 0.53 |
| ref-man, ref-woman (soft, upscaled) | 0.23, 0.23 (0.21-0.26 at 2x, blurred or not) |
| photo-10 (52 px face) | 0.18 |

Threshold 0.4: passes 1 px of blur, fails 2 px and the soft references.

**Real test photos** (no real photo passes all checks; none has a 500 px face):

| photo | fails |
|---|---|
| ref-man, ref-woman | size (303, 314), smile (0.72, 0.40), blurred |
| ref-man / ref-woman 2x (607, 627 px) | smile, blurred |
| portrait | size 189, mouth open/smile 0.96, hair cut off |
| photo-16 | size 214, chin (pitch 12.6) |
| photo-10 | size 52, chin, smile |
| 1179x1174 jfif | size 145, turned (yaw 49.5), roll, blurred, light split 0.40 |

## Tested in the in-app browser

No camera there (`NotFoundError`), so the panel opened on the per-step photo
path. Fed through the step's file input: the smiling reference (held, two checks
named, override offered), the avatar render straight on (kept), turned the wrong
way (held: "Wrong way"), turned left 35 (kept), right 20 (held: "Turn further:
14°, needs 25"), right 45 (kept). Status: "Front photo and 2 turned taken, every
check passed." Generate sent front + side1 + side2 to `/generate`, and the
full-body avatar's sidecar says `"side_views": 2`: compose fused both.

The live camera loop (the same `assess` per frame, a half-second hold on every
step; the old keep-the-widest-turned-frame logic is gone, the window replaces
it) was not run here: no camera. Worth one try on a real webcam.

Screenshots (not committed: they show test photos and a likeness):
`side-session-notes/local_caf6a96b-.../guided-capture/` in the shared notes folder.

## Limits

- A 500 px face with the hair in frame needs a frame about 1000 px tall: a 720p
  webcam cannot do both. The page says so and the override takes it anyway.
- Turn limits are MediaPipe readings, calibrated on one rendered head; a person
  whose straight-on reading is offset a few degrees gets a slightly shifted window.
- The old flow kept the widest turned frame up to a 50 read, because on a render
  two shots at 50° beat two at 35° for depth. This one caps at 40 read (~46°
  true), per the 30-45° brief. If real turned shots hold their landmarks at
  50°, raise `SIDE_TURN[1]` to about 42.
- Gaze is not checked on turned shots.
