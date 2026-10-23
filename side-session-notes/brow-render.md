# Rendered brows high and thick: where it comes from

Five test faces (ref-man, ref-woman, photo-10, 20261019-175001-images = "p14",
photo-16), rendered as the studio's `renderPng(url, false, true)` does (headless
Chrome, same scene-setup.js, same camera), MediaPipe on photo and render.
Units: % of the iris spacing (468-473) unless said. "Whole-face" is the mean
landmark error after a similarity fit on all 468 points, averaged over four
renders of each build (plain, anisotropy 16, no mipmaps, 2x size): one render
alone moves by 0.02-0.09 per face from settings that change nothing visible
(anisotropy alone moved photo-10 by 0.19).

Tools (scratch, not in the repo): the shared notes folder's `brow-render-tools/`.

## Ruled out, one factor at a time

| factor | brow height (inner, + = high) | MediaPipe thickness | verdict |
|---|---|---|---|
| studio render (base) | +0.9 / +2.4 / +2.1 / +3.6 / +3.3 | 29.6 / 33.2 / 36.1 / 37.0 / 37.7 (photo 28.7 / 28.7 / 32.5 / 29.8 / 29.8) | |
| GTAO off | within 0.4 | within 1.1 | not AO |
| anisotropy 16, no mipmaps, 2x render size | within 0.5 | within 1 | not texture filtering or resolution |
| photo's expression off | 0.3-2.0 higher still | same | not the ARKit expression |
| level instead of photo pose | same, or up to 0.7 higher | same | not the pose |
| bind pose (no rig pose at all) | brow band as wide as posed (18.8 vs 18.8, p16) | | not the rig |
| unlit (texture only, no lights) | +0.3 / +2.2 / +1.5 / +2.7 / +2.8 | 3-4 thinner | the lighting adds a little thickness, not the height |
| flat colour, studio light (geometry only) | -3.6 / -0.0 / +1.2 / -4.0 / -5.1 | 17-22 from shading alone | the brow ridge's own shadow reads low, not high |
| UV overlap at the brows | 8 texels | | not shared texels |
| bake resampling (texture mapped back into the photo) | dip centre same as the bake input | width +0 to +1.6 | the bake doesn't spread it |

So it's in the paint, not the renderer. The rendered brows read high and thick
with the lights off.

## What it is: two things, both in the paint

Brow profiles from pixels (not MediaPipe's brow points): down each brow's three
middle columns, the dark dip's full width at half depth and its centre above the
photo's upper lid, at each stage, all in the photo's frame.

| p16 | width | centre above lid |
|---|---|---|
| photo | 11.4 | 17.2 |
| after delight, fill, lashes | 8.7 | 18.6 |
| after `tighten_brows` (bake input) | 12.1 | 17.6 |
| baked texture, mapped back | 13.7 | 17.0 |
| render | 19.7 | 19.3 |

**1. `appearance.tighten_brows` darkened the lid.** Its region (the brow's hull
dilated by 12% of the brow's width) reaches down onto the lid. Every pixel there
darker than the region's 80th percentile, which includes the lid crease and the
socket's shading, was pushed 1.6x darker, up to a hard straight edge across the lid.
It also left the brows 14-34 L* darker than the photo's on all five faces (L*
153/120/87/68/69 against 167/142/101/101/101). The skin has been written for the
studio's tone curve since this was tuned, and the brows no longer render light.

**2. The paint is stretched 1.2-1.35x vertically through the brow and lid**
(compose.py, see the proposal). A line grid baked in place of the photo: rows 8
px apart in the photo land 8 px apart on the forehead, then 9-11 px apart through
the brow down to the lid (p16 at its pose; ref-woman about 1.15). That widens
the brow band and pushes its top up, and MediaPipe's brow contour goes up with it.

## Fix kept: `tighten_brows` stops at the brow, gain 1.6 -> 1.25

`BROW_BELOW = 0.06`: nothing below the brow's lower contour (plus 6% of its
width) is darkened. The skin level is still read over the whole region.
`BROW_GAIN` 1.6 -> 1.25.

| face | whole-face | brows | inner brow | brow L* (photo) | band width (photo) |
|---|---|---|---|---|---|
| ref-man | 1.82 -> 1.83 | 1.54 -> 1.43 | +1.1 -> +0.9 | 153 -> 166 (167) | 12.7 -> 11.1 (11.2) |
| ref-woman | 1.81 -> 1.83 | 3.39 -> 3.47 | +2.6 -> +2.5 | 120 -> 136 (142) | 11.5 -> 11.0 (8.7) |
| photo-10 | 2.24 -> 2.20 | 2.50 -> 2.31 | +1.9 -> +2.0 | 87 -> 98 (101) | 17.1 -> 15.9 (12.7) |
| p14 | 1.34 -> 1.33 | 3.89 -> 3.84 | +3.6 -> +3.2 | 68 -> 80 (101) | 20.0 -> 15.7 (11.4) |
| photo-16 | 1.33 -> 1.35 | 3.66 -> 3.69 | +3.4 -> +3.1 | 69 -> 82 (101) | 19.7 -> 15.8 (11.4) |
| mean | 1.71 -> 1.71 | 2.99 -> 2.95 | +2.5 -> +2.3 | | |

Whole-face is unchanged (per face -0.04 to +0.02, inside the four renders' spread).

Variants tried before this one, same measure:

| variant | whole-face | brows | brow L* | |
|---|---|---|---|---|
| no `tighten_brows` at all | 1.73 | 2.99 | within -16..+5 of the photo | woman +0.07 |
| gain 1.25 only | 1.72 | 3.03 | | |
| stop at the brow only | 1.71 | 3.08 | | |
| both (kept) | 1.71 | 2.95 | | |

Not moved: the eyebrow_audit-style thickness (darker than skin - 25 L* in a
column; 37.7 -> 38.6 on p16). On renders that measure is mostly the brow
ridge's shadow, which flat grey geometry alone reads at 17-22. The pixels'
centre of the band rises about 1 (p16 19.3 -> 20.7, photo 17.2): the false
lower half is gone. MediaPipe's inner brow moved down (3.4 -> 3.1).

## Where the stretch comes from (compose.py; proposal below)

The sheet the photo is baked through is MediaPipe's own image landmarks, turned
upright by MediaPipe's full head rotation (`generate.upright`). The render turns
the head back by that rotation **less `LEVEL_READ_PITCH` (3.3 degrees)**. So the
paint is laid on a head 3.3 degrees off the one the render draws. Points forward
of the eyes (brow ridge) and back of them (the lid in its socket) move apart
vertically.

- At the landmarks, each sheet stage turned by the render's pose against the
  photo: brow-to-lid gap upright +0.6 on all five, placed +0.1 to +0.7, after
  the mouth carry +0.3 to +0.8, after the eye carry +0.9 to +1.5. Turned by
  MediaPipe's full pose, the upright sheet is exact (0.0).
- Between the landmarks it's larger. A prediction from the probe data alone
  (each mesh point drawn at its posed position, carrying the photo pixel the
  sheet gives it) has paint 3-7% high at the brow's top, falling to 0-2% at the
  lid. It's the same with the mesh's depth replaced by the sheet's, so it isn't
  depth mismatch.
- `LEVEL_READ_PITCH = 0` (the render at MediaPipe's full pose): brows 2.95 ->
  2.66, inner +2.3 -> +1.4, but whole-face 1.71 -> 2.11. The pose was tuned for
  the rest of the face; not this way.
- **The sheet turned by +3.3 degrees about x (the eyes' centre) before the
  carries**, the render untouched: brows 2.95 -> 2.41, better on all five
  (1.43 -> 0.87, 3.47 -> 3.27, 2.31 -> 1.88, 3.84 -> 3.06, 3.69 -> 2.96), inner
  brow +2.3 -> +1.2. Whole-face 1.71 -> 1.75 (ref-man +0.13, others +0.05 or
  less). With -3.3 degrees the brows get worse (3.58), so the sign holds.
- **The same turn above the eyes only** (full at the eye line, fading to none
  at the nose tip, so the nose and mouth paint stay put):

  | face | whole-face | brows | inner brow | brow above lid (photo) |
  |---|---|---|---|---|
  | ref-man | 1.83 -> 1.91 | 1.43 -> 0.82 | +0.9 -> -0.5 | 22.4 -> 21.7 (20.1) |
  | ref-woman | 1.83 -> 1.86 | 3.47 -> 3.15 | +2.5 -> +1.3 | 27.8 -> 27.3 (24.0) |
  | photo-10 | 2.20 -> 2.19 | 2.31 -> 1.77 | +2.0 -> +0.8 | 25.9 -> 25.0 (22.2) |
  | p14 | 1.33 -> 1.33 | 3.84 -> 3.10 | +3.2 -> +2.0 | 28.3 -> 27.5 (21.8) |
  | photo-16 | 1.35 -> 1.31 | 3.69 -> 2.93 | +3.1 -> +1.9 | 28.2 -> 27.5 (21.8) |
  | mean | 1.71 -> 1.72 | 2.95 -> 2.35 | +2.3 -> +1.1 | |

  ref-man's brow overshoots to 0.5 low, and his whole-face goes up 0.08.

Also tried, not proposed: the brows taking the upper lid's pull in the eye
carry, instead of the whole eye's. Inner brow +2.3 -> +2.9 (worse).

## Proposal for compose.py (not edited here)

Lay the paint above the eyes on the head the render draws: turn the placed
sheet by `LEVEL_READ_PITCH` about x, round the eyes' centre, in full from the
eye line up and fading to nothing at the nose tip, before the mouth and eye
carries. Measured above as "above the eyes only": brows 2.95 -> 2.35 on all
five, whole-face 1.71 -> 1.72.

```python
def level_upper_face(sheet: np.ndarray) -> np.ndarray:
    """The placed sheet above the eyes turned by LEVEL_READ_PITCH.

    The sheet is the photo's landmarks turned upright by MediaPipe's whole head
    rotation, but the render turns the head back by that rotation less
    LEVEL_READ_PITCH. Paint laid on the one head and drawn on the other moves
    the brow ridge (forward of the eyes) and the lid (behind them) apart: the
    brows rendered 1-3.6% of the eye spacing high and a fifth thicker. Turned
    in full from the eye line up, and not at all from the nose tip down, so the
    nose and mouth stay where they were placed."""
    from scipy.spatial.transform import Rotation
    turn = Rotation.from_euler("x", LEVEL_READ_PITCH, degrees=True).as_matrix()
    eyes = sheet[[33, 133, 263, 362]].mean(axis=0)
    turned = (sheet - eyes) @ turn.T + eyes
    share = np.clip((sheet[:, 1] - sheet[1, 1]) / (eyes[1] - sheet[1, 1]), 0.0, 1.0)[:, None]
    return sheet + share * (turned - sheet)
```

and in `compose()`:

```python
    full_face = place_face(face, vertices, groups)
    full_face.vertices = level_upper_face(full_face.vertices)
    full_face.vertices = mouth_onto_mesh(full_face.vertices, vertices, corr)
```

Worth knowing before adopting:
- ref-man's brows overshoot (inner -0.5) and his whole-face goes up 0.08. His
  photo is near level (pitch 1.0), so the 3.3 constant is too much for him.
  `LEVEL_READ_PITCH` is one number for every avatar (its own ponytail note), and
  the per-avatar level reading it says to measure would serve both uses.
- Change `LEVEL_READ_PITCH` and this follows it. `test_rig.check_eye_carry`
  should still pass (the turn happens before the carry), but run it.
- The turn moves the sheet's depth too, which `FaceSampler` checks against
  (tolerance 0.6). Nothing failed on the five faces.

The rest of the height (inner +1.1 after this; MediaPipe's brow top 27.5 above
the lid against the photo's 21.8 on p16) stays, partly the eye carry (+0.3 to
+0.8 of gap at the landmarks) and partly the brow ridge's shadow, which flat grey
geometry already reads as a 17-22 band (scene-setup.js lighting, the light
direction session's area).

## Notes for whoever measures brows next

- Use pixels as well as MediaPipe. MediaPipe's thickness counted the ridge's
  shadow, and it didn't move when the paint got 20% thinner.
- The studio is a page, so the renders here come from headless Chrome over the
  DevTools protocol (`brow-render-tools/shoot.mjs`). The in-app browser stalled
  while its pane was hidden.
- Port 8823 on this machine is another local service. Python's http.server on
  Windows binds a taken port without complaint (`allow_reuse_address`), so the
  first renders hit that service. `serve.py` now refuses a taken port.
- Twice, every python.exe of this session was killed mid-build at once, with no
  traceback. Possibly another session's cleanup; rerunning was enough.
