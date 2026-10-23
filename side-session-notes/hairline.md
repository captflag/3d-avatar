# Hairline and forehead from the photo

The avatar's hair now ends where the photo's does across the whole forehead, not
only at its middle: receding temples recede, a straight line stays straight.

## What changed

1. **The photo's hairline as a curve** (`appearance.read_hair` -> `hairline_points`).
   At 11 columns across the forehead (`HAIRLINE_ACROSS`, -0.85..0.85 of half the
   face's width either side of landmark 9), the lowest segmented hair above the
   brows, in photo pixels (median over a column band 0.1 eye spacings wide).
   Columns with no hair are left out.
2. **Onto the head through the landmark sheet** (`compose.hairline_on_head`): an
   affine from the photo's landmark pixels to the placed sheet's x, y, fitted over
   the upper third of the face, as `FaceSampler` does past the sheet. Its centre
   agrees with the old single reading (brows + `hairline` eye spacings) to within
   0.02 units on all five faces.
3. **The cut follows the curve** (`garments.sweep_back` given `(xs, ys)`). Each
   column of the front is turned back by what brings its own lowest opaque hair
   (above the brow line, `above=`, so side hair hanging by the temples does not
   count as a low hairline) up to the curve. The turn fades with height to none at
   the crown, so only the front moves; it fades to none two steps past the
   curve's ends, and round the head past `SWEEP_SPAN`. Hair already above the
   curve is left alone. Nothing is cut away: lifted, the cards at the temples go up
   and back as combed hair does.
4. **Tint and cap follow the same curve** (`appearance.hairline_tint(hairline=)`).
   On the front of the head the tint's edge is never below the photo's hairline, so
   cards hanging lower do not paint root shade on forehead the photo shows bare.
   The scalp cap is cut from the tint (`hair_below`), so it follows.
5. **Forehead skin**: no change needed. `forehead_skin` already bakes the photo's
   skin up to the segmenter's hairline between the temples; an unlit render shows
   the uncovered temples carrying the photo's skin.

## How it was measured

Five faces built with compose.py (ref-woman female), rendered as the studio's
`renderPng(url, false, true)` does (headless Chrome, the studio's scene-setup.js,
head framing, the photo's pose). The render is aligned to the photo by a
similarity fit on the 468 landmarks. Hair = segmenter HAIR or class 5 (it calls
a card cut an accessory).

- **sil**: hair IoU above the chin. **fhIoU**: the same inside the forehead/temple
  box (brows up 2 eye spacings, temple to temple).
- **ratio**: forehead height, brows to the hairline at mid-forehead over brows to
  chin. `|d|` = against the photo's.
- **curveErr**: mean |render - photo| hairline height (eye spacings) over 35
  columns across the forehead.
- **whole**: mean landmark error after the fit, % of the iris spacing.

Tools (not in the repo): the shared notes folder's `hairline-tools/` (`hl.py` is
the measure, `build.sh`, `shoot.sh`, `lab.html` the studio's drawAvatar). Side by
sides: `hairline-tools/hairline-sides.png` (photo | before | after, photo's hair
outline green, render's red).

## Result (base -> kept)

| face | sil | fhIoU | ratio (photo) | curveErr | whole |
|---|---|---|---|---|---|
| ref-man | 0.735 -> 0.762 | 0.792 -> 0.827 | 0.431 -> 0.477 (0.396) | 0.123 -> 0.078 | 1.86 -> 1.76 |
| ref-woman | 0.757 -> 0.796 | 0.743 -> 0.805 | 0.431 -> 0.445 (0.452) | 0.087 -> 0.045 | 1.74 -> 1.68 |
| photo-10 | 0.751 -> 0.772 | 0.786 -> 0.819 | 0.391 -> 0.412 (0.391) | 0.116 -> 0.078 | 1.84 -> 1.80 |
| p14 | 0.709 -> 0.772 | 0.714 -> 0.794 | 0.379 -> 0.425 (0.425) | 0.138 -> 0.070 | 1.29 -> 1.20 |
| photo-16 | 0.707 -> 0.768 | 0.711 -> 0.789 | 0.375 -> 0.434 (0.425) | 0.145 -> 0.072 | 1.31 -> 1.21 |
| mean | 0.732 -> 0.774 | 0.749 -> 0.807 | \|d\| 0.030 -> 0.024 | 0.122 -> 0.069 | 1.61 -> 1.53 |

Hairline height at u = -0.8 / -0.4 / 0 / 0.4 / 0.8 (eye spacings above the brows):
photo-16 photo 0.67 0.61 0.66 0.69 0.21; before 0.34 0.64 0.59 0.53 0.25; after
0.52 0.69 0.68 0.73 0.35. The rounded arch the stock cut drew across the temples is
now close to the photo's squared, receding corners.

p14 and photo-16 are near-identical shots of the same man (different bytes, same
reading); count them as one face and the wins still hold on all four.

## Steps, one at a time (means over the five)

| step | sil | fhIoU | \|d ratio\| | curveErr | whole | kept |
|---|---|---|---|---|---|---|
| base | 0.732 | 0.749 | 0.030 | 0.122 | 1.61 | |
| c1 curve, each column turned whole | 0.639 | 0.672 | 0.086 | 0.107 | 1.59 | no: 40-68 degree turns at the temples stood the cut up in a block |
| c2 + only the edge above the brows sets the turn | 0.643 | 0.673 | 0.088 | 0.111 | 1.59 | no, same |
| c3 + turn fading to none at the crown | 0.761 | 0.788 | 0.025 | 0.088 | 1.53 | yes |
| c4 c3 with volumise before the sweep | 0.729 | 0.753 | 0.064 | 0.133 | 1.68 | no |
| c5 c3, fade over two steps past the ends | 0.767 | 0.796 | 0.024 | 0.083 | 1.52 | yes |
| c6 c5, volumise ramped from the opaque edge | 0.731 | 0.761 | 0.031 | 0.076 | 1.57 | no |
| c7 c5 + tint floor at the photo's hairline | 0.774 | 0.807 | 0.024 | 0.069 | 1.53 | yes |

Whole-face moves 0.02-0.09 between renders of the same build (brow-render notes),
so its changes here are inside that; the hair measures moved on every face.

## Left as it is

- **ref-man's middle ends 0.12 eye spacings high** (ratio 0.477 against 0.396).
  The sweep puts the opaque edge on the curve (checked on c3: within 0.04 units
  for |x| < 0.55, 0.13-0.18 short at the curve's two ends), then `volumise` lifts it: its ramp starts at the cut's lowest
  *vertex*, the transparent card tips, so the visible edge already gets a share of
  the lift, and ref-man's tall hair asks for the largest lift. Ramping from the
  opaque edge (c6) fixed him but put the other four 0.04-0.08 low; the render's
  visible edge is not the opaque points' edge. A per-face fix wants volumise to
  ramp from each column's own visible edge.
- The temple corners still fall 0.1-0.15 short at u = +-0.8 on four faces: the
  sides of the stock cut come forward over the temples below the curve's ends, and
  past the ends the turn fades out.
- The photo's hair stands higher and wider than the cut (volume, widen): not this
  change.

## Tests

`test_rig.check_hairline_sweep`: a receding curve lifts the temples (0.37) further
than the middle (0.17), the middle lands on the curve, the crown and back stay put.
`check_likeness_details`: with the photo's hairline above the cards' edge, the skin
below it is not tinted. test_rig.py passes in full.
