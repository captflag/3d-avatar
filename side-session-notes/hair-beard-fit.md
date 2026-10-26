# Hair and beard to the photo's shape, from its segmentation

Three changes, each measured on its own against the same base on the five test
faces at the photo's pose. Kept: the hair's outline fit. Not kept: the beard's
depth and the hair's shades. The identity score (identity_score.py, SFace) is
the main keep/drop measure; a change is kept only if it raises it beyond noise
without the hair-mask overlap getting worse.

## Kept: the cut's outline fitted to the photo's hair (`garments.fit_hair`)

After the cut is chosen, swept back to the hairline, volumised and widened, its
outline is fitted the way `morph.fit_outline` fits the jaw:

1. The head is turned **as the photo-pose render turns it**
   (`compose.photo_head_pose`, not MediaPipe's raw rotation: on the laughing
   photo-10 the two differ by 9 degrees of pitch, on the others by the 3.3 of
   LEVEL_READ_PITCH) and drawn: the head's silhouette, and the cut's opaque
   points (`appearance.opaque_hair`) over it.
2. The photo's segmented hair (largest connected part, holes closed) is brought
   into that picture by a similarity fit of the photo's landmarks onto the
   face placed on the head.
3. Along rays from the cranium's centre every 4 degrees, 100 either side of
   straight up, each outline's last pixel is found. The gap is smoothed round
   the head (12 degrees), and the hair between the skull's outline and its own
   is stretched in the picture's plane to end on the photo's: points on the
   skull stay, the outer edge moves the whole gap. Three rounds.
4. Never past the skull's outline plus 0.04 eye spacings in the picture, and
   no moved point ends nearer the scalp than it was (or than that floor): a
   card over the forehead pulled down in the picture's plane met the brow in
   depth (3 mm into the scalp on a test with almost no photo hair).
5. The drawn opaque points reach past the hair the render's segmentation finds
   by 0.035 eye spacings (`HAIR_DRAWN_MARGIN`; 0.01-0.06 on the five faces,
   flat all round the head, measured against the base renders), so the drawn
   outline is fitted to end that far past the photo's.

The scalp cap needs nothing of its own: it is cut from the hairline tint, which
is computed from the fitted cut. The fringe edge is left to `sweep_back` (the
photo's hairline curve), which already fits it.

The first attempt drew in MediaPipe's rotation and took the drawn outline as
the rendered one: man 0.763 -> 0.853 but woman 0.793 -> 0.676 and photo-10
0.769 -> 0.623. Both causes were found by mapping the base render's own
segmented hair into the fit's picture: it sat 0.09-0.15 eye spacings inside the
drawn outline on those two, and 0.01-0.06 once drawn at the render's pose.

## Measures

Renders: the studio's scene (side lights from side-views c6d3afe), head
framing, the photo's pose, headless Chrome. **identity**: SFace cosine,
photo vs render. **sil**: hair IoU above the chin, render aligned on the 468
landmarks. **fhIoU**: the same in the forehead/temple box. **whole**: mean
landmark error, % of the iris spacing. Build-to-build noise, measured by
building the base twice (base, base2): identity up to 0.016 on one face, 0.001
on the mean; sil 0.000 on the mean; whole 0.01.

| face | identity base / base2 -> fit | sil base -> fit | fhIoU | whole |
|---|---|---|---|---|
| ref-man | 0.602 / 0.603 -> 0.618 | 0.760 -> 0.867 | 0.831 -> 0.884 | 1.89 -> 1.93 |
| ref-woman | 0.345 / 0.343 -> 0.365 | 0.787 -> 0.825 | 0.794 -> 0.844 | 1.65 -> 1.66 |
| photo-10 | 0.394 / 0.378 -> 0.380 | 0.779 -> 0.814 | 0.820 -> 0.840 | 1.95 -> 1.71 |
| photo-14 | 0.405 / 0.421 -> 0.432 | 0.771 -> 0.876 | 0.801 -> 0.897 | 1.28 -> 1.18 |
| photo-16 | 0.417 / 0.417 -> 0.426 | 0.771 -> 0.870 | 0.798 -> 0.891 | 1.28 -> 1.23 |
| mean | 0.433 / 0.432 -> 0.445 | 0.774 -> 0.850 | 0.809 -> 0.871 | 1.61 -> 1.54 |

Hairline curve error unchanged (0.065); forehead ratio |d| 0.024 -> 0.027
(photo-10's mid-hairline 0.02 higher). An earlier build of the same fit
without the scalp guard (hf2) scored identity 0.449, sil 0.849.

Whole-face error moves with the hair alone: rendered with the hair hidden,
photo-14 and photo-16 read 1.41/1.41 and 1.48/1.48 before and after (hf2,
measured before the side lights came in), so the
face's geometry is untouched and the changes are MediaPipe reading the face
differently with different hair round it.

Images (shared notes folder, `hair-beard-fit-tools/`): `hair-fit-before-after.png`
(photo | base | fit; photo's hair outline green, render's red).

## Not kept: the beard's depth below the jaw

`beard_depth`: along the ray from the nose tip through each jaw-outline
landmark, how far the segmented face skin (a beard is labelled face skin) runs
past the landmark, in eye spacings, weighted by the facial hair just inside it
(as outline_trust reads it). ref-man 0.03-0.17, photo-10 0.03-0.32 (chin
deepest), the three clean jaws 0.00-0.03. Used only where jaw_beard > 0.4.

- **b1**: `fit_outline`'s targets moved out by that depth. The jaw moved, the
  render's lower-face outline did not: the landmark sheet the photo is baked
  through ends at MediaPipe's jaw, so the new rim showed the generic carried
  beard, which the segmenter does not call face. Identity man 0.602 -> 0.592,
  photo-10 0.394 -> 0.367.
- **b2**: b1 plus the sheet carried out to the beard's edge (a point at that
  depth past each jaw landmark, on the sheet and in the photo), so the photo's
  beard is painted out to the new rim. The lower face's silhouette now matches:

| face | lower-face IoU | edge error (eye sp.) | jaw landmarks | whole | identity |
|---|---|---|---|---|---|
| ref-man | 0.790 -> 0.875 | 0.106 -> 0.061 | 4.60 -> 1.45 | 1.89 -> 1.69 | 0.602 -> 0.619 |
| photo-10 | 0.833 -> 0.941 | 0.087 -> 0.025 | 3.96 -> 7.03 | 1.95 -> 2.64 | 0.394 -> 0.378 |

  Identity +0.017 on one and -0.016 on the other, both within a face's build
  noise; mean over five 0.433 -> 0.435. Not kept by the rule. On photo-10 the
  render's jaw landmarks now follow the beard's edge while the photo's sit
  inside it (MediaPipe sees the bone through a dark beard on the photo, not on
  the render), so the landmark measure counts the matched silhouette against
  it. Image: `hair-beard-fit-tools/beard-b2-before-after.png`. The code is in
  `hair-beard-fit-tools/all-experiments.patch` (appearance.beard_depth,
  morph.fit_outline `beyond`, facetexture.extend_sheet).

## Not kept: the photo's shades in the hair texture

`read_hair` read 8 shades of the photo's hair (equal shares by lightness, each
offset from the base colour as from the median); `dye_shades` gave each opaque
texel the shade at its lightness rank, so strands keep their pattern. Colour
measured over each image's segmented hair: |d| of mean L*, sdL, split (mean L
of the lightest quarter minus the darkest), and the a*b* distance; score is
their sum.

| variant (on base) | L | sdL | split | score | identity |
|---|---|---|---|---|---|
| base | 2.9 | 2.8 | 5.8 | 13.2 | 0.433 |
| s1: shades in full | 9.2 | 0.6 | 2.8 | 13.5 | 0.434 |
| s5: half mixed in | 5.7 | 1.5 | 2.8 | 11.1 | 0.436 |

Full shades fixed the spread but lifted dark hair 9-10 L* (the texture's own
mean sits low; its strands are std 40 against the photo's narrower spread). So
the mean was then held at the one dye's (calibrated to render as the photo's
base), half mixed, on top of the kept fit (hs against fit):

| | L | sdL | split | score | identity | sil |
|---|---|---|---|---|---|---|
| fit | 3.3 | 2.3 | 4.8 | 12.3 | 0.445 | 0.850 |
| fit + shades | 3.3 | 2.0 | 3.9 | 11.2 | 0.450 | 0.848 |

Closer in colour (spread and light/dark split on the two brown-haired faces),
identity +0.005 (not lower on any face), under the 0.008 render-noise bar. Not
kept. Code in the same patch (read_hair `shades`, garments.dye_shades).

## Tests

`test_rig.check_hair_outline`: a photo outline 0.2 units taller lifts the
crown 0.1-0.3; a photo with almost no hair pulls the cut in without any moved
point going into the scalp. test_rig.py passes in full.

Tools (not in the repo): shared notes folder `hair-beard-fit-tools/`
(`hl.py` hair, `bd.py` lower face, `hc.py` colour, `ids.py` identity,
`calib.py` drawn vs rendered outline, `hfviz.py` replays the fit offline from
inputs captured by `capture.py`, `queue.sh` builds one face at a time).
