# What the multiclass segmenter adds to the pipeline

The segmenter is **already in the pipeline**: `appearance.segment` (compose.py
~863) runs it, and `read_hair` takes the haircut, hair colour, hairline and
width from its hair mask. `seg.py` does not load the model a second time. It
measures that same mask against the landmarks:

```
python seg.py                     # self-check on a drawn face
python seg.py uploads/ref-man.png # numbers + out/seg/<photo>.png overlay
```

In the overlays, red dots are MediaPipe's 17 jaw landmarks, green dots are the
face-skin edge found along the ray from the nose tip (landmark 1) through each
landmark, and yellow lines join each pair. The face region is the largest
face-skin component with its holes filled.

## (a) Jaw landmarks against the segmented face edge

The gap is in face widths (234 to 454). Positive means the segmented face edge
lies outside the landmark. Landmark order: 132, 58, 172, 136, 150, 149, 176,
148, **152**, 377, 400, 378, 379, 365, 397, 288, 361.

| photo | beard (`jaw_beard`) | gap at chin 152 | range | past the edge |
|---|---|---|---|---|
| ref-man | full, 0.76 | **+0.104** | 0.00 to +0.109 | neck (body-skin) under the chin, background at the sides |
| photo-10 | full, 0.89 | **+0.190** | -0.012 to +0.190 | neck under the chin, background at the sides |
| ref-woman | none, 0.06 | -0.006 | -0.012 to +0.013 | neck, background |
| 20261019-175001-images | stubble, 0.27 | +0.017 | -0.004 to +0.029 | neck, background |
| photo-16 | stubble | +0.017 | -0.004 to +0.029 | same as above |

(photo-16 and the .jfif give the same numbers to within 0.007. Both are 477x642
and decode to nearly the same pixels: they are the same photo, not two views.)

What this shows:

- **Clean-shaven or stubbled jaw:** the landmarks sit on the segmented edge,
  within about ±0.015 face widths. That is 1 to 1.5 mask pixels, because the
  mask is 256 px and upsampled (on ref-man one mask pixel is 3.3 photo pixels,
  about 0.011 face widths). Here segmentation **confirms** the jaw landmarks
  and adds nothing new.
- **Full beard:** the landmarks sit **inside** the beard silhouette, by 0.10
  (ref-man) to 0.19 (photo-10) face widths at the chin, falling to about 0
  where the jaw meets the cheeks. In both overlays the landmarks follow the jaw
  under the beard, and the segmented edge follows the beard's lower outline.
  This is the **opposite** of what `appearance.outline_trust` assumes ("on a
  full beard [the outline] is the beard's edge, a centimetre or two outside
  the jaw"). On these two photos MediaPipe did not draw the outline round the
  beard. Down-weighting the jaw outline to 0.15 on bearded faces may therefore
  be discarding the best jaw evidence available. Worth re-checking against
  `morph.fit_outline` before anyone trusts either side.
- So segmentation **cannot replace the jaw landmarks under a beard**: its edge
  is the beard. What it does give is the beard's outer silhouette, and the gap
  measures **how far the beard hangs below the jaw**.

## (b) Hair colour

`read_hair` on the segmenter's hair mask returns exactly what the builds
recorded in `out/<face>.json` `hair_colour`: ref-man (104,72,53), ref-woman
(143,97,67), photo-10 (23,21,14), the .jfif (30,25,24). compose.py already
reads hair colour from this mask (`hair_reading["colour"]`, compose.py ~897)
and falls back to `catalogue.hair_colour` only when the mask holds too little
hair. **There is nothing to gain here.**

## (c) Beard: hair or face-skin?

**Face-skin.** Between the lower lip and the jaw outline, both full beards are
100% face-skin (ref-man 1.00, photo-10 1.00), the same as the clean faces.
The hair class stops at the sideburns: ref-man's `hair_reading.bottom` is
-0.63, ending well above the chin. Segmentation therefore cannot detect a
beard or read its colour. That stays with `facetexture.facial_hair`, which
works from colour. Only the beard's outer silhouette comes out, as part of
the face-skin edge.

## Proposal for compose.py (for the main session; compose.py not edited)

1. **Beard depth, the one new signal.** Where `jaw_beard > 0.4` (compose.py
   ~1114, where the beard geometry is decided), take `seg.jaw_edges(parts,
   landmarks)` and use the positive gaps along 176-148-152-377-400 as the
   beard's thickness below the jaw, in face widths. It could size
   `appearance.beard_underside`, or a beard-volume offset, from the photo
   instead of a constant. `parts` is computed at ~863. To use it there or
   earlier, move `parts = appearance.segment(photo_rgb)` up beside
   `outline_trust` (~802). It is a single call either way.
2. **Sanity check for `fit_outline` on clean jaws.** When `jaw_beard < 0.4`,
   a gap beyond about ±0.04 face widths at any jaw landmark means a bad
   landmark fit (hand, collar, hair across the jaw). Lower that landmark's
   trust instead of fitting the silhouette to it.
3. **Revisit `outline_trust`'s beard rule** with the overlays above
   (`out/seg/ref-man.png`, `out/seg/photo-10-face.png`). If the landmarks
   follow the jaw under a beard, as they do here, the 0.85 down-weight of
   `JAW_OUTLINE` on bearded faces should shrink or go.
4. Do not use the hair class for beards, and do not touch hair colour: both
   are settled above.

Caveats: two bearded photos, and one of them (photo-10) has a face only
about 60 px wide. The model card reports the 256 model's mean IoU at 77, and
68 on the darkest skin tones (see segmentation-licence.md).
