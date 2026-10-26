# Lips and nose

Branch `claude/lips-nose`, on top of `side-views` (e904f23). Five commits; the rest of what
was tried is written up below with its numbers and left out.

## How it was measured

Every change built for the five test faces (ref-man, ref-woman, photo-10, 20261019-175001-images
= "p14", photo-16) and rendered at the photo's pose and expression, head framing, the way
`studio.renderPng(url, false, true)` draws it. Each render was aligned onto its photo by a similarity fit on
the 468 landmarks; the photo and the aligned render were both scaled to 140 px between the irises
(photo-10's lips are 9 px tall as shot). p14 and p16 are the same photo, so the gap between their numbers
is a fair read of render noise (up to 0.16 on mouth error, 0.06 on whole-face).

- **whole / mouth / nose**: mean landmark error, % of the iris spacing (468-473).
- **width, upper, lower**: 61-291, 0-13, 14-17, % iris spacing. **lift**: corners above the lips'
  middle (13/14), +up; per side in the tables.
- **line**: darkest L (OpenCV, 0-255) across the seam against the lips' median. **strength**: that dip
  summed across the seam (L x %), which does not punish a sharp line against a soft one.
- **lip / tip Lab**: median CIE L*a*b* of the vermilion (seam excluded) and of a disk on the nose tip,
  minus the cheeks'. dE is the CIE76 distance from the photo's.
- **teeth / inside**: render with the head grey, body green, teeth red: red and green pixels in the
  mouth's box.
- **nostrils**: dark blobs (18 L under the nose base's median) inside the nose base; the painted ones
  measured on an unlit render, the mesh's own on an untextured one.

Tools (build, render, measure; standalone, run from the worktree) are in the shared notes folder,
`lips-nose-tools/`. `lipnose.py tag...` writes the table and the mouth crops. The images stay outside
the repo, since they are renders of real people and the repo is public. They are in
`lips-nose/` in the shared notes folder: `mouth-<face>.png` (photo | before | after, mouth),
`nose.png` (photo | before | after for four faces), `painted-mouths.png` (teeth red, inside green,
before then after), `lip-binding-c2b-c2d-c2e.png` (the binding variants on ref-man, photo-10, photo-16).

## Before and after (side-views e904f23 -> claude/lips-nose)

| face | whole | mouth | width (photo) | upper (photo) | lower (photo) | lift (photo) | lip dE | tip dE | teeth px | inside px |
|---|---|---|---|---|---|---|---|---|---|---|
| ref-man | 1.77 -> 1.68 | 1.61 -> 1.57 | 94.0 -> 94.6 (91.7) | 7.5 -> 6.6 (6.8) | 10.2 -> 9.3 (11.1) | 6.6 -> 5.3 (1.0) | 8.7 -> 8.3 | 3.5 -> 2.7 | 0 -> 0 | 56 -> 0 |
| ref-woman | 1.77 -> 1.71 | 2.44 -> 1.89 | 80.4 -> 82.7 (85.8) | 7.5 -> 9.2 (10.0) | 18.3 -> 18.2 (16.8) | 1.1 -> 2.5 (2.4) | 8.4 -> 6.8 | 5.0 -> 6.4 | 13 -> 0 | 8 -> 0 |
| photo-10 | 1.77 -> 1.66 | 1.75 -> 1.53 | 109.1 -> 110.7 (109.4) | 7.1 -> 6.6 (7.8) | 15.1 -> 15.9 (14.5) | 7.1 -> 5.6 (6.6) | 5.0 -> 3.4 | 8.0 -> 5.4 | open mouth | 48 -> 38 |
| p14 | 1.20 -> 1.17 | 1.94 -> 2.11 | 79.5 -> 80.1 (73.2) | 8.1 -> 8.4 (7.3) | 17.6 -> 18.4 (16.5) | 5.2 -> 5.8 (7.7) | 3.6 -> 2.3 | 6.9 -> 6.2 | 0 -> 0 | - |
| photo-16 | 1.21 -> 1.11 | 1.94 -> 1.85 | 79.7 -> 79.2 (73.1) | 8.3 -> 8.7 (7.3) | 17.9 -> 18.4 (16.5) | 5.3 -> 5.9 (7.7) | 3.0 -> 2.9 | 5.2 -> 5.5 | 0 -> 0 | 0 -> 0 |

Seam: line (photo / before / after) ref-man -50/-50/-42, ref-woman -37/-131/-114, p16 -66/-63/-57;
strength 261/297/304, 181/736/617, 269/415/386. Nose width 129-358 (photo / after): 65.5/70.1,
61.0/61.8, 68.3/74.0, 62.6/65.8. Smirk: photo-16's corners lift 6.3 / 9.0 (right / left) in the photo,
4.8 / 7.0 rendered: the 2.2-2.7 lean is kept, both corners sit ~1.8 low. The full table is
`lips-nose/final-table.txt`.

## What is in

1. **Lips closed at rest** (fd8c46c). `seal_lips` moved the inner-lip landmark vertices, which sit
   on the lips' fronts; the slit between MakeHuman's lip rims (1 mm at the middle, 2.7 mm towards the
   corners) stayed open, and on ref-woman 355 px of teeth showed through it on the bind-pose head (13
   in her render). Now the rims (`morph.UPPER_RIM` / `LOWER_RIM`, the vertices where the head's UV
   island ends at the mouth) meet at their middle, each lip following over 3 mm. The mouth's inside
   (`appearance.mouth_inside`: the faces off the head's texture joined to the rims, 418 vertices) is
   painted a shadowed dark red and set back 1.5 mm into the mouth. Sealed, it met the lips' skin at
   the seam, and a pressed smile on ref-man's turned head showed it as a pale chip below his lower lip
   (56 px). Whole-face 1.77/1.77/1.77/1.20/1.21 -> 1.74/1.74/1.75/1.20/1.21. `test_rig.check_lip_seal`.
2. **Lips corrected at full trust, on the vertices where they meet** (ea674c7).
   MediaPipe's mouth corners were bound to vertex 528, 3.6 mm in and 2.4 mm up from where the
   rims meet; the inner lips to the lips' fronts. `morph.lip_corr` puts them on the rims; the 40 lip
   landmarks count fully in `correct`. Mouth error on ref-woman 2.73 -> 1.99, photo-10 1.85 -> 1.55.
3. **The photo's lips painted onto those corners** (57547ac). `mouth_onto_mesh` with the same
   binding: ref-woman's mouth 80.1 -> 83.6 wide (photo 85.8), corner lift 1.5 -> 2.6 (2.4).
4. **Lip colour** (3af0200). Against the cheek the rendered lips were 4.2 L* lighter and 2.4 b*
   yellower than the photos'. `appearance.lip_tone` shifts the photo's lips before the bake: b* in
   full, L* by half. The yellow was alike on all five faces, the lightness not (8 L* on ref-man and
   ref-woman, 0-2 on the others), and taking it all moved ref-woman's landmarks 0.07. Lip dE
   9.0/9.1/4.0/2.5/2.5 -> 7.8/7.1/3.4/2.2/2.9.
5. **One set of nostrils** (3bbfb1f). The photo's nostrils were painted on ref-woman's nose tip,
   5.5% of the iris spacing above the dark her mesh's nose casts under it: two sets.
   `appearance.fill_nostrils` inpaints the photo's nostril darkness from the skin round it, facial hair
   excepted (filling ref-man's moustache cost him 0.1). Whole-face 1.63/1.75/1.69/1.17/1.17 ->
   1.68/1.71/1.66/1.17/1.11.

## Tried and left out (numbers against the kept state before each)

- **Rim normals smoothed** (the brows' broad-normal trick along the seam, 3 mm): ref-woman's seam
  -130 -> -127. Her seam's darkness is not shading of the rims. Untextured she renders -138; the
  texture alone gives -33 (photo -37). It is the upper lip's underside, curling in and facing down,
  seen at her photo's pitch. Shadows off changed nothing either (-129).
- **Lips to the photo less the ARKit expression the file applies** (instead of the MakeHuman units the
  fit took off): ref-man's corners 7.6 -> 3.8 (photo 1.0), but photo-10's laugh 1.64 -> 2.31 and
  p14/p16 +0.11. Also with only the non-opening shapes, and on the better binding below: photo-10
  1.96-2.12 every time.
- **Outer lip landmarks on the lip's real border** (`lip_corr` binding the outer contour to the
  front-most skin within 3 mm of the average face's landmark; nearest-vertex put landmark 0 on the
  man's philtrum, 3.5 mm above his lip, and the outer corners 10 mm behind the lip). The best single
  lip-shape result on the three closed-mouth faces: mouth error 1.98 -> 1.59, 2.03 -> 1.49,
  2.07 -> 1.38, photo-16's upper lip exactly the photo's 7.3. But ref-man 1.66 -> 2.06 and photo-10
  1.64 -> 1.84. On the true border, the file's smile shapes lift and spread the lips more than the
  fit took off (ref-man's corners 7.6 above the middle, the photo's 1.0). Neither lower trust nor
  painting with the old binding fixed it. This is the follow-up with the most in it: make the
  expression the file shows match the one the fit removed (`morph.fit`'s MakeHuman units against
  `photo_expression`'s ARKit weights), then bind the border.
- **Per-lip volume** (upper and lower targets from each lip's own height against the skull's border):
  ref-woman 1.73 -> 2.16, p14/p16 +0.21. With `correct` pinning the outline, the targets mostly
  change bulk, and the per-lip ratios against the base did not predict the render's errors.
- **Vermilion border sharpened** (unsharp mask in a band along the outer lip line, 0.8):
  border gradient error 3.25 -> 3.08 on average. Photo-10/14/16 render softer than their photos
  (11.5-12.2 against 14.9-15.2) but ref-man and ref-woman sharper (geometry), so they got worse.
  Not worth the code.
- **Lip roughness / highlight**: lips already have their own (0.35 against the cheeks' 0.6). The
  highlight error goes both ways (ref-woman's photo 41 over the lips' median, render 27; p14/p16 34
  against 40), so no single value wins.
- **Nose base carried onto the mesh's** (the nose's 16 base landmarks added to `mouth_onto_mesh`):
  ref-woman's painted nostrils barely moved (-14.5 -> -15.2 % above the subnasale), as the
  landmarks' vertices are where the paint already lands. p14 +0.07.
- **Nose tip roughness 0.32 and colour shift** (mean offset L* -1.9, a* -1.4, b* -0.8): tip highlight
  p14/p16 31 -> 38 (photo 45), but ref-woman and photo-10 overshoot, tip dE 5.23 -> 5.68 (photo-10's
  tip is off the other way), ref-man +0.09. Not kept.

## Not done

- **Nostril width, flaring and angle fitted to the photo's nostril blobs.** The mesh's alar dark sits
  4-8% of the iris spacing wider than the photo's nostrils on ref-man (left), ref-woman (right) and
  photo-16, and noses render 2-9% wide (129-358) while the fit holds `nose-width2-incr` at 0.75-0.98
  on three faces. The alar landmarks are not where MediaPipe finds the alae on the render. Fitting the
  targets needs the mesh's nostrils located from its own geometry or a render. That overlaps
  refine.py's render-in-the-loop fit, so it is left for it.
- **Ref-woman's seam** is still 3.4x the photo's line strength (617 against 181), from the upper
  lip's underside (see above). Lip shape, not paint.
- **The grey smudge round ref-woman's alae** is the mesh's alar-groove shading (it is there
  untextured). Nose shape again.
