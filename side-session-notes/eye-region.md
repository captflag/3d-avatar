# The eye region: brows, lids, lashes, whites, highlight

Built on side-views 9744847. Five test faces (ref-man, ref-woman, photo-10,
20261019-175001-images = "p14", photo-16), rendered by the studio
(`window.__studio.renderPng(url, false, pose)`, at the photo's pose and level),
scored with `identity_score.py` (SFace cosine, photo vs render).

**Noise.** Renders are not deterministic: the same build rendered twice moved
one face by up to 0.017 and the five-face mean by 0.004. Every number below is
the mean of three renders (six where marked) of each build, which brings the
mean's noise to about 0.0025 (0.002 for a difference of six-render means).

Tools (scratch, not in the repo): the shared notes folder's `eye-region-tools/`
(build, render, score, crops, paint stages, profiles). Images: its
`eye-region/` (they contain the test photos, so they stay out of the repo).

## Kept

| | pose | level | man | woman | p10 | p14 | p16 (pose) |
|---|---|---|---|---|---|---|---|
| base (9744847) | 0.446 | 0.412 | 0.618 | 0.367 | 0.378 | 0.441 | 0.429 |
| 1. `tighten_brows` taken out (4718bfb) | 0.470 | 0.447 | 0.639 | 0.400 | 0.395 | 0.456 | 0.459 |
| 2. upper lids lifted, `lift_lids` (5a35071) | 0.480 | 0.466 | 0.652 | 0.392 | 0.407 | 0.478 | 0.469 |

All five faces gain at both poses from base to 2 (level: 0.608/0.328/0.368/0.380/0.377
-> 0.643/0.385/0.407/0.450/0.444).

**1. Brows left as the photo has them.** `tighten_brows` pushed every pixel
darker than the skin round each brow 1.6x darker, over a region that ran onto
the lid. Its brows rendered 14-34 L* darker than the photos' and the lid's
crease and socket went dark with them: the brown band under every brow, and
ref-woman's double brow. The brow session's fix (stop at the brow, gain 1.25,
d5a2e4d, taken back out in cb25937 on a landmark measure) scores 0.464/0.434;
no darkening at all scores 0.470/0.447, higher on level for every face. Brow
L* is now 173/145/110/98/101 against the photos' 167/142/101/101/101, and the
band under the brow against the forehead -34/-76/-50/-73/-74 against
-55/-87/-57/-73/-72 (base: -53/-120/-81/-99/-100).

**2. The upper lids lifted.** After 1, the lids still rendered 26-33 L* darker
than the photos' on photo-10/14/16 (just above the lid: p16 132 in the photo,
103 rendered). Split three ways (`lit-unlit-flat.png`): untextured, the
geometry darkens the lid only a little (lid/cheek 0.94); texture only, the lid
is too dark against the cheek (0.55-0.65 against the photos' 0.76-0.92). Paint
stages on p16 (`p16-paint-stages.png`) show fill, lashes and de-lighting keep
the lid in proportion; the photo's lid is a thin band in the brow's shade,
and the mesh's taller lid stretches that shade over it (brightness profiles
above the eye: the photo's light fold just above the lash line, +10 against
the cheek on photo-10, renders -78). `appearance.lift_lids` pulls the paint
between each eye's upper contour and its brow's lower contour half way to the
forehead's L*, never darker. Lid L* after: p14/p16 126/128 (photo 133/132).

| lid lift variant | pose | level |
|---|---|---|
| none (after 1) | 0.470 | 0.447 |
| 0.5 towards the forehead (kept) | 0.481 | 0.465 |
| 0.8 towards the forehead | 0.479 | 0.464 (ref-woman -0.04) |
| 1.0 towards the band's own 85th / 70th percentile | 0.476 / 0.475 | 0.466 / 0.468 |
| 0.5, starting 0.08 (not 0.03) below the brow | 0.474 | 0.457 |
| 1.0 to 70th pct, 0.08 below the brow | 0.474 | 0.460 |

Ref-woman loses 0.008 at pose: her photo's lids are genuinely dark (139 L*
against a 208 cheek) and now render 172. The lift reads the forehead, not her
lid; reading her lid's own percentile kept her level but gained less overall.

## Tried, not kept

| change | pose | level | eye measure |
|---|---|---|---|
| the opening filled column by column, lid colour above to the colour below, not inpainted (`fill_vert`) | -0.002 | 0.000 | lid unchanged: the dark lid is not the fill |
| the same, upper colour held for 40% of the opening | -0.009 | -0.005 | |
| upper lid dropped 6 degrees (`UPPER_LID_DROP` 4 -> 6) | +0.002 | -0.001 | p14/p16 lid cover 23 -> 25 (photo 22), man/p10 still more open than the photos |
| brows moved down 0.04 / 0.08 eye spacings on the sheet | -0.013 / -0.058 | -0.007 / -0.047 | brow height above the lid matched the photos (blid 18-23 against 20-24), identity worse |
| lash colour scaled as the de-lit skin round it (it is the photo's as shot, the skin is written darker for the renderer) | +0.004 | +0.003 | lash L* 115/96/89/47/47 (photo 96/74/83/32/32), six renders |
| eye highlight: clear coat 0.3, eye roughness 0.4 | +0.002 | +0.001 | highlight over the iris 92/95/103/119/129 (photo 66/44/59/129/128, before 153-224); iris L* 106/105/85/38/35 (photo 110/121/83/33/34, before 95/97/74/24/21), six renders |
| both of the last two | +0.001 | +0.001 | six renders |
| clear coat 0 + roughness 0.4; clear coat 0.3 alone; specular 0.5; cornea roughness 0.12 / 0.3 | +0.004 at most | +0.003 at most | three renders |
| under-eye band pulled 0.3 / 0.6 towards the cheek's L* | +0.005 / -0.002 | -0.003 / -0.006 | p14/p16 under-eye -44 / -27 against the cheek (photo -55): overshoots |

The lash colour and the highlight move the measures to the photos' but not the
score beyond noise, and they do not add. Worth re-trying together with a
change that raises the eyes' share of the score (they are small in a 900 px
render: the iris is 8-10 px across).

## Measures at the end (photo's pose, one render; photo in brackets)

Units: L* on OpenCV's 0-255 scale; lengths in % of the iris spacing.

| | man | woman | p10 | p14 | p16 |
|---|---|---|---|---|---|
| upper lid over the iris (% of its diameter) | 12.2 (15.6) | 10.8 (10.9) | 11.9 (17.7) | 23.0 (21.7) | 23.7 (21.8) |
| sclera L* | 169 (166) | 156 (166) | 161 (134) | 135 (140) | 138 (141) |
| iris L* | 95 (110) | 97 (121) | 74 (83) | 24 (33) | 21 (34) |
| highlight over iris median | 153 (66) | 154 (44) | 174 (59) | 218 (129) | 224 (128) |
| lash line L* | 118 (96) | 97 (74) | 94 (83) | 49 (32) | 49 (32) |
| under-eye L* minus cheek | -19 (-31) | -2 (-17) | -21 (-17) | -61 (-55) | -60 (-55) |
| brow thickness | 22.0 (28.7) | 29.7 (28.7) | 29.2 (32.5) | 34.4 (29.8) | 34.8 (29.8) |
| brow above the lid | 21.7 (20.1) | 26.2 (24.0) | 24.5 (22.2) | 27.6 (21.8) | 27.6 (21.8) |
| brow L* | 180 (167) | 156 (142) | 121 (101) | 106 (101) | 108 (101) |
| under-brow band minus forehead | -27 (-55) | -45 (-87) | -43 (-57) | -60 (-73) | -58 (-72) |
| brow landmark height error (+ = high) | +0.4 | +2.1 | +2.0 | +2.7 | +2.7 |

Base for the same rows: brow L* 158/121/93/79/81, under-brow band
-53/-120/-81/-99/-100, lash 120/92/74/42/42.

## Still open

- The lift lightens the brows' lower edge too (brow L* now 5-20 above the
  photos'); keeping it off the brow scored lower (table above), so whatever the
  score likes there is not the brow's darkness alone.
- The highlight is still 2-3x the photos' on every face; matching it did not
  move the score. Same for the lash line (15-22 L* light).
- The brows still sit 2-6% of the iris spacing high above the lid; lowering
  them on the sheet scored worse. The stretch between brow and lid that
  brow-render.md measured is still there; the lids' lift is the part of it
  that paint can reach.
- Photo-16's under-eye bags render 5 L* darker against the cheek than the
  photo's and warmer; lifting them overshot without gain.
- Renders are not deterministic: scoring one render per build cannot resolve
  changes under about 0.01.
