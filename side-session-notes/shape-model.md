# A learned correction to the face fit, from our own synthetic heads

Branch `claude/shape-model-a4345b`, which has origin/side-views (cb25937) merged in. New: `shape_model.py`.
Not edited: morph.py and compose.py.

## Verdict

Yes, it helps. A ridge regression from what the fit saw (placed landmarks, fitted
weights, MediaPipe's head pose, body) to what it got wrong cuts the true shape
error on held-out synthetic heads by about a third. The outline and the nose
improve most. An MLP did worse, so no CNN was tried. On real photos it stays
in the range it saw in training. Whether it is right there is not proven
(see "Real faces").

The same data showed one more thing: on these renders MediaPipe's jaw outline
sits 5.9% of the eye spacing outside the true jaw. `morph.fit_outline` fits the
jaw to that outline, and against the true head it makes the jaw worse.

## Data

- 5,000 heads from `synth.py -n ... --seed 1` (data/synth, git-ignored).
  MediaPipe found a face in 4,962. 4,466 are used for training; the 496
  numbered `...0` are held out.
- Speed: 2 s a head on this machine, so about 3 h for 5,000. Building the head
  is the limit, at about 20 CPU-seconds each across 8 workers. The README's
  0.68 s is out of date.
- Setting BLAS to one thread matters for the fit itself: `morph.fit`'s bvls
  solve took 4 s with numpy's threads competing for cores and 0.7 s with one
  thread (`extract` sets this). compose pays the same 4 s whenever the machine
  is busy.

## The target: what the fit should have found

On synthetic data the true target weights are not the right answer:

1. A photo shows neither the head's size nor where it sits on the neck.
   `normalise` moves and scales the true head so its eyes, nose, mouth and
   chin sit on the body's, as `compose.place_face` places a photo.
2. compose builds on a binary man or woman. The synthetic heads use a
   continuous macro blend (beta 0.6, 0.6). `reachable` finds the bounded
   target weights on the chosen body that come closest to the true head over
   the face's skin (vertices within 1 cm of a landmark, QR-reduced bvls).
   Targets that the skin cannot see (ears, back of the head) keep the fit's
   value. This "best" is the floor any weight-space model can reach.

The model predicts best − fit (all 270 targets). Its prediction is added to
the fit's weights and clipped to [0, 1], so every weight is still a real
MakeHuman setting.

## Results on 300 held-out heads

Vertex error at the 468 landmark vertices against the normalised true head,
% of the true eye spacing (33-263), with the faces lined up by their mean.
- **outline**: 21 vertices along the jaw and cheeks (234 … 152 … 454).
- **jaw edge**: the jaw's silhouette as the photo saw the head, against the
  true head's, measured along rays from the nose through
  `morph.JAW_OUTLINE`.
- **nose long**: the 168-2 length, as REALISM-GAP measures it.

Meshes are built as compose builds them: fit, `lip_volume`, `correct` on lips
and lids, `seal_lips`, then (in the first block) `fit_outline`.

| | face (xy) | outline (xy) | jaw edge | nose long, mean (mean abs) |
|---|---|---|---|---|
| fit, as side-views builds it | 6.06 (4.66) | 10.17 (7.52) | 7.43 | -3.66 (5.23) |
| fit + model | 4.40 (3.31) | 7.21 (5.30) | 7.01 | +1.68 (2.69) |
| best (floor) | 2.44 (2.35) | 3.40 (3.30) | 6.85 | +2.13 (2.21) |
| *without fit_outline:* | | | | |
| fit | 6.04 (4.65) | 9.94 (7.28) | 4.61 | -3.66 (5.23) |
| fit + model | **4.32 (3.22)** | **6.37 (4.32)** | **3.29** | +1.68 (**2.69**) |
| best (floor) | 2.22 (2.12) | 1.12 (0.93) | 1.21 | +2.13 (2.21) |

The 157 clearly male or female heads (blend under 0.2 or over 0.8) give the
same picture: fit 5.89 / 9.63 / 4.65, fit + model 4.25 / 6.21 / 3.38.

Weights, per MakeHuman slider (opposite ends signed), mean |w − best|: fit
0.282, fit + model 0.146.

**Learning curve** (validation slider error): 250 heads 0.165, 500 0.159,
1,000 0.154, 2,000 0.150, 4,466 0.146. It is flattening. More heads of the
same kind would buy little; the gap to the floor (4.3 against 2.2) is
in what the landmarks fail to say.

**Model choice**: ridge with λ = 1000 on standardised features (1,678
inputs: 1,404 landmark offsets, 270 fitted weights, 3 pose angles, the body).
A two-layer MLP (256 or 512 wide, trained on vertex error through the QR
basis) reached a test vertex loss of 6.4-6.6 against ridge's 4.9 at 1,600
heads; it overfits. The brief said to try image features from a small CNN
only if this failed. It did not fail, so no CNN was tried. The remaining
2-point gap to the floor is where image features would have to earn their
place.

## The nose: the lip correction lengthens it

Nose length against the true head (60 held-out heads, % eye spacing), with
and without compose's lip-and-lid `correct`:

| weights | after `correct` | targets alone |
|---|---|---|
| fit | -3.12 | -4.41 |
| fit + model | +2.41 | +0.33 |
| best | +2.38 | -0.18 |

`correct` pulls the upper lip to MediaPipe's, and its Gaussian (reach 0.12
eye spacings) drags landmark 2, under the nose tip, with it. That adds
+2.5% to the nose's length on every head. This is probably at least part of
the "nose 2-3% long" in REALISM-GAP. A fix would keep the lip correction away
from 2 and its neighbours (a cut-off or a fixed point at 2 in `correct`). I
did not try it: it is in morph.py.

## MediaPipe's jaw on renders, and fit_outline

`python shape_model.py jaw-bias data/synth` projects each true head through
the render's own camera (perspective, the JSON's position, look_at and fov;
checked: the projected eyeball centres fall within 1-6% of the iris
landmarks). It then compares the head's silhouette with MediaPipe's jaw
outline along rays from the nose. On 100 held-out renders the outline lies
**+5.9% of the eye spacing outside the true edge** (mean |gap| 6.2), at every
landmark (+3.5 to +8.9, largest at the jaw's corners).

`fit_outline` moves the jaw onto that outline, and the table above shows the
result: the jaw edge worsens from 4.61 to 7.43 for the fit, from 3.29 to 7.01
for fit + model, and from 1.21 to 6.85 even for the floor. The side-views
measurement of fit_outline compared MediaPipe on the render with MediaPipe on
the photo, so it cannot see a bias MediaPipe shares between the two.

Caveat: these are renders, not photographs. If MediaPipe reads a real jaw
with the same outward offset, fit_outline makes real jaws too wide by about
that much. The check would need real photos with a known jaw: a head scan, or
the side-views fused profile.

## Real faces (sanity check)

`python shape_model.py real uploads/ref-man.png:male uploads/ref-woman.png:female uploads/photo-10.jpg:male uploads/20261019-175001-images.jfif:male uploads/photo-16.jpg:male`
The meshes include fit_outline. There is no ground truth here, so these
measure how far the model moves each face, beside the same measures on
synthetic heads. "Gap" is the fit's own measure: mesh landmark vertices
against the photo's landmarks, in the plane.

| | mean slider change | face moved | outline moved | nose length | landmark gap | outline gap |
|---|---|---|---|---|---|---|
| synthetic (mean of 100) | 0.229 | 3.67 | 6.52 | +5.12 | 3.00 → 4.21 | 5.15 → 7.02 |
| ref-man | 0.200 | 2.15 | 4.49 | +6.15 | 2.77 → 3.49 | 5.55 → 6.21 |
| ref-woman | 0.235 | 4.10 | 5.48 | -4.53 | 2.53 → 3.95 | 3.44 → 6.43 |
| photo-10 | 0.235 | 3.79 | 4.95 | +3.06 | 3.86 → 5.21 | 6.00 → 6.93 |
| 20261019-175001 | 0.211 | 2.34 | 3.90 | +1.85 | 2.42 → 3.18 | 3.96 → 4.84 |
| photo-16 | 0.211 | 2.32 | 3.91 | +2.00 | 2.42 → 3.17 | 4.03 → 4.83 |

- **Plausible:** every real face moves less than or about as much as the
  synthetic average. No face is pushed outside the range the model saw.
- **Moves away from the photo's landmarks**, as it does on synthetic heads,
  where the true error falls at the same time. The model has learned that
  MediaPipe's reading of a render is biased (short nose, wide jaw). If
  MediaPipe reads real photos the same way, the correction carries over; if
  not, it adds error. Only a render-based likeness check or `score.py`
  can settle that, comparing with and without the model.
- **Nose:** it lengthens four noses (+1.9 to +6.2) and shortens ref-woman's
  (-4.5). On renders the fit's noses come out 4.4% short before the lip drag.
  REALISM-GAP says the rendered avatar's nose comes out 2-3% long on real
  photos, which pulls the other way. Put the two together: with the lip drag
  fixed, the nose is probably right on real faces too. Without the fix, the
  model plus the drag would overshoot.

## How compose would call it (proposal, not applied)

compose.compose, straight after `morph.lip_volume`, on single-photo builds
only (the model never saw fused side-view depth):

```python
    shape_weights = morph.lip_volume(shape_weights, shape_names, base, corr, person)
    if not side_views and shape_model.MODEL.exists():
        # A correction learned on synthetic heads: what this fit gets wrong,
        # read from what it saw (side-session-notes/shape-model.md).
        shape_weights = shape_model.predict({"gender": gender, "person": person[:len(corr)],
                                             "weights": shape_weights,
                                             "pose": shape_model.head_pose(result)})
    vertices = morph.apply(base, shape_weights)
```

Alternatively, `morph.fit` could take an optional `correction` callable and
apply it to its returned identity weights. But `lip_volume` runs after the
fit, and the model was trained on the weights after it, so the call belongs
in compose. Cost: one 1,678 × 270 matrix product, under 1 ms. The model file
is 3.6 MB (`data/shape/model.npz`, git-ignored).

Before switching it on I would:
1. fix the lip drag on landmark 2 (above), since the model's nose works
   against it;
2. check `fit_outline` against the jaw bias above, and switch it off or
   offset it by that bias if the bias holds on real faces;
3. compare likeness on the five test faces with and without the model,
   using a render or `score.py`, since the landmark gap cannot judge it.

## Reproduce

```
python synth.py -n 5000 --seed 1 --workers 8          # ~3 h
python shape_model.py extract data/synth --workers 10   # 12 min -> data/shape/fits.npz
python shape_model.py train --measure 300               # ~15 min with meshes -> data/shape/model.npz
python shape_model.py real <photo>:<gender> ...
python shape_model.py jaw-bias data/synth
python shape_model.py                                   # self-check
```

## Known limits

- The synthetic shape prior is a guess (synth.py's ACTIVE and SPREAD). The
  model partly learns that prior, and transfers to real faces only as far as
  real faces resemble it.
- Neutral expression only, no beards, no glasses, and MediaPipe reading a
  CG face. The jaw bias above shows that this reading can differ from the
  truth by several percent.
- The binary body choice costs little here (the floor is the same for
  blended and clear heads). A continuous macro blend in compose would remove
  it completely.
