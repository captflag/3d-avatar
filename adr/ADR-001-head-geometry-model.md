# ADR-001: MakeHuman CC0 targets as the head geometry model, not FLAME

**Status:** Accepted and implemented in `morph.py`. Legal sign-off pending.
**Date:** 2026-10-16
**Deciders:** engineering lead; legal counsel before any commercial release

## Context

The avatar engine needs a morphable head: a parametric model whose shape can be
fitted to one selfie, so an avatar is the person rather than a photo laid over a
generic skull. Without one, likeness, texture alignment and expression handling
all stay broken (REALISM-GAP.md, causes 1, 3, 4 and 6).

External advice has twice recommended the research standard: FLAME geometry,
with MICA or DECA to regress it from a photo and SMPLitex to complete the
texture. Technically that is the strongest stack available. It has to be
evaluated against the one requirement it does not consider: **this is a product
we intend to sell.**

Verified licence position (sources in ARCHITECTURE.md):

- **FLAME** is licensed for non-commercial research only, and the licence also
  forbids using it to train networks. Commercial use needs a separate agreement
  with the Max Planck Institute.
- **MICA** regresses FLAME and is initialised from **ArcFace**, whose pretrained
  weights are non-commercial. Two independent blockers.
- **DECA** outputs FLAME parameters, so it inherits FLAME's licence.
- **SMPLitex** is built on SMPL, whose commercial use is a paid Meshcapade
  licence.
- **MakeHuman** base mesh, skeleton, skin weights and roughly 1,280 shape targets
  are CC0; each target file states its CC0 release explicitly.

Constraints: one GTX 1660 with 6 GB, no 3D scan dataset, a small team.

## Decision

Use MakeHuman's CC0 face targets as the head's morphable model. Fit about 270 of
them (nose, mouth, eyes, chin, cheeks, forehead, head shape, brows, ears) to the
difference between the person's landmarks and MediaPipe's canonical average
face, with bounded least squares.

When finer detail is needed, learn a residual displacement on top of this basis,
trained only on data we own — chiefly synthetic renders from the same MakeHuman
generator. Do not adopt FLAME-family models unless a commercial licence is in
hand.

## Options Considered

### Option A: FLAME with MICA/DECA, SMPLitex for texture

| Dimension | Assessment |
|-----------|------------|
| Complexity | High: PyTorch3D, several research codebases, GPU inference |
| Cost | Unbounded: commercial terms are by negotiation, and not on offer for every component |
| Scalability | Good quality per avatar once running |
| Team familiarity | Low |

**Pros:** strongest published quality; expression and pose separated cleanly;
large research ecosystem.
**Cons:** **not shippable as licensed** — FLAME, ArcFace weights and SMPL each
block it independently; FLAME's licence forbids training on it, which also rules
out fine-tuning; needs a GPU this team does not have for inference at scale.

### Option B: FLAME or SMPL-X under a paid commercial licence

| Dimension | Assessment |
|-----------|------------|
| Complexity | High, as Option A |
| Cost | SMPL family €1,500–25,000 a year by tier; FLAME by negotiation |
| Scalability | As Option A |
| Team familiarity | Low |

**Pros:** keeps the research-grade model legally.
**Cons:** recurring cost before any revenue; licences the base model only — MICA
still needs ArcFace replaced and retrained; annual tier re-evaluation and vendor
dependency on the core of the product.

### Option C: MakeHuman CC0 targets (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low: one bounded least-squares solve, about 200 lines |
| Cost | None |
| Scalability | CPU, a fraction of a second per avatar |
| Team familiarity | High: same base mesh, rig and UV layout the pipeline already uses |

**Pros:** shippable; fits the existing pipeline with no topology conversion;
targets move the eye and teeth helpers with the face, so the rig and eyes stay
correct; the generator doubles as an unlimited source of owned training data.
**Cons:** linear and hand-authored, so it captures proportions, not asymmetry or
fine folds; fewer identity dimensions than FLAME; no separate expression space.

### Option D: Train our own 3DMM from scratch

| Dimension | Assessment |
|-----------|------------|
| Complexity | Very high |
| Cost | Scan data and GPU time |
| Scalability | Unknown until built |
| Team familiarity | Low |

**Pros:** fully owned, potentially best fit to our users.
**Cons:** needs a licensed dataset we do not have; months before it beats C.

## Trade-off Analysis

The decisive dimension is not quality, it is whether the product can be sold.
A and D lose on that axis or on timeline; B trades a recurring fee and vendor
dependency for quality we cannot yet measure the value of.

C gives up identity detail. That loss is real, and it is the same critique the
advice makes of linear models. Measured on two test photos, the fit activated
66 and 75 targets and explained 48% and 55% of each face's deviation from
average — a large improvement on the 0% of a fixed skull, well short of a scan.
C also leaves the door open: a learned residual on the same basis can close most
of that gap later, and B remains available if that residual stalls.

## Consequences

- **Easier:** shipping commercially; running on CPU; keeping one topology across
  head, body, rig, garments and UV; generating our own synthetic training data.
- **Harder:** matching research-grade likeness; anything that assumes FLAME
  parameters, such as off-the-shelf expression transfer.
- **Revisit:** if a learned residual on this basis plateaus below acceptable
  likeness, price Option B. A "FLAME 2023 Open" with different terms is
  mentioned secondhand online; ask ps-license@tue.mpg.de before assuming it.

## Action Items

1. [x] Fit MakeHuman face targets to landmarks (`morph.py`).
2. [x] Record each avatar's fit in its sidecar (explained fraction, active targets).
3. [ ] Legal sign-off on every component in ARCHITECTURE.md section 2.
4. [ ] Build a synthetic data generator: randomise targets, skin, lighting and
       camera; render paired photo and ground-truth shape.
5. [ ] Train a residual displacement model on that data; measure likeness on a
       held-out set of real, consented selfies.
6. [ ] Re-evaluate Option B only if step 5 plateaus.

Answer to the next recommendation of FLAME, MICA, DECA or SMPLitex: see this
record. The technical case for them is good; the licence position has not changed.
