# Avatar engine: target architecture

A review of the proposed four-module ML pipeline, rebuilt so that every component
can legally ship in a commercial product and runs on hardware we can get.

**Short version.** The proposal's decomposition is right: geometry, texture,
hair and garments are the four separable problems, and each should produce its
own layer. Most of the specific components it names cannot ship: FLAME, SMPL-X,
ArcFace weights, FFHQ-UV, MetaHuman training data, FaceLift weights and
Hunyuan3D are each blocked for us by licence, territory or price. This document
keeps the structure, swaps the parts, and says what is built, what is next, and
what needs hardware we do not have.

---

## 1. Requirements

**Functional**
- One selfie in; a rigged, dressed, textured GLB out, plus a 2D render.
- Likeness: the avatar is recognisably the person, front and profile.
- Layered output: body, face, hair, each garment and the eyes as separate
  primitives on one skeleton, so items swap without regenerating the avatar.
- Engine-ready: GLB that imports into Unity, Unreal, Blender and three.js.

**Non-functional**
- Under 10 seconds per avatar on a CPU for the internal studio.
- Under 2 MB per avatar.
- **Commercially licensable end to end.** This is the binding constraint, and
  the one the proposal does not account for.

**Constraints**
- One local GPU: a GTX 1660 with 6 GB. Enough for inference of small models, not
  for training diffusion models or running 4-billion-parameter generators.
- A small team, and an existing Python, MediaPipe, MakeHuman and three.js stack.

---

## 2. Licence audit of the proposed components

Every row below was checked against its source; links are at the end. "No" means
not usable in a product we sell without a separate commercial agreement.

| Proposed component | Licence | Usable? | Replace with |
|---|---|---|---|
| FLAME head model | Non-commercial; also forbids training networks with it | **No** | MakeHuman targets (CC0) — **built** |
| SMPL-X body model | Academic; commercial via Meshcapade, €1,500–25,000/yr | Paid | MakeHuman base and macro targets (CC0) |
| ArcFace (InsightFace) weights | Code MIT, pretrained weights non-commercial | **No** | DINOv2 features; own identity model later |
| DINOv2 | Apache 2.0 (relicensed from CC BY-NC) | Yes | — keep |
| FFHQ-UV training data | Built on FFHQ, CC BY-NC-SA | **No** | Own synthetic data from MakeHuman |
| MetaHuman synthetic data | Unreal EULA bars use as AI training input | **No** | MakeHuman synthetic generator |
| FLUX.1 dev | Non-commercial | **No** | FLUX.1 schnell (Apache 2.0) or SD 1.5 (OpenRAIL-M) |
| FLUX.1 schnell | Apache 2.0 | Yes | — |
| FaceLift | Code Apache 2.0, weights Adobe Research Licence | **No** (weights) | Authored hair now; own model later |
| GAGAvatar | Licence not found | Unverified | Treat as unusable until checked |
| Hunyuan3D 2.x | Community licence excluding EU, UK, South Korea | Restricted | TRELLIS.2 |
| TRELLIS / TRELLIS.2 | MIT | Yes, but needs a 24 GB GPU | Rented GPU |
| 3D scan vendors | Per-vendor; AI-training clauses common | Check each | Only with AI training granted in writing |

Two of these deserve emphasis because the proposal's plan depends on them.

- **The proposed geometry base and both body models are off the table.** FLAME's
  licence forbids commercial use *and* using it to train networks, which rules
  out the proposal's Module A as written.
- **The proposed training-data plan is barred twice over.** FFHQ-UV is
  non-commercial, and Epic's EULA expressly prohibits using MetaHumans as input
  to train generative AI.

MakeHuman fills both holes. Its base mesh, skin weights, skeleton and roughly
1,280 shape targets are CC0, each target file states it, and because it is a
generator we control, it can produce unlimited labelled synthetic training data
that nobody else has a claim on.

---

## 3. High-level design

```
                    ┌──────────────── one selfie ────────────────┐
                    ▼                                             │
        ┌──────────────────────┐                                  │
        │ Landmarks & pose     │  MediaPipe (Apache-2.0)          │
        │ 478 points, head pose│                                  │
        └──────────┬───────────┘                                  │
                   ▼                                              │
  ┌─────────────────────────────────┐                             │
  │ A. GEOMETRY                     │  MakeHuman targets (CC0)    │
  │ fit ~270 face targets to the    │  bounded least squares      │
  │ landmark difference from average│  ── BUILT                   │
  │ [next: learned displacement]    │                             │
  └──────────┬──────────────────────┘                             │
             │ shaped base mesh                                   │
     ┌───────┼───────────────────────────────┬────────────────────┘
     ▼       ▼                               ▼
┌──────────┐ ┌───────────────────────────┐ ┌──────────────────────────┐
│ C. HAIR  │ │ B. TEXTURE                │ │ D. GARMENTS              │
│ skull cap│ │ landmark-warp photo,      │ │ cut from MakeHuman       │
│ ── built │ │ bake into UV, Poisson     │ │ helper shells ── built   │
│ [next:   │ │ blend, LAB skin ── BUILT  │ │ [next: authored meshes + │
│ authored │ │ [next: delighting and UV  │ │ RBF weight transfer +    │
│ cards]   │ │ completion model]         │ │ body alpha masks]        │
└────┬─────┘ └────────────┬──────────────┘ └────────────┬─────────────┘
     └────────────────────┼─────────────────────────────┘
                          ▼
            ┌───────────────────────────┐
            │ RIG & EXPORT              │  MakeHuman skeleton and
            │ 163 bones, rest pose,     │  weights (CC0)
            │ one skin, layered GLB     │  ── BUILT
            └───────────────────────────┘
```

The proposal's diagram has the same four boxes. What changed is what sits inside
them, and the fact that each box has a working classical version today with a
learned version as a drop-in later, behind the same function.

---

## 4. Module by module

### A. Geometry — built, licence-clean

`morph.py` fits about 270 MakeHuman face targets (nose, mouth, eyes, chin,
cheeks, forehead, head shape, brows, ears) to the photo.

- The person's face and MediaPipe's canonical average face are placed on the
  skull in the same way, and the fit targets the **difference** between them, so
  placement error cancels rather than being baked into the shape.
- Bounded least squares keeps every weight a real MakeHuman setting in [0, 1];
  a light ridge stops opposing targets stacking up to chase noise; depth is
  trusted at a third of width and height, since it is the least reliable thing a
  single photo tells you.
- Measured on the two test photos: 66 and 75 targets active, explaining 48% and
  55% of each face's deviation from average. The fit picks plausible traits
  unprompted, including "older" for the older subject.

**Ceiling:** a linear basis captures proportions, not asymmetric folds or fine
structure — exactly the proposal's critique of linear models, and it is fair.
**Next:** a learned displacement on top of this same basis (proposal's Module A),
trained on our own synthetic renders. DINOv2 is a clean feature backbone for it.

### B. Texture — classical version built

`facetexture.py` fits the photo's camera from the landmarks, warps the photo
with a thin-plate spline so its features land on the skull's (the skull's
feature positions come from its eyeball and teeth helpers), bakes into the base
mesh's UV layout, Poisson-blends into LAB-matched skin, and paints the eyeballs
from the same aligned photo.

**Ceiling:** the photo's lighting is baked in, and anything the camera did not
see is filled with flat skin. **Next:** the proposal's Module B — delighting into
albedo, plus UV completion for the unseen 60%. Train on our synthetic data, with
FLUX.1 schnell or SD 1.5 inpainting as the base, never FLUX.1 dev.

### C. Hair — placeholder built

A skull cap cut from the base mesh, coloured from just above the hairline. It
is the weakest part of the avatar and looks it. **Next:** authored hair cards
(MakeHuman community CC0 assets, checked item by item, or commissioned). The
proposal's Gaussian-splat route depends on FaceLift-class models whose weights we
cannot ship; it is a research direction, not a plan.

### D. Garments — import, weight transfer and alpha masks built

Built: garments can be loaded from files and rigged by weight transfer
(`garments.py`), validated against true weights to under 1 cm at the 95th
percentile for fitted garments under a stride pose; skin under clothes is
alpha-masked rather than deleted, with a band of skin kept under every edge;
every item must carry a licence or it is refused. MakeHuman's own MHCLO garments load too, rebuilt on each shaped body and
rigged exactly from their reference vertices (two CC0 garments prove the path).
**Next:**
1. **Authored garments** with normal and occlusion maps, preferably as MHCLO so
   they follow body shape; dropped in as files otherwise.
2. **Validate the loose path** against the first real skirt or coat.
3. **Generated garments** with TRELLIS.2 (MIT) on a rented 24 GB GPU.

---

## 5. Compute and cost

| Stage | Where it runs | Why |
|---|---|---|
| Everything built today | The GTX 1660, CPU-bound, a few seconds per avatar | Classical methods, no networks to run |
| Learned displacement, delighting, UV completion — training | Rented A100/H100 | 6 GB cannot train diffusion models |
| Same — inference | The GTX 1660, if the models are kept small | Distilled models fit in 6 GB |
| TRELLIS.2 garment generation | Rented 24 GB GPU, offline batch | Needs 24 GB; run to build a catalogue, not per avatar |

Generating garments offline into a catalogue, rather than per user request, is
the decision that keeps per-avatar cost flat.

---

## 6. Trade-offs, made explicit

- **Linear MakeHuman basis over a learned 3DMM.** Loses fine identity detail;
  gains a model we can ship, that runs on CPU, and that needs no training data.
  The right order: ship this, then learn a residual on top.
- **Cut-shell clothes over authored garments.** They fit perfectly and cost
  nothing, but they will never have a collar. They exist to prove the pipeline
  until real garments arrive.
- **Face patch on by default.** Stronger likeness from the front; a visible edge
  from the side. `--no-face-patch` gives a seamless head with weaker likeness.
  The shaped skull narrows that gap; a delit, completed texture closes it.
- **Synthetic data over scans.** Unlimited, labelled, ours — but synthetic
  avatars look synthetic, so a model trained only on them may underperform on
  real selfies. A small set of real captures with explicit consent and an AI
  training grant is the hedge.

On the proposal's claim of beating established platforms: with one 6 GB GPU and
no dataset yet, out-training them on raw quality is not the near-term edge. The
credible edges are a licence-clean stack, full control of the pipeline, and
integration with the rewards platform this repository already is.

---

## 7. What to revisit as this grows

- Before shipping commercially: written legal sign-off on every row of section 2.
  FLAME's site hints at a "FLAME 2023 Open" with different terms; that claim is
  secondhand and unverified, and worth an email to ps-license@tue.mpg.de if a
  research-grade head model ever becomes worth paying for.
- When avatars come from user uploads at volume: a quality gate, and a queue in
  front of generation instead of the studio's inline call.
- When the first learned model lands: a held-out set of real selfies and a
  likeness metric, so "better" is measured rather than eyeballed.

---

## Sources

- FLAME licence: [flame.is.tue.mpg.de/license.html](https://flame.is.tue.mpg.de/license.html), [model licence](https://flame.is.tue.mpg.de/modellicense.html), [discussion of FLAME 2023 Open](https://huggingface.co/3DAIGC/LAM-20K/discussions/2)
- SMPL-X: [smpl-x.is.tue.mpg.de/modellicense.html](https://smpl-x.is.tue.mpg.de/modellicense.html), [Meshcapade body model pricing](https://meshcapade.com/assets/body-models)
- InsightFace: [repository README mirror](https://github.com/yinheng-gh/insightface), [Model Zoo note](https://awesome.ecosyste.ms/projects/github.com%2Fdeepinsight%2Finsightface)
- DINOv2: [licence history](https://hn.svelte.dev/item/44904993), [model listing](https://thegtmdirectory.com/models/facebook-dinov2-small/dinov2-small)
- Hunyuan3D: [Hunyuan3D-2 LICENSE](https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE), [licence text](https://open.windriver.com/info/uni-license-list/licenses/tencent-hunyuan-3d-2.0-cla.html)
- TRELLIS: [repository](https://github.com/pb-coding/microsoft-trellis), [TRELLIS.2 notes](https://comfyui-wiki.com/en/models/trellis/trellis-2)
- FFHQ-UV: [paper](https://arxiv.org/pdf/2211.13874), [dataset listing](https://hyper.ai/en/datasets/30113)
- MetaHuman / Unreal EULA: [EULA change log](https://www.unrealengine.com/en-US/eula-change-log/unreal), [CG Channel, June 2025](https://www.cgchannel.com/?p=167592)
- FLUX.1: [licence guide](https://www.promptzone.com/zuzanna_choi/flux-ai-license-key-insights-2g8d), [local model licences](https://localaimaster.com/blog/local-image-model-licenses-commercial-use)
- FaceLift: [repository](https://github.com/weijielyu/FaceLift)
- AI-training clauses in 3D asset licences: [CGTrader discussion](https://ai.cgtrader.com/forum/technical-q-a/does-the-term-non-ai-license-refer-only-to-the-3d-model-or-also-to-rendered-images)

None of this is legal advice. Get counsel to sign off before shipping.
