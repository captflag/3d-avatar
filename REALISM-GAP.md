# Why our avatars look fake, and what each cause costs to fix

Written against a side-by-side with MetaPerson's marketing render: two people in
hoodies and jeans, photoreal, posed. Ours: a photo-faced mannequin in a flat
green shirt.

The honest summary first. **The gap is not a tuning problem.** Ours is a photo
pasted onto a generic body; theirs is a head whose *shape* is reconstructed from
the photo, textured with a lighting-free skin map, wearing garments a human
modelled. Those are different pipelines, and no amount of adjustment turns the
first into the second. Most of what follows is either engine work or art.

Causes are grouped and each is marked:
**[fixed]** done, **[engine]** needs real pipeline work, **[art]** needs assets.

---

## A. The head and face — the biggest single cause

1. **The head's shape was not the person's. [partly fixed — see ADR-001]**
   This is the one that matters most. The skull used to be the same for
   everybody with the photo pasted on, which is why avatars read as "a photo of
   X on a dummy" rather than "X". The competitor deforms a morphable head until
   it matches the face. Ours now does the same with ~270 CC0 MakeHuman face
   targets, explaining about half of each face's deviation from average on the
   test photos. The other half — asymmetry, fine folds — needs a learned
   residual on top of that basis.

2. **The face is a separate surface laid over the skull. [partly fixed]**
   It was a floating patch with a hard rim, no ears and a bald skull behind it.
   The head is now textured from the photo so the surrounding skin matches, and
   the real head supplies ears, jaw and neck. The patch's edge is softer but
   still visible, because the patch and the skull do not share a surface. A
   proper wrap deformation, where the skull's face region is pulled onto the
   fitted landmarks, removes the seam entirely and is the next step.

3. **Projected texture smears across the face. [engine]**
   With the head's features in the wrong places, an orthographic projection puts
   the photo's eyes and mouth where the mesh does not have them. This is a
   symptom of cause 1, not a separate bug: fit the shape and the projection
   lands.

4. **The photo's lighting is baked into the skin. [engine]**
   We paint the photo exactly as shot: its highlights, its shadow under the
   chin, the colour cast of wherever it was taken. Light the avatar from a
   different angle and the face disagrees with the scene, which the eye reads
   instantly as pasted. Real pipelines *delight* the photo into a flat albedo and
   let the renderer put the light back.

5. **No eyes as geometry. [art + engine]**
   Our eyes are painted onto a flat surface. Real eyes are a sphere, a cornea
   with a specular highlight, an iris with depth. Flat painted eyes are one of
   the strongest "dead" signals a face can give. The base mesh has eyeball
   helpers we are not using, because painted eyes and geometric eyes fight.

6. **The expression is whatever the photo caught. [engine]**
   A grin or a squint is baked into both texture and geometry. Theirs
   neutralises to a rest face, which is what you need before animating. We detect
   the expression already; removing it needs the morphable model from cause 1.

7. **No teeth, tongue or mouth interior. [art]**
   An open-mouthed smile paints teeth flat onto the lips surface.

8. **No skin micro-detail. [partly fixed]**
   The body now has a generated skin texture built to the person's own cheek
   colour statistics (LAB mean and spread), with fine grain in a normal map to
   take the plastic sheen off. Real pores need a tiled detail map at many times
   the atlas density, and subsurface scattering does not survive glTF export, so
   both remain open.

## B. Hair

9. **Hair is a coloured skull cap. [art]**
   A shell offset a few millimetres from the scalp with one flat colour and a
   hard lower edge — a swim cap. Real hair is a modelled mesh of cards or strands
   with an alpha-cut silhouette and an anisotropic highlight that travels as the
   head turns. Hair is usually the single most expensive asset on a character,
   and there is no cheap version that passes.

10. **One hairstyle, and it ignores the person's. [art]**
    Everyone gets the same cap. Theirs has a catalogue, and their pipeline can
    also generate a haircut that follows the photo.

11. **No eyebrows, eyelashes or facial hair as geometry. [art]**
    Ours are whatever the photo painted. Theirs are modelled, which is why beards
    catch light at the edge.

## C. Materials and rendering

12. **Everything was lit flat. [fixed]**
    One ambient light and one weak directional meant every surface returned the
    same value from every angle: no falloff down a sleeve, no sheen on hair, no
    sense that skin has depth. There is now an environment map for the materials
    to reflect, a key and a fill, and ACES tone mapping so highlights roll off
    instead of clipping. This was the cheapest real gain available.

13. **One roughness for every surface. [fixed]**
    Skin, cotton, hair and shoe leather had the same finish. They now differ.

14. **No normal, roughness or occlusion maps anywhere. [art]**
    Every material is a flat base colour. Theirs ship a full PBR set per garment,
    which is where fabric weave, stitching and worn edges come from.

15. **No ambient occlusion and no contact shadow. [engine]**
    Nothing darkens where an arm meets a torso or a shoe meets the ground, so the
    figure looks like it is hovering and lit from inside.

## D. Clothing

16. **Garments are slices of the body, not clothes. [art]**
    Each is a copy of the body's own shell pushed out a few millimetres. That is
    why the shirt has the torso's exact silhouette, no collar, no cuffs, no
    placket, no thickness and no folds. A real shirt is modelled with seams,
    hems, a collar stand and simulated drape.

17. **The hems are jagged. [fixed-ish / art]**
    Cutting the shell by height leaves a zig-zag edge along the waist and
    sleeves, clearly visible in the screenshots. The shirt now overlaps the
    waistband to hide the worst of it, but the edge is still a cut, not a hem.

18. **Flat colours, no fabric. [art]**
    No weave, no print, no normal map. Flat-coloured cloth looks like painted
    clay at any lighting quality.

19. **Shoes are a slab. [art]**
    No sole, no laces, no tongue.

## E. Body and proportion

20. **One body for everybody. [engine]**
    Nothing reads height, build, weight or age from the photo. Male and female
    differ only by which outfit we put on them. Theirs shapes the body.

21. **The body has no skin texture of its own. [partly fixed]** Generated
    mottled skin now, matched to the face; still no nail or knuckle detail.
    Hands, neck and forearms are flat colour with no nail, knuckle or vein
    detail.

## F. Pose and hands

22. **The rest pose was a mannequin stance. [fixed]**
    Arms out and fingers splayed is the base mesh's modelling pose, and it was
    the strongest non-facial fake signal. The rig now ships a relaxed rest pose:
    arms brought down to the sides, elbows softened, fingers curled. The figure
    is 6.1 units wide instead of 9.9, and the sleeves deform with the arms, which
    also proves the skinning across garments. Theirs are posed further still,
    with weight on one leg, which is a per-shot pose rather than a rest pose.

23. **No hand detail. [art]**
    Splayed, flat-shaded fingers are a strong fake signal because eyes go to
    hands after faces.

## G. Pipeline

24. **No quality gate.** Nothing checks whether a result is acceptable before it
    reaches a user: a bad photo yields a bad avatar silently.
25. **No per-person variation.** Two different people in the same outfit differ
    only by face texture and skin tone.

---

## What would actually close the gap, in order

1. **A morphable head model and a fitting solve.** Causes 1, 3, 4 and 6 are all
   the same missing component. Nothing else on this list changes the verdict as
   much, and everything in section A stays broken until it exists. This is the
   engine decision the clone plan flagged, and it is still the blocker.
2. **Delighting**, so the skin is an albedo rather than a photograph.
3. **Authored garments** with PBR materials, replacing the body-slice clothes.
4. **Authored hair**, with alpha cards.
5. **Eyes, brows and teeth as geometry.**
6. ~~A relaxed rest pose.~~ Done in this pass.

Items 2 to 5 are an art budget, not a sprint. The competitor's catalogue is
commissioned work, and the reverse-engineering notes show how much of it there
is: about a hundred haircuts, dozens of outfits, per-garment texture sets.

## Second pass: what external advice got right, and where it would mislead us

Advice received recommended four things. Three were sound and are now in; one
would break the product.

**Taken:**

- **Rim light, soft shadows and a contact shadow.** A rim light behind the
  figure lifts the shoulders, hair and jaw off a dark background, and a
  shadow-only ground plane means the feet touch something. The ground is
  transparent except for the shadow, so 2D PNG exports keep a clean alpha.
- **Landmark warping.** The fix for the smeared projection in cause 3. The photo
  is now bent so its eyes, nose, mouth and chin land where the generic skull
  keeps its own, before it is painted on. The skull's features come from the
  base mesh's eyeball and teeth helpers, so no hand annotation was needed.
- **Poisson blending in UV space**, via OpenCV's seamlessClone, so the photo's
  colours meet the synthetic skin with no step. Skin tone is now taken as a
  median in LAB, which keeps the undertone when part of the cheek is shadowed.

**Also done:** eyeballs are back, textured from the same aligned photo, so the
sockets are no longer holes; and garment cuts are snapped flat, so the hem is a
clean line rather than a zig-zag of face edges.

**Not taken, and why it matters:** the advice recommends DECA, EMOCA or FLAME for
head reconstruction. All three are built on FLAME, which is licensed for
non-commercial research only and needs a paid licence from its institute to
ship. This product is meant to be sold, and every component so far was chosen to
avoid exactly that trap. Using one would make the head reconstruction — the
centre of the product — unshippable. The direction is right; those specific
models are not usable here without a commercial licence in hand.

**Not taken yet, but correct:** alpha masks on the body instead of deleting faces
under clothes, authored garments with normal and occlusion maps, and hair cards.
The first is code we can write; the rest are the art budget described above.

## Third pass: a review that measured better than I had

A third review made four claims. Each was checked against the code rather than
taken on trust.

- **"The face sits too high: mouth on the nose, chin on the lips." Correct.**
  Measured at 3 to 5 cm too high on both test photos. Placement by bounding box
  with hand-tuned ratios was to blame, and it had also been feeding the shape
  fit's correspondences. The face is now placed by anatomy — eyes, nose tip,
  mouth and chin aligned to the skull's own by least squares — and lands within
  0.1 to 0.3 cm. `test_rig.py` now guards it. The shape fit improved on one test
  face as a side effect, from 55% to 65% of deviation explained.
- **"Use a dense landmark warp, not a box." Correct, and built** — in 3D rather
  than UV space, because this mesh's face is split across several UV islands and
  a warp drawn in UV space would tear along each seam. Two further faults turned
  up while doing it: the face had only about 130 pixels of the shared texture
  (the head now has a 1024 texture of its own), and Poisson blending was
  dragging the face's colour towards the generated skin (now feathered instead).
- **"The hair is a swim cap." Correct.** The hairline now follows the head:
  above the forehead, down at the temples to the tops of the ears, down to the
  nape at the back. One attempt overshot and covered the brows; it is driven by
  angle around the head now. Still no strands.
- **"Camera FOV is 60–75; add a rim light." Already done** — the camera is 32
  degrees and a rim light was added in the second pass.

Found along the way: the eyes were the photo's eye region painted onto a sphere,
which read as skin. They are generated now, with the person's iris colour, and
sized from anatomy after a first attempt with too small an iris gave a stare.

## What changed in the first pass

- Environment lighting, key and fill, ACES tone mapping, per-surface roughness.
- The head is textured from the photo rather than vertex-coloured, so the whole
  head carries the person's colouring and the face no longer sits on a bald
  mannequin skull. Ears, jaw and neck are real geometry now.
- Shirts overlap the waistband so the cut hem is less obvious.
- A relaxed rest pose: arms down, fingers curled, instead of the modelling stance.

Honest verdict on this pass: the avatar is better lit and better integrated, and
it is still plainly not photoreal. The reason is cause 1, and that is a model,
not a setting.
