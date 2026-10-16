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

4. **The photo's lighting is baked into the skin. [mostly fixed — fifth pass]**
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

6. **The expression is whatever the photo caught. [mostly fixed — seventh pass]**
   A grin or a squint was baked into both texture and geometry. Theirs
   neutralises to a rest face, which is what you need before animating. Ours now
   fits the expression alongside the head shape, builds the mesh without it and
   warps the photo back to a closed, relaxed mouth before baking. Mouth width
   stays with identity; see the seventh pass.

7. **No teeth, tongue or mouth interior. [art]**
   An open-mouthed smile painted teeth flat onto the lips surface. Since the
   seventh pass the mouth is closed, so the teeth no longer show; an avatar that
   opens its mouth still has nothing inside it.

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

## Fourth pass: MetaPerson's own selfies, run through ours

Their two marketing selfies (a bearded man, a woman with short auburn hair) were
run through our pipeline in matching outfits. The cheap gaps, closed:

- **The beard was shaved off.** The head bake fades the photo into generated
  skin across the outer 18% of the face so no edge shows, which is exactly where
  a beard grows. Pixels clearly darker than the person's cheek skin, below the
  nose, now keep full strength through the fade (`facetexture.facial_hair`).
- **Hair read as bald.** Nine pixels above the forehead hit the lit crown, and
  auburn came out as (219, 171, 134), skin colour, on a skull cap. Hair colour
  is now read from a band of hair with backdrop and skin rejected, giving
  (169, 121, 84) for her and (104, 72, 53) for him. It colours MakeHuman's own
  hair cards (CC0) over a darker cap that hides the scalp between strands.
- **One body for both.** `--gender` only changed the outfit. MakeHuman's
  regional macro targets, blended equally as MakeHuman does by default, now
  make a man's or a woman's figure (17.3 and 15.9 units, about 173 and 159 cm)
  before the face is fitted.
- **Clothes were body slices.** Defaults are now MakeHuman system outfits:
  collared, buttoned shirts and jeans with fabric, normal and occlusion maps,
  leather shoes, plus eyebrows and eyelashes. All CC0; outfits with a MakeHuman
  logo print were left out.
- **Mannequin stance.** A relaxed stance is the default: weight on one leg,
  hip and shoulders tilted against each other, arms at the sides, elbows soft.
- **The face patch** turned out to sit behind the shaped skull everywhere but
  the forehead, where it drew a hard edge. It is off by default.

Not closed, and why:

- **Iris colour.** Sampling is better (the whole iris, no lid shadow) but on
  these photos the iris pixels are genuinely grey: LAB b* 126 for the "blue"
  eyes, 128 being neutral. They look blue in the photo by contrast with warm
  skin. Matching that needs a perceptual rule or an eye-colour palette.
- **Pale faces.** A bright studio photo is baked in as it is. Fixed in the
  fifth pass.
- **Hairstyle is not picked from the photo**, and long hair has no asset.
- **No green shirt.** Garment recolouring is not built.

Size went from 1.6 to about 3.5 MB per avatar, almost all of it the outfit and
hair textures (capped at 1024).

## Fifth pass: the photo's lighting taken out of the skin

The face still read as a pale mask: a bright studio photo came out near-white,
its shading doubled with the renderer's, and the neck — generated skin in the
body's shared texture — looked like bark next to it. Now, before anything is
baked (`delight.py`):

- **Lighting fitted and divided out.** MediaPipe gives every landmark a depth,
  so the face is a 3D surface in the camera's frame with a normal per pixel.
  Skin being near one colour, its brightness against that normal is the
  lighting: an ambient term plus one direction, fitted robustly so brows,
  beard, lips and eyes drop out, then divided out so the face is shaded as if
  lit from the front. Second-order lighting was tried and was worse — a face
  shows only half the sphere, and gains hit 11x at the jaw. The correction is
  smoothed to the broad scale and applied at 70%, clamped to 0.75-1.4x,
  because the landmark mesh's normals are rough at the scale of a nostril.
- **Exposure capped.** Real skin reflects at most about half the light on it.
  Cheeks brighter than 0.45 linear luminance are scaled down: the bright test
  photos by 0.64 and 0.74; a well-exposed photo of darker skin is untouched.
- **Shine pulled back.** Specular shine on a forehead or nose is the lamp's
  reflection, not skin, and baked in it added to the renderer's own highlight.
  Pixels brighter than the skin around them are blended back towards it, with
  eyes and mouth excluded so teeth and the whites of the eyes survive.
- **Generated skin matched to the whole face**, not ten cheek points: in a
  smile the cheeks catch the light and read brighter than the forehead, jaw and
  temples the generated skin borders.
- **Pore grain scaled per texture.** The head's texture is its own (a texel is a
  fraction of a millimetre); the body shares one map from neck to toe, where the
  same grain magnified into bark. Normal strength is 0.5 on the head, 0.2 on
  the body.

Hair, brow and iris colour are still read from the photo as shot.

Still open: no white balance (a warm or blue cast stays), no occlusion (the
nostrils and the corners of the eyes stay as dark as photographed), and the
underexposed case is not lifted, since dark skin and a dark photo cannot be
told apart without a reference. The whites of the eyes are a natural one.

## Sixth pass: hair, eyes and clothes read from the photo; occlusion

Worked through the partly open gaps one at a time, each root-caused before it
was fixed.

- **Grey hair came out near-white.** Reproduced on the portrait: hair colour
  (169, 169, 163). Isolated: his face sits near the top of the frame, so the
  sampling band was mostly the bright window behind him, and the backdrop
  filter only knew the photo's corners — red curtain. Fixed by modelling the
  backdrop from every pixel beside the head at the same height: (88, 82, 78).
  The man and woman did not move. Then superseded for the defaults by the next
  item, which reads hair directly.
- **The haircut is chosen from the photo.** MediaPipe's multiclass selfie
  segmenter (Apache-2.0, per its model card; out of scope only for
  surveillance and identity) labels hair, face skin, body skin and clothes.
  From the hair mask: how high it stands above the face, where it ends against
  the chin, whether it covers the forehead, and how much there is. A close crop
  (the portrait) gets `short01`, fuller short hair `short02`, a fringe
  `short03`, jaw-length a bob, below the chin `long01`, and a bald head no hair
  at all. Hair colour now comes from the hair pixels themselves. Colour alone
  was tried first and failed: dark skin and a dark backdrop both passed as dark
  grey hair beside the jaw. The long-hair rule has not been checked against a
  real long-haired photo; none of the test photos has long hair.
- **Blue eyes read grey.** Hypothesis tested: the whites of the eyes measure
  warm in all three photos (a* +6 to +11, b* +8 to +9); against them the man's
  iris moves to (-9, -10), clearly blue. People judge iris colour against the
  white beside it, so the iris is now read relative to the sclera and snapped
  halfway to the nearest natural eye colour (dark brown, brown, hazel, green,
  grey, blue-grey, blue). Results: blue-grey, grey, dark brown.
- **The same correction for skin was rejected.** Applied to the skin, the
  sclera's warmth turned light skin sallow (a* +10 to +4): in small photos it is
  mostly pink lid and vessel bleeding into the white, not a photo cast. The
  colour cast stays open.
- **Outfits can be dyed.** An outfit file is one mesh and one texture; each
  piece of cloth is its own UV island, so islands below the waist and near the
  body's centre line are trousers and the rest is the top. Each part is dyed in
  LAB, scaling lightness and moving hue, so weave, folds, seams and baked
  occlusion survive. The defaults are now MetaPerson's own reference looks:
  green shirt with grey trousers, light blue shirt with navy trousers.
- **Sharper faces from real selfies.** The face fills about 42% of the head
  texture's width; when the photo's face is wider than that in pixels, the head
  is baked at 2048 instead of 1024. Small or upscaled photos stay at 1024.
- **Ambient occlusion in the viewer and studio renders** (GTAO, screen space):
  under the chin, inside the collar, where arms meet the body, in cloth folds.
  `review.html?ao=0` shows the render without it.

Still open from this pass: the hair cards' front edge is a hard, ragged line on
the forehead (alpha-tested cards); jeans authored with a stacked hem pool at the
ankle on a shorter figure; the colour cast; nostril and eye-corner darkness.

## Seventh pass: a resting face from a smiling photo

MetaPerson's avatars have neutral faces; ours kept the photo's smile twice over.
The shape fit read a grin as identity — mouth-angles-up at full strength, a wider
mouth, fuller cheeks — and the texture was baked through the smiling landmarks,
teeth and all. Now (`morph.fit`, `compose.py`):

- **Expression is fitted alongside identity, then left off the mesh.**
  MakeHuman's expression units (CC0, fetched by `fetch_assets.py`, each header
  checked) are extra columns in the same bounded least squares. Each is capped
  at 1.5x the MediaPipe blendshape score for the same movement (smile, jaw
  open, lip raise, brow down, squint, sneer), so a unit the photo does not show
  cannot borrow from identity. Scores under 0.1 count as noise. The three
  regional sculpts of each unit are blended equally, as the body is.
- **Identity cannot argue with a shown expression.** Along an expression's own
  axis the landmarks cannot tell a smile from a mouth shaped that way. Identity
  targets that nearly repeat or reverse a shown unit (|cos| > 0.6 at the
  landmarks) sit the fit out. Without that, identity cancelled the smile's lift
  with mouth-angles-down and the resting face frowned: corners 4.8 mm below
  the lip line on the big grin. Now 1.2 mm below, against 0.7 mm on
  MakeHuman's own face.
- **The photo is warped back to rest.** The fitted expression's displacement is
  subtracted from the landmark sheet the texture is baked through, before it
  is placed on the skull. The lips are then closed, each moving half the gap
  with its thickness kept, and the sampler looks each lip up on its own side of
  the seam. A triangulation across a closed seam blended upper lip into lower
  through the teeth; this took the share of mouth texels that sample the
  photo's teeth from 5.4% to 0.3%.
- **Eyelids measured where they are.** MediaPipe's eyelid landmarks sit about a
  centimetre behind MakeHuman's lids, on a rim the blink and squint units do
  not move, so those units read as doing nothing (0.1 cm or less across all
  landmarks). Expression motion is measured on the front-most skin under each
  landmark instead (0.8 to 1.1 cm), and inner lip landmarks take their outer
  lip's vertex, because a closed mouth has no inner contour of its own.
- **The sidecar records what was removed**: `expression_removed`, unit by unit.

| Photo | Detected | Removed | Corner vs lip line |
|---|---|---|---|
| ref-man | smile 0.64-0.72, chin raise 0.78, brow down 0.48 | smile 0.29, brow down 0.37, chin raise 0.07 | -1.0 -> -1.8 mm |
| ref-woman | smile 0.28-0.40, squint 0.29 | lip stretch 0.22 | +1.5 -> +1.2 mm |
| portrait | smile 0.95, lip raise 0.76, squint 0.74 | smile 0.68, lip raise 0.37, squint 0.37, brow down 0.27 | +4.5 -> -1.2 mm, mouth closed |

`test_rig.py` checks it: MediaPipe's average face, smiling and squinting by
MakeHuman's own units, fits back with 1% taken as identity (87% without the
expression columns) and neutralises to within 1% of average.

Still open:

- **Mouth width stays identity.** MakeHuman's smile widens the mouth about 8 mm
  at full strength; the fitted mouths are 9 to 19 mm wider than MakeHuman's on
  all three photos, slight smile or not, so most of that is a gap between
  MediaPipe's average mouth and real ones rather than the smile. The neutral
  mouth is still wide and the lips full on a broad grin.
- **The slight smile mostly stays.** ref-woman's 0.3-0.4 smile scores did not
  find a corner lift in the landmarks to take off.
- **Squint geometry only.** The squint is taken off the lids, but crow's feet
  and raised cheeks in the photo's texture stay where they were.
- **One tuning set, three photos.** Gain, regularisation and the 0.6 axis
  threshold were set on these three. A wider sweep wants a set of neutral and
  smiling photos of the same people.

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
