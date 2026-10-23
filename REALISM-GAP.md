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

## Eighth pass: the leftovers

- **Teeth on a closed mouth.** The seventh pass closes the lips, but only the
  landmark sheet moved; the sampler fills the gap between the lips by
  interpolating across the photo's open mouth, so the skull's lip seam picked
  up teeth. The opening is now painted over in lip colour (inpainted from the
  lips, pulled to the darker half of their colour) before baking.
- **Colour cast, second attempt.** The whites of the eyes failed as a reference
  (sixth pass). Skin is used instead, only to catch what is clearly wrong: every
  skin tone falls between 40 and 70 degrees of LAB hue (the test photos sit at
  53 to 60), so a face outside that band — a green fluorescent or blue shade
  cast — is rotated to the band's edge with its saturation restored to at least
  14. Natural faces are untouched. On artificial casts it recovers about half
  the error, always in the right direction; it cannot know the exact original.
- **Dark corners.** Under the nose and in the inner eye corners, pixels darker
  than the skin around them are lifted by 60% of the difference, inside those
  landmark zones only, so nostrils stay darker than skin but no longer black,
  and brows, lips and beard are untouched.
- **The hair's hard front edge** is geometric — the cards end on bare
  forehead — so it is fixed on the skin: the hair's lower edge is found at each
  angle round the head and the skin above it is tinted towards a darker shade
  of the hair colour, fading out 3 cm below, the root shadow of a real
  hairline. Distance to the nearest hair vertex was tried first and missed,
  because the cards stand 4 to 5 cm off the forehead. The viewer also renders
  alpha-tested cards with alpha-to-coverage, and the occlusion pass now renders
  multisampled: adding it had quietly removed antialiasing from every edge.
- **"Bunching" jeans** were two things. The white socks that are part of
  MakeHuman's shoe models poked through the trousers as the legs posed; they
  are dropped when trousers reach the ankle. And a men's cut is loose on a
  woman's figure: garments can now be taken in (`"slim": {"bottom": 0.6}`),
  each vertex pulled towards the body point it hangs from.
- **File size.** The head texture is kept opaque (it was PNG only to cut out
  skin under a collar) and recoloured hair is stored as a 256-colour palette:
  2.6 to 2.9 MB per avatar, 3.0 MB for a large selfie (5.3 before).
- **Long hair, tested.** No long-haired photo was available, so our own avatar
  was rendered wearing the long cut, a bob and a fringe, and each render was
  uploaded through the studio as a photo. The pipeline chose long, bob (with
  fringe, correctly, for the side-swept bob) and fringe. That proves the
  segmentation and the rule on clean renders, not on real photos.

Not attempted: a trained head-shape model (gap 9). It needs a synthetic-data
generator and training on a larger GPU than this machine's 6 GB.

## Ninth pass: an outside review, checked claim by claim

A review of `portrait.glb` named five bugs. It had read the expression
session's test builds (`portrait-neutral.glb`, `portrait-smiling.glb`), not the
current file, so each claim was re-checked against the current avatar first.

- **"The smiling photo distorts the closed mouth." Correct, for a reason the
  review did not name.** Teeth were already painted out. But on a broad grin the
  shape fit still took the smile's width as the person's: three mouth-width
  targets sat at their 1.0 limit, with lip volume 0.63 and laugh lines 0.83, so
  the resting mouth came out wide and full. Mouth and cheek shape are now pulled
  towards average in proportion to how much the photo's mouth moves (from a
  smile score of 0.3). The mouth fill also darkens towards its middle, the line
  of closed lips. DECA/SMIRK, which the review suggests, are FLAME-based and not
  licensed for commercial use (ADR-001).
- **"The head is 25% too small; scale it 1.2x." Not borne out.** Head height is
  13.3% of stature (adults: about 13%), head breadth with ears 9.7%; shoulders
  are 2.6 to 3.0 head-breadths. The review's "2.2 to 2.4" is not the adult
  norm. Left as is: a 1.2x head would read as a caricature.
- **"The hair is white."** Partly. The hair is recoloured in its texture, so a
  white colour factor is correct, but the reading was too light for short hair:
  scalp and lit strands lifted the median. Hair colour now comes from the darker
  10-40% of hair pixels: (86, 80, 76) for the portrait instead of (105, 98, 94).
  A buzz cut painted on a skull cap was tried for close crops and rejected — it
  read as a swim cap; the cards in the darker colour read as short greying hair.
  Alpha-to-coverage was already on; the file keeps the standard 0.5 cutoff.
- **"Green spot on the neck."** Correct, and a real bug, but not texture bleed:
  windows were cut in the neck where the shirt's list of covered vertices met a
  collar that sits just behind it on our shaped, posed body, and the collar's
  inside showed through. The neck is now never cut away, and a garment's own
  list is only trusted more than 3 cm from its edges.
- **"Raccoon eyes."** Partly: the crease between brow and lid and the hollow
  under the eye carried the photo's top light. Both are now crevice zones lifted
  towards the surrounding skin, brows excluded. The review's whole-face
  bilateral flattening was not used: it would also lighten brows and beard.
- **"Zombie arms."** The numbers were misread (the forearm turn was 16 degrees,
  not 90, and the 0.96 is a bone length), but the finding was right: the base
  mesh is modelled with forearms reaching 47 degrees forward, and the relaxed
  pose added to it. Forearms now hang 24 degrees forward, the upper arms at 30
  degrees, and the forearm-twist bone turns the palms to face the thighs.

## Tenth pass: a laughing, bearded man in a black shirt (photo-10)

A 236 x 293 photo, the face about 60 pixels across, a full laugh, a thick
beard, a black embroidered kurta and someone standing behind. A third review
named seven faults; each was measured on the build first.

- **"No beard, only paint." Correct.** A full beard baked into the skin read
  as a dirty jaw with no edge. The bearded skin is now also a shell 5 mm off
  the face, cut out by the beard mask with fine noise so its edge is stubble.
  The jaw's underside, which the photo never sees, had been baked as plain
  skin and showed as a pale ridge under the shell; unseen faces now take the
  beard's colour.
- **"An open mouth on a closed rig." Mostly not.** Teeth were already painted
  out. What the laugh did break was the face's width: a laugh pulls the jaw
  outline and cheeks out, and the fit took that as the person's shape. Jaw
  outline points are now trusted less as the mouth moves and where a beard
  hides the jaw (`outline_trust`); head width came down 2% and the
  horizontal head scale from 0.80 to 0.69.
- **"The outfit is hardcoded." Correct.** The top's colour is now read from
  the segmenter's clothes mask, only the piece of cloth under the chin, so the
  person behind does not count: (34, 34, 34) here, and the default shirt is
  dyed to it.
- **"Untinted hair."** Wrong: the hair is recoloured in its texture (mean
  (14, 13, 9) here), so its white colour factor is correct.
- **"Zombie pose." Correct, in three ways the earlier passes missed.** The
  forearm twist of -70 degrees, chosen on renders of straight fingers where
  palm and back look alike, turned the palms up and forward: measured on the
  posed palm normal it faced (-0.56, 0.63, 0.54). It is now -20, palms on the
  thighs. The finger curl turned about the vertical axis, which for fingers
  that hang down and forward is a sideways swing: the hands were paddles. Each
  joint now bends towards the palm (the side the thumb sits), more towards the
  little finger. And the base mesh stands with its ankles twice its hip width
  apart; they are now 1.24 hip widths, forearms 14 degrees forward instead of
  24, so the hands hang at the thighs' sides. The asymmetric knees are the
  contrapposto and stay.
- **Found on the way: square pixels.** The bake copied the nearest photo pixel
  into each texel, so a small photo's face was painted in visible squares. It
  now samples bilinearly.

What is still wrong, and why: the face is a blurred likeness at best. Sixty
pixels of face cannot carry pores, a hairline or a mouth shape, and from a
full laugh the resting mouth is a guess. The forehead under the hair reads
grey where the photo's fade meets the hair's shadow. The real fixes are a
larger, neutral photo, and a learned texture completion for what the photo
does not show; neither is a tuning problem.

## Eleventh pass: FlexAvatar, read for ideas

[FlexAvatar](https://github.com/tobias-kirschstein/flexavatar) (CVPR 2026, TUM)
turns one photo into a 3D Gaussian-splat head. It cannot be used here, as code
or weights: it is CC BY-NC 4.0, it is driven by FLAME through the Pixel3DMM and
SHeaP trackers (ADR-001), it is trained on NeRSemble and Ava256, and it makes
heads only, as splats, not a rigged mesh body. How it works: DINOv2 features of
the photo feed a transformer over a template head's UV map, which decodes
Gaussians, conditioned on FLAME expression codes; a learned dataset token
lets monocular video (which only shows part of a head) and studio captures
(which show all of it) train one model that always completes the head; and a
200-step fit tunes the avatar code and an expression offset to the photo.

Two ideas carry over without the model. Identity apart from expression we
already had: the photo's expression is fitted and warped out before the bake.
The other is its partial supervision: learn only from what a view really shows,
and complete the rest from a prior. Applied to our face, it found four faults:

- **The beard was in the skin's statistics.** The generated skin (neck, ears,
  scalp, body, hands) is built to the face's skin statistics, and on photo-10 a
  third of the face is beard: the body came out at L 150 against the cheeks'
  189. The beard is now left out (170, the face's average skin).
- **The beard was in the lighting fit.** A dark lower face reads as shadow, so
  the fit put the light above (y -0.72) and the correction greyed the beard.
  Without it the light comes out nearly frontal (y -0.36).
- **The forehead stopped short.** The landmarks end a finger's width below the
  hairline and the forehead above was generated skin, faded and tinted: a grey
  band under the hair. Past the landmarks the bake now takes the forehead the
  segmenter marks as skin, mapped by an affine fitted to the upper face, and only
  on surface facing the camera. (Without that last test, a front view crushed the
  scalp onto the hairline's few pixel rows and smeared them across it.)
- **The hairline tint judged hair by geometry.** Hair cards are alpha-cut and
  their transparent tips hang below where the hair visibly ends, so the tint
  painted dark hair colour on forehead that shows bare. Hair now covers what its
  opaque texels cover, and below that the tint leaves the photo's skin alone.

Found on the way, in the 2D export: alpha-to-coverage leaves each covered sample
with the fragment's own alpha, so wherever a cut-out card or the beard shell lay
over skin the PNG was part transparent: 1,007 such pixels inside the head, read
as pale specks and a dotted outline round the beard on a white page. The export
now renders the figure's coverage in plain white and uses it as alpha.

Still open after this pass: the haircut (taken up in the twelfth pass). And a
learned completion prior, FlexAvatar's real strength, would need a
licence-clean multi-view dataset to train on.

## Twelfth pass: a short back and sides

Every stock haircut carries the same volume all round the head, so short hair
with clipped sides, the commonest men's cut, came out as a bowl (short02) or a
skull cap (short04). Rather than fetch community cuts with licences of their own,
`hair_short_sides` is made from short02 (CC0, so the variant is too) by
`garments.short_sides`:

- Round the sides and back, below half-way from the eyes to the crown, the hair
  is laid down on the scalp, never nearer than 0.08 so it stays above the scalp
  layer; it returns to full volume above. The cards' own strands then read as
  clipped hair. Cutting them away was tried first and left bare skin above the
  ears and the scalp layer showing at the back as a glossy cap.
- Heights are taken on the scalp under each hair vertex, not the MHCLO anchors:
  hair hangs from a helper shell that already stands off the head, and drawing
  it to its anchors barely moved it.
- Cards over the ears are cut away, and at the front the cards end at 0.55 of
  the way to the crown, so the forehead shows to its hairline instead of under
  a forward sweep.
- The scalp layer is matte (roughness 0.95) for every cut: at 0.45 it showed as
  a shiny cap wherever cards part.

The photo picks it by a new reading, `sides`: how far hair stands out past the
face's outline at the temples, in face widths. Photo-10 reads 0.08; the
reference man and woman, with full sides, 0.13 and 0.16. Fuller short hair
under 0.11 gets the short back and sides; the close crop (portrait, also 0.08
but only 0.33 high) still gets short01.

## Thirteenth pass: a fourth review, of photo-10

Written against builds from earlier in the session, so each claim was checked
on the current one first.

- **"Charcoal forehead, black ears."** Fixed in the eleventh pass (the beard in
  the skin statistics and the hairline tint on bare forehead). The review's fix,
  clipping the bake 5 px below the forehead landmarks, would bring the band back:
  the landmarks already stop short of the hairline.
- **"Swollen pink mouth." Right about the colour.** On a laugh the fill took the
  reddest pixels round the opening, which are wet inner lip and gum, and the
  resting mouth spread that over full lips. Lip colour now comes from the
  person's own cheek, darker and redder by the average measured on the four
  test photos (L -15, a* +9, b* -4), and the visible lips are pulled halfway to
  it. Their volume stays the template's: a laugh hides the person's own.
- **"Beard and hair should alpha-BLEND."** Not taken. Alpha-to-coverage already
  softens their edges, and blending needs sorting the cards cannot get right;
  the pale outline the review saw was the 2D export's alpha, fixed in the
  eleventh pass.
- **"Bowl cut."** Fixed in the twelfth pass. Hair roughness stays 0.8: at 0.45
  the crown went white, as the second pass found.
- **"Staring eyes." Right.** The model's upper lid sits on the top of the iris.
  The rest pose lowers it 4 degrees (orbicularis03, the lid; the review's
  oculi01 is the brow), which covers the iris's top edge; 7 looked sleepy.
- **"Arms: a 44-degree forward elbow bend."** Misread: the base mesh's forearm
  reaches about 46 degrees forward and that turn brings it back to 14, which
  `check_review_fixes` measures. The quaternions offered would undo it.
  **The thumbs, though, were wrong**: hanging, they pointed across the thighs at
  the fly. The finger curl was too strong for fingers the model already bends
  about 25 degrees, and the thumb is now turned down at its root, worked out on
  the posed skeleton: from 28 to 50 degrees below horizontal.

Seen while checking: the eyebrow cards sit below the photo's own darker brows,
which reads as a double brow. Taken up in the fourteenth pass.

## Fourteenth pass: one brow, not two

The brow cards drew a thin brow at the lid crease under the photo's darker
one. Measured in the front view, the cards' centre line sat this far below
where the photo's brow is darkest on the head (1 mesh unit is about 10 cm):

| | before, right / left | after |
|---|---|---|
| photo-10 | 9.1 / 10.6 mm | within 0.6 mm |
| ref-man | 6.3 / 7.6 mm | within 0.6 mm |
| ref-woman | 5.6 / 6.3 mm | within 0.6 mm |

Of the three suspected causes, one held:

- **The cards hang from the template's brow.** On MediaPipe's average face,
  placed on the head the way every photo is, MakeHuman's cards sit 9 mm below
  the middle of MediaPipe's brow on the male body and 3.6 mm on the female.
  The shape fit cannot see that: it fits how a face differs from the average,
  so an offset the average already has is never in its residual. Its brow
  targets moved the cards with the face (eyebrows-trans-up 0.45 to 0.88) and
  left the template's gap where it was.
- **Not the expression.** Taking the fitted brow-down off moved the photo's
  brows less than 1 mm, and upwards.
- **Not the sampler.** The baked eyes sit at or 0.6 to 5 mm below the skull's
  eyes, never above, so nothing round the brow is mapped high.

And one that was not suspected: MediaPipe's brow contours run high. Halfway
between them, which the first fix aimed at, is the brow's top edge, and the
cards then sat on top of the photo's brow. The hair is darkest 0.6 to 1.05 of
the way down from the upper contour to the lower, depending on the person.

So `facetexture.brow_depth` reads, per brow, how far down its contours the
photo is darkest, and `compose.brows_to_photo` bends each card's centre line
onto that line (quadratics in x, so it takes the photo's arch), then brings it
back in depth to its own height off the skin. Moved straight up, a card sank
into the brow ridge, which bulges above the template's brow; slid along the
skin's tangent, it overshot the curve and stood 3 mm proud. Dropping the
cards when the photo brow is dark enough was not needed: on the brow, the
card's strands read as the brow's own hair. `check_brows` holds the cards to
the line within 1 mm and to their own height off the skin.

The same high contours had set the cards' colour: `brow_colour` sampled round
the landmarks, half of it forehead, and read ref-man's light brown brow as
(198, 147, 121) against skin of (250, 204, 182), so his cards were pale
streaks. It now samples along the darkest line: (166, 116, 92); ref-woman
(182, 137, 112) became (143, 97, 75), photo-10 (87, 49, 27) became (67, 32, 14).

Checked by sending the card vertices back through the bake's own sheet to the
photo: before, they lay on the upper lid; now on the brow, on all three.
Still open: on a light brow the strands render a little lighter than the
baked brow, since the recolour lets a strand reach 1.5 times the tint; and
under the brow the socket stays dark, which is the "raccoon eyes" of the
ninth pass, not a second brow.

## Detail audit: where fine detail falls short of MetaPerson

Ref-man, ref-woman and photo-10 rendered region by region (eyes, mouth, ears,
hair from the front, side and back, hands, collar, buttons, shoes) and set
against MetaPerson's renders of the same two selfies. The file itself, per
part:

| Part | Triangles | Colour map | Other maps |
|---|---|---|---|
| Head | 6,232 | 1024 JPEG, 44-73 KB | the generic pore normal map |
| Body (all skin below the neck) | 20,524 | one 1024 PNG | the same pore normal map |
| Outfit | 16,672 | 1024 JPEG | 1024 normal |
| Hair cut | 2,061 vertices | 1024, 256-colour palette | none |
| Eyes | 280 | 512 JPEG, 3 KB | none |
| Teeth, tongue | none | | |
| Blendshapes | 0 | | |

No part has a roughness, ambient-occlusion or cavity map; skin is roughness 0.5
everywhere.

**Severity 4: faults a viewer sees at once**

1. *A bald patch at the back of the head.* Ref-man's cut has gaps between its
   cards at the back, and the plain brown scalp layer under them reads as a
   hole. Black hair (photo-10) hides it. Fix: give the scalp layer the cut's own
   strand texture.
2. *Hands inside the trousers.* The woman's hands sink into her trouser legs,
   and specks of trouser show through the backs of her hands: the arm pose was
   tuned on the male body and her hips are wider. A toe also shows through a
   shoe sole. Fix: set the arms' spread per body from the hip and trouser
   width, and keep feet inside shoes.
3. *Eyes.* A flat 3 KB iris with an oversized pupil, no ring at the iris edge,
   no fibres, no cornea, no wet line or shadow under the upper lid; the lower
   lashes hang as separate spikes. Eyes are where a viewer looks first. Fix:
   a generated iris (fibres, limbal ring, the photo's colour), a clear cornea
   layer, an eye-shadow shell, denser lower lashes.
4. *Beard edges.* On photo-10 the beard's top edge is a drawn, dotted line
   across the cheek; on ref-man a hard line along the jaw. Inside, the beard is
   the photo's own blur, greenish on a small photo. Fix: a wider, softer edge
   ramp and a generated strand texture tinted to the beard's colour.

**Severity 3: detail clearly below theirs**

5. *Skin.* One pore pattern for everyone, head and body alike; one roughness;
   no cavity shading; no subsurface glow, so it reads as matte plastic. The
   woman's hands are peach against a grey-pale face. Fix: roughness by region
   (oily T-zone, matte cheeks), cavity from the mesh's curvature, a warm
   subsurface approximation in the renderer, body skin matched to the face.
6. *Hair rendering.* 256-colour textures, no normal or flow map, no
   anisotropic highlight, a ragged shelf for a hairline, no stray strands. The
   woman's pixie cut is the men's short02. Fix: full-colour textures, a flow
   map and anisotropic sheen (three.js supports it), a soft fringe row, more
   cuts, women's short styles first.
7. *Mouth.* No teeth, tongue or gums; lips without their fine vertical lines;
   no blendshapes, so the avatar cannot smile or speak, where MetaPerson's can.
   Fix: MakeHuman's CC0 teeth and tongue, and the expression units we already
   fit exported as morph targets.
8. *Clothes.* Buttons are painted smudges, folds are painted, collars and cuffs
   are tubes, cloth has no sheen, and one 1024 map covers a whole outfit. The
   woman wears the men's shirt, jeans and leather shoes. Fix: women's garments
   from the CC0 pack, cloth sheen, 2K garment maps, buttons in the normal map.
9. *Hands.* Smooth, with no nails or knuckle creases, at the body atlas's low
   texel density. Fix: a hand detail normal map and nail tint.

**Severity 2: polish**

10. A seam from ear to jaw where the photo meets generated skin, seen in three
    quarter view (ref-man).
11. Ears grey and opaque-looking; real ears glow warm against light.
12. A stray brow strand floats above the woman's right brow.
13. The render: one front key light. Theirs has a soft wrap light, skin glow
    and a contact shadow.
14. Texture budget: the 2K head only switches on for faces wider than 480 px,
    and the body shares one 1024 map. Selfie-resolution input should always
    get a 2K head and body.

### The four severity-4 faults, fixed

1. *Bald patch: not gaps.* With the scalp layer hidden the back of the head
   was full hair. The layer stands 0.055 off the skin and the cards at the back
   lie closer than that, so it covered them. It dates from before the hairline
   tint, which already colours the scalp under the cards, so it is no longer
   worn under a real haircut. (`synth.py` still adds it to its training
   renders.)
2. *Hands and feet.* `arms_clear` skins the hand and the hip and thigh skin
   into the pose and turns each arm out from the shoulder, 2 degrees at a time,
   until the hand stands 0.12 off the skin: enough for the cloth over it.
   Measured against the skin because garment normals are not reliably outward.
   The woman's hands went from 0.27 inside her thighs to 0.22 clear with 6
   degrees; the men's left arm moves 2. The specks on her hands were trouser
   fabric showing through. Feet: below the lowest point of a shoe's opening
   the foot is enclosed, and its skin is now hidden there; the distance test
   had kept 2,166 of 2,642 foot vertices under the shoes drawn, as they sit
   near the sole's open edges.
3. *Eyes.* The iris is generated with radial fibres, a collarette and a narrow
   dark limbal ring; the pupil is at a daylight size; the white dims and warms
   towards the corners; the upper lid shades the top of the eye. Texture 1024
   instead of 512. The lower lash cards are drawn in to 45% of their length.
4. *Beard edges: the beard shell is gone.* An alpha-cut copy of the bearded
   skin, lifted off the face, drew its edge as a torn line across the cheek and
   down the jaw however it was softened, since alpha cut is all or nothing per
   sample. Faced squarely, the photo's own beard is in the skin already. What
   the shell was for, the underside of the jaw that the photo never sees, is
   now painted into the head texture: each surface turned away from the camera
   takes the beard of the nearest one the camera saw squarely, fading over
   3.5 cm and before the neck seam, in the beard's own colour with strands.

Still there: the seam from ear to jaw (item 10), which on photo-10's profile
reads as a diagonal line between beard and neck.

### Items 10 and 12, fixed

*The seam from ear to jaw* was four faults on top of each other, found by
painting the head's generated skin magenta and the body's cyan:

1. The photo's beard was kept at full strength through the face sheet's soft
   border so it would not fade with the skin, and so it stopped dead on the
   sheet's outline. It now fades over the outer 6% of the face (`KEEP_EDGE`).
   The forehead map that shared that override has its own path.
2. Past the outline, along the side of the jaw towards the ear, nothing carried
   the beard on: the reliable beard inside the sheet lay 5 to 6.5 cm away. The
   sheet's outline, read just inside, now counts as beard too, and the carried
   beard is full strength for its first 4 cm, blended over the eight nearest
   sources so its edge does not streak, and never more than 1.5 cm above them.
3. Below the head's texture the neck is the body's, which never had beard: it
   ended on that seam. The body's neck is painted by the same rule.
4. Head and body are separate parts, and each computed its normals from its own
   triangles, so light broke along the neck seam in angular patches. Normals now
   come from the whole skin. Island edges are bled a few texels so filtering
   does not draw the seam as a pale line.

Not touched: the head's pores are stronger than the body's (0.5 against 0.2),
on purpose, since the body's map is coarser.

*The stray brow strand* was the thin brow card itself. MakeHuman's eyebrow006
is authored for blending; its alpha tops out at 187 of 255, and cut at half it
kept 1,832 texels as a few clumps: a dotted line and a fleck. A cut-out texture
whose alpha never reaches opaque is now stretched to the full range, which
keeps 4,134 texels and draws a continuous thin brow.

### Items 5 to 9 (severity 3)

New CC0 assets from MakeHuman's system pack, each checked as CC0 in the pack's
own index before use: three women's outfits, four pairs of shoes, teeth and a
tongue.

- **Mouth (7).** All 24 MakeHuman expression units (smile, jaw open, blink,
  brow raise, squint, pucker and the rest) are exported as glTF blendshapes on
  every part. Parts hung from the base mesh by MHCLO (brows, lashes, teeth,
  clothes) take each unit through the same vertices and weights they hang
  from, so the brows rise with a raised brow and the lower teeth drop with the
  jaw (its helper moves 3.3 cm). Offsets are stored sparse; all 24 cost about
  190 KB. Teeth and a tongue go in every mouth, set back 5 mm: the model's
  resting lips leave a slit, and teeth 1 to 3 mm behind it drew a white line
  across a closed mouth.
- **Clothes (8).** Women get a women's cut by default (female_elegantsuit01,
  dyed by part like the men's outfit) and ankle boots. A new plain dye takes a
  printed pattern out before dyeing (the blouse is pinstriped), keeping folds
  and occlusion. Clothing keeps its authored 2048 maps: halved to 1024,
  buttons and stitching smeared. Cloth carries KHR_materials_sheen; at 0.35
  it washed navy to slate grey, so it is 0.12. The woman's blouse comes out
  grey because the top colour is read from her photo's grey blazer.
- **Hair (6).** Hair had no normal map. One cut (short02) ships one its own
  material file never names, now picked up; for the other six one is derived
  from the texture's strand brightness (`garments.strand_normals`), with about
  the same relief. Hair carries an anisotropic
  highlight (KHR_materials_anisotropy) along the strands, which run along the
  texture's v axis. Kept as a 256-colour palette: on a single-hue texture it
  loses nothing visible. Not done: a women's short cut; the pack has none
  short and side-swept.
- **Skin (5).** Creases shaded from the mesh's own shape (ears, nostrils, eye
  corners, knuckles), measured against each neighbourhood's curvature so
  convex fingers still show their creases. Roughness by region on the face
  (T-zone 0.45, lips 0.35, cheeks 0.6) as a roughness map, and skin's own
  index of refraction (1.4, not glTF's 1.5). The woman's face and hands
  measured the same colour (L 183 and 184); the peach-against-grey of the
  audit was the old plastic reflection.
- **Hands (9).** Fingernails found from the rig: the back of each finger's
  last joint past 30 to 45% of its length, measured square to the bone (the
  model's fingers are curled), 30 to 45 vertices each; painted lighter and
  pinker with a fold, roughness 0.3. They are small on screen: the hands get
  little of the shared 1024 body texture (about 8 by 8 texels a nail), item 14.

Files grew from about 2.7 MB to 3.2 to 4.2 MB, mostly the 2048 clothing maps.

### Items 11, 13 and 14 (severity 2)

- **Texture budget (14).** Head and body skin are always 2048 (`SKIN_SIZE`).
  The head used to drop to 1024 for small photos on the grounds that the
  photo had nothing more to give, but ears, scalp, neck, the carried beard and
  the hairline are generated and were soft regardless; the body's one map runs
  neck to toe, where a nail was 8 texels wide. The pore normal map stays 1024:
  it is noise, and at 2048 it cost 885 KB for nothing visible. Body pores 0.25
  (from 0.2), since its texels are now half the size. Files are 3.8 to 4.8 MB.
- **Ears (11).** No vertex group marks the ears, but MakeHuman's own ear-shape
  targets move the ears and nothing else, so their reach is the mask. Where
  the photo does not cover them, ears are tinted redder and a little darker
  (LAB L -4, a* +9, b* +3): thin, with blood showing through. Under the old
  rim light they read grey.
- **Lighting (13).** The render was a strong warm key (1.7), a weak cool fill
  (0.4, a quarter of the key) and a strong rim (1.4): faces split into a lit
  half and a grey half, and ears and jaw got a cold white edge. Now the key is
  1.35 and more frontal (about 30 degrees off the nose), the fill 0.75 from
  the other side and in front, a hemisphere light adds soft wrap from above
  with a warm bounce from below, and the rim is 1.0. This changes the studio
  and review renders only; the GLB carries no lights.

Of the audit's 14 items, only the woman's hair cut (a women's short,
side-swept style is not in the CC0 pack) and the pale wedges at the sides of
her blouse (baked into its texture by the armholes) remain.

### The leftovers, and the bigger gaps

Fixed:

- *Pale wedges on the blouse:* not the texture. Hidden, the body showed they
  were skin through the cloth. Skin is hidden under clothes except in a band
  along a garment's open edges, so a gap at a cuff shows skin, not a hole; but
  a garment sewn from pieces has open edges along every seam too, and the
  blouse's armhole seams kept skin showing. `garments.open_rim` now counts an
  edge as an opening only if no other piece's edge runs within 3 mm: the
  blouse had 709 open-edge vertices, 306 of them real openings. The plain dye
  also now covers texels just past each piece's faces (a cuff's stripes lay
  there), pads the background round each piece with the dye so filtering
  reads no old print, and halves the baked occlusion contrast.
- *Hairline:* the tint's fade breaks into fine scattered strands instead of
  a smooth shadow under the cards' hard edge.
- *Skin glow:* glTF has no subsurface scattering; a faint red-orange sheen on
  skin (0.20, 0.07, 0.04) gives the warm edge glow at grazing angles.
- *File size:* 4.8 to 3.8 MB for the men, 3.8 to 2.9 MB for the woman.
  Normals as signed bytes and UVs as unsigned shorts (KHR_mesh_quantization,
  now a required extension; three.js, Blender and Unity read it): about 390
  KB. Cut-out textures as a 255-colour dithered palette plus one transparent
  slot: the body's skin 947 to 432 KB at 1.4 delta-E as seen. Teeth and tongue
  textures 512.
- *Photo advice:* each avatar's info carries `advice`, shown in the studio's
  status line, when the face is under 250 px across or the photo smiles
  broadly (photo-10 gets both; ref-man the smile; ref-woman none).

Tried and not kept:

- *Buttons:* a detector for buttons printed on garment textures (small,
  compact blobs off the cloth's colour) could not tell them from fragments of
  stripes, which also cluster at the pieces' edges where the plackets are.
  Buttons need to be marked by hand per garment, or modelled.
- *Women's short cut:* bob01 and bob02 trimmed with `short_sides`, fringe cut
  above the brows. From the front the fringe version came close to the
  woman's side-swept style; from the side it was a hard-edged bowl over a
  bare band above the ear. short02 stays. This needs an authored hair asset.
- *Mouth from a smiling photo:* letting the fit read upper-lip fullness however
  wide the smile, on the idea that a smile stretches the corners but not the
  upper lip's height, pushed photo-10's upper lip to its fullest setting when
  his real lip is stretched thin over his teeth. The laugh still leaks in.
  Reverted; the mouth stays near average on a broad smile, and the photo
  advice asks for a relaxed one.
- *Upscaling small photos:* MediaPipe reads a face at a fixed internal size,
  so landmarks wandered the same under 1-pixel and 3% jitter at 1, 2, 3 and 4
  times (4.8% of the face's width on photo-10). Nothing gained.

Still open: likeness itself. Closer needs larger, relaxed photos (now asked
for) or a learned prior trained on licence-clean data. That said, the next
section found the shape fit had been leaving likeness on the table.

## Why the faces didn't look like their photos

Measured, not judged by eye. MediaPipe read each photo and each render, and the
distances a face is recognised by were compared as ratios to eye spacing, with
the fitted mesh checked between the two.

- *Shape: the fit measured from the wrong face.* It fitted how the photo
  differs from MediaPipe's average face, then applied that to MakeHuman's
  base, which is not that average. Its chin sits 6% of the eye spacing higher
  and its jaw corners 5% further in, so every avatar inherited a short chin
  and a narrow jaw. On top of that, the few landmarks that set those
  distances were outvoted by the other 460. No target was at its limit: the
  basis could reach the face, but the fit didn't aim for it. Every result came
  out with eyes 4–6% too far apart and a chin 5–11% short, which reads as a
  child's face. Now the fit measures from the skull's own landmarks in the
  picture's plane (depth still from the average), and fits eye spacing, face
  and jaw width, face height and nose to chin as distances as well
  (`morph.reference`, `morph.PROPORTIONS`). The fitted mesh hits all five
  within 0.6% on the three test faces. Read back off the renders, nose to chin
  went from −8% to +1%, the woman's face height from −11% to −5%, and jaws
  improved by 2–4 points.
- *A beard pulls the whole lower face out.* MediaPipe draws the jowls on the
  beard as well as the outline. Measured from the skull, those fully trusted
  landmarks widened the jaw 8% past the beard. Landmarks on facial hair now
  count less too, on bearded faces only (shadow under a clean-shaven chin
  reads as hair otherwise).
- *The camera doubles the photo's perspective.* The head render stands about
  four head-depths away at 32°, so the face's sides, behind the eyes, shrink by
  about 4%. The photo's landmarks already carry its own perspective. The head render
  now uses 8° from four times the distance (same framing), and reads 3–4%
  wider at face and jaw. Full-body renders keep 32°.
- *The skin loses half its colour in the render.* The baked texture keeps the
  photo's chroma: forehead a*/b* 7.8/13.5 against the photo's 8.4/15.3. The
  render measures 3.5/6.8. Three.js's ACES curve, applied to the texture at
  about 1.5× light, gives exactly that (86.3, 2.0, 6.0). Key, fill, hemisphere
  and environment add up to over-exposure, and ACES desaturates what it
  compresses. Measured over forehead, cheeks and nose, at 100/75/60% light:
  ACES at best dE 12, AgX 16, Khronos PBR Neutral 4.7 at today's light, with
  chroma ×1.02 of the photo. Dimming only darkened every curve. Neutral is
  now the tone mapping. On the three test renders, skin went from dE 12–18 to
  3–8, and chroma from ×0.5 to ×0.96–1.06.
- *Also:* the photo's smile is dropped for a neutral face. Hair is a template
  cut. The beard is paint with no volume, and its ginger is lost to the same
  tone curve. Eyes are within 10% in shape but generated.

## Brows and the hairline band

- *The brows were drowned by an eyeshadow band, not missing.* The photo's own
  brows are baked into the head texture with about the photo's contrast. But
  the skin between brow and upper lid came out 53–60 L* darker than the
  forehead in the render, against 16–25 in the photos and in the unlit
  texture. Ambient occlusion, the key light's shadow, the normal map and sheen
  each changed it by under 3 L*. The cause is the shape: MakeHuman's eye socket
  is deep, its skin under the brow faces the floor, and every studio light
  comes from above, so the photo's own under-brow shading was shaded a second
  time. Under the brows only (`appearance.under_brow`, fading out by a fifth of
  the eye spacing), the normals now keep the face's broad form
  (`broad_normals`, 120 rounds). Smoothing the whole photo area was tried
  first and tipped cheeks and chin towards the neck. The band is now 36–37
  L* for the woman and ref-man and 19 for photo-10. Brow contrast came out at
  30/26/50 against the photos' 32/19/52. The eyes also read 2–3% less
  oversized, since a dark socket had made them look bigger.
- *The brow cards are off by default.* MakeHuman's brow cards drew thin
  strands along the brow's top edge: pale streaks on a fair brow, stray dark
  dashes on the others. With the band gone, the photo's painted brows read as
  full brows. The cards remain in the catalogue as wearable items.
- *The hairline band was strands painted sideways, and too far down.*
  `hairline_tint` breaks its fade into strands by stretching noise down the
  texture. The head's UV island lies on its side, so the strands ran across
  the forehead as dashes. They now run along whichever texture axis climbs
  the head, read from the forehead's own UVs. The skin also counted as "under
  the hair" 6 mm past the hair's visible edge, which painted a band of strands
  down the photo's own forehead. At 1.5 mm (`HAIR_COVER_REACH`), the dark
  share of the strip just under the edge fell from 26/33/15% to 10/10/10%, and
  no skin shows between the cards.

## The photo's expression, and the eyes

- *The avatar shows the face the photo was making.* MediaPipe scores every
  photo under the same 52 names the avatar's blendshapes carry, but those
  scores can't be used as weights directly. MediaPipe scores a neutral avatar
  above zero on some shapes (browOuterUp 0.26–0.45, squint 0.13–0.28), and its
  response to our shapes is far from straight (a smile scores 0.35 at a
  quarter weight and 0.81 at full). Taken raw, the woman's faint smile came
  out a broad one. So each shape was rendered at five weights on two neutral
  avatars and read back, and a photo's score now goes back through that curve
  (`morph.ARKIT_RESPONSE`). Twenty-three shapes MediaPipe hardly reads on an
  avatar (frown, press, brow down, sneer …) take the score as it is.

  The eyes are left to the shape fit. Gaze and blink readings on photos were
  MediaPipe's own lean, and squint moved the eye landmarks further from the
  photo on all three faces.

  Against the photos, with the expression on: ref-man's landmarks are off
  2.3 → 2.0% of the eye spacing (mouth 3.4 → 2.0), the woman's are about
  even (2.9 → 3.0, her smile now 0.41/0.33 against the photo's 0.40/0.28), and
  photo-10's mouth is 5.0 → 3.6. Photo-10's MediaPipe scores move further from
  the photo's (0.071 → 0.087), because his "neutral" face already carries the
  laugh the shape fit let in. The file's own weights stay at rest; the studio
  applies `photoExpression`.
- *The eyes match the photos by every measure we have.* Iris size against the
  eye's width (0.44/0.46/0.42 against 0.45/0.46/0.42), the opening, and the
  white above and below the iris, read from the pixels: none above, in render
  or photo. MediaPipe's iris landmarks had suggested the iris sat high, but a
  3° downward turn of the eyeballs (0.008 units at the iris) did not move them
  at all, so they track the lid contour on a render, not the iris. Reverted.

## The laugh in the neutral face, hair colour, light brows

- *The laugh was in the paint, not the shape.* photo-10's neutral avatar
  read as smiling (MediaPipe 0.63/0.55), but with the head painted one flat
  colour its shape read 0.03: the photo's grin was baked onto a closed mouth.
  The bake's landmark sheet is relaxed by the fitted expression, which cannot
  reach a broad laugh, so its mouth stayed wider than the mesh's (15% on
  photo-10, 12% even on ref-man), and the grin's corners were painted out on
  the cheeks. Now the sheet's lip line is moved onto the mesh's own before the
  bake (`compose.mouth_onto_mesh`): a wide falloff for the skin round the mouth,
  then a narrow, damped Gaussian radial-basis fit that lands each lip point.
  An average alone left a third of a laugh's width. Neutral faces now read
  0.04 (photo-10) and 0.00 (ref-man), and the photo's expression is no longer
  doubled. The pale patch in photo-10's chin beard is his own: the photo's
  chin centre is 21 L* lighter than the beard beside it.

  Cost: the expression view's mouth is narrower than a broad photo smile.
  MakeHuman's smile unit widens the mouth less than a real one does, and the
  old leak had widened it for free (ref-man's mouth landmarks 2.0 -> 3.6% of the
  eye spacing with the expression on). MediaPipe's response to our shapes was
  measured again after this and the lighting change (it depends on both). One
  curve averaged over two faces now overshoots the woman's faint smile
  (0.56/0.52 against 0.40/0.28) and fits the men's.
- *Hair colour read from the body of the hair.* The base colour came from
  the darkest band of the photo's hair (10-40%): rendered hair was 7 L* darker
  than the photo's, and the woman's strawberry blonde, whose shadows are
  redder, came out copper (hue 55 against 60). Read from 25-60%, and the dye
  given a sixth less saturation (the dyed texture adds that), the woman's
  hair renders L* 49.8 / chroma 29.5 / hue 61 against the photo's 48.6 / 27.8 /
  60, ref-man's 35.3 / 20.8 / 58 against 36.5 / 20.8 / 59. photo-10's black
  hair stays lighter than his photo (13.7 against 8.2): its own highlights.
- *Light brows, and the eye sockets, were the room's light.* The brow paint
  is faithful: ref-man's brow is 0.80 of the forehead's brightness in the
  texture and in the photo. Lit, it fell to 0.72. Roughness, occlusion, sheen
  and the normals each changed it by under 1 L*; switching off the
  environment light brought it to the photo's. The room is lit from the
  ceiling, which lights the upturned forehead far more than the brow and the
  sockets. Tipping it 30 degrees towards the face (`environmentRotation`)
  brings ref-man's brows from 23.5 to 15.3 L* under the forehead (photo
  17.9), the under-brow band from 38 to 24 (photo 16) and the woman's from 39
  to 28 (photo 25), and skin colour from dE 5.1 to 3.8. Tipped further it
  matched better still but flattened nose and lips into a pasted-on look.
  photo-10's harsh-lit photo fit the overhead room better.

## A wider smile, and a response curve over three faces

- *The smile now widens the mouth.* MakeHuman's corner puller alone widened
  it 3-12% at full weight; ref-man's smiling photo is 20% wider than his
  neutral mouth and photo-10's laugh 33%. `mouthSmile` now takes the
  retraction unit too (corners drawn outwards): 8-21% at full weight.
- *MediaPipe reads the same smile differently per face.* At equal weight both
  faces' corners rise the same (9.4% of the eye spacing at full), yet a quarter
  weight scored 0.44 on the woman and 0.09 on ref-man. One curve cannot fit
  every face; it is now averaged over all three test avatars (photo-10's
  neutral is neutral since the mouth fix). With the expression on, against the
  photos: the woman's smile reads 0.47/0.46 (photo 0.40/0.28, was 0.56/0.52), her
  mouth landmarks 2.2% of the eye spacing (best yet); ref-man's 0.62/0.72
  (photo 0.64/0.72), mouth 3.6 -> 2.9; photo-10's 0.84/0.84 (photo 0.86/0.80). His
  laughing mouth stays narrower than the photo's (0.67 against 0.80 of the
  eye spacing): an open laugh is more than a smile.

## Tried and set aside: outline correction, a curved short-cut hairline, 16-bit blendshapes

- *The last face-shape gaps are not in the fit.* Read off the renders, the
  woman's face is 4.6% narrow and 3-4% short, ref-man's 4% short, photo-10's jaw
  7% narrow. But the mesh's own landmark vertices match the photo to 0.4% (face
  width 1.597 against 1.591 of the eye spacing, jaw exactly). MediaPipe itself
  wanders: the same photo mirrored reads up to 2.6% different. Hiding the hair
  takes ref-man's face height from -3.5% to -1.4%: the template fringe hides
  the hairline that landmark 10 sits on. What is left is not systematic (face
  width -4.5, +0.2, -3.3% on the three faces with the hair hidden), so a fixed
  correction table would mend one face and break another. Closing it needs a
  render in the loop: render, read with MediaPipe, adjust, render again.
  photo-10's jaw is his beard's outline, which the fit rightly trusts little;
  that is beard volume's to fix.
- *A curved hairline for the short-sides cut* (lowest mid-forehead, rising at
  the temples, running down to the sideburn) only made the fringe a narrower
  tab with the same straight edge. Any cut through hair cards leaves a hard
  geometric edge; a natural one needs strands that thin out (vertex alpha
  feathered under alpha-to-coverage) or an authored cut. Reverted.
- *Blendshapes as normalized 16-bit integers* (KHR_mesh_quantization allows
  SHORT normalized for morph POSITION; min/max then in integers) cut a file
  8% (3323 -> 3060 KB), and three.js r162 rendered it exploded: it reads
  normalized sparse values about 32767 times too large. Valid by the spec,
  broken in the viewer we ship with; reverted. Revisit with dense accessors or
  a later three.js.
- *Feathering the hair's lower edge with vertex alpha* (COLOR_0, fading over
  the last 0.1 above the edge, under alpha-to-coverage) moves the edge, it
  does not soften it. three.js turns alpha-to-coverage into a one-pixel ramp at
  the 0.5 cutoff, and these cards are mostly opaque, so texture × vertex alpha
  crosses 0.5 on one smooth line: a hard edge, set back, over the painted
  band beneath (dark band under the edge 6.2 -> 8.5% on the woman, 7.6 -> 12.3%
  on ref-man). Reverted. Strand-by-strand thinning needs alpha that varies
  along the strands (texture) or blended hair with sorting.
- *The bigger hairline gap is where the hair ends, not how.* In all three
  photos the hair ends 0.57-0.66 eye spacings above the brows; on the avatars
  0.23-0.33. The template fringe covers a third of an eye spacing of forehead
  the person shows, which is also part of why face height reads short.

## The fringe swept back to the photo's hairline

`read_hair` now reads the photo's hairline (the lowest hair above the brows,
mid-forehead, in eye spacings), and `garments.sweep_back` turns the haircut's
front up and back about the cranium's centre until its visible edge sits that
high on the shaped head: no cards cut, the turn easing off towards the sides
(30-75 degrees round) so the back does not move. Turned on a circle, the hair
stood off the flatter forehead and showed its dark inside from the side;
each moved point is now set back to its old distance from the scalp. Against
the photos:

| | hair ends above the brows (photo / before / after) | face height (before -> after) |
|---|---|---|
| woman | 0.65 / 0.24 / 0.53 | -3.6% -> -2.6% |
| ref-man | 0.58 / 0.25 / 0.55 | -4.0% -> -1.9% |
| photo-10 | 0.58 / 0.34 / 0.46 | -2.6% -> -0.7% |

Whole-face landmark error 3.0 -> 2.7 (woman), 2.7 -> 2.5 (ref-man), photo-10
unchanged at 3.5% of the eye spacing. photo-10's short-sides block still ends
in a straight edge: that cut's own front, not the sweep.

## Hair: the photo's height, not a different stock cut

Every short cut in the catalogue was built on all three test faces, rendered,
and its hair lined up with the photo's by the stable face landmarks; the
score is the overlap of the two (intersection over union) above the eyes.
No stock cut came close (0.41-0.46 at best), and they all missed the same
way: height. In the photos the hair stands 1.20, 1.53 and 1.18 eye spacings
above the brows; every MakeHuman cut lies flat at 0.90-1.17. Width at the
temples was already right (short02 2.01 against 2.08), and the hairline was
fixed by the sweep.

`read_hair` now also reads the `rise` (the top of the hair over the
forehead's middle, a low percentile so flyaways do not set it), and
`garments.volumise` lifts the cut along the scalp's normals until its top
reaches that height: fully over the top and front, easing off towards the
eyes, sides and back, and from nothing at the front edge, so the hair rises
from the hairline as a quiff does. Lifted right at the edge it stood off the
forehead as a shell, its dark inside showing under the fringe; over a 0.3
ramp the painted scalp still showed as a band under ref-man's quiff; 0.6
closes it.

| | overlap with the photo's hair (stock -> now) | top above the brows (photo / now) |
|---|---|---|
| woman | 0.44 -> 0.53 | 1.20 / 1.19 |
| ref-man | 0.34 -> 0.62 | 1.53 / 1.52 |
| photo-10 | 0.37 -> 0.67 | 1.18 / 1.15 |

Still open: the cut's own shape (a side parting, a swept fringe, a curl) is
the stock card layout; ref-man's temples show small notches at 45 degrees;
photo-10's short-sides front keeps its straight edge.

## Depth from turned photos

One photo gives each landmark's place across the picture and only a guess at
its depth, so the fit trusts depth at a third and profiles come out average.
`views.fuse` takes the front photo and one or two turned 30-45 degrees and
solves each landmark's 3D position from all of them: image x and y count
fully, MediaPipe's depth guess a tenth, a view only as far as the landmark
faces it. Each side view's turn is searched round MediaPipe's reading, which
read a 35 degree turn as 28.6 and so made every depth a fifth too deep. The fit
then trusts depth as far as a side view saw it, except at the lips: read off a
painted lip line, their fused depth stood a centimetre proud.

The fit also needed a free shift in depth. Placement sets depth by the nose tip
alone, so a measured face whose nose stands out reads as one whose every other
landmark sits back, and with depth trusted, the targets bent the whole face
forward to match: side views first made the profile worse than one photo.

Measured on a ground truth: ref-man's avatar with a stronger nose, chin and
mouth added, rendered front, 35 degrees each way, rebuilt from the renders,
heads compared vertex for vertex (same topology), aligned on the back of the
head. Depth error over the face front, in mm:

| rebuilt from | RMS | nose | mouth | chin |
|---|---|---|---|---|
| (ref-man without the added profile) | 6.9 | -10.9 | -9.1 | -3.8 |
| the front photo | 4.7 | -7.2 | -6.0 | +0.2 |
| front + both sides, before the depth shift | 7.5 | +0.8 | +3.4 | +5.7 |
| front + one side | 3.4 | -5.9 | -4.1 | -0.2 |
| front + both sides | 3.5 | -3.5 | -2.1 | +0.5 |
| front + both sides, lips trusted too | 4.4 | -3.8 | -0.8 | +2.4 |

The nose still comes out 3-4 mm short; why is not yet measured.
Not yet tried on real turned photos of a person, only on renders.

## Turned photos: painting, angle, and what did not help

**The sides painted from the turned photos** (`views.paint_sides`). The front
photo reaches the face's outline and no further; the cheeks' sides and the jaw
towards the ears were generated skin. `fuse` now returns each turned photo's
camera, the shaped head is projected into it, and each texel the front photo
did not cover takes the turned photo's colour as far as it faces that camera
squarely and that photo's segmentation calls it skin, colour-matched to the
front on the cheek both see. Not the ears: no landmark finds them, and a few
millimetres' misfit painted the photo's ear half beside the mesh's, as a pink
blotch on the rendered test head. Checked on renders only; real turned photos
of a person are the test still to run.

**Wider turns, not more of them.** On the rendered test head (depth error over
the face front, mm):

| turned views | RMS | nose | mouth | chin |
|---|---|---|---|---|
| none | 4.7 | -7.3 | -6.0 | +0.2 |
| +-20 degrees | 5.4 | +1.0 | +2.4 | +5.3 |
| +-35 degrees | 3.6 | -3.7 | -2.1 | +0.6 |
| +-50 degrees | 2.4 | -0.6 | -1.8 | +0.5 |
| +-20, 35 and 50 together | 4.8 | -9.4 | -5.9 | -1.0 |

A slight turn shows depth as a small sideways shift, which MediaPipe's own
average face outweighs, and it drags the fusion back towards average.
Weighting views by how far they turn (sin squared) rescued the six-view set
(3.3) but cost the pairs (+-50: 2.8): not kept. So the camera does not upload
a video's worth of frames: it watches the turn and keeps the widest frame on
each side that passes (MediaPipe reads true turns of 20, 35 and 50 degrees as
about 20, 29 and 42; it now asks for a reading of 30-50, taking it at 40).

**A free-form correction after the target fit: lips only** (`morph.correct`).
What MakeHuman's targets cannot reach is spread from the landmarks over the
mesh by a damped Gaussian radial basis. Tried everywhere, it made things worse:
- in depth, from the turned views, the temples sank up to a centimetre and the
  chin stood 8 mm proud (depth RMS 3.5 -> 6.4). MediaPipe's outline landmarks
  follow the silhouette as the head turns, so their fused depth is 5-6% of the
  eye spacing too deep; even kept to the face's midline it lost (4.9).
- across the picture, the mouth came closer to the photo but the nose further
  on all four test faces, and two of four were worse overall: MediaPipe's nose
  points do not sit on the mesh's, and the outline's slide.
On the lips alone, landmark error on the head renders against the photos
(mouth / whole face, % of eye spacing):

| | before | lips corrected |
|---|---|---|
| ref-man | 4.06 / 2.54 | 2.70 / 2.56 |
| ref-woman | 3.14 / 3.85 | 2.57 / 3.76 |
| photo-10 | 5.64 / 3.89 | 2.44 / 3.26 |
| photo-14 | 5.40 / 3.51 | 3.86 / 3.41 |

The nose reads a little further off (ref-man 1.8 -> 2.4): partly the lip pull
reaching the nose's base, partly re-detection noise. Depth on the test head is
unchanged (3.6).

**The haircut widened to the photo's outline** (`garments.widen`). With the
quiff, the hair still stood narrower over the temples than the photos': on
ref-man and photo-14 by 0.13-0.19 eye spacings at 0.5-0.75 above the brows.
`read_hair` reads the hair's half-width at four heights; the cut's sides are
pushed out along the scalp's normals until its outline has the photo's shape
(each height against the width at the brows, since faded strand tips make a
render 0-10% narrower than its vertices), at most 20% wider, never pulled in.
Hair overlap with the photo above the eyes, quiff only -> widened: ref-man 0.66
-> 0.73, ref-woman 0.59 -> 0.60, photo-10 0.72 -> 0.73, photo-14 0.56 -> 0.71.
photo-14 now runs a little wide low on the temples (1.16 against 1.02).

**SigLIP (Apache-2.0) to choose the haircut: not used.** On six photos it
picked the right cut 4 times to the rules' 5, and beard styles 2 of 6; its raw
scores carry no usable confidence, and it brings about 800 MB of torch.

**Depth Anything V2 Small (Apache-2.0; Base and Large are CC-BY-NC) for
one photo's depth: not used.** Its relative depth, sampled at the landmarks and
scaled to MediaPipe's, against the rendered test head's true depth (in front
of the eye corners, % of eye spacing, RMS over all landmarks / the central face):

| | all | central | nose | subnasale | chin 152 | jaw 172 |
|---|---|---|---|---|---|---|
| MediaPipe's own guess | 5.9 | 6.0 | -4.4 | -8.0 | -15.9 | +6.0 |
| Depth Anything | 11.1 | 8.6 | -4.1 | +8.4 | +8.8 | +35.1 |
| Depth Anything, scaled on the central face | 13.0 | 10.3 | -16.1 | -3.2 | +1.5 | |
| half and half on the central face | 7.1 | 7.2 | -10.2 | -5.6 | -7.2 | |

It flattens the face's sides (the jaw corner a quarter as deep as it is) and
puts the chin forward where MediaPipe puts it back; no mix of the two beat
MediaPipe alone. One caveat: the test head is a render, and a model trained on
photos may do better on photos, which have no ground truth here. Turned photos
(above) remain the way to measured depth.

**Scoring by a face-recognition model** (`score.py`): AWS Rekognition
CompareFaces, photo against head render, 0-100. Wired up; no scores yet, for
want of credentials.

## Combined: jaw fit, skin exposure from the tone curve, stubble

The jaw fit (below) is merged with the forehead/stubble work and the
segmentation measurements. That work changes two things: delight.py sets
exposure so the cheeks render at the photo's brightness under the studio's
tone curve (`delight.rendered`, in place of the fixed 1.35), and
`appearance.stubble` adds hair-sized grain where the de-lit photo is darker
than the cheeks below the nose (nothing below jaw_beard 0.15). Built
together:

Landmark error at the photo's pose (whole face):

| face | start of round | jaw fit | combined |
|---|---|---|---|
| ref-man | 1.99 | 1.88 | 1.83 |
| ref-woman | 1.83 | 1.78 | 1.81 |
| photo-10 | 2.39 | 2.24 | 2.18 |
| photo-14 | 1.36 | 1.27 | 1.26 |
| photo-16 | 1.39 | 1.27 | 1.29 |

Brightness, render minus photo, L* (0-255) at forehead / cheeks / chin, on
level renders (the default):

| face | before | combined |
|---|---|---|
| ref-man | -3 / -4 / +34 | -3 / -4 / +28 |
| ref-woman | 0 / +8 / +35 | -4 / +1 / +27 |
| photo-10 | -1 / +9 / +26 | -5 / +1 / +16 |
| photo-14 | +22 / +10 / +23 | +7 / -13 / -3 |
| photo-16 | +24 / +11 / +25 | +9 / -12 / -2 |

On photo-14/16 the forehead and chin come close, but the cheeks swing from
11 too light to 12 too dark. At the photo's pose those two read +21 / -19 /
-14 (from +29 / +4 / +13): the studio's lights are more top-down than the
photo's, and pitching the head moves the light across it. Lighting that
matches the photo's direction is the open item.

## The jaw as the camera sees it; brows, pose and hair colour tried

Measured at each photo's head pose (the studio's "Photo's head pose"),
landmark error in % of the eye spacing, whole face and jaw.

**The jaw fitted as a silhouette (`morph.fit_outline`), kept.** The fitted
head (without the neck: faces bound mostly to head and jaw bones,
`morph.head_only`) is turned by the photo's rotation and drawn. Each of the
17 jaw landmarks is met along the ray from the nose tip. The silhouette
vertex there is moved by the gap in the picture's plane and the move is spread
like `correct`'s (reach 0.25 eye spacings). This repeats three times. On
photo-16 the gaps go from up to 4.1% to 0.4% or less (one point 1.1).
Depth moves are left out: with them, ref-woman's mouth read 0.1 worse.

| face | before | jaw fit (beard landmarks at 0.15) | jaw fit (full trust) |
|---|---|---|---|
| ref-man | 1.99 / jaw 3.24 | 1.98 / 3.21 | 1.87 / 3.05 |
| ref-woman | 1.83 / 2.85 | 1.81 / 2.66 | (no beard: same) |
| photo-10 | 2.39 / 3.98 | 2.34 / 3.76 | 2.21 / 3.09 |
| photo-14 | 1.36 / 2.55 | 1.26 / 2.44 | (same) |
| photo-16 | 1.39 / 2.49 | 1.26 / 2.27 | (same) |

Full trust on beards: the segmentation session found MediaPipe's jaw running
along the jaw *under* a full beard (the segmented face edge 0.10-0.19 face
widths outside it), the opposite of what `outline_trust` assumes. The rest
of the fit still uses `outline_trust`; worth re-testing there.

One regression: ref-woman's mouth reads 2.22 -> 2.89. Her lips themselves
move under 0.1%. Her chin moves 2.8%, and the crease under her lower lip
reads lower on the render.

Level renders, compared with tilted photos as before: ref-man 2.15 -> 1.87
and ref-woman 2.97 -> 2.52. The three tilted photos get slightly worse
(+0.05 to +0.13), as expected now the jaw matches the photo as shot.

**Brow height: three causes ruled out.** The rendered brows sit 1.5-6.7%
above the photo's (inner ends most), level or posed.
- *The fit:* brow-to-lid rows (105-159, 334-386) in PROPORTIONS left the
  rendered brow height unchanged (photo-16 26.3 -> 26.5 above the lid,
  photo 21-23) and the whole face slightly worse on three faces. Not kept.
- *The bake:* traced through the build, the painted brow sits 18.7/19.3
  above the lid against the photo's 18.3/18.9 (eye-corner spacings). The
  paint is where the photo has it.
- *The lid drop (4 degrees):* rendered without it, ref-man's brows read the
  same.
- *`tighten_brows` thickening the fringe:* darkening only the brow's core
  thinned photo-16's brows by 1 (36 -> 35, photo 27). The whole face got
  worse on three faces, and the brows lighter than the photo's. Not kept.
Still open: the rendered brows read 25-30% thicker than the photo's and
higher, while the texture is right. That points at how the brow area
renders: the ridge's shading and ambient occlusion under the brow.

**Head pose from the forehead, eyes and nose bridge: set aside.** On the
laughing photo-10, MediaPipe's head transform reads 11.2 degrees of pitch
where a rigid fit of the average face to 30 landmarks a laugh does not move
reads 3.0. With that reading: photo-10 2.24 -> 1.93, ref-woman 1.78 -> 1.73,
but ref-man 1.88 -> 2.21. The level bias of the stable reading varies by face
(-2.2 to 2.8 degrees), so one constant does not fit. Taking the same reading
off the avatar's own landmark vertices was worse on all five. Kept: MediaPipe's
transform less 3.3 degrees. A fix needs the level render read per avatar.

**Hair colour from the photo check, kept.** The studio now passes the
check's hair colour name with the haircut. A reading outside that name's
lightness range (`photo_check.HAIR_LIGHTNESS`) is pulled to its nearer end,
keeping its hue. On sk-22, "black" takes the read (75, 70, 80) to (55, 51,
60). Without a check, nothing changes.

## Where the face is lost: reading, fitting, rendering

Each stage measured on its own, in the upright front-on frame (% of the eye
spacing). The fit compares the mesh's landmark vertices with the photo's
landmarks. The render compares the landmarks found on the render with the
photo's.

- **Fit**: about 3.5 across the face. Eyes 1.2-1.3, mouth 1.3-1.7 (3.5 on
  the laugh), nose 2.2-2.9, outline 5-7, brows 4-7. The outline is the
  largest error the fit leaves.
- **Render**: about 3-5. The nose (2.5 -> 3.6-6.6) and the mouth (1.5 ->
  2.2-8.2) are lost here, between the paint and the mesh. The brows gain:
  the painted brows follow the photo and hide the mesh's 4-7.
- **Reading**: three small photos of one man (29-41 px faces) disagreed
  with each other by 5-8, more than the fit or the render lose. On photos
  like those, the reading is the bottleneck.

**The head was turned 4 degrees.** `relaxed_pose` turned the neck to the
side. Every render's nose line sat off-centre, one cheek 17-20% wider than
the other whatever the photo (photos: -11 to +10). The mesh itself is
balanced to 1-3. With the head square, renders balance within 2, and the
landmark error is better on all five faces:

| face | whole face | jaw | nose | brows |
|---|---|---|---|---|
| ref-man | 2.38 -> 2.12 | 4.83 -> 3.37 | 2.41 -> 2.64 | 3.03 -> 2.11 |
| ref-woman | 3.75 -> 2.94 | 8.19 -> 6.24 | 5.53 -> 3.88 | 3.85 -> 3.29 |
| photo-10 | 3.52 -> 2.30 | 8.07 -> 5.66 | 4.85 -> 2.98 | 4.00 -> 3.44 |
| photo-14 | 3.01 -> 2.89 | 6.33 -> 5.52 | 3.91 -> 4.14 | 4.67 -> 4.51 |
| photo-16 | 3.03 -> 2.91 | 6.00 -> 5.36 | 4.09 -> 4.22 | 4.34 -> 4.66 |

Part of the face-width and jaw gap the earlier rounds put down to MediaPipe
wandering was this turn.

**Outline correction, tried again, set aside.** The jaw and cheek outline
(234, `JAW_OUTLINE`, 454) was added to `morph.correct` at full trust. The
mesh's outline vertices moved to the photo's (fit outline 5.5 -> 3.0 on
photo-16), but the outline the detector finds on the render did not follow
them. On the squared head: ref-man 2.12 -> 2.04, ref-woman flat, photo-10
2.30 -> 2.35, photo-14 2.89 -> 3.02, photo-16 2.91 -> 3.03, and the jaw was
worse on three faces. Taken out. The outline needs the render in the loop:
fit the silhouette the camera sees, not the landmark vertices.

**A barely parted mouth rendered open.** sk-21's photo scores jawOpen
0.08. The response row turned that into weight 0.39, and the render read
0.45: lips apart on the teeth, mouth error 7.08. The row was stale. Re-measured
at weights 0-1, jawOpen reads 0.33 on ref-woman and 0.80-0.85 on photo-16 and
sk-21 at full weight. The new row is their average, and 0.08 now falls below
the shown threshold. On sk-21: render jawOpen 0.06, mouth 7.08 -> 2.71,
whole face 8.20 -> 6.80. The five test faces score no jawOpen, so they do
not change.

**The nose: mostly the photo's pitch, not the avatar.** In the photos' own
frame the rendered nose tip was off by -3.6 to +5.8% of the eye spacing, in
step with the photo's head pitch (4-12 degrees). Shot from slightly above, a
photo shows the tip 6.4 above the nose's base where the level render shows
9.3. With each avatar's neck turned by the difference between the photo's
pose and the level render's (both as MediaPipe reads them), the landmark
error is:

| face | level render | at the photo's pose | nose |
|---|---|---|---|
| ref-man | 1.78 | 1.80 | 2.22 |
| ref-woman | 2.99 | 1.61 | 1.58 |
| photo-10 (laughing) | 2.34 | 2.60 | 3.56 |
| photo-14 | 2.81 | 1.38 | 0.84 |
| photo-16 | 2.95 | 1.43 | 0.84 |

Measured this way, the nose is within 0.8-2.2 on four faces. On photo-10,
MediaPipe probably misreads the laughing face's pitch. Comparing a level render with a
tilted photo had been charging the pose to the nose, the jaw and the whole
face. Future rounds should measure at the photo's pose.

**The 2D render at the photo's head pose, as an option.** Each GLB now
carries `photoHeadPose`: MediaPipe's head rotation for the photo, less the
3.3 degrees of pitch it reads on our own level renders (1.5-4.5 across the
test faces; `LEVEL_READ_PITCH`). The studio's "Photo's head pose" box turns
the neck by it for the 2D image. The default stays level, as does the file's
own pose. Default render against the option:

| face | level (default) | photo's pose |
|---|---|---|
| ref-man | 2.15 | 1.98 |
| ref-woman | 2.97 | 1.84 |
| photo-10 (laughing) | 2.28 | 2.37 |
| photo-14 | 2.90 | 1.39 |
| photo-16 | 2.95 | 1.38 |

ref-woman reaches 1.61 with her own level reading (1.9 degrees) in place of
the constant.

Tried and set aside in this round:
- *Nose length (168-2) as a proportion row.* The fit had left the nose
  2-3% long (the base nose is 6-7% longer than the photos'), and the row
  brought the fitted length to within 0.2. Level renders were mixed (ref-man
  2.18 -> 1.78, the others within noise). At the photo's pose: 1.80 -> 1.83,
  1.61 -> 1.64, 2.60 -> 2.58, 1.38 -> 1.51, 1.43 -> 1.46. Not kept.
- *Ambient occlusion off.* The renderer's ambient occlusion draws a dark rim under the nose:
  without it the nose improved on four faces, but the eyes got worse on all five
  (ref-woman 1.74 -> 2.45). Kept on.

Tried first and set aside: opening the jaw until the mesh's inner lips part
as far as the photo's, for the mouth's width. It still overshot: weight 0.17
rendered a lip gap of 0.21 against the photo's 0.12, and the mouth error
was 9.8. The vertex landmark 14 lands on barely moves with the jaw (0.006
where its neighbours drop 0.3), so the lip landmarks' vertices are a poor
gauge of the visible opening.

## The last open items, each tried

- **Brow height**: the eye carry widened to 0.7 eye spacings and faded
  out from the eyes to the nose tip, so the brows come with the eyes and
  the mouth stays put. Brow landmark error better on all four faces
  (3.29 -> 3.03, 4.00 -> 3.85, 4.23 -> 4.00, 5.18 -> 4.67); whole-face
  error a wash. Unfaded, 0.7 improved the whole face on three faces
  (3.71 -> 3.51, 3.04 -> 2.90), but by carrying the painted mouth with the
  eyes: a lead that the painted mouth sits a little off, not a fix. The
  brow still sits 28 above the lid by the render's own landmarks (photo
  21-23): that is the mesh's brow ridge.
- **Lower lip roll**: each lip's volume from its own height (edge to
  seam) instead of both from the pair: photo-16's lips +4.6 -> +6.9,
  ref-woman's whole face 3.71 -> 4.20. Taken out.
- **photo-10**: the cheek rows at a third of their weight: 3.66 -> 3.60,
  ref-man 2.27 -> 2.35. A wash; taken out.
- **Forehead +24**: the de-lighting gain capped at 1.2 left it at +25 and
  the brow ridge at +26. On this one person only: the key light from above
  on the photo's forward-tilted forehead. No lever without relighting.
- **Nose depth from one photo**: a forward bias on the nose targets took
  the rendered test head's nose from -7.3 to +1.2 mm, but the test head was
  made by adding nose depth, so the test cannot tell a better fit from a
  bigger nose for everyone. Not shipped; turned photos remain the answer.
- **Ears, the swept-back cut**: no landmarks for the one, no CC0 asset for
  the other.

## The lower face: cheeks, lighting, mouth

Seventeen things flagged on photo-16's cheeks, mouth, chin and jaw;
fixed in the order they were listed, each measured before and after on
all five faces (render minus photo, % of iris spacing or L*).

**Mid-face width.** The cheeks rendered 5-9 off in either direction (a
broad face oval, a narrow one broad). Cheekbone (50-280) and mid-cheek
(123-352) width joined the fitted proportions: ref-man +8.1 -> +4.9,
ref-woman +7.5 -> +4.5. On the narrow faces nothing moved, and the fit's
own landmarks turned out to match the photo's widths exactly (cheekbone
106.9 against 106.8, jaw 119.0 against 118.8): what the detector reads as
narrow on the render is shading, not shape. photo-10 (laughing) came out
narrower (-1.4 -> -6.1) and its whole-face error 3.25 -> 3.66; the
proportion rows read a laugh's cheeks as the face's.

**Lighting symmetry.** One cheek rendered 42 L* brighter than the other
relative to the photo, on three of five faces. Not the texture (the
de-lit face is 10 L* the other way), not the mesh (left/right pairs within
1%), not the key and fill (moving the fill up changed it by 3). A flat
grey head with every directional light off still split 39: the room
environment map is the lopsided light, and removing it left the head dark
and symmetric. Rotated about the vertical, the grey head's split went
+70 at -20 degrees, +39 at 0, -15 at +20; the environment is now turned
+14. Split on the five faces: +24 / +1 / +40 / +42 / +43 -> +12 / -6 /
+22 / +14 / +13. Skin brightness elsewhere unchanged.

**Crevice lift.** Lowering it to 0.5 changed nothing on the render: the
light nasolabial folds and philtrum (+35 / +50) are not the lift. In the
de-lit photo they are as dark as the photo's; it is the bake that loses
them, a 214-pixel photo's creases resampled over the texture. An unsharp
mask on L* after de-lighting (`sharpen_creases`, 0.5 of the detail at a
fifteenth of the eye spacing, off the stubble and the lips' seam) brings
the fold back from 16 to 19 L* below the cheek against the photo's 23. At
0.8 it matched but the stubble broke into blotches.

**Lips.** The lips are less saturated in the texture than in the photo
(a* +13 against +16/+21) and read more saturated on the render (+25);
the skin's sheen, the one red-orange thing in the material, is not it
(sheen off: unchanged). The detector's lip patch takes in the seam, which
the sealed lips and the occlusion draw darker than the photo's (-37
against -13 at the start of the round, after the sharpening). Not resolved.

**Mouth set.** The photo's pressed, drawn-in mouth was taken off as
expression and rendered as a flat line, corners 7-8 higher than the
photo's. Below a smile of 0.15 the press and retraction units now count as
the mouth's shape (`SET_SMILE`): corners +7.3 -> +6.4 and +8.6 -> +6.4,
lip height +5.3 -> +3.2 on one run and +5.0 on another: the detector's
noise on a 43-pixel mouth is about 2.

**Two more, tried and taken out.** The closed mouth's painted seam lifted
halfway towards the lip colour before the renderer's occlusion darkens the
crease: the lip-line reading did not move, and on ref-woman it swung from
+87 to -110 between two renders of the same thing, so that measure is the
detector's noise on a few pixels and says nothing. The cheek width rows
faded out with a smile, for photo-10: its widths came in (-6.1 -> -2.4) and
its whole-face error went out (3.66 -> 3.76). photo-10's cheek rows stay
as they are; its laugh is the limit.

**Stubble.** jaw_beard 0.28 is below the beard carry, so the stubble is
the photo's own specks stretched by the bake: the jaw's darkest tenth 76
L* below its skin against the photo's 48. Lifting faint stubble towards
the skin measured nothing and was taken out. A resolution limit.

**The solver.** With the cheek rows added, scipy's trf stopped short of
the optimum: a face differing from the skull only by a depth shift bent
head-fat-decr to 0.23 where nothing asked for it (the rig check caught
it). The fit now uses bvls, which solves the bounded problem exactly, at
about a second a fit.

Whole-face landmark error, start of the round -> now: ref-man 2.40 ->
2.27, ref-woman 3.88 -> 3.71, photo-10 3.25 -> 3.66, photo-14 3.07 ->
3.04. Brows and mouths better on all four; photo-10's nose and jaw worse.
The cheek-width reading on the renders is not a shape measure (the fit
matches the photo's widths exactly) and swings with the shading, so it is
not quoted per face.

## Eyes and brows against the photo

Measured on photo-16, each eye and brow against the photo's (iris spacing
as the unit; L* 0-255):

- **The lash line was gone.** Along the upper lid the photo reads L* 50-57;
  the avatar's lid there read 98-105, bare skin, since fill_eyes paints the
  opening and the lashes over with lid skin, and the eyeballs sat on it and
  stared. `appearance.paint_lashes` reads the photo's own colour along the
  upper lid (the darker half of the pixels there) and draws it back on the
  rim after the fill, 0.07 of the eye's width wide, with a thinner, fainter
  line along the lower lid. In the painted photo the line is L* 48; on the
  render it reads 61-76, the rim's curl and a 2-pixel line in a 214-pixel
  photo taking the rest.
- **The brows had spread.** Painted from a photo whose brows are a few
  pixels tall, they came out 40% thicker (27/32 -> 39/37) and lighter
  (L* 98 -> 123 on one side). `appearance.tighten_brows` pulls each brow's
  pixels darker than the surrounding skin by 1.6 about that skin level:
  L* 123 -> 96 and 106 -> 81 against the photo's 98 / 104. The thickness
  reads the same (38 / 37), the measure taking in the socket's shadow under
  the brow, and the brow sits 28 above the lid against the photo's 21-23.
  The catalogue's brow card worn over the paint changed no number.
- **The whites were a flat bright ellipse.** The lids' shadow in the eye
  texture is deeper (0.4 -> 0.6 above, and 0.25 below): sclera L* 158 ->
  148 on the one side, the other unchanged. The iris highlight was two hard
  blobs: eye roughness 0.05 -> 0.18.
- **Under the brow** the lid read 12 L* darker than the photo's relative to
  the cheek, which merged the brow into a thick band: the crevice lift in
  delight is 0.8 of the deficit instead of 0.6.
- Unchanged: the eye tilt (one eye 4 degrees more than the photo's), the
  hooded fold itself, and the brow's height.

## Hair: the facets, and the creases under them

The hair rendered as a cap of grey triangles: a highlight patch per
triangle, which no amount of colour or shape work changed. Built with each
suspect off in turn on photo-16, the facets stayed without the strand
normal map and without the sweep, lift and widening, and vanished without
the anisotropic highlight. The file carried no tangents: a viewer then
derives one per triangle from its texture coordinates, and the highlight,
which runs along the tangent, broke along them. `skinned_glb.tangents`
now writes per-vertex tangents (along u, across the normal, with the
handedness) for every part with an anisotropic material.

What was left after that was creases: the lifts moved each hair vertex
along the normal of the single nearest skin vertex, so two neighbouring
points on one card could take two different normals and the card folded
between them. `garments.scalp_normals` averages the nearest twelve by
inverse distance (the normal now turns at most 23 degrees along a card
edge). Hair L* against photo-16's, mean / spread / brightest tenth, with
the fine-scale noise (L* left after a 3-pixel blur):

| | mean | spread | p90 | fine-scale |
|---|---|---|---|---|
| photo | 38 | 29 | 78 | 15.0 |
| facets (before) | 41 | 31 | 81 | 16.7 |
| tangents | 29 | 29 | 57 | 18.9 |
| tangents + smooth lifts | 38 | 31 | 78 | 17.5 |
| isotropic highlight, roughness 0.5 | 34 | 35 | 78 | 19.0 |

The brown and blonde heads keep their earlier numbers (their variation
is strand colour, not sheen).

**The seams.** A card mesh keeps a vertex per texture island, so 306 of
short02's positions are two vertices, each on its own card with its own
normal and tangent, and the shading stepped at every seam. Normals are now
taken from the mesh with coincident positions merged (`welded_normals`)
and tangents summed per position, so the copies agree exactly (the file's
byte normals differ by 8.6 degrees at most, from rounding). Fine-scale
noise 17.5 -> 16.6 against the photo's 15.0, brightness unchanged
(41 / 31 / 80).

**The gap under the cards: a cap.** A swept-back cut's lifted front stands
off the forehead, and between its sparse lowest strands the tinted skin
showed, lit as skin: a brown band under black hair, read as a gap between
hair and skull. Tried on the band first, in order: the tint's root shade
divided by the renderer's lift like the skin (right, kept, but the band
stayed brown: 143 -> 142 L*); a longer solid plateau and shorter fade
(kept: band 0.20 -> 0.17 eye spacings); full-strength tint (the strip under
the hair went darker than the photo's, 112 against 137: not kept); coarse
strands drawn across the whole tint (brown stripes: not kept). What closed
the gap was geometry: `compose` now adds a cap (`scalp_cap`) cut from the
head's own faces 0.012 off the skin, above the brows, in the hair's root
shade, blended by a smooth alpha (the exporter's RGBA path quantises alpha
to a cut-out, so blended parts keep theirs) that is opaque under the cards
and fades over 0.06 units from 0.1 above their lowest strands. Cut at half
alpha instead, the wisps drew its edge as a sawtooth, and opaque down to the
lowest strands it put solid hair 0.15 eye spacings lower than the photo's
(the strip under the hair 85 against 137). Now: solid hair starts 0.88
above the brows (photo 0.93), the strip 118 (photo 137; 125 before the
cap), its edge 6.9 L* a row (photo 10.6). Nothing of it shows from the
side or the back. What is left is the cards' own silhouettes and the alpha
cut at their tips.

## One face audited to the detail: photo-16

A fifth test photo, a tanned man with heavy upper lids, thin lips and a tall
slicked-back cut, 214 pixels across the face. Twenty issues were flagged
against it (shape by feature, colour by region, zoomed crops) and the five
largest fixed, each measured before and after on all five faces. Shape in %
of the photo's iris spacing, colour as L* on 0-255.

**Skin written for the renderer's lift.** Every face's cheek rendered at
L* 197-212 whatever the photo's (173-222): the de-lighting left the
brightness wherever its gain put it, and the renderer's lights brighten skin
by a third in linear luminance (a flat grey head rendered at 1.3-1.4 times
its albedo at the cheek; the sheen adds under 2%). Light faces hit the
albedo ceiling and happened to come out right; the tan face rendered 33 L*
light. `delight` now keeps the cheeks at the brightness the photo showed,
divided by RENDER_LIFT (1.35), then the ceiling. photo-16 cheeks +33 -> +15,
chin +40 -> +16, under the eyes +38 -> +14; the three light faces unchanged
(their photos are flash-lit above the ceiling). The forehead stays +24: the
key light from above.

**Eyelids corrected to the photo's contours.** Every face rendered its eyes
1-4% more open than the photo's (lower lids low). The eye contours now join
the lips in the free-form correction. Eye opening, aligned on eye corners
and nose bridge: ref-man +1.0/+1.5 -> 0.0/+0.1, ref-woman +2.1/+1.8 ->
+0.2/+1.1, photo-10 +1.8/+3.1 -> -0.1/+0.9, photo-16 +3.0/+3.4 -> +1.6/+2.2
(all before the eye texture darkened, which the detector reads as a smaller
eye). Treating the photo's eyeSquint (0.57 on a relaxed face) as identity
instead of expression was tried first and moved nothing: the identity targets
gave the same lids. Not done: the hooded upper-lid fold itself, which is a
texture the lid skin never gets.

**Eye texture and rim.** The eye texture is written darker by the same lift
(the whites rendered at 246 against the photos' 216, the iris a shade light
for "dark brown"), and the lid skin is painted in 0.2 of the eye's width past
the opening (0.12 left a rim of lashes that the lid stretched into a dark
ring: under-eye L* -35 against the photo on one side; now +14).

**Lips.** The mesh's lip contour was already right (20.2 against the photo's
19.6, in the fit's frame) but the lips rendered as a full, parted pout: (1)
the closed mouth was taken for open, the inner-lip polygon rasterising to a
line of pixels that passed the area test on a 38-pixel mouth, and the lips
repainted flat pink (now judged by the gap at the mouth's middle,
MOUTH_OPEN 0.05 of the width: 0.000-0.007 closed, 0.16-0.19 laughing); (2)
the landmarks see no lip bulk, and the fit had set lowerlip-height-decr 0.44
against -incr 0.38 (`morph.lip_volume` now sets the volume and height
targets from the lip-height ratio, LIP_GAIN 2.5; at 4.0 ref-man's lips came
out 2.9 too thin and mouths widened); (3) the inner lips met 1.4 apart and
teeth showed (`seal_lips`). The detector still reads photo-16's lips 7-10
taller than the photo's; to the eye they are now thin, with the lower lip's
roll shaded below. ref-woman's lips read +7 as before.

**Hairline band.** A pale strip of forehead with a few wisps between the
cards and the painted hairline, 0.28 iris spacings tall against the photo's
0.13 transition. Three causes, in order found: the tint was computed per
vertex and the forehead is four vertices tall, so the edge smeared across
the triangles (now per texel); the fade was 0.32 units from the edge itself
(now solid for 0.06 below it, fading over 0.15); and the photo-forehead
sparing reached up through the hair's sparse lowest strands (now off in
that plateau). Band 0.28 -> 0.16, solid hair starting 1.04 -> 0.90 above
the brows against the photo's 0.93. The strip that remains is the wisps.

Whole-face landmark error, base -> now: ref-man 2.40 -> 2.48, ref-woman
3.88 -> 3.61, photo-10 3.25 -> 3.51, photo-14 3.07 -> 3.00. Brows better on
all four (3.78 -> 2.40, 5.74 -> 4.59, 4.50 -> 3.75, 7.04 -> 4.79); photo-10's
nose and mouth read worse (4.28 -> 4.61, 2.41 -> 2.79), the laughing mouth
pulled narrower by the lip targets.

**Tried and taken out: the cut's top fitted column by column to the
photo's silhouette** (`read_hair` reading the hair's top at eleven columns,
a `garments.crown` pushing each column's vertices up to it). Measured from
the vertices the cut's top was already within 0.2 eye spacings of the
photo's in every column, so the lift was small, and what it did push (along
the scalp's normals, divided by their upward share) heaped the hair into a
lopsided mass. The pompadour is not a silhouette problem from the front: it
is the front-to-back sweep and the strands, which a stock cut's cards have
no texture for.

The hairline's wisps were drawn along two arcs across the forehead: the
cut-off of the tint where the photo's forehead takes over stepped in 0.05
units and left a row of dots, and the wisps' contrast was 0.9. Eased to 0.12
and 0.6.

A measuring note: aligning render to photo on the eye corners stopped
working once the lids were corrected to the photo (the corners move), and
read the whole face 6-10% too large; the scripts now align on the nose
line, forehead top, chin and face sides. The hair overlap measure, which
also warps by those points, is unstable on two faces and not quoted here.

**Hair sheen.** The rendered hair had weaker highlights than every photo's:
the brightest tenth of its pixels at L* 54 against the photo's 78 on the
black hair, 130 against 150 and 167 against 194 on the brown and blonde.
The cuts' roughness (0.8 in the catalogue) is now 0.68: black hair's
brightest tenth 81 (photo 78), spread 31 (29); the lighter heads move only
4 and 2, their variation being strand colour rather than shine. At 0.55 the
black hair overshot to 101.

**Lid crease.** The lid skin is painted in 0.1 of the eye's width above the
opening instead of 0.2 (still 0.2 below, where the lash rim made the ring),
so the fold a hooded eye creases at is left in the paint. Under-eye L* and
the ring unchanged; the crease itself is barely more visible at 214 pixels.

**A swept-back cut from the community packs: none usable.** MakeHuman's
community "Hair 01" pack (hair01_cc0.zip, 25 cuts, every one listed CC0)
is mostly bobs, braids and long styles; the two short men's cuts in it are
`culturalibre_hair_02`, whose own .mhclo header says AGPL3 whatever the
pack page says (not used), and `culturalibre_hair_05` (CC0 in its header,
by swift502): a low-poly cut of separate clumps that renders as black spikes
with scalp between them, at half the photo's height. Overlap with photo-16's
hair 0.06 against short02's 0.59. The "Hair 02" and "Hair 03" packs are
CC-BY, outside the licence rule. A pompadour needs a cut modelled for it.

**The two arcs across the forehead, ruled out one by one.** They survive
with the hairline tint off altogether, with the landmark warp eased into
the forehead affine at the sheet's rim, and with the crease shading off; a
flat grey head with no maps shows none, and the mesh's midline profile is a
smooth bulge. So they are in the colour texture, and the one source left is
the photo's own forehead through the affine map (its faint lines, or the
segmenter's eroded skin edge), which a sharper photo would settle. The
easing and the other two changes were taken out again, having moved nothing.

**Lip depth in the correction** (the one-photo trust for the lip contours,
so the lower lip's roll would follow the photo's guess): lips +6.2 -> +6.4
by the detector. Taken out.

Still open on photo-16: the pompadour (no usable cut), the lid fold as a
shape, the forehead +24 under the key light, the lower lip's roll, the
arcs, the ears, and the 214-pixel face.

## Measured gaps: what each turned out to be

Signed errors, head render against photo, % of eye spacing:

- **Most of the jaw "error" is the photo's head turn.** The avatar's outline
  sat 4-13 inside the photo's on one side and 3-8 outside on the other, on
  every face: a slightly turned head against a front-on render. With the two
  sides averaged the jaw is within 0-5, and what is left is the temples, 2-5
  too narrow. Forehead width (54-284) added to the fitted proportions moved
  them 0.1-0.4 and left two of four faces worse: not kept.
- **The brows sat high above the lids** (aligned on eye corners and nose
  bridge: +4.1 / -0.5 / +3.9 / +8.0). Not the photo's expression (rendered
  without it, the same) and not the shape fit (the mesh's brow landmarks are
  within 2 of the photo's): the brows are painted and carded from the
  landmark sheet, whose eyes did not land on the mesh's lids.
  `compose.eyes_onto_mesh` carries the sheet's eyes onto the mesh's, the
  brows with them (EYE_CARRY 0.5 eye spacings). With the hair change below,
  brow landmark error 3.78 -> 2.39, 5.74 -> 4.55, 4.50 -> 3.63, 7.04 -> 4.95;
  whole face 2.40 -> 2.18, 3.88 -> 3.69, 3.25 -> 3.41, 3.07 -> 2.94
  (photo-10's nose reads worse).
  photo-14 still shows two brow lines, the card above the painted brow.
- **The nose is not at its targets' limit** (no nose target above 0.91), and
  with turned views at 50 degrees its depth is within 0.6 mm. The 3-4 mm
  shortfall was the 35 degree views; from one photo, MediaPipe's depth guess
  is the limit.
- **Hair widened too far on photo-14** (1.16 against the photo's 1.02 just
  above the brows): matching only the outline's shape widened a cut whose
  photo is narrower than it at the brows. Matched in absolute width now, a
  rendered cut standing 6% wider than its vertices (1.01-1.10 measured):
  photo-14 1.09 / 1.08 / 1.00 against 1.02 / 1.05 / 1.01.
- **Not fixed:** the eyes read 1-4 too open (the lower lids low), and
  ref-woman's lower face renders 6-12 short of the photo's. Ears and small
  photos as before.

## Upscaling a small face, against ground truth

photo-16 (477 x 642, face 214 px cheek to cheek) was shrunk (Lanczos) until
its face was 40 px and 80 px wide. Each small version was then read three
ways: as it is, enlarged back to full size with bicubic, and enlarged with
Lanczos. An avatar was built from each and compared with the avatar built from
the original photo.

Real-ESRGAN was not run. Its code is BSD-3, but the released weights were
trained on DF2K + OST: DIV2K's licence says it is "made available for academic
research purpose only", and Flickr2K is scraped Flickr under mixed licences.
It fails the training-data rule, as does nearly every learned upscaler
(DIV2K is the field's standard set).

| | landmarks | depth | head mm (mean / worst) | face texture dE | fine detail | brow colour dE | iris dE |
|---|---|---|---|---|---|---|---|
| 40 px, as is | 0.71 | 0.76 | 1.14 / 6.5 | 2.9 | 0.80 | 11.4 | 4.4 |
| 40 px, bicubic | 0.69 | 0.45 | 0.97 / 6.4 | 2.0 | 0.84 | 8.7 | 3.7 |
| 40 px, Lanczos | 0.71 | 0.30 | 0.99 / 6.0 | 1.9 | 0.85 | 7.0 | 3.3 |
| 80 px, as is | 0.29 | 0.25 | 0.38 / 1.2 | 1.8 | 0.86 | 4.9 | 2.7 |
| 80 px, bicubic | 0.30 | 0.29 | 0.28 / 0.8 | 1.3 | 0.89 | 3.9 | 3.2 |
| 80 px, Lanczos | 0.33 | 0.26 | 0.34 / 2.2 | 1.2 | 0.91 | 2.6 | 2.1 |
| 80 px, Lanczos to 250 px | | | 0.51 / 2.6 | 1.5 | 0.90 | | 2.4 |

The landmark and depth columns are the mean error against the original
photo's landmarks, as % of eye spacing, with position and size taken out.
Head mm is the head mesh against the original avatar's. Face texture dE is
the mean CIE76 colour difference over the face in the baked head texture.
Fine detail is how well the texture's high-pass correlates with the
original's (1 = all of it). Brow colour is `facetexture.brow_colour` on the
photo. Iris is from the sidecar. Skin and hair colour stayed within 1.5 dE in
every row, with no trend.

- **Landmarks: no gain**, as found on photo-10 earlier ("Upscaling small
  photos"). Read as it is, the 40 px face sits 2.8 off, evenly across the
  face: half a small pixel of offset, a shift the fit ignores. With position
  and size taken out, every reading of the 40 px face lands at 0.7 and every
  reading of the 80 px face at 0.3, whichever upscaler was used.
- **Head shape: no gain we can tell from noise.** Upscaling moved the head
  0.1-0.15 mm closer, but enlarging the same 80 px face to 250 px instead of
  214 moved it 0.17 mm *further*. Differences of that size come from the
  scale alone. The 40 px avatar stays 1 mm off on average and 6 mm at
  worst, whatever the upscaler.
- **Texture: smoother, not sharper.** The face texture's colour error falls
  2.9 -> 1.9 dE at 40 px and 1.8 -> 1.2 at 80, below the roughly 2.3 dE that
  can be seen. No detail comes back. In the texture strip
  (out/upscale/textures.png) every small version is equally soft, with no
  brow hairs and no stubble.
- **Brow colour looks like the largest gain, but is not one**: 11.4 -> 7.0
  dE at 40 px and 4.9 -> 2.6 at 80. The small photo's brow pixels really are
  lighter: the darkest point across each brow is (81, 47, 39) on the
  original, (87, 57, 48) at 80 px and (98, 70, 61) at 40 px, and today's 40 px
  reading (95, 69, 59) is already about that dark. Enlarging darkens it only
  because Lanczos and bicubic overshoot at edges, going darker than any real
  pixel of the small photo. A patch sized in face widths was tried and did not
  help either: no single size kept the full-size photos where they are, and
  the 40 px colour stayed about 11 dE off at every size. The colour only tints
  brow cards, and photo-16's avatar wears none.

**Verdict: no upscaler wins clearly.** Lanczos is slightly ahead of bicubic
on texture, brows and iris. Both beat reading the small photo as it is only
in colour sampling, by amounts at or below what can be seen. Neither moves
the shape or brings back detail. What a 40 px face loses could only be
guessed back by a learned restorer, and none trained on licence-clean data
was found. One photo, shrunk cleanly: a real small photo also carries camera
blur and JPEG loss.

A pre-step was written and tested, but not merged. `fit_face(min_face_px=250)`
enlarges a smaller face with Lanczos and finds it again, and the advice still
reports the face at the size it was shot. The brow colour a 40 px face
loses is gone from its pixels, so neither enlarging nor a different patch
brings it back.

## The photo check on the three sk photos

`photo_check.py` (claude-sonnet-5-5, off without `ANTHROPIC_API_KEY`) on
uploads/sk-20.png, sk-21.png and sk-22.png: three 148x148 photos of one man,
the set whose builds disagreed by 5-8% of the eye spacing. Beside it, what the
pipeline measures (face width in pixels from the landmarks, head pose from
MediaPipe's transform):

| photo | pipeline | check: issues | check: hair, face |
|---|---|---|---|
| sk-20 | face 35 px; yaw 23, pitch 3, roll -12 | sunglasses, face too small, head turned, side lighting | short, swept up and back (hair_short02); black; stubble; sunglasses |
| sk-21 | face 45 px; yaw 15, pitch 12, roll -5 | face too small, blurred, head turned | short, swept back (hair_short02); black; none |
| sk-22 | face 43 px; yaw -11, pitch 11, roll -6 | face too small, head turned, head tilted, blurred | short, swept up and back (hair_short02); black; stubble |

All three come back unusable, which is right: none should be built from as it
is. Where the pipeline can measure, the two agree: every face is far under
250 px, and every head is turned over 10 degrees. What MediaPipe says nothing
about, the check catches: the sunglasses on sk-20 (until now the build fitted
eyes behind dark lenses without a word), blur on sk-21 and sk-22, and hard
light from one side on sk-20. It did not call sk-21 tilted at a 12 degree
pitch, though it did call sk-22 tilted at 11, so near the threshold it is a
judgement call.

The haircut agrees with the segmentation rules (hair_short02 on all three).
Here the segmentation found hair to measure, so the check's cut is not used.
It is used only where the reading finds none, and on the five test faces a
hint changes nothing (identical GLBs). Its "stubble" on sk-20 and sk-22 sits
beside jaw_beard 0.25 and 0.49, so only sk-22's beard is carried. Its
"black" hair is closer than the pipeline's read on sk-22, which is grey
(75, 70, 80) where the others read 16-32: a soft, highlit photo. Neither is
acted on yet.

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
