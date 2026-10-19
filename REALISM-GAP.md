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
