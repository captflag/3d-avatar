# MediaPipe multiclass selfie segmenter: licence

**Verdict: usable commercially.** Google's model card says the weights are under
Apache License 2.0. The repo already relies on this: `appearance.py` and the
README cite it.

## Sources (read 2026-10-23)

- Model card PDF, "Mediapipe Multiclass Segmentation", dated 15/03/2023:
  https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf
  - The "LICENSED UNDER" field reads "Apache License, Version 2.0".
  - Author: Adel Ahmadyan, Google. Model date: May 10, 2023.
  - The card covers two models: 256x256 (ours) and 512x512.
- Docs page, which links the card from the "SelfieMulticlass (256 x 256)" row:
  https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter
  (redirects to developers.google.com/edge/...). The page's own CC-BY-4.0 /
  Apache-2.0 footer covers the docs and sample code, not the model.
- Model file:
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
  The file in `assets/` has the same SHA-256 as a fresh download:
  `c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`.
- MediaPipe repo LICENSE (https://github.com/google-ai-edge/mediapipe) is
  plain Apache-2.0 and does not mention models. The model card is what
  licenses the weights.

## Intended use and limits (model card)

- Applications: "Human segmentation from images/videos in interactive
  entertainment, and video conferencing applications". Domains: AR, video
  conferencing, entertainment. An avatar builder fits.
- Out of scope: "Any form of surveillance or identity recognition", plus
  human life-critical decisions and applications that need pixel-perfect
  masks. These are intended-use statements, not licence terms. This project
  does none of them.

## Training data

The card says almost nothing about it:

- "Model is trained on thousands of samples with diverse identities (i.e.
  unique subjects), various blendshape combinations representing both common
  and random expressions and transformations (e.g. rotations)."
- It does not state the source, how the data was collected, consent, licensing
  or annotation. "Blendshape combinations" hints at synthetic or rendered data,
  but the card does not say so.
- Fairness evaluation set: 918 images, 1902 people, labelled with perceived
  gender and Monk skin tone. Mean IoU for the 256 model is 77.2 overall,
  73.5 on Monk tone 8, and **68.3 on tones 9-10** (only 31 people). Expect
  rougher masks on very dark skin.

This is the same disclosure level as the FaceLandmarker the pipeline already
uses: weights under Apache-2.0, training data undisclosed but not under
restrictive terms. Apache-2.0 asks for the licence text and NOTICE when the
model is redistributed, so include them if a product bundles the .tflite. The
repo does not ship the file; the README downloads it.
