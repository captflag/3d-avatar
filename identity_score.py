"""How much an avatar looks like its photo, as a local face-recognition model sees it.

The keep/drop measure for likeness changes. Landmark error says whether the
shape fits; three changes that brought the skin's brightness closer to the
photos made MediaPipe's landmarks read slightly worse, and only a "same
person?" score could say which mattered. This is that score, free and local.

SFace (MobileFaceNet, OpenCV model zoo, weights Apache-2.0) embeds the
largest face found by YuNet in each image; the score is the cosine
similarity of the two. OpenCV's same-person threshold is 0.363. On this
repo's photos: different people 0.15; two real photos of the same man
0.48-0.67; the same build rendered twice differs by 0.008 at most.

**Internal yardstick only.** SFace was trained on research face datasets
whose terms bar commercial use. Never ship it or import it from the product.

Model files (git-ignored, into assets/):
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

    .venv/Scripts/python.exe identity_score.py photo.jpg render.png [photo2 render2 ...]
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
DETECTOR = HERE / "assets" / "face_detection_yunet_2023mar.onnx"
RECOGNISER = HERE / "assets" / "face_recognition_sface_2021dec.onnx"
SAME_PERSON = 0.363

_models = None


def _load():
    global _models
    if _models is None:
        missing = [p.name for p in (DETECTOR, RECOGNISER) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing in assets/: {', '.join(missing)} (see this file's docstring)")
        _models = (cv2.FaceDetectorYN.create(str(DETECTOR), "", (320, 320), 0.6),
                   cv2.FaceRecognizerSF.create(str(RECOGNISER), ""))
    return _models


def embedding(path) -> np.ndarray | None:
    """SFace's embedding of the largest face in the image, or None. A
    transparent render is flattened onto grey first."""
    detector, recogniser = _load()
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
    flat = Image.new("RGBA", rgba.size, (200, 200, 200, 255))
    flat.alpha_composite(rgba)
    img = cv2.cvtColor(np.asarray(flat.convert("RGB")), cv2.COLOR_RGB2BGR)
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    if faces is None:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])
    return recogniser.feature(recogniser.alignCrop(img, face))


def similarity(a, b) -> float:
    """Cosine similarity of two images' faces; NaN if either has none."""
    ea, eb = embedding(a), embedding(b)
    if ea is None or eb is None:
        return float("nan")
    return float(_load()[1].match(ea, eb, cv2.FaceRecognizerSF_FR_COSINE))


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    for photo, render in zip(argv[0::2], argv[1::2]):
        s = similarity(photo, render)
        print(f"{s:.3f}  {'same person' if s >= SAME_PERSON else 'below threshold'}  {photo}  {render}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
