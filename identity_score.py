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

A second recogniser, so no change is kept for one model's quirks: dlib's
ResNet (weights released CC0, trained on partly non-commercial and scraped
photos, so internal only too), on the same YuNet face box aligned by dlib's
5-point model (not its non-commercial 68-point one). Its score is the
Euclidean distance of the two 128-d descriptors: lower is closer, under
DLIB_SAME_PERSON the same person.

Model files (git-ignored, into assets/):
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2   (unpacked into assets/dlib/)
    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2        (unpacked into assets/dlib/)
    and the dlib-bin package (uv pip install dlib-bin)

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
DLIB = HERE / "assets" / "dlib"
DLIB_SAME_PERSON = 0.6

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


def _face(path):
    """The image (BGR) with a transparent render flattened onto grey, and the
    largest face YuNet finds in it, or None."""
    detector, _ = _load()
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
    flat = Image.new("RGBA", rgba.size, (200, 200, 200, 255))
    flat.alpha_composite(rgba)
    img = cv2.cvtColor(np.asarray(flat.convert("RGB")), cv2.COLOR_RGB2BGR)
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    return img, (None if faces is None else max(faces, key=lambda f: f[2] * f[3]))


def embedding(path) -> np.ndarray | None:
    """SFace's embedding of the largest face in the image, or None."""
    img, face = _face(path)
    if face is None:
        return None
    return _load()[1].feature(_load()[1].alignCrop(img, face))


_dlib = None


def dlib_embedding(path) -> np.ndarray | None:
    """dlib's 128-d descriptor of the largest face in the image, or None."""
    global _dlib
    if _dlib is None:
        import dlib
        _dlib = (dlib, dlib.shape_predictor(str(DLIB / "shape_predictor_5_face_landmarks.dat")),
                 dlib.face_recognition_model_v1(str(DLIB / "dlib_face_recognition_resnet_model_v1.dat")))
    dlib, shape, recogniser = _dlib
    img, face = _face(path)
    if face is None:
        return None
    x, y, w, h = (int(round(v)) for v in face[:4])
    rgb = np.ascontiguousarray(img[..., ::-1])
    marks = shape(rgb, dlib.rectangle(x, y, x + w, y + h))
    return np.array(recogniser.compute_face_descriptor(rgb, marks))


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
