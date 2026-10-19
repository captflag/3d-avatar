"""How much an avatar looks like its photo, as a face-recognition service sees it.

Landmark error says whether the shape fits; it does not say whether someone
would take the render for the person. AWS Rekognition's CompareFaces does: a
0-100 similarity between the largest face in the photo and the face in the
render, from a recognition model trained for exactly that question, licensed
for commercial use and paid per call (a fraction of a cent).

The render is the studio's own 2D head image: in the studio pick "2D image",
untick "Full body", Download. Credentials come from the usual AWS chain
(`aws configure`, or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in the
environment); this script never reads them itself.

    .venv/Scripts/python.exe score.py photo.jpg render.png [photo2 render2 ...]
"""
from __future__ import annotations

import io
import pathlib
import sys

from PIL import Image

# Rekognition takes at most 5 MB per image; a face needs nowhere near this many pixels.
MAX_SIDE = 1600


def image_bytes(path: pathlib.Path) -> bytes:
    """JPEG bytes of the image, any transparency flattened onto mid grey (a
    transparent render's background is otherwise read as black)."""
    with Image.open(path) as im:
        im = im.convert("RGBA")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        flat = Image.new("RGB", im.size, (128, 128, 128))
        flat.paste(im, mask=im.getchannel("A"))
    out = io.BytesIO()
    flat.save(out, "JPEG", quality=92)
    return out.getvalue()


def similarity(client, photo: pathlib.Path, render: pathlib.Path) -> float | None:
    """CompareFaces similarity, 0-100, or None when no face matched at all."""
    reply = client.compare_faces(SourceImage={"Bytes": image_bytes(photo)},
                                 TargetImage={"Bytes": image_bytes(render)},
                                 SimilarityThreshold=0)
    matches = reply.get("FaceMatches", [])
    return max(m["Similarity"] for m in matches) if matches else None


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    import boto3
    session = boto3.Session()
    client = session.client("rekognition", region_name=session.region_name or "us-east-1")
    for photo, render in zip(argv[::2], argv[1::2]):
        score = similarity(client, pathlib.Path(photo), pathlib.Path(render))
        shown = "no face matched" if score is None else f"{score:5.1f}"
        print(f"{shown}  {pathlib.Path(photo).name}  vs  {pathlib.Path(render).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
