"""What the photo's segmentation says about the jaw, the hair and a beard.

The masks come from appearance.segment (MediaPipe's multiclass selfie
segmenter, Apache-2.0); this measures them against the landmarks:

  python seg.py uploads/ref-man.png ...   (writes out/seg/<photo>.png overlays)
  python seg.py                            (self-check on a drawn face)

`jaw_edges` walks outward from the nose through each of MediaPipe's 17 jaw
landmarks and finds where the face skin ends: a positive gap means the
segmented face edge lies outside the landmark, negative inside it.
"""
import pathlib
import sys

import cv2
import numpy as np

from appearance import FACE_SKIN, HAIR, read_hair, segment

JAW = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
CHIN = [136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 17]   # closed by the lower lip
NAMES = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]
COLOURS = np.array([(0, 0, 0), (0, 80, 255), (255, 200, 150), (0, 220, 220), (60, 200, 60), (255, 0, 255)], np.uint8)


def _px(landmarks, i, w, h):
    return np.array([landmarks[i].x * w, landmarks[i].y * h])


def face_region(parts: np.ndarray) -> np.ndarray:
    """The largest piece of face skin, holes (a beard or lips read as something
    else) filled, so an edge is the face's outline and not a hole's."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((parts == FACE_SKIN).astype(np.uint8))
    if n < 2:
        return np.zeros(parts.shape, bool)
    face = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(np.zeros_like(face), contours, -1, 1, cv2.FILLED).astype(bool)


def jaw_edges(parts: np.ndarray, landmarks, reach: float = 0.3):
    """Per jaw landmark: (gap in face widths, edge point, class at the
    landmark, class just past the edge). The gap is None where no face skin
    lies within `reach` face widths along the ray."""
    h, w = parts.shape
    face = face_region(parts)
    face_w = np.linalg.norm(_px(landmarks, 454, w, h) - _px(landmarks, 234, w, h))
    centre = _px(landmarks, 1, w, h)
    steps = np.arange(-reach * face_w, reach * face_w, 0.5)
    out = []
    for i in JAW:
        p = _px(landmarks, i, w, h)
        d = (p - centre) / max(np.linalg.norm(p - centre), 1e-6)
        pts = p + steps[:, None] * d
        xs, ys = np.clip(pts[:, 0].astype(int), 0, w - 1), np.clip(pts[:, 1].astype(int), 0, h - 1)
        at = NAMES[parts[int(np.clip(p[1], 0, h - 1)), int(np.clip(p[0], 0, w - 1))]]
        on = np.nonzero(face[ys, xs])[0]
        if not len(on):
            out.append((None, None, at, None))
            continue
        k = on.max()
        past = NAMES[parts[ys[min(k + 4, len(ys) - 1)], xs[min(k + 4, len(xs) - 1)]]]
        out.append((steps[k] / face_w, pts[k], at, past))
    return out


def chin_classes(parts: np.ndarray, landmarks) -> dict:
    """Share of each class between the lower lip and the jaw outline: where a
    beard would be."""
    h, w = parts.shape
    poly = np.array([_px(landmarks, i, w, h) for i in CHIN], np.int32)
    mask = cv2.fillPoly(np.zeros((h, w), np.uint8), [poly], 1).astype(bool)
    labels = parts[mask]
    return {NAMES[c]: round(float((labels == c).mean()), 2) for c in np.unique(labels)}


def overlay(photo_rgb, parts, landmarks, edges) -> np.ndarray:
    h, w = parts.shape
    img = cv2.addWeighted(photo_rgb, 0.55, COLOURS[parts], 0.45, 0)
    for i, (gap, edge, _, _) in zip(JAW, edges):
        p = tuple(_px(landmarks, i, w, h).astype(int))
        if edge is not None:
            cv2.line(img, p, tuple(edge.astype(int)), (255, 255, 0), 1)
            cv2.circle(img, tuple(edge.astype(int)), 3, (0, 255, 0), -1)
        cv2.circle(img, p, 3, (255, 0, 0), -1)
    return img


def check():
    """A drawn face: a skin ellipse whose bottom fifth is 'hair' (a beard), on
    background. Landmarks on the ellipse's outline should sit on the edge
    except under the beard, where the filled face reaches them anyway."""
    class P:
        def __init__(self, x, y):
            self.x, self.y = x, y
    h = w = 400
    parts = np.zeros((h, w), np.uint8)
    cv2.ellipse(parts, (200, 200), (100, 140), 0, 0, 360, FACE_SKIN, -1)
    landmarks = [P(0.5, 0.5)] * 478
    landmarks[234], landmarks[454] = P(100 / w, 0.5), P(300 / w, 0.5)
    for k, i in enumerate(JAW):
        a = np.radians(180 - 180 * k / (len(JAW) - 1))   # left cheek round the chin to the right
        landmarks[i] = P((200 + 0.9 * 100 * np.cos(a)) / w, (200 + 0.9 * 140 * np.sin(a)) / h)
    landmarks[17] = P(0.5, 280 / h)                          # the lower lip
    gaps = [g for g, *_ in jaw_edges(parts, landmarks)]
    # 10% inside a 100-140 px ellipse is 10-14 px, about 0.05-0.07 face widths.
    assert all(0.03 < g < 0.09 for g in gaps), gaps
    parts[300:, :][parts[300:, :] == FACE_SKIN] = HAIR       # a beard below y=300
    assert jaw_edges(parts, landmarks)[8][3] == "hair"       # past the chin's edge: beard
    assert chin_classes(parts, landmarks).get("hair", 0) > 0.2
    print("seg check ok")


def main(paths):
    from generate import stand_up
    out = pathlib.Path("out/seg")
    out.mkdir(parents=True, exist_ok=True)
    for path in map(pathlib.Path, paths):
        photo, result = stand_up(path)
        landmarks = result.face_landmarks[0]
        photo_rgb = np.asarray(photo)
        parts = segment(photo_rgb)
        edges = jaw_edges(parts, landmarks)
        reading = read_hair(photo_rgb, landmarks, parts)
        gaps = [None if g is None else round(float(g), 3) for g, *_ in edges]
        print(f"== {path.name} {photo.size}")
        print("  gaps (face widths):", gaps)
        print("  at landmark:", [a for _, _, a, _ in edges])
        print("  past edge:  ", [b for *_, b in edges])
        print("  chin classes:", chin_classes(parts, landmarks))
        print("  hair colour:", None if reading is None else reading["colour"])
        cv2.imwrite(str(out / f"{path.stem}.png"),
                    cv2.cvtColor(overlay(photo_rgb, parts, landmarks, edges), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(sys.argv[1:]) if sys.argv[1:] else check()
