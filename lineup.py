"""The people lineup test: does a person's avatar pick them out of four photos?

The likeness rulers (SFace, dlib) are trained on real photos and are weak
judges of a stylized face; people are the real ruler (workflow stage 0,
reports/likeness-gap-workflow.md). Each round shows one avatar (its stylized
portrait, out/renders/style/<name>-<variant>.png, from style_score.py) beside
four photos: the person's own and three others, of the same gender where the
set has them. The tester picks one. Answers go to out/lineup/answers.jsonl.

Served by app.py on this PC only (/lineup): the test photos never leave it,
as their consent requires.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
ANSWERS = OUT / "lineup" / "answers.jsonl"
CHOICES = 4
PHOTO_SIZE = 360        # px, the longer side, as the page shows them


def people(variant: str) -> list[dict]:
    """Everyone with a photo and a portrait in `variant`: the dev faces, and
    the example (out/example/front.png) when it is there."""
    from score_run import TESTSET, load_faces
    found = [{"name": f["name"], "photo": HERE / f["photo"], "gender": f.get("gender", "male")}
             for f in load_faces(TESTSET / "test-set.json", "dev")]
    if (OUT / "example" / "front.png").exists():
        found.append({"name": "example", "photo": OUT / "example" / "front.png", "gender": "male"})
    return [p for p in found if portrait(p["name"], variant).exists()]


def portrait(name: str, variant: str) -> pathlib.Path:
    return OUT / "renders" / "style" / f"{name}-{variant}.png"


def trials(everyone: list[dict], seed: int) -> list[dict]:
    """One round per person, in a seeded order: their avatar and CHOICES names
    (theirs among them, at a seeded place), the others of their gender first."""
    rng = random.Random(seed)
    rounds = []
    for p in rng.sample(everyone, len(everyone)):
        same = [o["name"] for o in everyone if o["name"] != p["name"] and o["gender"] == p["gender"]]
        other = [o["name"] for o in everyone if o["name"] != p["name"] and o["gender"] != p["gender"]]
        rng.shuffle(same)
        rng.shuffle(other)
        options = [p["name"]] + (same + other)[:CHOICES - 1]
        rng.shuffle(options)
        rounds.append({"person": p["name"], "options": options})
    return rounds


def image(kind: str, name: str, variant: str) -> bytes | None:
    """A round's picture as JPEG bytes: the avatar's portrait or the person's
    photo (upright, shrunk to PHOTO_SIZE). None for a name not in the set."""
    from PIL import Image
    match = next((p for p in people(variant) if p["name"] == name), None)
    if match is None or kind not in ("avatar", "photo"):
        return None
    if kind == "avatar":
        src = Image.open(portrait(name, variant))
    else:
        from score_run import upright
        src = Image.open(upright(match["photo"]))
    src = head(src.convert("RGB"))
    src.thumbnail((PHOTO_SIZE, PHOTO_SIZE))
    buf = io.BytesIO()
    src.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def head(picture):
    """The head alone, cut the same way from a photo and an avatar (MediaPipe's
    landmarks: room for the hair above, the chin's below, at 3:4): the avatar's
    shirt is dyed the photo's top's colour, and shown it picked the person
    without a look at the face. The whole picture where no face is found."""
    import numpy as np
    from generate import detect
    found = detect(picture)
    if not found.face_landmarks:
        return picture
    w, h = picture.size
    px = np.array([[l.x * w, l.y * h] for l in found.face_landmarks[0]])
    top, bottom = px[:, 1].min(), px[:, 1].max()
    tall = bottom - top
    y0, y1 = top - 0.55 * tall, bottom + 0.08 * tall
    half = 0.375 * (y1 - y0)
    cx = px[:, 0].mean()
    return picture.crop((int(cx - half), int(y0), int(cx + half), int(y1)))


def record(answer: dict) -> None:
    keep = {k: answer.get(k) for k in ("session", "variant", "person", "picked", "ms")}
    keep["time"] = dt.datetime.now().isoformat(timespec="seconds")
    ANSWERS.parent.mkdir(parents=True, exist_ok=True)
    with ANSWERS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(keep) + "\n")


def results(variant: str | None = None) -> dict:
    """Right answers over all, per person, and per tester session; chance is 1 in CHOICES."""
    rows = [json.loads(line) for line in ANSWERS.read_text(encoding="utf-8").splitlines()] if ANSWERS.exists() else []
    rows = [r for r in rows if variant is None or r.get("variant") == variant]
    right = lambda rs: round(sum(r["picked"] == r["person"] for r in rs) / len(rs), 3) if rs else None
    by = lambda key: {k: {"right": right([r for r in rows if r[key] == k]), "answers": sum(r[key] == k for r in rows)}
                      for k in sorted({r[key] for r in rows})}
    return {"variant": variant, "answers": len(rows), "right": right(rows), "chance": 1 / CHOICES,
            "per_person": by("person"), "per_session": by("session")}


def check() -> None:
    """The rounds: one per person, their own name among the choices, gender kept where it can be."""
    everyone = [{"name": f"m{i}", "gender": "male"} for i in range(4)] + [{"name": f"f{i}", "gender": "female"} for i in range(5)]
    rounds = trials(everyone, seed=3)
    assert sorted(r["person"] for r in rounds) == sorted(p["name"] for p in everyone)
    for r in rounds:
        assert r["person"] in r["options"] and len(set(r["options"])) == CHOICES
        assert all(o[0] == r["person"][0] for o in r["options"]), r     # 4 of each gender: no mixing needed
    assert trials(everyone, seed=3) == rounds and trials(everyone, seed=4) != rounds
    print("ok  lineup: one round each, own photo among four, same gender, repeatable by seed")


if __name__ == "__main__":
    check()
