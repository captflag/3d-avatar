"""Is this photo good enough to build from? Asked of Claude, before building.

Poor photos lose more likeness than the fit or the render do: three photos of
one man, small faces, one in sunglasses and two turned, disagreed by 5-8% of
the eye spacing. MediaPipe says where the landmarks are, not that the eyes are
behind dark lenses or the hair under a hat, so the photo goes to the Claude API
(vision) and comes back as fixed issue codes, which become fixed advice, plus
what the pipeline can use: hair, facial hair, glasses.

Off unless ANTHROPIC_API_KEY is set: then check_photo returns None and nothing
else changes. The key is read by the SDK from the environment and never logged.

    .venv/Scripts/python.exe photo_check.py photo.jpg [photo2 ...]
"""
from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import sys

from PIL import Image, ImageOps

from catalogue import catalogue

MODEL = "claude-sonnet-5-5"
# The API scales anything larger down to about this long edge anyway.
MAX_SIDE = 1568

ADVICE = {
    "sunglasses": "Take off the sunglasses: the eyes are fitted from the photo.",
    "glasses": "Take off the glasses if you can: the frames hide the eye corners.",
    "face_too_small": "Move closer: the face should fill about a third of the frame.",
    "blurred": "Hold the camera still, in more light: the face is blurred.",
    "head_turned": "Face the camera straight on.",
    "head_tilted": "Hold your head level.",
    "mouth_open": "Close your mouth and relax your face.",
    "side_lighting": "Face the light: one side of the face is in shadow.",
    "hair_covered": "Take off the hat or hood so the hair shows.",
    "no_single_face": "Use a photo with one face in it.",
}


def haircuts() -> dict[str, str]:
    """The catalogue's haircuts by id, with their labels."""
    return {i["id"]: i["label"] for i in catalogue()["items"] if i["category"] == "hair" and "mhclo" in i}


def schema() -> dict:
    def enum(*values):
        return {"type": "string", "enum": list(values)}
    return {
        "type": "object",
        "properties": {
            "usable": {"type": "boolean"},
            "issues": {"type": "array", "items": enum(*ADVICE)},
            "hair_length": enum("bald", "buzzed", "short", "chin", "shoulder", "long", "unknown"),
            "hair_style": {"type": "string"},
            "hair_cut": enum(*haircuts(), "bald", "unknown"),
            "hair_colour": enum("black", "dark brown", "brown", "light brown", "blonde", "red",
                                "grey", "white", "dyed", "unknown"),
            "facial_hair": enum("none", "stubble", "moustache", "goatee", "beard"),
            "glasses": enum("none", "glasses", "sunglasses"),
        },
        "required": ["usable", "issues", "hair_length", "hair_style", "hair_cut",
                     "hair_colour", "facial_hair", "glasses"],
        "additionalProperties": False,
    }


def prompt(size: tuple[int, int]) -> str:
    cuts = "\n".join(f"- {k}: {v}" for k, v in haircuts().items())
    return f"""This photo will be turned into a 3D avatar of the person in it. The face shape is fitted from \
face landmarks, the skin is baked from the photo, and a haircut is picked from a catalogue.

Report every issue that would cost likeness:
- sunglasses / glasses: anything over the eyes
- face_too_small: the face is under about 250 pixels across in the original, which is {size[0]}x{size[1]} pixels
- blurred: motion blur, out of focus, or heavy compression on the face
- head_turned: the head is turned away from the camera by more than about 10 degrees
- head_tilted: the head is tilted sideways or up/down by more than about 10 degrees
- mouth_open: the mouth is open or the teeth show
- side_lighting: one side of the face is clearly darker than the other
- hair_covered: a hat, hood or scarf hides the hair
- no_single_face: no face, or more than one
usable is false when an issue would visibly change the avatar (any of these except a mild one).

Describe the hair and face as shown: hair_style is a few words (e.g. "short, swept back, faded sides"). \
hair_cut is the closest of these catalogue haircuts, "bald" for no hair or a shaved head, \
"unknown" when the hair is hidden:
{cuts}"""


def image_block(path) -> tuple[dict, tuple[int, int]]:
    """The photo upright, at most MAX_SIDE, as a base64 JPEG block; and its original size."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        size = im.size
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=90)
    data = base64.standard_b64encode(out.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}, size


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def check_photo(path, client=None) -> dict | None:
    """The check's verdict on the photo (a path or an open file), with `advice`
    in plain words; None when the check is off (no key) or the API gave no answer."""
    if client is None:
        if not enabled():
            return None
        import anthropic
        client = anthropic.Anthropic()
    block, size = image_block(path)
    reply = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema()}},
        messages=[{"role": "user", "content": [block, {"type": "text", "text": prompt(size)}]}],
    )
    if reply.stop_reason != "end_turn":
        return None
    verdict = json.loads(next(b.text for b in reply.content if b.type == "text"))
    verdict["advice"] = [ADVICE[i] for i in verdict["issues"]]
    return verdict


# The lightness (L*, 0-100) each hair colour the check names can have. A
# reading from the photo outside it is pulled to its nearer end: on a soft,
# highlit photo the hair mask read black hair as grey (L* 30).
HAIR_LIGHTNESS = {"black": (0, 22), "dark brown": (12, 35), "brown": (25, 48), "light brown": (40, 62),
                  "blonde": (55, 88), "grey": (45, 80), "white": (75, 100)}


def hair_colour_hint(verdict: dict | None) -> str | None:
    """The hair colour the check saw, when it is one HAIR_LIGHTNESS knows."""
    colour = (verdict or {}).get("hair_colour")
    return colour if colour in HAIR_LIGHTNESS else None


def within_colour(rgb, colour: str):
    """`rgb` (0-255) with its lightness pulled into the named colour's range,
    its hue and chroma kept."""
    import cv2
    import numpy as np
    lab = cv2.cvtColor(np.uint8([[np.clip(rgb, 0, 255)]]), cv2.COLOR_RGB2LAB).astype(np.float64)
    low, high = HAIR_LIGHTNESS[colour]
    lab[..., 0] = np.clip(lab[..., 0], low * 2.55, high * 2.55)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0].astype(np.float64)


def hair_hint(verdict: dict | None) -> str | None:
    """The catalogue haircut the check saw, "bald", or None when it has none."""
    cut = (verdict or {}).get("hair_cut")
    return cut if cut in haircuts() or cut == "bald" else None


if __name__ == "__main__":
    if not enabled():
        sys.exit("Set ANTHROPIC_API_KEY to run the photo check.")
    for name in sys.argv[1:]:
        print(name, json.dumps(check_photo(pathlib.Path(name)), indent=1))
