"""The photo check against a mocked API reply: no key, no network.

Run: .venv/Scripts/python.exe test_photo_check.py
"""
from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace

from PIL import Image

import photo_check

REPLY = {"usable": False, "issues": ["sunglasses", "head_turned"], "hair_length": "short",
         "hair_style": "short, swept back", "hair_cut": "hair_short02", "hair_colour": "dark brown",
         "facial_hair": "stubble", "glasses": "sunglasses"}


class FakeClient:
    def __init__(self, reply: dict, stop_reason: str = "end_turn"):
        self.sent = None
        self.messages = self
        self.reply = SimpleNamespace(stop_reason=stop_reason,
                                     content=[SimpleNamespace(type="text", text=json.dumps(reply))])

    def create(self, **request):
        self.sent = request
        return self.reply


def photo() -> io.BytesIO:
    out = io.BytesIO()
    Image.new("RGB", (3000, 2000), (120, 100, 90)).save(out, "PNG")
    out.seek(0)
    return out


def main():
    client = FakeClient(REPLY)
    verdict = photo_check.check_photo(photo(), client)
    assert verdict["advice"] == [photo_check.ADVICE["sunglasses"], photo_check.ADVICE["head_turned"]]
    assert photo_check.hair_hint(verdict) == "hair_short02"

    sent = client.sent
    assert sent["model"] == "claude-sonnet-5-5"
    image, text = sent["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    assert "3000x2000" in text["text"]                 # the original size, not the downscaled one
    schema = sent["output_config"]["format"]["schema"]
    assert "hair_long01" in schema["properties"]["hair_cut"]["enum"]
    assert set(schema["required"]) == set(schema["properties"])

    # Only catalogue haircuts or "bald" reach compose.
    assert photo_check.hair_hint({"hair_cut": "bald"}) == "bald"
    assert photo_check.hair_hint({"hair_cut": "unknown"}) is None
    assert photo_check.hair_hint({"hair_cut": "../etc"}) is None
    assert photo_check.hair_hint(None) is None

    # A refusal or a cut-off reply is no verdict, not a crash.
    assert photo_check.check_photo(photo(), FakeClient(REPLY, "refusal")) is None

    # Off without a key: nothing is sent.
    key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        assert photo_check.check_photo(photo()) is None
        from app import app
        reply = app.test_client().post("/check", data={"photo": (photo(), "x.png")})
        assert reply.get_json() == {"enabled": False}
    finally:
        if key is not None:
            os.environ["ANTHROPIC_API_KEY"] = key
    print("photo check: ok")


if __name__ == "__main__":
    main()
