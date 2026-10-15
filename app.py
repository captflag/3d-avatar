"""Avatar studio — the local tool the team uses to try the engine on photos.

Drop a photo on the page, get a 3D avatar next to it. Runs on one PC and is
reachable from the office network. Generation happens inline because it takes a
couple of seconds; there is no queue and no database, by design.

ponytail: no auth, no accounts, no job queue. This is an internal testing tool on
a trusted network, not the product. The product path is the Fastify API.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image

from compose import compose
from generate import NoFaceFound, generate_avatar, get_landmarker

HERE = pathlib.Path(__file__).resolve().parent
UPLOADS = HERE / "uploads"
OUT = HERE / "out"

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def safe_stem(name: str) -> str:
    """A filename we are willing to put on disk and in a URL."""
    stem = pathlib.Path(name).stem[:48]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")
    return stem or "photo"


def entries() -> list[dict]:
    """Everything generated so far, newest first."""
    items = []
    for sidecar in OUT.glob("*.json"):
        glb = sidecar.with_suffix(".glb")
        if not glb.exists():
            continue
        try:
            info = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Uploads are named after the entry; anything made from the command line
        # points at its source photo by name instead.
        photos = sorted(UPLOADS.glob(sidecar.stem + ".*"))
        source = info.get("source_photo", "")
        if photos:
            photo_url = f"/uploads/{photos[0].name}"
        elif source and (UPLOADS / source).exists():
            photo_url = f"/uploads/{source}"
        elif source and (HERE / "samples" / source).exists():
            photo_url = f"/samples/{source}"
        else:
            photo_url = None
        items.append({
            "id": sidecar.stem,
            "model": f"/out/{glb.name}",
            "photo": photo_url,
            "created": dt.datetime.fromtimestamp(glb.stat().st_mtime).strftime("%d %b %H:%M"),
            "expression": list(info.get("expression", {}).items())[:3],
            "size_kb": info.get("size_kb"),
        })
    return sorted(items, key=lambda i: i["id"], reverse=True)


@app.post("/generate")
def generate():
    upload = request.files.get("photo")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a photo first."), 400

    suffix = pathlib.Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return jsonify(error=f"{suffix or 'that file'} is not an image we read. Use JPG, PNG or WebP."), 400

    UPLOADS.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    entry_id = f"{stamp}-{safe_stem(upload.filename)}"
    photo_path = UPLOADS / f"{entry_id}{suffix}"
    upload.save(photo_path)

    # Trust the bytes, not the extension. Pillow signals a damaged file in
    # several ways (SyntaxError for a bad PNG chunk among them), and every one
    # of them means the same thing to us, so catch broadly rather than guess.
    try:
        with Image.open(photo_path) as probe:
            probe.verify()
    except Exception:
        photo_path.unlink(missing_ok=True)
        return jsonify(error="That file is not a readable image."), 400

    full_body = request.form.get("mode") == "body"
    gender = "female" if request.form.get("gender") == "female" else "male"
    try:
        if full_body:
            info = compose(photo_path, OUT / f"{entry_id}.glb", gender)
        else:
            info = generate_avatar(photo_path, OUT / f"{entry_id}.glb")
    except NoFaceFound:
        photo_path.unlink(missing_ok=True)
        return jsonify(error="No face found in that photo. Try a clear, front-facing one."), 422
    except Exception:
        app.logger.exception("generation failed for %s", entry_id)
        photo_path.unlink(missing_ok=True)
        return jsonify(error="Generation failed. Check the server log."), 500

    return jsonify(
        id=entry_id,
        model=f"/out/{entry_id}.glb",
        photo=f"/uploads/{photo_path.name}",
        review=f"/review.html?model=/out/{entry_id}.glb&photo=/uploads/{photo_path.name}",
        info=info,
    )


@app.errorhandler(413)
def too_large(_err):
    """Flask rejects an oversized body before our handler runs, and its default
    reply is HTML the page cannot read. Phone photos hit this, so answer in the
    shape the page expects."""
    mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify(error=f"That photo is over {mb} MB. Shrink it and try again."), 413


@app.get("/out/<path:name>")
def serve_out(name: str):
    return send_from_directory(OUT, name)


@app.get("/uploads/<path:name>")
def serve_upload(name: str):
    return send_from_directory(UPLOADS, name)


@app.get("/samples/<path:name>")
def serve_sample(name: str):
    return send_from_directory(HERE / "samples", name)


@app.get("/review.html")
def serve_review():
    return send_from_directory(HERE, "review.html")


@app.get("/gallery.json")
def gallery():
    return jsonify(entries())


@app.get("/")
def index():
    return send_from_directory(HERE, "studio.html")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    get_landmarker()  # pay the model load now, not on the first upload
    print("Avatar studio on http://localhost:8777  (and this PC's LAN address)")
    app.run(host="0.0.0.0", port=8777, threaded=True)
