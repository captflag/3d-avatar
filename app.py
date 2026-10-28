"""Avatar studio — the local tool the team uses to try the engine on photos.

Drop a photo on the page, get a 3D avatar next to it. Runs on one PC and listens
on this PC only; `python app.py --lan` opens it to the office network. Generation
happens inline because it takes a couple of seconds; there is no queue and no
database, by design.

ponytail: no auth, no accounts, no job queue. This is an internal testing tool on
a trusted network, not the product. The product path is the Fastify API.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import sys

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image

import photo_check
import skinned_glb
from compose import compose
from generate import NoFaceFound, generate_avatar, get_landmarker

HERE = pathlib.Path(__file__).resolve().parent
UPLOADS = HERE / "uploads"
OUT = HERE / "out"

# .jfif and .jpe are JPEG too: Windows saves web images as .jfif.
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".jpe", ".png", ".webp", ".bmp"}
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
            "kind": info.get("kind", "face"),
            "rigged": bool(info.get("rigged")),
            "model": f"/out/{glb.name}",
            "photo": photo_url,
            "created": dt.datetime.fromtimestamp(glb.stat().st_mtime).strftime("%d %b %H:%M"),
            "_mtime": glb.stat().st_mtime,
            "expression": list(info.get("expression", {}).items())[:3],
            "size_kb": info.get("size_kb"),
            "advice": info.get("advice", []),
        })
    # Newest first by when the model was written. Sorting by id looked right
    # while every id began with a timestamp, but anything made from the command
    # line is named by hand and sorted alphabetically to the top.
    return sorted(items, key=lambda i: i["_mtime"], reverse=True)


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

    # Turned photos from the camera, for the profile (views.py). One that is not
    # a readable image is dropped rather than failing the front photo's avatar.
    side_photos = []
    for k, side in enumerate(request.files.getlist("side")[:2], start=1):
        side_suffix = pathlib.Path(side.filename or "").suffix.lower()
        if side_suffix not in ALLOWED_SUFFIXES:
            continue
        side_path = UPLOADS / f"{entry_id}-side{k}{side_suffix}"
        side.save(side_path)
        try:
            with Image.open(side_path) as probe:
                probe.verify()
            side_photos.append(side_path)
        except Exception:
            side_path.unlink(missing_ok=True)

    full_body = request.form.get("mode") == "body"
    gender = "female" if request.form.get("gender") == "female" else "male"
    # The haircut the photo check saw (/check), if the page ran one: only a
    # catalogue id or "bald" gets through.
    hair_hint = photo_check.hair_hint({"hair_cut": request.form.get("hair_hint")})
    hair_colour = photo_check.hair_colour_hint({"hair_colour": request.form.get("hair_colour")})
    try:
        if full_body:
            info = compose(photo_path, OUT / f"{entry_id}.glb", gender, side_photos=side_photos or None,
                           hair_hint=hair_hint, hair_colour_hint=hair_colour)
        else:
            info = generate_avatar(photo_path, OUT / f"{entry_id}.glb", side_photos)
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


@app.post("/check")
def check():
    """The photo check's verdict (photo_check.py), asked before building.
    `enabled` is false when no API key is set; nothing is kept on disk."""
    upload = request.files.get("photo")
    if upload is None:
        return jsonify(error="Choose a photo first."), 400
    if not photo_check.enabled():
        return jsonify(enabled=False)
    try:
        verdict = photo_check.check_photo(upload.stream)
    except Exception as err:
        # The type only: an API error's text is not ours to log in full.
        app.logger.warning("photo check failed: %s", type(err).__name__)
        verdict = None
    if verdict is None:
        return jsonify(enabled=True, error="The photo check gave no answer.")
    return jsonify(enabled=True, **verdict)


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


@app.get("/compat/<name>")
def serve_compat(name: str):
    """An avatar without KHR_mesh_quantization, for old viewers (skinned_glb.dequantised)."""
    path = OUT / name
    if not name.endswith(".glb") or path.resolve().parent != OUT.resolve() or not path.exists():
        abort(404)
    data = skinned_glb.dequantised(path.read_bytes())
    return app.response_class(data, mimetype="model/gltf-binary",
                              headers={"Content-Disposition": f'attachment; filename="{path.stem}-compat.glb"'})


@app.get("/download/<name>")
def serve_download(name: str):
    """An avatar for download wearing the haircut the studio's picker chose:
    ?hair=<catalogue id or bald>&colour=r,g,b (skinned_glb.with_hair), the
    other cuts the file carries taken out; ?compat=1 also as serve_compat."""
    path = OUT / name
    if not name.endswith(".glb") or path.resolve().parent != OUT.resolve() or not path.exists():
        abort(404)
    data = path.read_bytes()
    if "hair" in request.args:
        colour = None
        if request.args.get("colour"):
            colour = [int(c) for c in request.args["colour"].split(",") if c.strip().isdigit()]
            if len(colour) != 3 or max(colour) > 255:
                abort(400)
        data = skinned_glb.with_hair(data, request.args["hair"], colour)
    if request.args.get("compat"):
        data = skinned_glb.dequantised(data)
    return app.response_class(data, mimetype="model/gltf-binary",
                              headers={"Content-Disposition": f'attachment; filename="{path.name}"'})


@app.get("/uploads/<path:name>")
def serve_upload(name: str):
    return send_from_directory(UPLOADS, name)


@app.get("/samples/<path:name>")
def serve_sample(name: str):
    return send_from_directory(HERE / "samples", name)


@app.get("/lib/<path:name>")
def serve_lib(name: str):
    return send_from_directory(HERE / "lib", name)


@app.get("/models/face_landmarker.task")
def serve_landmarker():
    # The same face model the pipeline reads photos with, for the studio's
    # live camera checks: what passes there is what the pipeline will see.
    return send_from_directory(HERE / "assets", "face_landmarker.task")


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
    # Uploads are faces: only this PC sees them unless --lan is asked for.
    lan = "--lan" in sys.argv[1:]
    print("Avatar studio on http://localhost:8777" + ("  (and this PC's LAN address)" if lan else ""))
    app.run(host="0.0.0.0" if lan else "127.0.0.1", port=8777, threaded=True)
