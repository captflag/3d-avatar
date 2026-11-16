"""The stylized portrait, generated: the person's photo re-drawn as a 3D animated
character (workflow stage 3, reports/likeness-gap-workflow.md).

Stable Diffusion 1.5 image-to-image from the photo itself (its background made
the portrait's plain peach), the person's face carried by IP-Adapter
plus-face (a CLIP image encoder; no InsightFace). The face's structure comes
from the photo: starting from our 3D render instead kept the render's
geometry, and SFace against the photo stayed at the render's (0.16-0.18).

Two steps, two environments:

    .venv/Scripts/python portrait.py prepare PHOTO -o DIR      # start.png, face.png, eyes.txt
    .venv-gen/Scripts/python portrait.py draw DIR -o OUT.png [--seed 7]

`prepare` needs MediaPipe (the main environment); `draw` needs CUDA PyTorch
and diffusers (.venv-gen, full precision: the GTX 1660 gives NaN in fp16,
and ControlNet does not fit beside the UNet in 6 GB in fp32; with it, 11-13
minutes a portrait and no better likeness).

Licences: SD 1.5 CreativeML OpenRAIL-M, IP-Adapter Apache-2.0. SD 1.5 was
trained on scraped images: an exception the user allowed for this step only
(2026-11-04).
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image

SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IP_ADAPTER = ("h94/IP-Adapter", "models", "ip-adapter-plus-face_sd15.safetensors")
BACKDROP = (222, 165, 130)              # the stylized portrait's (lib/scene-setup.js PORTRAIT)
# The look of the reference cartoon avatar, in words: no studio's name. The
# eyes' colour read from the photo (`prepare`): unnamed, they came out blue.
PROMPT = ("portrait of a stylized 3D animated character, head and shoulders, {eyes} eyes, gentle closed-mouth smile, "
          "relaxed eyebrows, smooth soft skin with warm subsurface glow, soft studio lighting, "
          "plain warm peach background, high quality 3D render")
NEGATIVE = ("photo, photograph, realistic skin pores, wrinkles, blemishes, frown, angry, text, watermark, "
            "deformed, blurry, lowres, multiple people, brick wall, textured background")
SIZE = 512
STEPS = 30
GUIDANCE = 6.0
# How far the photo is re-drawn (0 keeps it, 1 ignores it) and how hard the
# face image pulls. Measured on the example (SFace against its photo; the
# reference cartoon avatar 0.485, dlib 0.595): strength 0.5 0.67 but half a
# photo, 0.55 a 3D character at 0.42 (dlib 0.50: both call it the same
# person), 0.6 0.33; face weight 0.8-0.9 no better than 0.7.
# On the 8 dev faces (against their crop at 512 px: SFace alone loses three
# of them at that size, 0.07-0.21 on the untouched crop) 0.55 keeps the
# person, 0.56-0.74, mean 0.62; 0.6 0.58, 0.7 at face weight 0.5 0.56. Their
# photos are sharp and base SD 1.5 keeps them photographic even at 0.7: a
# polished portrait, not the cartoon (the user's call, 2026-11-16, over a
# style model whose licence or training data would not be clean).
STRENGTH = 0.55
FACE_WEIGHT = 0.7


def prepare(photo: pathlib.Path, out: pathlib.Path) -> None:
    """The photo as the portrait starts from: upright, background the
    backdrop's peach (MediaPipe's segmenter), cropped head and shoulders at
    3:4; its face, squared, for IP-Adapter; and the iris colour's name."""
    import cv2
    import appearance
    from generate import stand_up
    image, found = stand_up(photo)
    if not found.face_landmarks:
        raise SystemExit(f"no face in {photo}")
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    landmarks = found.face_landmarks[0]
    person = cv2.GaussianBlur((appearance.segment(rgb) != 0).astype(np.float32), (0, 0), 2)[..., None]
    flat = (rgb * person + np.array(BACKDROP, np.float32) * (1 - person)).astype(np.uint8)
    px = np.array([[l.x * w, l.y * h] for l in landmarks])
    cx, top, bottom = px[:, 0].mean(), px[:, 1].min(), px[:, 1].max()
    y0, y1 = max(int(top - 0.75 * (bottom - top)), 0), min(int(bottom + 0.85 * (bottom - top)), h)
    half = int(0.375 * (y1 - y0))
    start = Image.fromarray(flat[y0:y1, max(int(cx) - half, 0):min(int(cx) + half, w)])
    side = int(0.62 * start.width)
    fx, fy = start.width // 2, int(0.42 * start.height)
    face = start.crop((fx - side // 2, fy - side // 2, fx + side // 2, fy + side // 2)).resize((224, 224))
    _, eyes = appearance.iris_colour(rgb, landmarks)
    out.mkdir(parents=True, exist_ok=True)
    start.save(out / "start.png")
    face.save(out / "face.png")
    (out / "eyes.txt").write_text(eyes or "dark brown", encoding="utf-8")


def pipeline():
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD, torch_dtype=torch.float32, variant="fp16",
                                                          safety_checker=None, requires_safety_checker=False)
    pipe.load_ip_adapter(IP_ADAPTER[0], subfolder=IP_ADAPTER[1], weight_name=IP_ADAPTER[2],
                         image_encoder_folder="models/image_encoder")
    pipe.set_ip_adapter_scale(FACE_WEIGHT)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.enable_model_cpu_offload()       # 6 GB: each model on the GPU only while it runs (4.4 GB peak)
    return pipe


def draw(pipe, prepared: pathlib.Path, seed: int = 7, strength: float = STRENGTH,
         prompt: str = PROMPT) -> Image.Image:
    import torch
    start = Image.open(prepared / "start.png").convert("RGB")
    start = start.resize((SIZE, int(SIZE * start.height / start.width) // 8 * 8))
    eyes = (prepared / "eyes.txt").read_text(encoding="utf-8").strip()
    return pipe(prompt.format(eyes=eyes), negative_prompt=NEGATIVE, image=start,
                ip_adapter_image=Image.open(prepared / "face.png").convert("RGB"), strength=strength,
                num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                generator=torch.Generator("cpu").manual_seed(seed)).images[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    prep = sub.add_parser("prepare", help="start image, face crop and eye colour (main environment)")
    prep.add_argument("photo", type=pathlib.Path)
    prep.add_argument("-o", "--out", type=pathlib.Path, required=True)
    make = sub.add_parser("draw", help="the portrait (.venv-gen)")
    make.add_argument("prepared", type=pathlib.Path, help="a folder from `prepare`")
    make.add_argument("-o", "--out", type=pathlib.Path, required=True)
    make.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.photo, args.out)
    else:
        out = draw(pipeline(), args.prepared, args.seed)
        if np.asarray(out).max() < 8:
            raise SystemExit("the portrait came out black")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.save(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
