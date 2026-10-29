"""Add a person to the test set (TESTSET.md): their photos filed under the
next pseudonym, a row in the consent register, an entry in the set file.

    python add_person.py FOLDER --signed 2026-10-29 --gender female [--photo-check]

FOLDER holds the photos named as TESTSET.md lists them (front, left35,
right35, left50, right50, left20, o1, o2, smile, body; any image type). Run it
only once the person's signed consent form is on file: --signed is its date.

Each photo is checked against its name before anything is filed: a face
found, big enough, and the head turned as the name says. The test set's first
face's "side" photos were turned 0.3 and 7.4 degrees, and no step noticed;
the face-shape scores built on them would have meant nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TESTSET = HERE / "data" / "testset"
# How far each photo's head should be turned (degrees, either way), and the
# face's smallest width across the cheeks in pixels (TESTSET.md: 250 the floor).
TURNS = {"front": (0, 10), "left35": (25, 45), "right35": (25, 45), "left50": (40, 62), "right50": (40, 62),
         "left20": (12, 28)}
MIN_FACE_PX = 250
KINDS = list(TURNS) + ["o1", "o2", "smile", "body"]


def turn_problem(kind: str, yaw: float) -> str | None:
    """What is wrong with a `kind` photo whose head is turned `yaw` degrees, or None."""
    if kind not in TURNS:
        return None
    low, high = TURNS[kind]
    if not low <= abs(yaw) <= high:
        return f"turned {abs(yaw):.0f} degrees, wants {low}-{high}"
    return None


def read(path: pathlib.Path) -> tuple[float, float]:
    """(yaw in degrees, face width across the cheeks in pixels) of the photo's face."""
    from generate import fit_face
    _, marks, result, photo = fit_face(path)
    m = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
    yaw = float(np.degrees(np.arcsin(np.clip(-m[2, 0], -1.0, 1.0))))
    return yaw, abs(marks[454].x - marks[234].x) * photo.size[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=pathlib.Path)
    parser.add_argument("--signed", required=True, type=dt.date.fromisoformat,
                        help="the date on the person's signed consent form")
    parser.add_argument("--gender", required=True, choices=["male", "female"])
    parser.add_argument("--keep-days", type=int, default=365, help="photos deleted after this many days")
    parser.add_argument("--photo-check", action="store_true",
                        help="they also ticked sending the photos to the Claude API")
    parser.add_argument("--force", action="store_true", help="file it even where a photo fails its check")
    args = parser.parse_args()

    photos = {p.stem.lower(): p for p in args.folder.iterdir() if p.stem.lower() in KINDS and p.is_file()}
    if "front" not in photos:
        print(f"no front photo in {args.folder} (files named {', '.join(KINDS)})", file=sys.stderr)
        return 2
    problems, yaws = [], {}
    for kind, path in sorted(photos.items()):
        try:
            yaw, width = read(path)
        except Exception as err:                      # no face, unreadable file
            problems.append(f"{path.name}: {type(err).__name__} {err}")
            continue
        yaws[kind] = yaw
        found = [turn_problem(kind, yaw), f"face {width:.0f} px across, wants {MIN_FACE_PX}+" if width < MIN_FACE_PX
                 and kind != "body" else None]
        print(f"{path.name:16s} turned {yaw:+6.1f}  face {width:5.0f} px  " + ("; ".join(f for f in found if f) or "ok"))
        problems += [f"{path.name}: {f}" for f in found if f]
    for a, b in (("left35", "right35"), ("left50", "right50")):
        if a in yaws and b in yaws and np.sign(yaws[a]) == np.sign(yaws[b]):
            problems.append(f"{a} and {b} are turned the same way")
    if problems and not args.force:
        print("\nnot filed:\n  " + "\n  ".join(problems) + "\nretake those, or --force", file=sys.stderr)
        return 1

    TESTSET.mkdir(parents=True, exist_ok=True)
    set_path, register = TESTSET / "test-set.json", TESTSET / "consent.csv"
    faces = json.loads(set_path.read_text(encoding="utf-8")) if set_path.exists() else []
    number = max([int(f["name"][1:]) for f in faces if f["name"][1:].isdigit()] + [0]) + 1
    name, consent_id = f"p{number:02d}", f"c{number:02d}"
    folder = TESTSET / name
    folder.mkdir()
    filed = {kind: f"data/testset/{name}/{kind}{path.suffix.lower()}" for kind, path in photos.items()}
    for kind, path in photos.items():
        shutil.copy2(path, HERE / filed[kind])

    face = {"name": name, "photo": filed["front"], "gender": args.gender, "consent_id": consent_id,
            # About one person in five held back for sign-off (TESTSET.md).
            "split": "lockbox" if number % 5 == 0 else "dev", "tags": {}}
    if "left35" in filed and "right35" in filed:
        face["sides"] = [filed["left35"], filed["right35"]]
    for key, kinds in (("held_out", ["left20"]), ("teeth", ["smile"]), ("body", ["body"])):
        if kinds[0] in filed:
            face[key] = filed[kinds[0]]
    if [k for k in ("o1", "o2") if k in filed]:
        face["others"] = [filed[k] for k in ("o1", "o2") if k in filed]
    set_path.write_text(json.dumps(faces + [face], indent=1), encoding="utf-8")
    if not register.exists():
        register.write_text("consent_id,pseudonym,date,scopes,retain_until,withdrawn\n", encoding="utf-8")
    scopes = "build;score" + (";photo_check" if args.photo_check else "")
    with register.open("a", encoding="utf-8") as out:
        out.write(f"{consent_id},{name},{args.signed},{scopes},{args.signed + dt.timedelta(days=args.keep_days)},\n")
    print(f"\nfiled as {name} ({face['split']}), consent {consent_id}. Write {name} on their paper form; "
          f"re-run score_run.py --noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
