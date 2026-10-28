"""Before a merge: nothing private is tracked, everything compiles, the checks pass.

    python premerge.py          # all of it, with the assets and sample photos on this PC
    python premerge.py --ci     # what needs no assets (GitHub Actions runs this)

The repository is public. Photos, the people in them, the downloaded assets
and models, built avatars and keys stay on this PC: the guard fails if git
tracks any of them, by path and by file type (only names are looked at; .env
files are never opened). It also fails if anything but the scoring scripts
imports identity_score, whose model is research-only and must never ship.

Then every tracked Python file must compile, the asset-free checks must pass
(test_photo_check.py, test_score_run.py) and, without --ci, test_rig.py too,
where a check that skips for a missing file counts as a failure. Whether a
change is worth keeping is score_run.py --gate's call, not this script's.
"""

from __future__ import annotations

import pathlib
import py_compile
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PRIVATE_FOLDERS = ("uploads/", "out/", "assets/", "samples/", "data/", "graphify-out/", ".venv/")
# Photos, meshes, models and archives: none belongs in git, bar the studio's own drawings.
PRIVATE_TYPES = re.compile(r"\.(jpe?g|jfif|jpe|png|webp|bmp|gif|heic|tiff?|glb|gltf|obj|mhclo|fbx|onnx|task"
                           r"|tflite|npz|npy|pt|pth|ckpt|safetensors|zip|7z|tar|gz)$", re.I)
SHARED_FOLDER = "lib/"
SCORING = {"score_run.py", "shape_model.py", "identity_score.py"}
ASSET_FREE = ["test_photo_check.py", "test_score_run.py"]


def guard(tracked: list[str]) -> list[str]:
    problems = []
    for path in tracked:
        name = path.rsplit("/", 1)[-1]
        if path.startswith(PRIVATE_FOLDERS):
            problems.append(f"{path}: a private folder is tracked")
        elif name.startswith(".env") and name != ".env.example":
            problems.append(f"{path}: an environment file (keys) is tracked")
        elif PRIVATE_TYPES.search(path) and not path.startswith(SHARED_FOLDER):
            problems.append(f"{path}: a photo, mesh, model or archive is tracked")
    for path in tracked:
        if path.endswith(".py") and path not in SCORING and (HERE / path).exists():
            if re.search(r"^\s*(import|from)\s+identity_score\b", (HERE / path).read_text(encoding="utf-8"), re.M):
                problems.append(f"{path}: imports identity_score, which must never ship")
    return problems


def run(script: str, strict: bool = False) -> bool:
    done = subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True)
    print(done.stdout.rstrip())
    # "skip x check: ..." and "ok  x: ...; portrait check skipped" alike.
    skipped = [line for line in done.stdout.splitlines() if re.search(r"\bskip", line, re.I)]
    if done.returncode:
        print(done.stderr[-3000:].rstrip())
        print(f"FAIL {script}")
        return False
    if strict and skipped:
        print(f"FAIL {script}: {len(skipped)} checks skipped for a missing file")
        return False
    return True


def main() -> int:
    ci = "--ci" in sys.argv[1:]
    tracked = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True, text=True,
                             check=True).stdout.splitlines()
    problems = guard(tracked)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"ok  guard: {len(tracked)} tracked files, nothing private")
    broken = []
    for path in tracked:
        if path.endswith(".py") and (HERE / path).exists():
            try:
                py_compile.compile(str(HERE / path), doraise=True)
            except py_compile.PyCompileError as error:
                broken.append(str(error))
    print("\n".join(f"FAIL {b}" for b in broken) or "ok  every tracked Python file compiles")
    passed = [run(script) for script in ASSET_FREE]
    if not ci:
        passed.append(run("test_rig.py", strict=True))
    return 0 if not problems and not broken and all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
