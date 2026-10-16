"""Fetch the MakeHuman assets the catalogue uses, and nothing else.

MakeHuman's system assets (hair, eyebrows, eyelashes, clothes, shoes) ship as
one 280 MB zip. Every item in it is CC0 — its own manifest,
packs/makehuman_system_assets.json, marks each one so — but we use a few dozen
megabytes of it. A zip keeps its index at the end, so this reads the index with
an HTTP range request and then fetches only the folders catalogue.json names.

Also fetches the six regional macro targets `morph.body_for` blends to make a
man's or a woman's body, and the expression units `morph.fit` uses to tell a
smile from a wide mouth, each checked for its CC0 header.

Run: .venv/Scripts/python.exe fetch_assets.py
"""

from __future__ import annotations

import io
import json
import pathlib
import urllib.request
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"
PACK = ("https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/"
        "makehuman_system_assets_cc0.zip")
TARGETS = ("https://raw.githubusercontent.com/makehumancommunity/makehuman/master/"
           "makehuman/data/targets/macrodetails/")
EXPRESSIONS = ("https://raw.githubusercontent.com/makehumancommunity/makehuman/master/"
               "makehuman/data/targets/expression/units/")


class RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a URL, read by range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(head, timeout=30).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or not len(buffer):
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(request, timeout=120).read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def wanted_folders() -> set[str]:
    """Pack folders named by the catalogue, e.g. `hair/short02/`."""
    catalogue = json.loads((HERE / "catalogue.json").read_text(encoding="utf-8"))
    folders = set()
    for item in catalogue["items"]:
        path = item.get("mhclo", "")
        if path.startswith("mh_system/"):
            folders.add(path[len("mh_system/"):].rsplit("/", 1)[0] + "/")
    return folders


def main() -> None:
    folders = wanted_folders()
    pack = zipfile.ZipFile(io.BufferedReader(RemoteFile(PACK), buffer_size=1 << 20))
    members = [m for m in pack.infolist()
               if not m.is_dir() and (any(m.filename.startswith(f) for f in folders)
                                      or m.filename == "packs/makehuman_system_assets.json")]
    for member in members:
        target = ASSETS / "mh_system" / member.filename
        if not target.exists():
            pack.extract(member, ASSETS / "mh_system")
    print(f"mh_system: {len(members)} files for {len(folders)} catalogue items")

    out = ASSETS / "targets" / "macrodetails"
    out.mkdir(parents=True, exist_ok=True)
    for region in ("african", "asian", "caucasian"):
        for sex in ("female", "male"):
            name = f"{region}-{sex}-young.target"
            if not (out / name).exists():
                (out / name).write_bytes(urllib.request.urlopen(TARGETS + name, timeout=60).read())
    print("macrodetails: 6 body targets")

    from morph import EXPRESSION_DIR, EXPRESSION_UNITS, REGIONS
    for region in REGIONS:
        (EXPRESSION_DIR / region).mkdir(parents=True, exist_ok=True)
        for unit in EXPRESSION_UNITS:
            path = EXPRESSION_DIR / region / f"{unit}.target"
            if path.exists():
                continue
            data = urllib.request.urlopen(f"{EXPRESSIONS}{region}/{unit}.target", timeout=60).read()
            if b"released as CC0" not in data[:500]:
                raise SystemExit(f"{region}/{unit}.target has no CC0 header; not using it")
            path.write_bytes(data)
    print(f"expression: {len(EXPRESSION_UNITS)} units x {len(REGIONS)} regions")


if __name__ == "__main__":
    main()
