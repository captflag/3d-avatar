"""Fetch the MakeHuman assets the catalogue uses, and nothing else.

MakeHuman's system assets (hair, eyebrows, eyelashes, clothes, shoes) ship as
one 280 MB zip. Every item in it is CC0 — its own manifest,
packs/makehuman_system_assets.json, marks each one so — but we use a few dozen
megabytes of it. A zip keeps its index at the end, so this reads the index with
an HTTP range request and then fetches only the folders catalogue.json names.

Also from the system pack, though no catalogue item uses them yet: the skins
(to take real pore and mottling detail from, in place of skin.py's noise),
the eye materials' iris photos, the high-poly eyeball and the five teeth
shapes; and from the community skin packs (CC0) a few natural skins of tones
the system pack covers thinly. Each item is checked against its pack's own
manifest before it is used.

The haircuts from MakeHuman's community asset packs (Hair 01, CC0; Hair 02
and 03, CC-BY) are fetched the same way, folder by folder, each checked
against its pack's own manifest.

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


# The system pack's own skins, every age, region and sex, but not the toon,
# suit or second variants; the eye materials and high-poly eyeball; the teeth.
EXTRA_SYSTEM = ([f"skins/{age}_{region}_{sex}/" for age in ("young", "middleage", "old")
                 for region in ("african", "asian", "caucasian") for sex in ("female", "male")]
                + ["eyes/materials/", "eyes/high-poly/"] + [f"teeth/teeth_shape0{k}/" for k in range(1, 6)])
# Natural community skins (no makeup, tattoos or anatomy), for the middle and
# darker tones and the higher-resolution scans.
EXTRA_COMMUNITY = {"skins01_cc0": ["skins/callharvey3d_midtoned_female/", "skins/cutoff3d_indian_female_skin/"],
                   "skins02_cc0": ["skins/mindfront_aksel_skin/", "skins/mindfront_skin_male_african_middleage/"]}
SYSTEM_MANIFEST = "packs/makehuman_system_assets.json"


def item_name(path: str) -> str:
    """The manifest's name for the item a pack file belongs to: its folder,
    or for an eye material, the file (eyes/materials/brown_eye.png: brown)."""
    folder, file = path.rsplit("/", 1)
    if folder == "eyes/materials":
        return file.split(".")[0].removesuffix("_eye")
    return folder.rsplit("/", 1)[-1]


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


def wanted_folders(prefix: str = "mh_system/") -> set[str]:
    """Pack folders named by the catalogue under `prefix`, e.g. `hair/short02/`."""
    catalogue = json.loads((HERE / "catalogue.json").read_text(encoding="utf-8"))
    folders = set()
    for item in catalogue["items"]:
        path = item.get("mhclo", "")
        if path.startswith(prefix):
            folders.add(path[len(prefix):].rsplit("/", 1)[0] + "/")
    return folders


# The community's asset packs the catalogue takes items from: hair01_cc0 is
# CC0, hair02/03_ccby are CC-BY (each item's author credited, see README).
COMMUNITY = "https://files2.makehumancommunity.org/asset_packs/{base}/{pack}.zip"
COMMUNITY_LICENCES = ("CC0", "CC-BY")


def fetch_community() -> None:
    """The catalogue's items from the community packs, checked against each
    pack's own manifest: an item not marked CC0 or CC-BY there is refused."""
    catalogue = json.loads((HERE / "catalogue.json").read_text(encoding="utf-8"))
    packs = {item["mhclo"].split("/")[1] for item in catalogue["items"]
             if item.get("mhclo", "").startswith("community_packs/")} | set(EXTRA_COMMUNITY)
    for pack in sorted(packs):
        folders = wanted_folders(f"community_packs/{pack}/") | set(EXTRA_COMMUNITY.get(pack, []))
        url = COMMUNITY.format(base=pack.split("_")[0], pack=pack)
        archive = zipfile.ZipFile(io.BufferedReader(RemoteFile(url), buffer_size=1 << 20))
        manifest = next(m for m in archive.namelist() if m.startswith("packs/") and m.endswith(".json"))
        licences = json.loads(archive.read(manifest))
        for folder in folders:
            name = folder.rstrip("/").rsplit("/", 1)[-1]
            if licences.get(name, {}).get("license") not in COMMUNITY_LICENCES:
                raise SystemExit(f"{pack}/{name} is not marked {' or '.join(COMMUNITY_LICENCES)}; not using it")
        members = [m for m in archive.infolist() if not m.is_dir()
                   and (any(m.filename.startswith(f) for f in folders) or m.filename == manifest)]
        for member in members:
            if not (ASSETS / "community_packs" / pack / member.filename).exists():
                archive.extract(member, ASSETS / "community_packs" / pack)
        print(f"{pack}: {len(members)} files for {len(folders)} items")


def main() -> None:
    folders = wanted_folders() | set(EXTRA_SYSTEM)
    pack = zipfile.ZipFile(io.BufferedReader(RemoteFile(PACK), buffer_size=1 << 20))
    licences = json.loads(pack.read(SYSTEM_MANIFEST))
    members = [m for m in pack.infolist()
               if not m.is_dir() and (any(m.filename.startswith(f) for f in folders) or m.filename == SYSTEM_MANIFEST)]
    for member in members:
        if member.filename != SYSTEM_MANIFEST and licences.get(item_name(member.filename), {}).get("license") != "CC0":
            raise SystemExit(f"{member.filename} is not marked CC0 in the pack's manifest; not using it")
    for member in members:
        target = ASSETS / "mh_system" / member.filename
        if not target.exists():
            pack.extract(member, ASSETS / "mh_system")
    print(f"mh_system: {len(members)} files for {len(folders)} items")
    fetch_community()

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
