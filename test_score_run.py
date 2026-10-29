"""The keep-or-drop gate and the consent check in score_run.py, on made-up
numbers: no photos, models or assets needed (CI runs it).

Run: .venv/Scripts/python.exe test_score_run.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import tempfile

import score_run


def faces(n: int, value: float, held: float = 3.0) -> dict:
    return {f"p{i:02d}": {"others_p": value + 0.001 * i, "identity_p": 0.6, "lm_held": held} for i in range(n)}


def check_gate():
    noise = {f"p{i:02d}": {"others_p": 0.01, "identity_p": 0.01, "lm_held": 0.1} for i in range(12)}
    # Few people: a clear gain is kept, provisionally; a loss is dropped.
    assert score_run.gate("others_p", faces(3, 0.50), faces(3, 0.45), noise)
    assert not score_run.gate("others_p", faces(3, 0.45), faces(3, 0.50), noise)
    # A gain on average that costs one face more than two of its spreads is dropped.
    now = faces(3, 0.50)
    now["p01"]["others_p"] = 0.40
    assert not score_run.gate("others_p", now, faces(3, 0.45), noise)
    # Lower is better for landmark error.
    assert score_run.gate("lm_held", faces(3, 0.5, held=2.5), faces(3, 0.5, held=3.0), noise)
    # A shape gain that drops identity beyond its spread is dropped.
    worse = faces(3, 0.5, held=2.5)
    for row in worse.values():
        row["identity_p"] = 0.55
    assert not score_run.gate("lm_held", worse, faces(3, 0.5, held=3.0), noise)
    # Ten or more people: the tests must agree too. Half better, half worse fails.
    mixed = faces(12, 0.45)
    for i, name in enumerate(mixed):
        mixed[name]["others_p"] += 0.012 if i % 2 else -0.004
    assert not score_run.gate("others_p", mixed, faces(12, 0.45), noise)
    assert score_run.gate("others_p", faces(12, 0.47), faces(12, 0.45), noise)
    # No spread measured: no verdict.
    assert not score_run.gate("others_p", faces(3, 0.50), faces(3, 0.45), {})
    print("ok  gate: keeps clear gains, drops losses, one bad face and identity drops")


def check_level():
    noise = {f"p{i:02d}": {"identity_p": 0.02, "lm_held": 0.2} for i in range(4)}
    base = {n: {"identity_p": 0.5, "lm_held": 3.0} for n in noise}
    assert score_run.level({n: dict(r) for n, r in base.items()}, base, noise)      # unchanged: kept
    one = {n: dict(r) for n, r in base.items()}
    one["p02"]["lm_held"] = 3.5                                                     # one face 2.5 spreads worse
    assert not score_run.level(one, base, noise)
    drift = {n: {"identity_p": 0.48, "lm_held": 3.0} for n in base}                 # all a spread worse
    assert not score_run.level(drift, base, noise)
    better = {n: {"identity_p": 0.55, "lm_held": 2.0} for n in base}
    assert score_run.level(better, base, noise)
    print("ok  gate level: an unchanged or better run kept, one face or a drift worse dropped")


def check_add_person():
    import add_person
    assert add_person.turn_problem("left35", -33.0) is None and add_person.turn_problem("right35", 31.0) is None
    assert add_person.turn_problem("left35", 7.4), "a side photo turned 7 degrees passed"
    assert add_person.turn_problem("front", 14.0), "a turned front photo passed"
    assert add_person.turn_problem("o1", 40.0) is None, "another-day photos may be turned"
    print("ok  add person: side, held-out and front photos checked for their turn")


def check_spreads():
    got = score_run.spreads({"p01": {"x": 1.0}, "p01~0": {"x": 1.2}, "p01~1": {"x": 0.8}, "p02": {"x": 2.0}})
    assert abs(got["p01"]["x"] - 0.2) < 1e-6 and "x" not in got["p02"], got
    print("ok  spreads: per person, across its noise copies")


def check_consent():
    today = dt.date.today()
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        (folder / "consent.csv").write_text(
            "consent_id,pseudonym,date,scopes,retain_until,withdrawn\n"
            f"c1,p01,{today},build;score,{today + dt.timedelta(days=30)},\n"
            f"c2,p02,{today},build;score,{today + dt.timedelta(days=30)},{today}\n"
            f"c3,p03,{today},build;score,{today - dt.timedelta(days=1)},\n"
            f"c4,p04,{today},build,{today + dt.timedelta(days=30)},\n"
            f"c5,p05,{today},build;score,{today + dt.timedelta(days=30)},\n", encoding="utf-8")
        (folder / "test-set.json").write_text(json.dumps(
            [{"name": f"p0{i}", "photo": "x.jpg", "consent_id": f"c{i}"} for i in range(1, 6)] +
            [{"name": "p06", "photo": "x.jpg"}] + [{"name": "s01", "photo": "x.jpg", "synthetic": True}] +
            [{"name": "p05", "photo": "x.jpg", "consent_id": "c5", "split": "lockbox"}]), encoding="utf-8")
        dev = [f["name"] for f in score_run.load_faces(folder / "test-set.json", "dev")]
        assert dev == ["p01", "p05", "s01"], dev
        locked = [f["split"] for f in score_run.load_faces(folder / "test-set.json", "lockbox")]
        assert locked == ["lockbox"], locked
    print("ok  consent: withdrawn, expired, unscoped and unregistered faces left out; generated ones taken")


def check_rubric():
    assert score_run.lower_better("hair") and score_run.lower_better("hair_dE") and score_run.lower_better("lm_held")
    assert not score_run.lower_better("hair_iou") and not score_run.lower_better("identity_p")
    good = {"skin_dE": 3.0, "iris_dE": 6.0, "sclera_dE": 4.0, "teeth_dE": 5.0}
    rows = {f"p{i:02d}": dict(good) for i in range(5)}
    rows["p04"]["skin_dE"] = 7.0                        # one face in five misses: 80% still clears
    text = score_run.rubric(rows)
    line = lambda area: next(l for l in text.splitlines() if l.startswith(area))
    assert "8 reached" in line("skin, eyes, teeth") and "not scored" in line("hair"), text
    assert "not measured" in line("likeness") and "provisional" in text, text
    rows["p03"]["skin_dE"] = 7.0                        # two in five: 60%, below
    assert "below 8" in next(l for l in score_run.rubric(rows).splitlines() if l.startswith("skin"))
    print("ok  rubric: bars read the right way round, 80% of faces must clear each")


def check_robustness():
    scores = {"p01": {"identity_p": 0.55}, "p02": {"identity_p": 0.50}}
    good = {"p01~dark": {"kind": "dark", "result": "built", "expect": "photo is dark",
                         "advice": ["The photo is dark (the face at 40 of 255)"]},
            "p01~jpeg-30": {"kind": "jpeg 30", "result": "built", "expect": None, "advice": [], "identity_p": 0.53},
            "p02~small": {"kind": "small", "result": "refused", "expect": "pixels"}}
    verdict, text = score_run.robustness(scores, good)
    assert verdict == "8 reached", text
    crashed = {**good, "p02~alpha": {"kind": "alpha", "result": "crashed", "error": "Traceback"}}
    assert score_run.robustness(scores, crashed)[0] == "below 8"
    missed = {**good, "p02~blurred": {"kind": "blurred", "result": "built", "expect": "blurred", "advice": []}}
    verdict, text = score_run.robustness(scores, missed)
    assert verdict == "below 8" and "p02~blurred" in text, text
    lost = {**good, "p02~tilted-15": {"kind": "tilted 15", "result": "built", "expect": None, "advice": [], "identity_p": 0.30},
            "p02~alpha": {"kind": "alpha", "result": "built", "expect": None, "advice": [], "identity_p": 0.20}}
    assert score_run.robustness(scores, lost)[0] == "below 8"      # two of three builds lost over 0.05
    print("ok  robustness: a crash, a missed warning or builds that lose the face keep it below 8")


def check_review_page():
    import base64
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "renders").mkdir()
        names = [f"p{i:02d}" for i in range(8)]
        for name in names:
            for tag, shade in (("new", 200), ("old", 60)):
                Image.new("RGBA", (40, 40), (shade, shade, shade, 255)).save(root / "renders" / f"{name}-{tag}p.png")
        Image.new("RGB", (40, 40), (128, 90, 70)).save(root / "photo.png")
        old_out, old_upright = score_run.OUT, score_run.upright
        score_run.OUT, score_run.upright = root, lambda path: root / "photo.png"
        try:
            page = score_run.review_page("new", "old", [{"name": n, "photo": "x.jpg"} for n in names])
        finally:
            score_run.OUT, score_run.upright = old_out, old_upright
        text = page.read_text(encoding="utf-8")
    keys = [base64.b64decode(k).decode() for k in __import__("re").findall(r'data-key="([^"]+)"', text)]
    assert len(keys) == 8 and set(keys) == {"left", "right"}, keys     # shuffled, not always one side
    assert text.count("data:image/jpeg;base64,") == 8 * 3, "a face lacks its photo or a build, in the page"
    assert "-newp" not in text and "-oldp" not in text, "the page names the builds"
    print("ok  review page: both builds per face in the page, sides shuffled, the new one hidden")


if __name__ == "__main__":
    check_gate()
    check_level()
    check_add_person()
    check_spreads()
    check_consent()
    check_rubric()
    check_robustness()
    check_review_page()
