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
            [{"name": "p06", "photo": "x.jpg"}] +
            [{"name": "p05", "photo": "x.jpg", "consent_id": "c5", "split": "lockbox"}]), encoding="utf-8")
        dev = [f["name"] for f in score_run.load_faces(folder / "test-set.json", "dev")]
        assert dev == ["p01", "p05"], dev
        locked = [f["split"] for f in score_run.load_faces(folder / "test-set.json", "lockbox")]
        assert locked == ["lockbox"], locked
    print("ok  consent: withdrawn, expired, unscoped and unregistered faces left out")


if __name__ == "__main__":
    check_gate()
    check_spreads()
    check_consent()
