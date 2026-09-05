"""Record/replay round trip.

Deterministic replay is what makes filter tuning a measurement rather
than an opinion, so it is tested like a contract.
"""

import numpy as np
import pytest

from tlod.types import Frame
from tlod.vision.recording import Recorder, ReplayCamera


def make_recording(path, n=12, size=(64, 48)):
    with Recorder(path) as rec:
        for i in range(n):
            img = np.full((size[1], size[0], 3), i * 5, np.uint8)
            rec.add(Frame(img, stamp=i * 0.02, index=i + 1))
    return path


def test_round_trip_preserves_frame_count(tmp_path):
    path = make_recording(tmp_path / "s")
    cam = ReplayCamera(path, realtime=False)
    assert len(cam) == 12
    cam.start()
    seen = set()
    for _ in range(200):
        f = cam.read()
        if f:
            seen.add(f.index)
        if cam.exhausted:
            break
    assert len(seen) == 12


def test_replay_reports_resolution(tmp_path):
    cam = ReplayCamera(make_recording(tmp_path / "s"), realtime=False)
    assert cam.resolution == (64, 48)


def test_replay_is_deterministic(tmp_path):
    """The whole point: the same recording gives the same frames."""
    path = make_recording(tmp_path / "s")

    def run():
        cam = ReplayCamera(path, realtime=False)
        cam.start()
        out = []
        while not cam.exhausted:
            f = cam.read()
            if f:
                out.append(int(f.image[0, 0, 0]))
        return out

    assert run() == run()


def test_realtime_replay_respects_original_timing(tmp_path):
    """Timing is half of what the control loop is being tested on, so a
    realtime replay must not race ahead."""
    path = make_recording(tmp_path / "s", n=6)
    cam = ReplayCamera(path, realtime=True)
    cam.start()
    first = cam.read()
    immediately = cam.read()
    # Frame 2 is 20 ms out; it must not be served yet.
    assert immediately is None or immediately.index == first.index


def test_loop_mode_wraps(tmp_path):
    cam = ReplayCamera(make_recording(tmp_path / "s", n=4), realtime=False, loop=True)
    cam.start()
    for _ in range(20):
        cam.read()
    assert not cam.exhausted


def test_missing_recording_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayCamera(tmp_path / "nope")


def test_recorder_writes_extra_fields(tmp_path):
    import json

    path = tmp_path / "s"
    with Recorder(path) as rec:
        rec.add(Frame(np.zeros((4, 4, 3), np.uint8), 0.0, 1), extra={"note": "hi"})
    line = json.loads((path / "frames.jsonl").read_text().splitlines()[0])
    assert line["note"] == "hi"
