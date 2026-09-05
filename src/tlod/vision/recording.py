"""Record a session and replay it deterministically.

The reason this exists: tuning the tracker on a synthetic trajectory is
guesswork, and tuning it on a live camera is unrepeatable -- you cannot
wave your hand the same way twice, so you cannot tell whether a parameter
change helped or whether you just moved differently. Recording once and
replaying it a thousand times turns filter tuning from an opinion into a
measurement.

It is also how a bug seen once becomes a bug you can re-run.

Format is deliberately dumb: JPEG frames in a directory plus a JSONL
sidecar of timestamps. Not efficient, but inspectable with an image
viewer and a text editor, and immune to a codec being unavailable on the
deployment board.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from tlod.types import Frame
from tlod.vision.camera import Camera


class Recorder:
    """Writes frames and their true shutter timestamps to a directory."""

    def __init__(self, path: str | Path, quality: int = 90) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self._index = 0
        self._meta = (self.path / "frames.jsonl").open("w")
        self._t0: float | None = None

    def add(self, frame: Frame, extra: dict | None = None) -> None:
        if self._t0 is None:
            self._t0 = frame.stamp
        name = f"{self._index:06d}.jpg"
        cv2.imwrite(str(self.path / name), frame.image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        record = {"file": name, "t": frame.stamp - self._t0, "index": frame.index}
        if extra:
            record.update(extra)
        self._meta.write(json.dumps(record) + "\n")
        self._index += 1

    @property
    def count(self) -> int:
        return self._index

    def close(self) -> None:
        self._meta.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ReplayCamera(Camera):
    """Plays a recording back as if it were a camera.

    Two modes. `realtime=True` reproduces the original inter-frame timing,
    which is what you want when testing the control loop, since timing is
    half of what is being tested. `realtime=False` runs as fast as frames
    can be decoded, which is what you want when sweeping filter parameters
    over a recording a hundred times.
    """

    def __init__(self, path: str | Path, realtime: bool = True, loop: bool = False) -> None:
        self.path = Path(path)
        self.realtime = realtime
        self.loop = loop
        meta = self.path / "frames.jsonl"
        if not meta.exists():
            raise FileNotFoundError(f"no recording at {self.path}")
        self.records = [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
        if not self.records:
            raise ValueError(f"recording at {self.path} is empty")
        self._i = 0
        self._t0 = 0.0
        self._running = False
        self._current: Frame | None = None
        probe = cv2.imread(str(self.path / self.records[0]["file"]))
        self._resolution = (probe.shape[1], probe.shape[0])

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._i = 0
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def exhausted(self) -> bool:
        return not self.loop and self._i >= len(self.records)

    def read(self) -> Frame | None:
        if not self._running:
            return None
        if self._i >= len(self.records):
            if not self.loop:
                return self._current
            self._i = 0
            self._t0 = time.perf_counter()

        record = self.records[self._i]
        now = time.perf_counter()
        if self.realtime and (now - self._t0) < record["t"]:
            return self._current  # not due yet; index unchanged so consumers skip

        image = cv2.imread(str(self.path / record["file"]))
        if image is None:
            self._i += 1
            return self._current
        self._current = Frame(image=image, stamp=now, index=self._i + 1)
        self._i += 1
        return self._current

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def __len__(self) -> int:
        return len(self.records)
