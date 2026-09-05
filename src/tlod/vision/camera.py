"""Camera capture, optimised for latency rather than throughput.

The default OpenCV usage pattern -- `cap.read()` in your main loop -- is
wrong for this robot. V4L2/AVFoundation keep a queue of frames; if your
loop is even slightly slower than the camera, `read()` hands you the
*oldest* queued frame and you silently fall further behind. People measure
"30 fps" and never notice they are looking 150 ms into the past.

So: a dedicated thread does nothing but `grab()` as fast as the camera
emits, discarding the backlog, and `retrieve()`s only the newest frame.
The consumer always gets the freshest frame available and never blocks the
camera. Combined with BUFFERSIZE=1 and MJPEG this is the difference
between ~130 ms and ~40 ms of pipeline latency on a Logitech C922.

Shutter timestamping: `grab()` returns when a frame lands, so the shutter
opened roughly one frame period plus the USB transfer earlier. That offset
is `latency_offset`, and `tlod bench camera` measures it for your actual
camera instead of trusting this guess. Every downstream prediction depends
on it being approximately right.
"""

from __future__ import annotations

import abc
import logging
import threading
import time

import cv2
import numpy as np

from tlod.types import Frame

log = logging.getLogger(__name__)


class Camera(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> Frame | None:
        """Newest available frame, or None if none has arrived yet."""

    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[int, int]: ...

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class OpenCVCamera(Camera):
    def __init__(
        self,
        index: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        fourcc: str = "MJPG",
        latency_offset: float = 0.035,
        autofocus: bool = False,
        autoexposure: bool = False,
        exposure: float | None = None,
        backend: int | None = None,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.latency_offset = latency_offset
        self.autofocus = autofocus
        self.autoexposure = autoexposure
        self.exposure = exposure
        self.backend = backend

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._count = 0
        self._intervals: list[float] = []

    def start(self) -> None:
        cap = cv2.VideoCapture(self.index) if self.backend is None else cv2.VideoCapture(self.index, self.backend)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {self.index!r}")

        # Order matters: FOURCC before size before fps, or drivers ignore it.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Auto-anything runs on the camera before the frame ships, and the
        # adjustment itself costs milliseconds. Fixed settings also stop the
        # exposure hunting when a hand sweeps through frame, which otherwise
        # blurs precisely the motion we care about.
        if not self.autofocus:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if not self.autoexposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual on V4L2
        if self.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

        self._cap = cap
        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (self.width, self.height):
            log.warning("camera gave %dx%d, asked for %dx%d", *actual, self.width, self.height)
        self.width, self.height = actual

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        last = time.perf_counter()
        while self._running:
            if not self._cap.grab():
                time.sleep(0.001)
                continue
            now = time.perf_counter()
            ok, image = self._cap.retrieve()
            if not ok:
                continue
            self._count += 1
            self._intervals.append(now - last)
            if len(self._intervals) > 120:
                self._intervals.pop(0)
            last = now
            frame = Frame(image=image, stamp=now - self.latency_offset, index=self._count)
            with self._lock:
                self._frame = frame

    def read(self) -> Frame | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def measured_fps(self) -> float:
        if not self._intervals:
            return 0.0
        return 1.0 / float(np.mean(self._intervals))


class MockCamera(Camera):
    """Synthetic camera: a moving disc on a plain background.

    Enough to exercise the full perception -> control path, measure loop
    timing, and run tests in CI, without hardware. The disc moves on a
    known trajectory so a tracker's prediction can be scored against
    ground truth.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        speed: float = 1.0,
        radius: int = 45,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.speed = speed
        self.radius = radius
        self._t0 = 0.0
        self._count = 0
        self._running = False

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def truth(self, t: float | None = None) -> tuple[float, float]:
        """Ground-truth disc centre in pixels at time `t`."""
        t = (time.perf_counter() - self._t0) if t is None else t
        cx = self.width / 2 + 0.30 * self.width * np.sin(t * self.speed)
        cy = self.height / 2 + 0.20 * self.height * np.cos(t * self.speed * 0.7)
        return float(cx), float(cy)

    def read(self) -> Frame | None:
        if not self._running:
            return None
        now = time.perf_counter()
        img = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        cx, cy = self.truth(now - self._t0)
        cv2.circle(img, (int(cx), int(cy)), self.radius, (60, 90, 220), -1)
        self._count += 1
        return Frame(image=img, stamp=now, index=self._count)

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


def list_cameras(max_index: int = 6) -> list[int]:
    """Indices that open successfully. Noisy on some backends; best effort."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
