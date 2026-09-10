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
import re
import threading
import time

import cv2
import numpy as np

from tlod.types import Frame

log = logging.getLogger(__name__)

# Added to one frame period to estimate shutter-to-grab latency. Covers
# USB transfer and MJPEG decode. Rough; `tlod bench camera` explains how
# to measure the real thing.
TRANSFER_DECODE_ESTIMATE = 0.015


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

    def __enter__(self) -> Camera:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _cannot_open(index) -> str:
    """Why it did not open, and which index to use instead.

    "could not open camera 0" is true and useless: on this board index 0
    is a Rockchip codec that exists, opens far enough to fail, and is not
    a camera. What the operator needs is the index that *is* one, and the
    kernel already knows it.
    """
    lines = [f"could not open camera {index!r}."]
    try:
        nodes = capture_nodes()
    except Exception:
        nodes = {}
    if nodes:
        lines.append("  capture nodes this board is showing:")
        for i, name in sorted(nodes.items()):
            lines.append(f"    {i:>3}  {name}")
        lines.append("  Pass the one that is your camera as --camera N, or set")
        lines.append("  camera.index in the config. The Rockchip rkvdec/rkvenc/rga")
        lines.append("  nodes are hardware codecs, not cameras.")
    else:
        lines.append("  no v4l2 capture nodes found at all. Check `lsusb` for the")
        lines.append("  camera, and that it is not on a hub that just dropped it.")
    lines.append("  `tlod cameras` lists this too. Indices move across reboots and")
    lines.append("  replugs -- never trust last week's.")
    return "\n".join(lines)


class OpenCVCamera(Camera):
    def __init__(
        self,
        index: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        fourcc: str = "MJPG",
        latency_offset: float | None = None,
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
        # None means estimate it from the measured frame period. A fixed
        # constant was wrong in a way that hid itself: the default of
        # 35 ms was *below* one frame period on a camera actually
        # delivering 29.6 fps (33.5 ms), so every timestamp claimed the
        # shutter opened more recently than it possibly could have. The
        # prediction horizon rides directly on this number.
        self._fixed_offset = latency_offset
        self.latency_offset = latency_offset if latency_offset is not None else 0.05
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
            raise RuntimeError(_cannot_open(self.index))

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

        # Frame rate is negotiated too, and silently. A camera that
        # accepts fps=60 and delivers 30 halves the perception rate while
        # the config still claims 60, which then quietly invalidates the
        # prediction horizon. Measured on a MacBook's built-in camera:
        # asked 60, got 29. Warn loudly; `measured_fps` is the truth.
        reported = cap.get(cv2.CAP_PROP_FPS)
        if reported > 0 and abs(reported - self.fps) > 1.0:
            log.warning(
                "camera reports %.0f fps, asked for %d. Check measured_fps "
                "after starting; the configured value is a request, not a promise",
                reported, self.fps,
            )
        self.reported_fps = reported

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

            if self._fixed_offset is None and len(self._intervals) >= 30:
                # grab() returns once a frame has landed, so the shutter
                # opened at least one frame period earlier, plus transfer
                # and decode. This is a lower bound, not a measurement --
                # a true figure needs an external reference (film a
                # millisecond timer). It is, at least, a bound that
                # cannot be below the physically possible.
                period = float(np.median(self._intervals))
                self.latency_offset = period + TRANSFER_DECODE_ESTIMATE
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
    """Synthetic camera driven by a scene.

    Rate-limited to its configured fps. This matters more than it sounds:
    an unthrottled mock produced ~1700 fps, and the perception thread
    spinning that hard starved the control thread badly enough to push
    loop jitter to 70 ms. A simulator that does not respect frame timing
    quietly invalidates every timing conclusion drawn from it.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        scene=None,
        render: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.scene = scene
        self.render = render
        self._t0 = 0.0
        self._count = 0
        self._running = False
        self._next_frame = 0.0
        self._last: Frame | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._next_frame = self._t0
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def read(self) -> Frame | None:
        if not self._running:
            return None
        now = time.perf_counter()
        if now < self._next_frame:
            # Not yet due. Hand back the previous frame unchanged; its
            # index is unchanged too, so consumers skip it.
            return self._last
        # Schedule from the timeline, not from now, to avoid drift.
        period = 1.0 / self.fps
        self._next_frame = max(self._next_frame + period, now - period)

        t = now - self._t0
        if self.render and self.scene is not None:
            img = self.scene.render(t, self.width, self.height)
        else:
            # Tier A does not need pixels unless someone is watching.
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._count += 1
        self._last = Frame(image=img, stamp=now, index=self._count)
        return self._last

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


def capture_nodes() -> dict[int, str]:
    """`/dev/videoN` -> the name its driver reports, for capture nodes only.

    V4L2 only, and empty everywhere else; the caller falls back to probing.
    This exists because on this rig the index is not a guess worth making.
    An Orange Pi 5 enumerates its Rockchip codecs first -- rkvdec, rkvenc,
    rga, several nodes each -- so the USB camera lands somewhere above
    /dev/video10 and the low indices are hardware that opens, reports
    "Not a video capture device", and is not a camera. Probing indices
    alone cannot tell those apart; the driver name can.

    Nodes come in pairs: a UVC camera exposes one capture node and one
    metadata node, and only the first is openable. `V4L2_CAP_VIDEO_CAPTURE`
    is what separates them.
    """
    import glob
    import re

    nodes: dict[int, str] = {}
    for path in sorted(glob.glob("/dev/video*")):
        m = re.fullmatch(r"/dev/video(\d+)", path)
        if m is None:
            continue
        index = int(m.group(1))
        try:
            with open(f"/sys/class/video4linux/video{index}/name") as fh:
                name = fh.read().strip()
        except OSError:
            continue
        # A metadata node shares its parent's name, so the name alone
        # cannot rule one out. `index` under the same directory is 0 for
        # the capture node of a UVC device.
        try:
            with open(f"/sys/class/video4linux/video{index}/index") as fh:
                if fh.read().strip() != "0":
                    continue
        except OSError:
            pass
        nodes[index] = name
    return nodes


def stable_paths() -> dict[int, str]:
    """`/dev/videoN` -> a `/dev/v4l/by-id/...` path that survives a replug.

    The index is not a property of the camera. It is the order the kernel
    happened to enumerate things in, and on this board a USB camera shares
    that numbering with six Rockchip codec nodes -- so unplugging the
    camera, moving it and plugging it back in renumbers it, and every
    command that named an index is now pointing at a video decoder.
    Measured on this rig across one remount: the Arducam went from 1 to 0.

    udev builds by-id from the device's own descriptor, so it names the
    camera rather than its position in a queue. Passing that path as
    `camera.index` is the way to stop having this conversation.
    """
    import glob
    import os

    out: dict[int, str] = {}
    for link in sorted(glob.glob("/dev/v4l/by-id/*")):
        try:
            target = os.path.realpath(link)
        except OSError:
            continue
        m = re.fullmatch(r"/dev/video(\d+)", target)
        if m is not None:
            out.setdefault(int(m.group(1)), link)
    return out


def list_cameras(max_index: int = 24) -> list[int]:
    """Indices that open successfully. Best effort.

    Only capture nodes are probed. Opening every `/dev/videoN` in a range
    was how this used to work and it hangs here: the Rockchip codec nodes
    accept the open and then never answer, so `tlod cameras` printed the
    first line of its own output and stopped, which reads exactly like the
    camera being broken. `capture_nodes()` already knows which nodes claim
    V4L2_CAP_VIDEO_CAPTURE, and nothing else was ever going to be a camera.
    """
    nodes = capture_nodes()
    candidates = sorted(nodes) if nodes else list(range(max_index))
    found = []
    caps = []
    try:
        for i in candidates:
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            caps.append(cap)
            if cap.isOpened():
                found.append(i)
        return found
    finally:
        # Released even when the caller is interrupted, which is the case
        # that cost an evening. A V4L2 device allows several opens but only
        # one set of buffers, so a process *stopped* mid-probe -- Ctrl-Z,
        # or Ctrl-C caught somewhere unhelpful -- keeps its descriptor and
        # every later open fails at VIDIOC_REQBUFS with EBUSY. From the
        # outside that is indistinguishable from a broken camera, and the
        # process holding it does not look like it is doing anything.
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass
