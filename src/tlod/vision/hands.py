"""Hand detection and localisation in the robot's frame.

Split deliberately into two stages:

  HandDetector   image -> 2D landmarks. Swappable. MediaPipe on the host
                 today; a smart camera that does detection on-sensor
                 (IMX500, Grove Vision AI) drops in here later without
                 anything downstream noticing.

  HandLocator    2D landmarks -> a metric position in the robot base frame.
                 Shared by every detector, because turning pixels into
                 metres is a property of the camera geometry, not of
                 whichever neural network found the hand.

Note on MediaPipe versions: early 0.10 exposed `mp.solutions.hands`,
which is what essentially every tutorial online still uses. It was
dropped at 0.10.30 and is not in 1.0 either. This module uses the Tasks
API and a downloaded `.task` bundle, which is the only supported path
going forward.

That same 0.10.30 rewrote the Tasks API's insides -- pybind11 and
protobuf graph configs became ctypes calls into a bundled
`libmediapipe.so`/`.dylib` -- but not its surface, and 1.0 did not change
it either. Every name used below is spelled and behaves the same on
0.10.14 through 1.0.1: `mp.Image`/`mp.ImageFormat` (still re-exported
from the package root, now out of `tasks.python.vision.core`),
`BaseOptions` and its `Delegate` enum, `HandLandmarker`,
`HandLandmarkerOptions`, `RunningMode`, `detect_for_video(image, ms)` and
the result's `.hand_landmarks` / `.handedness[i][0].category_name`.
So this file needs no version check; which build gets installed is a
packaging question, and `pyproject.toml`'s `hands` extra explains it.
"""

from __future__ import annotations

import abc
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tlod.types import Frame, HandObservation

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")

# MediaPipe's 21-landmark topology.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
PINKY_MCP = 17
PINKY_TIP = 20

# Distance across the knuckles, index MCP to pinky MCP. Used to recover
# metric depth from apparent size. Adult range is roughly 7-9 cm; the
# error this introduces is proportional, so a 10% wrong palm gives a 10%
# wrong depth. Good enough to aim, not good enough to grasp.
PALM_WIDTH_M = 0.081


@dataclass(slots=True)
class Hand2D:
    """A detected hand in image coordinates."""

    landmarks: np.ndarray        # (21, 2) pixels
    confidence: float
    handedness: str              # "Left" | "Right" | "unknown"
    stamp: float

    @property
    def palm_center(self) -> np.ndarray:
        """Halfway between the wrist and the knuckle line -- the flat of the palm.

        This is where the arm aims, so it is worth being exact about which
        point it is. Fingertips are out: they move independently of the
        hand and a strike aimed at one lands on a moving target. The wrist
        alone is out too: it swings widely as the hand rotates.

        The obvious middle ground -- the plain centroid of the wrist and
        the four knuckles -- was what this returned, and it is wrong in a
        way that only shows up on hardware. Four of its five points are
        the knuckles, so it sits 80% of the way from wrist to knuckle
        line: not the palm, but the crease where the fingers begin. On the
        arm that reads as forever aiming at the edge of the index finger,
        and it is the worst place on the hand to aim, because the finger
        bases curve away and the paddle slides off them.

        Weighting the wrist against the knuckles as a whole puts it at
        50%, on the flat middle of the palm. Flat matters twice over: it
        is a more consistent height, and contact is judged from how far
        short the paddle stopped.
        """
        knuckles = self.landmarks[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)
        return 0.5 * self.landmarks[WRIST] + 0.5 * knuckles

    @property
    def palm_width_px(self) -> float:
        return float(np.linalg.norm(self.landmarks[INDEX_MCP] - self.landmarks[PINKY_MCP]))


class HandDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame: Frame) -> list[Hand2D]: ...

    def close(self) -> None:
        pass


class NullHandDetector(HandDetector):
    """Finds no hands, ever.

    Two uses, both deliberate rather than a placeholder: object-only runs,
    where loading MediaPipe to return nothing would be a heavy way to do
    it, and boards where MediaPipe will not import or will not build a
    landmarker. In the second case the rest of the pipeline -- camera,
    object detection, tracking, control, the publisher -- is unaffected
    and worth keeping alive, so `cli.build_detector` substitutes this and
    says so loudly instead of dying on an import traceback.
    """

    def detect(self, frame: Frame) -> list[Hand2D]:
        return []


def ensure_model(path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the landmark bundle if it is not already present."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand landmarker model to %s", path)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


class MediaPipeHandDetector(HandDetector):
    """CPU hand landmarks via MediaPipe Tasks.

    Runs in VIDEO mode rather than LIVE_STREAM. LIVE_STREAM delivers
    results on its own callback thread, which sounds faster but only moves
    the wait somewhere else and makes ordering nondeterministic; we already
    have a dedicated perception thread, so a synchronous call there is both
    simpler and easier to time.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        delegate: str = "cpu",
    ) -> None:
        # Imported here, not at module scope, so `tlod --help` and every
        # command that uses a different detector stay free of MediaPipe.
        # These five names are identical on 0.10.14 and on 1.0.x; see the
        # module docstring before reaching for a version check.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        model_path = ensure_model(model_path)
        # Force the CPU delegate by default. On macOS, MediaPipe otherwise
        # tries to open a Metal GPU service that is unavailable outside a
        # windowed app context and aborts the whole process with
        # "Check failed: service_" -- a hard crash, not an exception, so it
        # cannot be caught and retried. CPU is also fast enough here
        # (~10 ms at 720p) that the GPU path is not worth the fragility.
        # On macOS this flag stops mattering at MediaPipe 1.0, which takes
        # the Metal path regardless; that is a pin, not a setting, and
        # pyproject.toml's `hands` extra is where it lives.
        chosen = BaseOptions.Delegate.GPU if delegate.lower() == "gpu" else BaseOptions.Delegate.CPU
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=chosen),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame: Frame) -> list[Hand2D]:
        import cv2

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # Tasks VIDEO mode requires strictly increasing integer timestamps.
        ms = int(frame.stamp * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms

        if self._landmarker is None:
            raise RuntimeError("detector is closed")
        result = self._landmarker.detect_for_video(image, ms)
        h, w = frame.image.shape[:2]

        hands: list[Hand2D] = []
        for i, lms in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=float)
            label, score = "unknown", 1.0
            if i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].category_name
                score = float(result.handedness[i][0].score)
            hands.append(Hand2D(pts, score, label, frame.stamp))
        return hands

    def close(self) -> None:
        """Close the landmarker, and make sure it is closed only once.

        MediaPipe's HandLandmarker also closes itself from __del__, which
        on 1.0.x runs during interpreter shutdown -- by which point the
        module globals its dispatcher needs are already None, and it
        raises `TypeError: 'NoneType' object is not callable` from inside
        a destructor. Python prints that and continues, so nothing is
        wrong, but a clean run ends in a traceback and looks like one.

        Dropping our reference after closing lets it be collected while
        the interpreter is still whole.
        """
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()


class ScriptedHandDetector(HandDetector):
    """Replays a fixed pixel trajectory. Used by tests and the simulator so
    the whole perception -> control path can be exercised deterministically,
    including the timing, with no camera and no model."""

    def __init__(self, path_fn, palm_width_px: float = 90.0) -> None:
        self.path_fn = path_fn
        self.palm_width_px = palm_width_px

    def detect(self, frame: Frame) -> list[Hand2D]:
        cx, cy = self.path_fn(frame.stamp)
        half = self.palm_width_px / 2.0
        lms = np.tile(np.array([cx, cy]), (21, 1))
        lms[INDEX_MCP] = [cx - half, cy]
        lms[PINKY_MCP] = [cx + half, cy]
        return [Hand2D(lms, 1.0, "Right", frame.stamp)]


class HandLocator:
    """Lifts 2D hands into the robot base frame.

    `depth_mode` picks the assumption used to resolve the ray:

      "plane"  the hand is at a fixed height above the table. Exact if it
               really is, e.g. a palm resting flat for a slap game.
      "size"   depth from apparent palm width. Works anywhere in the volume
               and needs no assumption about height, at the cost of being
               only as accurate as PALM_WIDTH_M matches this particular
               human. Calibrate per player if it matters.
      "auto"   size-based, but clamped to a sane height band. The default:
               it degrades gracefully when a hand is partly occluded and
               the apparent width collapses.
    """

    def __init__(
        self,
        projector,
        depth_mode: str = "auto",
        hand_height: float = 0.06,
        height_band: tuple[float, float] = (0.0, 0.40),
        palm_width_m: float = PALM_WIDTH_M,
    ) -> None:
        self.projector = projector
        self.depth_mode = depth_mode
        self.hand_height = hand_height
        self.height_band = height_band
        self.palm_width_m = palm_width_m

    def _depth_from_size(self, hand: Hand2D) -> float | None:
        w_px = hand.palm_width_px
        if w_px < 1.0:
            return None
        fx = float(self.projector.intr.K[0, 0])
        return fx * self.palm_width_m / w_px

    def locate(self, hand: Hand2D) -> HandObservation | None:
        u, v = hand.palm_center

        if self.depth_mode == "plane":
            p = self.projector.pixel_to_plane(u, v, self.hand_height)
            if p is None:
                return None
        else:
            depth = self._depth_from_size(hand)
            if depth is None:
                return None
            origin, direction = self.projector.ray(u, v)
            # `depth` is distance along the optical axis; the ray is a unit
            # vector in base coordinates, so convert using the ray's
            # component along the camera's forward axis.
            forward = self.projector.extr.R[:, 2]
            cos_t = float(np.dot(direction, forward))
            if cos_t <= 1e-6:
                return None
            p = origin + direction * (depth / cos_t)

            if self.depth_mode == "auto":
                lo, hi = self.height_band
                if not (lo <= p[2] <= hi):
                    clamped = self.projector.pixel_to_plane(u, v, float(np.clip(p[2], lo, hi)))
                    if clamped is not None:
                        p = clamped

        landmarks_3d = None
        return HandObservation(
            position=p,
            stamp=hand.stamp,
            velocity=None,
            landmarks=landmarks_3d,
            handedness=hand.handedness,
            confidence=hand.confidence,
        )

    def locate_all(self, hands: list[Hand2D]) -> list[HandObservation]:
        out = []
        for h in hands:
            obs = self.locate(h)
            if obs is not None:
                out.append(obs)
        return out
