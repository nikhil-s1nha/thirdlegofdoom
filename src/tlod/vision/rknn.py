"""Hand detection on the Rockchip RK3588 NPU (Orange Pi 5).

Why this exists: MediaPipe costs ~19 ms per frame on an Apple M-series
CPU and considerably more on the Orange Pi's A76 cores, where it also has
to compete with the control loop for the same silicon. The RK3588's 6
TOPS NPU runs a detector in single-digit milliseconds and, more usefully,
runs it somewhere that is not the CPU.

The important caveat, stated up front: **a box detector is not a landmark
detector.** This returns a hand bounding box, from which a palm centre
and an apparent width are synthesised. That is entirely sufficient for
hand-slap, which only ever needs "where is the hand and how big does it
look". It is *not* sufficient for anything reading finger pose --
rock-paper-scissors, counting, pointing. Those need real landmarks, so
they keep MediaPipe on the CPU, or need a second landmark model on the
NPU. Do not quietly swap this in under a gesture game and wonder why it
cannot tell a fist from a palm.

  !! UNVERIFIED ON HARDWARE !!
  Written against the rknn-toolkit-lite2 API while the board was
  elsewhere. Model conversion, anchor decoding and the exact output
  tensor layout all depend on which detector is compiled, and must be
  checked against the model actually used. See docs/deployment.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tlod.types import Frame
from tlod.vision.hands import INDEX_MCP, PINKY_MCP, Hand2D, HandDetector

log = logging.getLogger(__name__)


class RknnHandDetector(HandDetector):
    """Box-only hand detector running on the RK3588 NPU."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        confidence: float = 0.5,
        nms_iou: float = 0.45,
        core_mask: int = 0,          # 0 = auto; RK3588 has three NPU cores
        palm_width_fraction: float = 0.62,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.confidence = confidence
        self.nms_iou = nms_iou
        # A hand box is not a palm. The knuckle span is roughly this
        # fraction of the box width for a hand seen face-on, and this
        # single number is what the size-based depth estimate rides on,
        # so it wants measuring against a ruler rather than trusting.
        self.palm_width_fraction = palm_width_fraction
        self._rknn = None

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no RKNN model at {self.model_path}. Convert one with "
                "rknn-toolkit2 on an x86 host; see docs/deployment.md"
            )
        self._open(core_mask)

    def _open(self, core_mask: int) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as e:  # pragma: no cover - board-only dependency
            raise RuntimeError(
                "rknn-toolkit-lite2 is not installed. It ships from Rockchip, "
                "not PyPI; see docs/deployment.md"
            ) from e

        rknn = RKNNLite()
        if rknn.load_rknn(str(self.model_path)) != 0:
            raise RuntimeError(f"failed to load {self.model_path}")
        if rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("failed to initialise the NPU runtime")
        self._rknn = rknn
        log.info("RKNN hand detector ready: %s", self.model_path.name)

    # -- inference ---------------------------------------------------------
    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize preserving aspect ratio, padding the remainder.

        Squashing to a square instead would distort the hand, and the
        apparent-width depth estimate reads directly off that width.
        """
        import cv2

        h, w = image.shape[:2]
        tw, th = self.input_size
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas, scale, (ox, oy)

    def detect(self, frame: Frame) -> list[Hand2D]:
        if self._rknn is None:
            return []
        import cv2

        padded, scale, (ox, oy) = self._letterbox(frame.image)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        outputs = self._rknn.inference(inputs=[np.expand_dims(rgb, 0)])
        boxes, scores = self._decode(outputs)

        hands: list[Hand2D] = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            # Undo the letterbox to get back to original image pixels.
            x1 = (x1 - ox) / scale
            x2 = (x2 - ox) / scale
            y1 = (y1 - oy) / scale
            y2 = (y2 - oy) / scale
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half = abs(x2 - x1) * self.palm_width_fraction / 2.0

            # Synthesise the minimum landmark set the rest of the stack
            # reads: the palm centroid and the two knuckles that define
            # apparent width. Every other landmark is the centre, which
            # is honest -- this detector does not know where they are.
            landmarks = np.tile(np.array([cx, cy]), (21, 1))
            landmarks[INDEX_MCP] = [cx - half, cy]
            landmarks[PINKY_MCP] = [cx + half, cy]
            hands.append(Hand2D(landmarks, float(score), "unknown", frame.stamp))
        return hands

    def _decode(self, outputs) -> tuple[np.ndarray, np.ndarray]:
        """Turn raw tensors into boxes and scores.

        Deliberately minimal and model-specific. A YOLO-family export
        gives (N, 5+) rows of xywh plus objectness; anchor-based models
        need their anchors applied first. Replace this to match whatever
        was actually compiled -- getting it subtly wrong yields boxes that
        look plausible and are systematically offset, which then shows up
        as a robot that consistently slaps two centimetres to the left.
        """
        raw = outputs[0]
        raw = raw.reshape(-1, raw.shape[-1]) if raw.ndim > 2 else raw
        if raw.shape[-1] < 5:
            return np.empty((0, 4)), np.empty(0)

        xywh, objectness = raw[:, :4], raw[:, 4]
        keep = objectness >= self.confidence
        xywh, objectness = xywh[keep], objectness[keep]
        if len(xywh) == 0:
            return np.empty((0, 4)), np.empty(0)

        boxes = np.stack([
            xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
            xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2,
        ], axis=1)
        order = self._nms(boxes, objectness)
        return boxes[order], objectness[order]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        order = np.argsort(-scores)
        keep: list[int] = []
        while len(order):
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
            order = rest[iou < self.nms_iou]
        return keep

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None
