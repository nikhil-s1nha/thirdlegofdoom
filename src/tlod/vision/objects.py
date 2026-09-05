"""Object detection on the tabletop.

Starts with colour segmentation, deliberately. A learned detector is the
obvious reach, but for coloured game pieces on a known table under
controlled lighting, HSV thresholding is ~1 ms, needs no model, no
training data and no accelerator, and is trivially debuggable when it goes
wrong. The interface is what matters: when a YOLO on the Orange Pi 5's NPU
replaces this, nothing downstream changes.

Objects are resolved onto the table plane, where ray-plane intersection is
exact rather than an assumption -- objects genuinely do rest on the table.
This is the one place monocular depth is not a compromise.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import cv2
import numpy as np

from tlod.types import Detection, Frame


@dataclass(slots=True)
class ColorSpec:
    """An HSV band. OpenCV hue is 0-179, not 0-359."""

    label: str
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    # Red wraps the hue origin, so it needs two bands.
    lower2: tuple[int, int, int] | None = None
    upper2: tuple[int, int, int] | None = None


DEFAULT_COLORS: list[ColorSpec] = [
    ColorSpec("red", (0, 120, 80), (8, 255, 255), (172, 120, 80), (179, 255, 255)),
    ColorSpec("green", (40, 90, 60), (85, 255, 255)),
    ColorSpec("blue", (95, 110, 60), (130, 255, 255)),
    ColorSpec("yellow", (20, 110, 110), (33, 255, 255)),
]


class ObjectDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame: Frame) -> list[Detection]: ...

    def close(self) -> None:
        pass


class ColorBlobDetector(ObjectDetector):
    def __init__(
        self,
        projector,
        colors: list[ColorSpec] | None = None,
        table_z: float = 0.0,
        min_area_px: int = 400,
        max_objects: int = 12,
        blur: int = 5,
    ) -> None:
        self.projector = projector
        self.colors = colors if colors is not None else DEFAULT_COLORS
        self.table_z = table_z
        self.min_area_px = min_area_px
        self.max_objects = max_objects
        self.blur = blur

    def detect(self, frame: Frame) -> list[Detection]:
        img = cv2.GaussianBlur(frame.image, (self.blur, self.blur), 0) if self.blur else frame.image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        out: list[Detection] = []
        for spec in self.colors:
            mask = cv2.inRange(hsv, np.array(spec.lower), np.array(spec.upper))
            if spec.lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(spec.lower2), np.array(spec.upper2))
            # Open then close: drop speckle, then fill the holes that
            # specular highlights punch in a glossy game piece.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area_px:
                    continue
                m = cv2.moments(c)
                if m["m00"] == 0:
                    continue
                u = m["m10"] / m["m00"]
                v = m["m01"] / m["m00"]
                p = self.projector.pixel_to_plane(u, v, self.table_z)
                if p is None:
                    continue
                # Radius from area, converted through the local scale of
                # the projection rather than assumed.
                edge = self.projector.pixel_to_plane(u + np.sqrt(area / np.pi), v, self.table_z)
                radius = float(np.linalg.norm(edge - p)) if edge is not None else 0.0
                out.append(
                    Detection(
                        label=spec.label,
                        position=p,
                        stamp=frame.stamp,
                        confidence=float(min(1.0, area / (self.min_area_px * 4))),
                        pixel=(float(u), float(v)),
                        radius=radius,
                    )
                )

        out.sort(key=lambda d: -d.confidence)
        return out[: self.max_objects]


class NullObjectDetector(ObjectDetector):
    def detect(self, frame: Frame) -> list[Detection]:
        return []
