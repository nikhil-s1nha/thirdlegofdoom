"""Verify the vision stack without looking at it.

On a headless board you cannot eyeball whether the boxes land on the
hand, and "it seems to work" is not a measurement. This produces numbers
instead, and separates two things that are easy to conflate:

**Precision** -- is the estimate stable and self-consistent? Detection
rate, frame-to-frame jitter, implausible jumps, depth that stays put as a
hand crosses the frame. All measurable with nothing but a camera. None of
it proves the answer is *right*: a badly calibrated camera produces
beautifully precise, consistently wrong positions.

**Accuracy** -- is the estimate actually correct? That needs ground
truth, and the arm is the only ground truth available: drive the tool to
a configuration, and forward kinematics says where the marker is to
within a millimetre. Detect it, compare. This is the check that catches a
bad extrinsic, and it is the reason `--with-arm` exists.

Run it with no arm to sanity-check the pipeline; run it with the arm to
find out whether the numbers mean anything.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Thresholds:
    """What counts as working. Deliberately loose; tighten once you know
    what your setup actually achieves."""

    min_detection_rate: float = 0.60      # fraction of frames with a hand
    max_jitter_mm: float = 25.0           # position noise on a still hand
    max_jump_rate: float = 0.05           # fraction of frames that teleport
    jump_distance_m: float = 0.25         # what counts as a teleport
    max_depth_spread_mm: float = 80.0     # depth wander as the hand crosses frame
    max_arm_error_mm: float = 30.0        # detected vs forward kinematics


@dataclass
class Result:
    name: str
    passed: bool
    value: float
    limit: float
    units: str
    note: str = ""

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  {mark}  {self.name:<26} {self.value:8.2f} {self.units:<4} (limit {self.limit:g}) {self.note}"


@dataclass
class Report:
    frames: int = 0
    detections: int = 0
    fps: float = 0.0
    results: list[Result] = field(default_factory=list)
    arm_points: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def text(self) -> str:
        lines = [
            "",
            f"  frames {self.frames}, hands detected in {self.detections} "
            f"({self.detections / max(self.frames, 1):.0%}), {self.fps:.1f} fps",
            "",
        ]
        lines += [r.line() for r in self.results]
        if self.notes:
            lines += ["", *(f"  note: {n}" for n in self.notes)]
        lines += ["", f"  {'ALL CHECKS PASSED' if self.passed else 'SOME CHECKS FAILED'}"]
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "frames": self.frames, "detections": self.detections, "fps": self.fps,
            "passed": self.passed,
            "results": [asdict(r) for r in self.results],
            "arm_points": self.arm_points,
            "notes": self.notes,
        }, indent=2))


def check_precision(
    camera,
    detector,
    locator,
    duration: float = 20.0,
    thresholds: Thresholds | None = None,
    save_dir: str | Path | None = None,
    on_progress=None,
    fixed_distance: bool = False,
    on_frame=None,
) -> Report:
    """Camera-only checks. Measures consistency, not correctness.

    `on_frame(image, hands)` is called for every frame read, detections
    or not. It exists so this can drive a live preview: the numbers here
    say whether tracking is *consistent*, and a camera that needs aiming
    is a problem you have to see rather than read. Frames with no hand in
    them are the interesting ones when the complaint is "it sees too
    many", so it fires before the `continue`.
    """
    import cv2

    thresholds = thresholds or Thresholds()
    report = Report()
    positions: list[np.ndarray] = []
    stamps: list[float] = []
    depths: list[float] = []
    pixels: list[tuple[float, float]] = []

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    last_index = -1
    saved = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        frame = camera.read()
        if frame is None or frame.index == last_index:
            time.sleep(0.001)
            continue
        last_index = frame.index
        report.frames += 1

        hands = detector.detect(frame)
        if on_frame is not None:
            on_frame(frame.image, hands)
        if not hands:
            continue
        observation = locator.locate(hands[0])
        if observation is None:
            continue

        report.detections += 1
        positions.append(observation.position)
        stamps.append(frame.stamp)
        pixels.append(tuple(hands[0].palm_center))
        depths.append(float(np.linalg.norm(observation.position - locator.projector.extr.t)))

        if save_dir and saved < 40 and report.detections % 5 == 0:
            annotated = frame.image.copy()
            u, v = hands[0].palm_center
            cv2.circle(annotated, (int(u), int(v)), 14, (120, 220, 130), 2)
            for pt in hands[0].landmarks:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 2, (235, 235, 235), -1)
            p = observation.position
            cv2.putText(annotated, f"{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f} m",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 130), 2)
            cv2.imwrite(str(Path(save_dir) / f"{saved:03d}.jpg"), annotated)
            saved += 1

        if on_progress and report.frames % 30 == 0:
            on_progress(report)

    elapsed = time.perf_counter() - start
    report.fps = report.frames / elapsed if elapsed else 0.0
    rate = report.detections / max(report.frames, 1)
    report.results.append(Result(
        "detection rate", rate >= thresholds.min_detection_rate,
        rate * 100, thresholds.min_detection_rate * 100, "%",
        "" if rate >= thresholds.min_detection_rate else "- lighting? hand in frame?",
    ))

    if len(positions) < 10:
        report.notes.append("too few detections for the remaining checks")
        return report

    P = np.array(positions)

    # Jitter, from consecutive-frame differences rather than the spread
    # about a mean -- the hand may legitimately drift, and a slow drift
    # should not read as noise.
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    jitter_mm = float(np.median(steps) * 1000)
    report.results.append(Result(
        "frame-to-frame jitter", jitter_mm <= thresholds.max_jitter_mm,
        jitter_mm, thresholds.max_jitter_mm, "mm",
        "hold your hand still for a true reading",
    ))

    jumps = float(np.mean(steps > thresholds.jump_distance_m))
    report.results.append(Result(
        "teleports", jumps <= thresholds.max_jump_rate,
        jumps * 100, thresholds.max_jump_rate * 100, "%",
        "" if jumps <= thresholds.max_jump_rate else "- detector losing and reacquiring",
    ))

    # Depth should not depend on where in the frame the hand is. If it
    # does, the intrinsics are wrong -- focal length most likely.
    #
    # This only means anything if the hand really was held at a constant
    # distance, which is an instruction to a human, not something the
    # code can verify. Reported either way, but only enforced when the
    # caller asserts the precondition was met -- otherwise it fails
    # honest runs where the hand legitimately moved in depth.
    spread_mm = float((np.percentile(depths, 90) - np.percentile(depths, 10)) * 1000)
    report.results.append(Result(
        "depth spread", (not fixed_distance) or spread_mm <= thresholds.max_depth_spread_mm,
        spread_mm, thresholds.max_depth_spread_mm, "mm",
        "" if fixed_distance else "- informational; pass --fixed-distance to enforce",
    ))

    if save_dir and saved:
        report.notes.append(f"wrote {saved} annotated frames to {save_dir}")
    return report


def check_against_arm(
    camera,
    controller,
    locate_marker,
    poses,
    report: Report,
    thresholds: Thresholds | None = None,
    settle: float = 0.4,
    move_time: float = 2.0,
) -> Report:
    """Score the vision against forward kinematics. The accuracy check.

    Drives the tool to each pose, locates the gripper marker in the image,
    and compares the vision's 3D answer against where the arm knows it
    is. Any constant camera-to-base error shows up here and nowhere else.
    """
    from tlod.arm import model
    from tlod.vision.calibrate_flow import wait_until_still

    thresholds = thresholds or Thresholds()
    errors: list[float] = []

    for i, pose in enumerate(poses, 1):
        if not controller.goto_pose(pose, duration=move_time):
            report.notes.append(f"pose {i} unreachable, skipped")
            continue
        wait_until_still(controller)
        time.sleep(settle)

        frame = camera.read()
        if frame is None:
            continue
        seen = locate_marker(frame.image)
        if seen is None:
            report.notes.append(f"marker not found at pose {i}")
            continue

        truth = model.fk(controller.state().q[:5])[:3, 3]
        error = float(np.linalg.norm(np.asarray(seen, float) - truth))
        errors.append(error)
        report.arm_points.append({
            "pose": i,
            "truth": [round(float(v), 4) for v in truth],
            "seen": [round(float(v), 4) for v in np.asarray(seen, float)],
            "error_mm": round(error * 1000, 2),
        })

    if not errors:
        report.notes.append("no usable arm points; accuracy NOT verified")
        return report

    worst_mm = float(np.max(errors) * 1000)
    mean_mm = float(np.mean(errors) * 1000)
    report.results.append(Result(
        "accuracy vs kinematics", worst_mm <= thresholds.max_arm_error_mm,
        worst_mm, thresholds.max_arm_error_mm, "mm",
        f"- mean {mean_mm:.1f} mm over {len(errors)} poses",
    ))
    return report
