# Roadmap

## Status

| | | |
|---|---|---|
| M0 | research, architecture | done |
| M1 | infrastructure — kinematics, control, vision, runtime | done |
| M2 | viewer, record/replay | done |
| M3 | real perception | done — real camera, real hand, on the board |
| M4 | strike primitives, safety | done |
| M5 | the game | done, playable |
| M6 | hardware | done for one board; the NPU is untouched |

Everything through M5 runs in simulation, and M3/M6 have now run on real
hardware: one Orange Pi 5 carrying the camera, mediapipe and the control
loop at 100 Hz, driving the arm over USB. `tlod touch --real` puts the
tool 2.5 mm from an object the camera located by itself. Details and
measurements in [deployment.md](deployment.md).

## Three tiers

| tier | camera | hand | arm |
|---|---|---|---|
| A | synthetic | scripted | simulated |
| B | your webcam | your hand | simulated |
| C | mounted | real | real |

Tier B is the one people skip. Real hands blur, get occluded and enter
frame at bad angles in ways no synthetic path reproduces, and none of it
needs the arm to exist.

Only four things genuinely need hardware: servo sign conventions, real
slew rate and command latency, physical-mount extrinsics, and NPU
benchmarks. The first three are now done; the fourth is not.

## What's left

**The Kalman tuning** is still fitted to a synthetic trajectory. Refit it
on a recording (`tlod record`, then `tlod replay`) now that recordings of
a real hand exist.

**Depth.** `depth_mode: plane` with the palm flat on the table sidesteps
the problem rather than solving it — one camera still cannot measure how
far away a hovering hand is, and palm-width depth is still unvalidated
against a ruler. Tracking through occlusion and edge-of-frame entry is
also still unmeasured.

**The game against a real person.** `eval` numbers are all against the
simulated opponent. Nobody has been slapped yet.

**The NPU.** RK3588S, `rknn-toolkit2`, `tlod/vision/rknn.py`. Untouched:
mediapipe on the CPU turned out to hold 31 fps alongside the control
loop, so the NPU stopped being on the critical path. The vendor's
YOLOv5n/YOLOv5s figures in [deployment.md](deployment.md) are still
vendor figures.

**The two-board split.** Still tested, still not the recommended layout —
see [deployment.md](deployment.md). It becomes interesting again if the
NPU or a learned policy lands.

No sidecar microcontroller. The servos carry torque and overload limits
in their own firmware, an inline switch on the 12 V line is a better
e-stop than any chip, and contact is read from `Present_Load` over the
bus already in use.

## Ideas, not commitments

- more games on the same primitives: rock-paper-scissors (needs finger
  landmarks, not boxes), quick-draw, pick-and-place
- learned policies via LeRobot — would justify the Orange Pi's NPU
- two-arm play
