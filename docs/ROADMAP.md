# Roadmap

## Status

| | | |
|---|---|---|
| M0 | research, architecture | done |
| M1 | infrastructure — kinematics, control, vision, runtime | done |
| M2 | viewer, record/replay | done |
| M3 | real perception | built, untested against a real hand |
| M4 | strike primitives, safety | done |
| M5 | the game | done, playable |
| M6 | hardware | written, nothing has run on a board |

Everything through M5 runs in simulation.

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
benchmarks.

## What's left

**M3 — real perception.** Everything is built; it needs a person in front
of a camera. The Kalman tuning was fitted to a synthetic trajectory and
should be refitted on a recording (`tlod record`, then `tlod replay`).
Also: validate palm-width depth against a ruler, and check tracking
through occlusion and edge-of-frame entry.

**M6 — hardware.** Bring-up order is in [deployment.md](deployment.md).
Arm: motor IDs, calibrate, `first-light`, verify reach, measure real
latencies, mount and calibrate the camera. Then Orange Pi 5 (RK3588S
NPU, `rknn-toolkit2`) and the Pico sidecar (piezo scoring, hardware
e-stop).

## Ideas, not commitments

- more games on the same primitives: rock-paper-scissors (needs finger
  landmarks, not boxes), quick-draw, pick-and-place
- learned policies via LeRobot — would justify the Orange Pi's NPU
- two-arm play
