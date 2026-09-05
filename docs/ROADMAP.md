# Third Leg of Doom — Roadmap

A tabletop game robot on an SO-ARM101 (SO-101) arm. Plays hand-slap and
other games against a human, using a fixed overhead camera.

## The constraint everything is designed around

Measured and estimated end-to-end, hand-moves to arm-moves:

| stage | cost |
|---|---|
| camera exposure → USB → decode (C922 @720p60, tuned) | 30–50 ms |
| MediaPipe hand landmarks (measured, CPU) | 19 ms |
| IK + control (measured) | <1 ms |
| serial sync_write @1 Mbaud | ~0.5 ms |
| servo response + mechanical travel (sim: 291 ms for a large move) | 120–300 ms |
| **total** | **~200–370 ms** |

Human visual reaction is ~200–250 ms *plus* their own hand travel. So a
robot that reacts to what it saw is at best tied and usually late.

**The design response is prediction, not speed.** A Kalman filter estimates
hand velocity so the arm aims where the hand *will be*. Measured: 4.8×
better aim at a 300 ms horizon (34.8 cm → 7.3 cm error). Everything else
in this project is in service of that loop being fast, honest about its
own timing, and safe.

---

## M0 — Research and architecture — **done**

Established the facts the rest depends on:

- **The arm is 5-DOF + gripper, not 6-DOF.** Six motors, one drives the
  gripper. Task space is position + tool pitch + tool roll; tool yaw is a
  dependent variable. This shapes the entire kinematics design.
- Geometry taken from the official URDF, not from measurements.
- Servo control table verified against the STS3215 datasheet.
- MediaPipe platform landmines identified (see M6).

## M1 — Infrastructure — **in progress**

The skeleton everything hangs on. Runs end-to-end in simulation with no
hardware, so all later work is testable before the arm arrives.

| piece | state |
|---|---|
| shared types, timestamped at the physical event | done |
| FK/IK from URDF — 0.45 ms/solve, 100% in tracking regime | done |
| backend interface + simulator with real slew dynamics | done |
| Feetech STS3215 backend | written, **unverified on hardware** |
| controller: safety guards, e-stop, min-jerk trajectories | done |
| low-latency camera capture + synthetic camera | done |
| calibration: intrinsics, extrinsics, pixel↔robot frame | done |
| MediaPipe hand detection (19 ms @720p) + 3D localisation | done |
| Kalman tracking and prediction | done |
| **runtime: perception/control threads, timing instrumentation** | todo |
| **config system, CLI, object detection baseline** | todo |
| **tests, docs** | todo |

**Exit criteria:** `tlod sim` runs a full perception→prediction→IK→arm loop
against a synthetic hand, reports per-stage latency, and the test suite
passes. No hardware required.

## M2 — Hardware bring-up

Blocked on the arm arriving. Everything here is verification, not design.

- Motor ID assignment and calibration (`lerobot-setup-motors`, or ours)
- **`tlod arm first-light`** — energise one joint at a time, confirm
  direction signs and limits before anything moves at speed
- Validate the Feetech backend against real servos; fix the sign and
  calibration conventions that cannot be checked in simulation
- **Measure the real numbers** that are currently estimates: servo slew
  rate, command-to-motion latency, camera pipeline latency, achievable
  control rate. The simulator gets retuned to match.
- Physical camera mount, then extrinsic calibration using the arm itself
  as the calibration target

**Exit criteria:** the arm moves to a commanded Cartesian point within a
few mm, and every latency figure in the table above is measured rather
than estimated.

## M3 — Perception in the real world

Synthetic hands are easy. Real ones are not.

- Hand tracking validated across the actual workspace, lighting, and skin
  tones — including the failure modes: motion blur on a fast slap,
  occlusion by the arm itself, hands entering frame at the edge
- Depth: validate palm-width ranging against ground truth; decide whether
  the plane assumption is good enough for the game
- Object detection: color/contour baseline first, learned model if needed
- Retune the Kalman filter on **recorded real hand motion** rather than a
  synthetic trajectory

**Exit criteria:** hand position tracked to a few cm across the workspace
at full frame rate, with measured prediction error at the real latency
horizon.

## M4 — Vision and control fused

The point of the project: closed-loop visual servoing with latency
compensation.

- Continuous visual servoing — arm tracks a moving hand in real time
- Latency compensation wired end to end: aim at `predict(measured_latency)`
- Motion primitives: **strike**, **retract**, **hover**, **track**, **grasp**
- Reachability and timing gating: given where the hand is going and how
  fast the arm is, decide *whether the swing is winnable* before starting
- Safety hardened: this machine moves quickly toward a human hand, so
  speed limits, keep-out volumes, and the e-stop get real scrutiny here

**Exit criteria:** the arm reliably intercepts a moving hand, and refuses
swings it cannot win instead of flailing.

## M5 — The games

- **Hand slap** — both roles. As slapper: predict the dodge, commit late.
  As dodger: detect the incoming strike and retract. These are different
  problems and the dodger is the easier one to make good.
- Difficulty tuning — a robot that always wins is not fun. Deliberate,
  tunable reaction handicap.
- Additional games reusing the same primitives: rock-paper-scissors
  (needs landmarks, not just boxes), quick-draw reaction, cup shuffle,
  pick-and-place challenges
- Game-agnostic scoring, rounds, and state machine

**Exit criteria:** a person can walk up and play a full game without an
operator.

## M6 — Standalone deployment

Take the laptop out of the loop.

- Port to **Orange Pi 5** (RK3588S, 6 TOPS NPU). It can run the whole
  stack — vision, IK, and the servo bus over USB — as a self-contained
  robot.
  - Detection models converted with `rknn-toolkit2` (YOLOv5n ~58 fps on
    this NPU); MediaPipe pinned to 0.10.18, the newest aarch64 wheel
  - Re-measure everything: A76 cores are slower than an M-series laptop,
    and the NPU path has its own latency profile
- **Raspberry Pi Pico as an I/O sidecar** — the right job for it, since it
  cannot do vision:
  - **Piezo impact detection** for scoring. Microsecond timestamps on a
    real hit, versus vision guessing at 20 ms granularity.
  - **Hardware e-stop** that cuts servo torque independently of whether
    Python is responsive. Real safety needs this.
  - Score display, LEDs, buzzer, start button
- Autostart, crash recovery, watchdog

**Exit criteria:** power it on and it plays, with no computer attached.

---

## What can proceed in parallel

The hardware-independent work is deliberately the majority, so waiting on
shipping costs as little as possible.

| can be done now, no hardware | needs the arm | needs the Orange Pi |
|---|---|---|
| all of M1 | M2 entirely | RKNN conversion |
| game logic and state machines (M5) | Feetech validation | on-device benchmarks |
| prediction tuning on recorded video | real latency numbers | autostart/deploy |
| object detection on still images | extrinsic calibration | |
| difficulty and scoring design | | |

The backend interface is what makes this work: `MockArm` and `FeetechArm`
satisfy the same contract, so game code written against the simulator runs
unchanged on the real arm.
