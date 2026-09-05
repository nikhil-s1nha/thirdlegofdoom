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

## Three fidelity tiers

Hardware is not in hand, so the plan is ordered to squeeze everything
possible out of the tiers that need none of it. Hardware handoff is last.

| tier | camera | hand | arm | what it proves |
|---|---|---|---|---|
| **A — full sim** | synthetic | scripted | simulated | logic, timing, IK, determinism. Runs in CI. |
| **B — hybrid** | **your real webcam** | **your real hand** | simulated | the entire perception stack, for real. Playable. |
| **C — hardware** | mounted camera | real hand | real arm | servos, calibration, true latency |

Tier B is the important one and is often skipped. Real hands blur, get
occluded, and enter frame at bad angles in ways no synthetic trajectory
reproduces — and none of that needs the arm to exist. Almost every
perception and game question can be answered and *played* on a laptop
before the parcel arrives.

What genuinely cannot be done before hardware, and nothing else:
servo sign conventions, real slew rate and command latency,
camera-to-base extrinsics for the physical mount, and NPU benchmarks.

---

## M0 — Research and architecture — **done**

Established the facts the rest depends on:

- **The arm is 5-DOF + gripper, not 6-DOF.** Six motors, one drives the
  gripper. Task space is position + tool pitch + tool roll; tool yaw is a
  dependent variable. This shapes the entire kinematics design.
- Geometry taken from the official URDF, not from measurements.
- Servo control table verified against the STS3215 datasheet.
- MediaPipe platform landmines identified (see M6).

## M1 — Infrastructure — **in progress** *(tier A)*

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
| runtime: threads, mailbox, fixed-rate loop, latency accounting | done |
| **config system, CLI, object detection baseline** | todo |
| **tests, docs** | todo |

**Exit:** `tlod sim` runs a full perception→prediction→IK→arm loop against
a synthetic hand, reports per-stage latency, and the tests pass.

## M2 — Simulation you can watch *(tier A)*

A loop that only prints numbers cannot be debugged by eye, and every
behaviour question from here on is a question about motion.

- Digital twin: render the arm from FK link frames, the tracked hand, the
  prediction horizon, and the safety volumes
- Playback and record: capture a session and replay it deterministically,
  so a bug seen once can be re-run exactly
- Synthetic scene generator — hands entering from varied angles at varied
  speeds, for repeatable stress tests
- Simulator fidelity: tune the mock arm's slew and latency to the
  datasheet, and mark clearly which numbers are estimates awaiting M6

**Exit:** watch the robot play, in a window, and replay any run.

## M3 — Real perception, simulated arm *(tier B)*

The laptop webcam becomes the sensor. Everything from photons to a
predicted 3D hand position is now real; only the arm is simulated.

- Real hand tracking through its actual failure modes: motion blur on a
  fast slap, occlusion, edge-of-frame entry, varied lighting and skin tones
- Validate palm-width depth ranging against measured ground truth, and
  decide whether the plane assumption suffices
- **Retune the Kalman filter on recorded real hand motion** rather than a
  synthetic trajectory — the current tuning is a placeholder
- Measure the true webcam pipeline latency and set the prediction horizon
  from data
- Object detection: colour/contour baseline, learned model if needed

**Exit:** real hand tracked to a few cm at full frame rate, with
prediction error measured at the real latency horizon.

## M4 — Vision and control fused *(tier B)*

The point of the project: closed-loop visual servoing with latency
compensation, driven by a real hand.

- Arm tracks a real moving hand in real time, in the viewer
- Latency compensation wired end to end: aim at
  `predict(measured_latency)` using the M3 number
- Motion primitives: **strike**, **retract**, **hover**, **track**, **grasp**
- Winnability gating: given where the hand is going and how fast the arm
  is, decide *whether a swing can land* before committing to it
- Safety hardened here, while the consequences are still pixels

**Exit:** simulated arm reliably intercepts your real hand, and declines
swings it cannot win instead of flailing.

## M5 — The games *(tier B — playable)*

- **Hand slap**, both roles. As slapper: predict the dodge, commit late.
  As dodger: detect the strike and retract. Different problems; the
  dodger is easier to make good.
- Difficulty tuning — a robot that always wins is not fun. Deliberate,
  tunable handicap.
- More games on the same primitives: rock-paper-scissors (needs
  landmarks, not boxes), quick-draw, cup shuffle, pick-and-place
- Game-agnostic scoring, rounds, state machine

**Exit:** you can play a full game against the on-screen robot with your
real hand, and it is fun.

## M6 — Hardware handoff *(tier C — last)*

Everything above is done and proven. This milestone is verification and
deployment, not design.

**Arm bring-up**
- Motor ID assignment and calibration
- `tlod arm first-light` — energise one joint at a time, confirm direction
  signs and limits before anything moves at speed
- Validate the Feetech backend; fix sign and calibration conventions that
  cannot be checked in simulation
- **Replace every estimated number with a measured one**: servo slew rate,
  command-to-motion latency, achievable control rate. Retune the simulator
  to match, so tier A stays trustworthy.
- Mount the camera; extrinsic calibration using the arm as its own target
- Re-verify the games against the real latency, which will differ

**Standalone deployment**
- Port to **Orange Pi 5** (RK3588S, 6 TOPS NPU) — vision, IK, and the
  servo bus over USB, self-contained
  - Detection converted with `rknn-toolkit2` (YOLOv5n ~58 fps); MediaPipe
    pinned to 0.10.18, the newest aarch64 wheel
  - Re-measure: A76 cores are slower than an M-series laptop
- **Raspberry Pi Pico as I/O sidecar** — the right job for a chip that
  cannot do vision:
  - **Piezo impact detection** for scoring. Microsecond hit timestamps
    versus vision guessing at 20 ms granularity.
  - **Hardware e-stop** cutting servo torque independently of whether
    Python is responsive. Real safety needs this.
  - Score display, LEDs, buzzer, start button
- Autostart, crash recovery, watchdog

**Exit:** power it on and it plays, with no computer attached.

---

## Why this order

Every milestone before M6 produces something testable today. The backend
interface is what makes it work: `MockArm` and `FeetechArm` satisfy the
same contract, so game code written against the simulator runs unchanged
on the real arm. The risk this ordering accepts is that hardware surprises
arrive late; the risk it avoids is a project that cannot progress at all
until a parcel shows up.
