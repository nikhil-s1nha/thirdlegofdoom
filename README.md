# Third Leg of Doom

A tabletop game robot on an SO-ARM101 (SO-101) arm. It watches your hand
through a camera and plays games against you — hand slap first.

**Status: milestone 1 complete.** The whole loop runs in simulation with no
hardware. See [docs/ROADMAP.md](docs/ROADMAP.md).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[hands,dev]"
tlod sim              # fully synthetic run
pytest                # 79 tests
```

## The problem this robot has

Hand-slap is decided in tens of milliseconds. The sense-to-motion loop is
not:

| stage | cost | measured? |
|---|---|---|
| camera → decode (C922 @720p60, tuned) | 30–50 ms | estimate |
| MediaPipe hand landmarks | **19 ms** | ✅ measured |
| IK + control | **0.45 ms** | ✅ measured |
| serial sync_write @1 Mbaud | ~0.5 ms | estimate |
| servo response + travel | 120–300 ms | sim: 291 ms |
| **total** | **~200–370 ms** | |

Human visual reaction is ~200–250 ms *plus* hand travel. A robot that
reacts to what it saw is late every time, and no amount of optimisation
closes that gap — the servo alone eats the budget.

**So the robot does not react. It predicts.** A Kalman filter estimates
hand velocity, and the arm aims where the hand *will be* when it arrives.
Measured: **4.8× better aim** at a 300 ms horizon (34.8 cm → 7.3 cm error).
Everything else here exists to keep that loop fast, honestly timed, and
safe.

## Three ways to run it

Hardware is not required for most of the work.

| | camera | hand | arm | command |
|---|---|---|---|---|
| **A** sim | synthetic | scripted | simulated | `tlod sim` |
| **B** hybrid | **your webcam** | **your hand** | simulated | `tlod hybrid` |
| **C** hardware | mounted | real | real | `tlod first-light` |

Tier B is the useful one before the parcel arrives: real photons, real
hands, real detection, real prediction — only the arm is simulated.

## What's here

```
src/tlod/
  types.py            values crossing module boundaries; every observation
                      carries the timestamp of the physical event
  arm/
    model.py          FK + IK from the official URDF
    backend.py        the interface MockArm and FeetechArm both satisfy
    mock.py           simulator with finite slew rate, not a teleporter
    feetech.py        STS3215 bus over the Feetech SDK  [unverified on hw]
    controller.py     safety guards, e-stop, min-jerk trajectories
  vision/
    camera.py         low-latency capture (grab-and-discard threading)
    calibration.py    intrinsics, extrinsics, pixel ↔ robot frame
    hands.py          MediaPipe Tasks + 3D localisation
    tracking.py       Kalman prediction — the thing that wins games
    objects.py        colour segmentation baseline
    scene.py          synthetic hand defined in the workspace
  runtime/
    signal.py         one-slot mailbox between threads (not a queue)
    loop.py           drift-free fixed-rate loop, latency statistics
    app.py            perception thread + control thread + policy
  cli.py              tlod sim | hybrid | bench | first-light | ...
```

## Design decisions worth knowing

**The arm is 5-DOF + gripper, not 6-DOF.** Six motors, one drives the
gripper. Task space is position + tool pitch + tool roll; tool *yaw* is a
dependent variable, fixed by whichever base pan reaches the target. `Pose`
deliberately has no yaw field. A general 6-DOF IK solver handed a full
pose target chases an orientation this arm cannot reach.

**IK is warm-started Levenberg–Marquardt on exact FK.** Adaptive damping,
because a fixed term cannot be both quick in the workspace interior and
stable at full extension — and full extension is exactly where a game
needs the arm. 0.45 ms/solve, 100% success in the tracking regime.

**Perception and control are separate threads joined by a one-slot
mailbox, not a queue.** A queue lets control fall behind into a backlog of
stale frames. Control should always read the newest estimate or nothing.

**Staleness is explicit.** `get_fresh()` returns `None` past a deadline.
Acting confidently on a 400 ms old estimate is worse than acting on none.

**The simulator models finite slew rate.** A backend that teleports gives
the reassuring, useless answer "the arm always gets there in time".

## Gotchas already hit, so you don't

- **`mediapipe` 1.0 removed `mp.solutions`** — the API every tutorial uses.
  Use the Tasks API with a downloaded `.task` bundle.
- **`mediapipe` 1.0.x hard-crashes on macOS arm64** (Metal calculators in
  the palm detector ignore the CPU delegate; the process aborts, it is not
  catchable). Pinned to 0.10.3x on darwin.
- **arm64 Linux only has mediapipe wheels up to 0.10.18** — relevant for
  the Orange Pi 5. Same Tasks API, so the code is unchanged.
- **`cv2.read()` hands you the oldest queued frame** when your loop lags.
  People measure 30 fps while looking 150 ms into the past.

## Hardware

Seeed SO-ARM101 Pro. Geometry comes from the official URDF, vendored at
`assets/so101_new_calib.urdf`. See [docs/hardware.md](docs/hardware.md)
for the servo control table and wiring.
