# Third Leg of Doom

A tabletop game robot on an SO-ARM101 (SO-101) arm. It watches your hand
through a camera and plays games against you — hand slap first.

**Status: playable in simulation, M1–M5 complete.** No hardware required
for any of it. See [docs/ROADMAP.md](docs/ROADMAP.md).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[hands,dev]"

tlod play --view              # play hand slap against a simulated human
tlod play --real-hand --view  # play it with YOUR hand, via the webcam
tlod move 0.22 0 0.12         # just move the arm to a point
tlod touch --view             # detect table objects and touch each one
tlod eval                     # measure the robot's win rate
pytest                        # 130 tests
```

## Branches

| branch | what it is |
|---|---|
| `main` | infrastructure through M2 |
| `arm-core` | + motion primitives, `tlod move`, `tlod reach` — the arm on its own |
| `gamification` | + hand slap, opponent, scoring — branched from `arm-core` |

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

**Two answers, and the second one matters more.**

*Predict, don't react.* A Kalman filter estimates hand velocity so the arm
aims where the hand *will be*. Measured: **4.8× better aim** at a 300 ms
horizon (34.8 cm → 7.3 cm error).

*And: let the robot go first.* Latency taxes only the **responder**. With
the robot as slapper and the human as dodger, the whole pipeline delay is
spent *before* the strike, where nobody is waiting on it — and a hand
waiting to be slapped is nearly stationary, so a stale estimate is still
a correct one. An 8 cm strike lands in ~210 ms.

## Then the measurements said the game was wrong

Making the robot fast enough to win turned out to be easy, and useless.
Measured against a simulated 250 ms human with feints off:

| strike | robot win |
|---|---|
| 180–350 ms | **100%** |
| 450 ms | **17%** |
| 550 ms+ | **0%** |

That is a step function, not a difficulty curve. The human's usable
window is only ~45% of the strike duration — contact fires at ~70% of the
travel, and motion onset costs the first ~25%. Beating an 8 cm strike
needs a sub-70 ms reaction. Slowing the arm to compensate takes ~650 ms
per strike, which stops reading as a slap, and striking from further away
is *both* slower and harder-hitting.

So the game changed instead of the arm. Real hand-slap is slapper-favoured
too; what makes it fun is that **the dodger is punished for flinching**:

| event | point to |
|---|---|
| strike lands | robot |
| strike dodged | human |
| feint draws a flinch | robot |
| feint held through | human |

Now the human reads intent instead of racing physics, difficulty becomes
a smooth dial (feint frequency), and safety stops fighting game design.
Calibrated result — **normal is an even match across the whole human
range**:

| preset | vs 180 ms | vs 250 ms | vs 350 ms |
|---|---|---|---|
| easy | 21% | 0% | 0% |
| **normal** | **57%** | **43%** | **50%** |
| hard | 86% | 79% | 64% |

Full reasoning in [docs/slap-analysis.md](docs/slap-analysis.md).

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
  game/
    handslap.py       the game: commit timing, feints, flinch scoring
    opponent.py       a simulated human that dodges, for testing without one
    contact.py        did it land? geometric / proximity / piezo
    touch.py          detect objects and visit each one
    base.py           state machine that is also a Policy
  viz/
    overlay.py        arm, hand, prediction and limits drawn on the frame
    viewer.py         the window (main thread, always)
  vision/
    camera.py         low-latency capture (grab-and-discard threading)
    calibration.py    intrinsics, extrinsics, pixel ↔ robot frame
    hands.py          MediaPipe Tasks + 3D localisation
    tracking.py       Kalman prediction — the thing that wins games
    objects.py        colour segmentation baseline
    scene.py          synthetic hand defined in the workspace
    recording.py      record once, replay a thousand times
    rknn.py           RK3588 NPU detector          [unverified on hw]
  runtime/
    signal.py         one-slot mailbox between threads (not a queue)
    loop.py           drift-free fixed-rate loop, latency statistics
    app.py            perception thread + control thread + policy
  cli.py              tlod play | move | touch | sim | hybrid | eval | ...
firmware/
  pico_sidecar.py     hardware e-stop + piezo scoring  [unverified on hw]
```

## What has actually been verified

Measured, not assumed. Everything here ran on a laptop; nothing has
touched the arm.

| | result |
|---|---|
| IK, tracking regime | 400/400 solved, 0.40 ms mean, 0.59 ms p95 |
| control loop | 100.4 Hz effective, **0.000 ms p95 jitter**, 0.2% overruns |
| MediaPipe, real 720p webcam frames | 20.2 ms mean, 24.4 ms p95, 113 ms tail spike |
| full loop, real camera | 73.5 ms p50 shutter→command |
| record → replay, real camera | 146 frames, 0 failures, deterministic |
| object touching | 0.2 mm mean placement error |
| **7-minute soak** | **RSS flat, +17 gc objects, latency drift −0.7 ms** |
| tests | 135 passing, ruff clean |

Every CLI path has been executed at least once, including `first-light`
rehearsed in simulation.

### Bugs this found

Worth listing, because each passed the test suite and would have surfaced
as "the robot feels wrong" rather than as a failure:

- the mock camera free-ran at **1745 fps**, starving the control loop to
  70 ms jitter — a simulator that does not respect frame timing quietly
  invalidates every timing conclusion drawn from it
- the synthetic hand's path was defined in pixels, so most of it lay
  outside the workspace: the demo was exercising the safety clamps, not
  the behaviour
- `latency_offset` defaulted to 35 ms, **below one frame period** on a
  camera actually delivering 29.6 fps — every timestamp claimed the
  shutter opened more recently than was physically possible
- the camera accepted `fps=60` and delivered 30, silently
- `MockArm._integrate` ran unlocked while called from three threads
- `Strike.abort()` never restored the torque limit, so every interrupted
  strike left the arm permanently weak
- `_write` checked the e-stop flag outside the lock
- a hand in frame was detected as a red object (skin reads as red to any
  colour segmenter — not a simulation artifact)
- `Difficulty.commit_delay` was set in two presets and never read

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

**Strike safety is structural, not advisory.** The drop is capped, the
commanded depth never goes below the target plane (so a wrong height
estimate stalls rather than presses), torque is lowered for the duration,
and IK is solved once so a fast move cannot switch branches halfway down.

**The real e-stop is a microcontroller.** Software e-stop stops working in
exactly the case you need it: a hung loop, a crashed process, a pulled
cable. The Pico cuts servo power in its interrupt handler and reports
afterwards.

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
