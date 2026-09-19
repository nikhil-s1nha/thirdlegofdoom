# 6 DOF Hand Slap

A tabletop game robot on an SO-ARM101 (SO-101) arm. A fixed camera watches
your hand; the arm plays hand-slap against you.

Runs fully in simulation — no hardware needed to try it.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.12+
pip install -e ".[hands,dev]"

tlod play --view              # hand slap vs a simulated human
tlod play --real-hand --view  # play it with your own hand, via webcam
tlod move 0.22 0 0.12         # move the tool to a point
tlod sim --view               # the whole loop, synthetic
tlod hybrid --view            # real webcam and hand, simulated arm
tlod hybrid --real            # ... and the real arm, hovering over your hand
tlod touch --real             # detect table objects and touch each one
pytest
```

New here, or have the hardware? Read [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

## Commands

| | |
|---|---|
| `move X Y Z` | move the tool to a point (sim or real) |
| `reach` | probe the reachable workspace |
| `play` | hand slap; `--real-hand` uses your webcam |
| `eval` | sweep opponent reaction time, measure win rate |
| `touch` | detect table objects and touch each one |
| `sim` / `hybrid` | run the loop synthetic / with a real camera, `--real` for the arm too |
| `calibrate intrinsics\|extrinsics` | lens, then camera-to-robot transform |
| `first-light` | verify a new arm one joint at a time |
| `bench` | measure IK, camera and loop latency |
| `record` / `replay` | capture a session, replay it deterministically |
| `vision-serve` / `control` | optional two-board split: vision on one, kinematics on the other |
| `leg` | the Arduino paddle: `open`, `close`, `home`, `slap`, `strike`, `monitor` |
| `power` | measure what a move actually costs the supply |
| `vision-check` | verify vision numerically \+ MJPEG preview; for headless boards |
| `probe` | read the arm with torque off; safest first hardware test |
| `cameras` / `ports` / `config` | discovery and setup |

## Branches

| | |
|---|---|
| `main` | **you are here** — arm, vision, calibration, and the game |
| `arm-core` | arm and vision only, no game |
| `gamification` | merged into `main`; kept for history |
| `UartComms` | an earlier UART transport, superseded by `--serial-port` on `main` |

## Layout

```
src/tlod/
  types.py        values crossing module boundaries; read this first
  arm/            model (FK/IK), backend, mock, feetech, controller, primitives
  vision/         camera, calibration, hands, tracking, objects, scene, recording
  runtime/        signal (mailbox), loop (fixed rate), app (threads + Policy)
  game/           handslap, opponent, contact, touch
  leg.py          the Arduino paddle on its own USB port
  viz/            overlay and viewer
```

Perception and control run on separate threads joined by a one-slot
mailbox, so control always gets the newest estimate and never a backlog.
`MockArm` and `FeetechArm` satisfy one interface, so simulation and
hardware are the same code path.

## Things that will bite you

- **The arm is 5-DOF, not 6.** Six motors, one drives the gripper. You get
  position + tool pitch + roll; yaw is fixed by the base pan.
- **Sense-to-motion is ~200–370 ms**, slower than human reaction. So the
  robot slaps and you dodge — latency only taxes whoever is responding.
  See [docs/slap-analysis.md](docs/slap-analysis.md).
- **`mediapipe` dropped `mp.solutions` at 0.10.30**, and 1.0 crashes on
  macOS arm64. Pinned per platform, and on arm64 Linux per interpreter
  too: Python 3.12 caps at 0.10.18, 3.13 has no 0.10.x wheel and takes
  1.0. Same Tasks API throughout, so this never reaches the code. The
  3.13/1.0.1 combination was a bet — upstream says nothing either way
  about aarch64 Linux — and it has now been run: 31 fps, 100% detection,
  2.4–5 mm jitter.
- **`cv2.read()` returns the oldest queued frame** when your loop lags.
- **Camera fps is a request, not a promise.** Check `tlod bench camera`.
- **`/dev/video*` indices move.** Across reboots and replugs. One evening
  here it moved twice, and `/dev/video5` was a hardware encoder on one
  boot and the camera on the next. Check with `tlod cameras`; never trust
  last week's index.
- **Colour detection takes the largest blob and does not doubt itself.**
  Both the calibration marker and the objects `touch` reaches for. It
  cannot tell a green marker from a green mug, and it does not fail when
  it picks wrong — it produces a confident answer about the mug. This cost
  four separate debugging sessions. `scripts/marker_view.py` shows you
  which blob would win, before the arm moves.
- **A leftover `vision-serve` silently wins.** The mailbox takes whatever
  datagram arrived last and cannot tell one publisher from another, so a
  process forgotten from an earlier session feeds the loop stale
  positions while the run you are watching looks fine. Check for one
  before believing anything strange.
- **A contact threshold does not survive changing the motion.** Both
  times hit detection was wrong, it was this: the threshold was
  calibrated against one version of the strike, the strike changed, and
  the threshold stayed. It is not noise and it is not calibration — the
  clusters move. See [docs/hit-detection.md](docs/hit-detection.md).
- **The third leg reports nothing.** Two hobby servos on an Arduino, so
  no encoder and no feedback: the board tells you it took the word, never
  where the paddle went. Three sharp edges in the sketch — `slap` leaves
  the paddle down until something sends `home`, `home` acknowledges with
  an *empty line*, and `open` blocks the board for 200 ms — and opening
  the port resets the board, so the first beat is what says it is ready.
  See [docs/hardware.md](docs/hardware.md).
- **`configs/default.yaml` is not a base layer.** `Config.load` reads the
  single file you pass; everything absent falls back to the dataclass
  defaults in `config.py`, not to `default.yaml`. Editing it does not
  affect a run started with another config, and it drifts from the code
  without complaint. Each config is standalone.

## Hardware

Seeed SO-ARM101 Pro. Kinematics come from the official URDF, vendored at
`assets/so101_new_calib.urdf`.

It runs on **one Orange Pi 5** — camera, mediapipe and the 100 Hz control
loop in one process, arm on USB. End to end, `tlod touch --real` puts the
tool 2.5 mm from an object the camera found by itself. The two-board
split (vision here, control on a Raspberry Pi, joined by UDP) still works
and is still tested, but the Pi could not hold the loop even with no
vision on it, so it is an option rather than the recommendation.

- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) — setup, usage, modification
- [docs/hit-detection.md](docs/hit-detection.md) — did the slap land, and the four wrong answers
- [docs/headless.md](docs/headless.md) — verifying vision with no screen
- [docs/hardware.md](docs/hardware.md) — servo control table, wiring
- [docs/power.md](docs/power.md) — current budgets, motion profiles, brownouts
- [docs/deployment.md](docs/deployment.md) — Orange Pi 5, one board
- [docs/ROADMAP.md](docs/ROADMAP.md) — what is and isn't built
