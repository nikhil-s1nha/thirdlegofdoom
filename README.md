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
| `vision-serve` / `control` | split across two boards: vision on one, kinematics on the other |
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
  1.0. Same Tasks API throughout, so this never reaches the code.
- **`cv2.read()` returns the oldest queued frame** when your loop lags.
- **Camera fps is a request, not a promise.** Check `tlod bench camera`.

## Hardware

Seeed SO-ARM101 Pro. Kinematics come from the official URDF, vendored at
`assets/so101_new_calib.urdf`.

- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) — setup, usage, modification
- [docs/headless.md](docs/headless.md) — verifying vision with no screen
- [docs/hardware.md](docs/hardware.md) — servo control table, wiring
- [docs/deployment.md](docs/deployment.md) — Orange Pi 5, standalone
- [docs/ROADMAP.md](docs/ROADMAP.md) — what is and isn't built
