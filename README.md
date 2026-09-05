# Third Leg of Doom

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
| `sim` / `hybrid` | run the loop synthetic / with a real camera |
| `calibrate intrinsics\|extrinsics` | lens, then camera-to-robot transform |
| `first-light` | verify a new arm one joint at a time |
| `bench` | measure IK, camera and loop latency |
| `record` / `replay` | capture a session, replay it deterministically |
| `cameras` / `ports` / `config` | discovery and setup |

## Branches

| | |
|---|---|
| `main`, `arm-core` | arm, vision, calibration. No game. |
| `gamification` | the above plus hand slap |

## Layout

```
src/tlod/
  types.py        values crossing module boundaries; read this first
  arm/            model (FK/IK), backend, mock, feetech, controller, primitives
  vision/         camera, calibration, hands, tracking, objects, scene, recording
  runtime/        signal (mailbox), loop (fixed rate), app (threads + Policy)
  game/           handslap, opponent, contact, touch
  viz/            overlay and viewer
firmware/         pico_sidecar.py — hardware e-stop + piezo scoring
```

Perception and control run on separate threads joined by a one-slot
mailbox, so control always gets the newest estimate and never a backlog.
`MockArm` and `FeetechArm` satisfy one interface, so simulation and
hardware are the same code path.

## Things that will bite you

- **The arm is 5-DOF, not 6.** Six motors, one drives the gripper. You get
  position + tool pitch + roll; yaw is fixed by the base pan.
- **Sense-to-motion is ~200–370 ms**, slower than human reaction. The arm
  aims where the hand *will* be, and it slaps rather than dodges —
  latency only taxes whoever is responding. See
  [docs/slap-analysis.md](docs/slap-analysis.md).
- **`mediapipe` 1.0 removed `mp.solutions`** and crashes on macOS arm64.
  Pinned per platform; arm64 Linux caps at 0.10.18.
- **`cv2.read()` returns the oldest queued frame** when your loop lags.
- **Camera fps is a request, not a promise.** Check `tlod bench camera`.

## Hardware

Seeed SO-ARM101 Pro. Kinematics come from the official URDF, vendored at
`assets/so101_new_calib.urdf`.

- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) — setup, usage, modification
- [docs/hardware.md](docs/hardware.md) — servo control table, wiring
- [docs/deployment.md](docs/deployment.md) — Orange Pi 5 + Pico
- [docs/ROADMAP.md](docs/ROADMAP.md) — what is and isn't built
