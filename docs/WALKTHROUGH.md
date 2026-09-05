# Getting this robot working

For someone with an SO-ARM101 in a box. Assumes a terminal and some
Python; no robotics background.

Read part 0 first.

---

## 0. Two things to know up front

### The arm is 5-DOF, not 6

Six motors, but one drives the gripper, so five arm joints:

```
shoulder_pan     yaw about the base vertical
shoulder_lift  ┐
elbow_flex     ├ three parallel pitch axes
wrist_flex     ┘
wrist_roll       roll about the tool axis
gripper          the jaw
```

Five joints cannot reach an arbitrary position *and* orientation. You get:

```
position (x, y, z)  +  tool pitch  +  tool roll
```

Tool **yaw** is fixed by whichever base pan reaches the target. That is
why `Pose` has no `yaw` field, and why a generic 6-DOF IK library will
chase orientations this arm cannot reach.

### The robot slaps; you dodge

Sense-to-motion is 200–370 ms. Human reaction is 200–250 ms plus hand
travel. A robot that *reacts* to your hand is always late, and the servo
eats most of the budget, so speed does not fix it.

The fix is who moves first. Latency taxes whoever is responding, so the
robot initiates and spends its delay before the strike. A hand waiting to
be slapped also barely moves, so a slightly stale position is still
correct — which is why prediction is a minor detail here, not the
mechanism. The Kalman filter stays for smoothing, for telling the game
whether your hand is settled, and for refusing to swing at an uncertain
estimate.

The game is about **feints**, not reflexes. Details in
[slap-analysis.md](slap-analysis.md).

---

## 1. Five minutes, no hardware

### Which branch

| branch | you get |
|---|---|
| `main`, `arm-core` | arm, vision, calibration. No game. |
| `gamification` | the above plus hand slap (`play`, `eval`) |

Bringing up hardware or writing your own behaviour? Start on `arm-core`.
Want to play? `git checkout gamification`.

### Install

```bash
git clone <repo> && cd thirdlegofdoom
python3 -m venv .venv && source .venv/bin/activate     # Python 3.12+
pip install -e ".[hands,dev]"
```

```bash
tlod sim --duration 5        # the whole loop, synthetic
tlod move 0.22 0 0.12        # move the tool to a point
tlod touch --view            # detect table objects and touch each one
pytest

# the game is on the `gamification` branch: tlod play --view
```

A healthy `tlod sim`:

```
  vision.detect             mean   0.15 ms
  shutter->servo command    mean   9.43 ms
  control 509 ticks, 2.0% overruns, jitter p95 0.000 ms
  IK: 505 commands, 0 failures, 0 safety-guard hits
```

`IK failures` and `safety-guard hits` should be 0, jitter near 0. Above
~10% overruns means your machine is struggling and every timing number
below will be pessimistic.

### With your own hand

```bash
tlod play --real-hand --view
```

Real camera, real hand, real detection, simulated arm — everything except
the servos. `space` pauses, `e` toggles e-stop, `q` quits.

---

## 2. How the code is organised

### Data flow

```
  ┌────── perception thread (camera rate, bursty) ──────┐
  camera ─▶ detect ─▶ locate in 3D ─▶ track ────────────┼─▶ ┌─────────┐
  └─────────────────────────────────────────────────────┘   │ Latest  │ one slot
                                                            └────┬────┘
  ┌────── control thread (fixed 100 Hz) ────────────────────────┼──┐
       policy ─▶ safety clamp ─▶ IK ─▶ rate limit ─▶ backend ◀──┘  │
  └────────────────────────────────────────────────────────────────┘
                                        backend = MockArm | FeetechArm
```

- **A mailbox, not a queue.** Behind a queue, control consumes stale
  frames. It should get the newest estimate or nothing.
- **Staleness is explicit.** `get_fresh(max_age)` returns `None` past a
  deadline. Acting on a 400 ms old estimate is worse than not acting.
- **Sim and real share one path.** `--real` is a flag, not a different
  program.

### Files

```
src/tlod/
  types.py          everything crossing a module boundary. Read first.
                    JOINT_NAMES, Pose, HandObservation. Observations carry
                    the timestamp of the physical event, not of the code.

  arm/
    model.py        FK and IK, geometry from the official URDF.
                    Warm-started Levenberg-Marquardt, 0.4 ms per solve.
    backend.py      the interface MockArm and FeetechArm both satisfy
    mock.py         simulator with a real slew rate; does not teleport
    feetech.py      STS3215 servos over the Feetech SDK
    controller.py   safety limits, e-stop, min-jerk. All commands pass here.
    primitives.py   hover / strike / retract / feint / goto. Steppable,
                    so a game can abandon one mid-flight.

  vision/
    camera.py       threaded grab-and-discard; cap.read() gives you the
                    oldest queued frame when your loop lags
    calibration.py  intrinsics, extrinsics, pixel <-> robot frame
    calibrate_flow.py  the interactive calibration procedures
    hands.py        MediaPipe landmarks, 3D localisation
    tracking.py     Kalman filter: smoothing, velocity, uncertainty
    objects.py      colour-blob object detection
    scene.py        synthetic scene for simulation
    recording.py    record once, replay deterministically
    rknn.py         Orange Pi 5 NPU detector          [unverified]

  runtime/
    signal.py       the one-slot mailbox
    loop.py         drift-free fixed-rate loop, latency stats
    app.py          wires it together; defines Policy

  game/
    touch.py        visit each detected object. Good calibration check.
    base.py         state machine that is also a Policy

  viz/              overlay and viewer (main thread, always)
  cli.py            every command
```

### The class you subclass

```python
class Policy:
    def start(self, robot): ...
    def update(self, robot, perception, dt): ...   # 100 Hz
    def stop(self, robot): ...
```

`perception` is `None` when vision is stale. Handle that case — it is
`None` rather than old data on purpose.

---

## 3. Hardware bring-up

In order. Each step is cheap to fail.

### 3.1 Assemble, set motor IDs

Follow the [official assembly guide](https://huggingface.co/docs/lerobot/so101).
IDs are assigned **one motor at a time**, before daisy-chaining:

```bash
pip install -e ".[robot]"
tlod ports                       # find the adapter, e.g. /dev/ttyACM0
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

Linux: `sudo usermod -aG dialout $USER`, then log out and back in.

### 3.2 Calibrate the joints

```bash
lerobot-calibrate --robot.type=so101_follower \
                  --robot.port=/dev/ttyACM0 --robot.id=my_arm
```

```yaml
arm:
  backend: feetech
  port: /dev/ttyACM0
  lerobot_id: my_arm
```

### 3.3 First light

```bash
tlod first-light
```

One joint at a time, ±0.2 rad, slowly. This is where you find an inverted
direction sign harmlessly instead of during a strike. Watch each joint; if
one goes the wrong way, its `sign` is wrong in the calibration.

### 3.4 Check accuracy

```bash
tlod move 0.22 0 0.12 --real
tlod reach
```

Measure where the tip actually landed. Within a few mm is good. A constant
offset means calibration centres are off; a scaling error means a wrong
`sign` or gear ratio.

### 3.5 Replace estimates with measurements

The simulator ships with datasheet servo figures.

```bash
tlod bench all
```

Update `arm.sim_max_speed`, `sim_accel`, `sim_latency` so simulation stays
trustworthy.

### 3.6 Camera

Fixed mount, angled down. Steeper is better — error from a wrong assumed
hand height scales with the tangent of the viewing angle.

Intrinsics (once per camera and resolution):

```bash
tlod calibrate intrinsics --camera 0 -o calib/intrinsics.npz
```

Move a chessboard around: corners, edges, near, far, tilted. Auto-captures
15 views. Aim for RMS below 1 px. `--pattern` counts **inner** corners, so
a printed 10×7 board is `9x6`.

Extrinsics (every time the camera or arm moves):

```bash
# rehearse, nothing moves
tlod calibrate extrinsics --sim --intrinsics calib/intrinsics.npz \
    -o calib/extrinsics.npz

# for real
tlod calibrate extrinsics --intrinsics calib/intrinsics.npz \
    -o calib/extrinsics.npz
```

Stick a green marker on the gripper. The arm drives to twelve poses and
finds the marker in each; forward kinematics supplies the 3D coordinates,
so the result is in exactly the frame the controller commands in. Expect a
few px RMS. One point far worse than the rest is a mislocated marker —
rerun.

```yaml
camera:
  source: opencv
  index: 0
  intrinsics: calib/intrinsics.npz
  extrinsics: calib/extrinsics.npz
```

Verify:

```bash
tlod touch --view
```

The drawn skeleton must land on the real arm. If it is offset, extrinsics
are wrong and nothing downstream will fix it.

### 3.7 Play

```bash
tlod play --difficulty easy --view
```

Stay out of reach for the first run. Read part 6 before putting a hand
under it.

---

## 4. Changing things

### Difficulty

Tuned by **how often it feints**, not by slowing the arm. A slower arm
hits softer and feels broken; more feints genuinely gives you more chances
to score.

```python
# src/tlod/game/handslap.py
"normal": cls(hover_height=0.08, strike_duration=0.21,
              feint_probability=0.45, mean_wait=1.8, settle_bonus=2.5),
```

| field | effect |
|---|---|
| `feint_probability` | ⬆ = easier. Feints are how you score. |
| `hover_height` | strike distance. ⬆ = more warning, harder impact |
| `strike_duration` | ⬆ = easier; stops feeling like a slap past ~350 ms |
| `mean_wait` | average hesitation before committing |
| `settle_bonus` | how much a motionless hand tempts a strike |

Measure, don't guess:

```bash
tlod eval --difficulty normal --reactions 0.18,0.25,0.35
```

Aim for ~50% against 250 ms.

### How hard it hits

`StrikeLimits` in `src/tlod/arm/primitives.py`:

```python
max_drop: float = 0.08        # the safety knob. Shorter = safer AND faster.
strike_speed: float = 3.5     # rad/s during a strike
torque_limit: int = 350       # of 1000, while striking
plane_margin: float = 0.005   # never command below the target plane
```

`max_drop` improves speed and safety together — a shorter strike lands
sooner and arrives slower. Reach for it first.

### A new game

```python
from tlod.game.base import StateMachine
from tlod.arm.primitives import GoToPose
from tlod.types import Pose

class WaveHello(StateMachine):
    name = "wave"
    initial_state = "wave_left"

    def _state_wave_left(self, robot, controller, dt):
        if self.motion is None:
            self.run_motion(GoToPose(Pose(0.22, -0.12, 0.20), 0.6), controller)
        if self.step_motion(controller, dt):
            self.transition("wave_right")

    def _state_wave_right(self, robot, controller, dt):
        if self.motion is None:
            self.run_motion(GoToPose(Pose(0.22, 0.12, 0.20), 0.6), controller)
        if self.step_motion(controller, dt):
            self.transition("wave_left")

    def update(self, robot, perception, dt):
        if robot.controller.estopped:
            return
        getattr(self, f"_state_{self.state}")(robot, robot.controller, dt)
```

`update` runs on the control thread at 100 Hz. Never sleep or block in it
— express waiting as a `Hold` motion or a deadline check.

### A new motion primitive

```python
class MyMotion(Motion):
    name = "mine"

    def _on_start(self, controller):
        self._q0 = controller.commanded.copy()

    def step(self, controller, dt) -> bool:      # True when finished
        controller._write(target_q, max_speed=..., dt=dt)
        return self.elapsed >= self.duration
```

If it changes a hardware setting (torque limit, speed), undo it in **both**
`step` when it finishes **and** `abort`. Forgetting `abort` is a bug this
codebase already had: interrupted strikes left the arm permanently weak.

### A different camera

```yaml
camera:
  source: opencv
  index: 0
  width: 1280
  height: 720
  fps: 60
  fourcc: MJPG
```

Frame rate is a request, not a promise; you are warned at startup if the
camera gives you less. Check with `tlod bench camera --force`. For another
camera type, implement the `Camera` interface (`start`, `stop`, `read`,
`resolution`) and construct it in `build_camera()`.

### A different hand detector

Implement `HandDetector.detect(frame) -> list[Hand2D]`. See `rknn.py` for
a worked example. Box detectors give no finger landmarks — fine for
hand-slap, not for anything reading finger pose.

### Different arm geometry

```bash
python scripts/extract_urdf.py assets/so101_new_calib.urdf
```

Paste into `model.py`, run `pytest tests/test_kinematics.py`. A test pins
FK at home, so it fails loudly if the geometry moved.

---

## 5. Configuration

`tlod config -o my.yaml` writes current settings; use with `-c my.yaml`.
Unknown keys raise rather than being ignored — a silently dropped typo is
how a safety limit fails to apply.

### `arm`

| field | default | |
|---|---|---|
| `backend` | `mock` | `mock` or `feetech` |
| `port` | `""` | empty auto-detects if there is exactly one |
| `lerobot_id` | `""` | read `lerobot-calibrate` output |
| `goal_acceleration` | 60 | 0 = instant and harsh, 254 = smooth |
| `torque_limit` | 800 | of 1000, normal operation |
| `sim_max_speed` | 3.5 | rad/s — estimate, measure yours |
| `sim_accel` | 25.0 | rad/s² — estimate |

### `safety`

| field | default | |
|---|---|---|
| `max_speed` | 2.0 | rad/s, normal motion |
| `strike_speed` | 5.0 | rad/s, explicit strikes only |
| `table_z` | 0.0 | table height in base coordinates |
| `min_height` | 0.015 | never drive the tool below this |
| `max_radius` | 0.33 | horizontal reach cap |
| `min_radius` | 0.08 | do not fold back into the base |
| `command_timeout` | 0.5 | hold position if commands go stale |

### `camera`

| field | default | |
|---|---|---|
| `source` | `mock` | `mock` or `opencv` |
| `latency_offset` | `None` | `None` estimates from the measured frame period. Never hardcode below one frame time. |
| `autofocus`, `autoexposure` | `False` | both add latency and hunt during motion |
| `intrinsics`, `extrinsics` | `""` | paths to your `.npz` files |

### `vision`

| field | default | |
|---|---|---|
| `depth_mode` | `auto` | `plane`, `size`, or `auto` (size, clamped) |
| `palm_width_m` | 0.081 | knuckle span; depth error is proportional |
| `process_noise` | 4.0 | Kalman responsiveness. Fitted on a synthetic path — refit on a recording. |

### `runtime`

| field | default | |
|---|---|---|
| `control_hz` | 100.0 | control loop rate |
| `perception_max_age` | 0.25 | past this, policies get `None` |
| `prediction_horizon` | 0.3 | used by `TrackHandPolicy` and the viewer, not the game |

---

## 6. When it does not work

| symptom | cause | fix |
|---|---|---|
| arm reaches past things | extrinsics wrong | `tlod touch --view`; skeleton must land on the real arm |
| constant offset one way | extrinsics | recalibrate; the marker must be the only green thing in frame |
| a joint moves backwards | inverted `sign` | `tlod first-light`, fix the calibration |
| `IK failures` climbing | target outside workspace | `tlod reach`; check `safety.max_radius` |
| `safety-guard hits` climbing | unreachable poses requested | not fatal, but the game is being clamped |
| tracking drops out | lighting, blur, frame edge | fix lighting first; it is usually lighting |
| hand seen as a red object | skin reads red to colour segmenters | handled by hand suppression; widen the radius |
| jitter high, overruns >10% | CPU starved | lower `control_hz` or camera resolution |
| latency worse than expected | camera gave 30 fps, not 60 | `tlod bench camera --force` |
| `no serial ports found` | power or permissions | check the supply; `usermod -aG dialout` |
| mediapipe crashes on macOS | 1.0.x aborts on arm64 | already pinned to 0.10.3x; check your install |
| a change did nothing | your config overrides the preset | `tlod config -o /tmp/x.yaml` and read what is in effect |

Reproduce anything odd:

```bash
tlod record -o recordings/weird --duration 20
tlod replay recordings/weird --view
```

---

## 7. Safety

This machine moves quickly toward a human hand.

Non-negotiable, on top of the software guards:

1. **Foam paddle, never the gripper.** Compliance matters more than speed:
   1 m/s into something soft is ~15 N over 10 ms; into something rigid,
   ~150 N over 1 ms.
2. **Padded target pad.** A hand on a bare table has nowhere to go.
3. **An inline switch on the 12 V supply, tested.** Normally closed, in
   series with the servo power, so it cuts through copper with no
   software in the path. Software e-stop (`Torque_Enable = 0`) stops
   working exactly when you need it — a hung loop, a crashed process, a
   pulled cable. Flip the switch mid-strike and confirm the arm goes limp
   *before* playing.
4. **Keep `max_drop` small** (~8 cm).
5. **Keep `torque_limit` low for strikes**, so the servo yields on
   unexpected contact.
6. **Start on `easy`, out of reach.**

What the software already does, so you know what is and is not protecting
you:

- strike depth never goes below the target plane — a wrong height estimate
  stalls rather than presses
- strike distance capped in code
- torque lowered during strikes and restored after, including on abort
- joint limits with a margin, plus a Cartesian keep-out volume
- watchdog holds position if commands go stale
- e-stop holds torque rather than dropping it — a limp arm falls, possibly
  onto the hand that triggered the stop

None of that replaces item 3.

---

- [slap-analysis.md](slap-analysis.md) — why it slaps rather than dodges
- [hardware.md](hardware.md) — servo control table, wiring
- [deployment.md](deployment.md) — Orange Pi 5, standalone
- [ROADMAP.md](ROADMAP.md) — what is and isn't built
