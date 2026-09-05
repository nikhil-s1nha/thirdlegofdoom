# Getting this robot working

A start-to-finish guide for someone who has an SO-ARM101 in a box and
wants it playing hand-slap. Assumes you can use a terminal and have seen
Python before. No robotics background needed.

Read part 0. It is short, and skipping it is how people waste a weekend.

---

## 0. Two things that will confuse you otherwise

### The arm is 5-DOF, not 6

It is sold as "6-DOF". It has six motors, but one of them drives the
gripper, so there are **five arm joints**:

```
shoulder_pan     yaw about the base vertical
shoulder_lift  ┐
elbow_flex     ├ three parallel pitch axes
wrist_flex     ┘
wrist_roll       roll about the tool axis
gripper          the jaw
```

Five joints cannot reach an arbitrary position *and* orientation. What
you can ask for is:

```
position (x, y, z)  +  tool pitch  +  tool roll
```

Tool **yaw** is not yours to choose — it is fixed by whichever base pan
reaches the target. This is why `Pose` in this codebase deliberately has
no `yaw` field, and why a general 6-DOF IK library pointed at this arm
will chase orientations it cannot reach and fail confusingly.

### Everything is slow, and that shaped the design

From a hand moving to the arm moving is roughly 200–370 ms. Human
reaction is ~200–250 ms *plus* their hand travel. So a robot that reacts
to what it saw is always late, and no optimisation fixes it — the servo
alone eats most of the budget.

Two consequences you will see everywhere in the code:

1. **It predicts rather than reacts.** A Kalman filter estimates hand
   velocity so the arm aims where the hand *will be*.
2. **The robot goes first.** It slaps; you dodge. Latency only taxes the
   responder, so by initiating, the robot spends its delay before the
   strike where nobody is waiting on it.

Full reasoning in [slap-analysis.md](slap-analysis.md).

---

## 1. Five minutes, no hardware

### First: which branch are you on?

The repository is split so you can work on the arm without the game
getting in the way.

| branch | what you get | commands |
|---|---|---|
| `main` / `arm-core` | the arm, vision, calibration — no game | `move` `reach` `sim` `hybrid` `touch`* `calibrate` `bench` `record` `replay` `first-light` |
| `gamification` | all of the above **plus** hand slap | + `play` `eval` |

\* `touch` needs the object detector, present on both.

If you want to play the game, you want `gamification`:

```bash
git checkout gamification
```

If you are bringing up hardware or building your own behaviour, start on
`arm-core` — fewer moving parts while you get the basics right.

### Install

Everything runs in simulation. Do this before the parcel arrives.

```bash
git clone <your repo> && cd thirdlegofdoom
python3 -m venv .venv && source .venv/bin/activate     # needs Python 3.12+
pip install -e ".[hands,dev]"
```

Then:

```bash
tlod sim --duration 5        # the whole loop, synthetic
tlod move 0.22 0 0.12        # move the tool to a point
tlod play --view             # play hand slap against a simulated human
pytest                       # 140 tests
```

`tlod sim` prints a latency breakdown. This is the shape of a healthy run:

```
  vision.detect             mean   0.15 ms
  vision.shutter->published mean   0.65 ms
  control.policy+ik         mean   0.61 ms
  shutter->servo command    mean   9.43 ms
  control 509 ticks, 2.0% overruns, jitter p95 0.000 ms
  IK: 505 commands, 0 failures, 0 safety-guard hits
```

**What to look at:** `IK failures` and `safety-guard hits` should both be
0. Jitter p95 should be near 0. If overruns are above ~10%, your machine
is struggling and every timing number below will be pessimistic.

### Play it with your own hand, still no arm

If you have a webcam, this is the useful one:

```bash
tlod play --real-hand --view
```

Real camera, real hand, real detection, simulated arm. It exercises
everything except the servos. Keys: `space` pauses, `e` toggles e-stop,
`q` quits.

---

## 2. How the code is organised

### The data flow

Two threads, joined by a one-slot mailbox. This is the whole system:

```
  ┌────────── perception thread (camera rate, bursty) ──────────┐
  camera ──▶ detect hands ──▶ locate in 3D ──▶ track & predict ─┼──▶ ┌─────────┐
                                                                │    │ Latest  │  one slot
  └────────────────────────────────────────────────────────────┘     │ mailbox │  last write wins
                                                                     └────┬────┘
  ┌────────── control thread (fixed 100 Hz, strict) ───────────────────────┼───┐
       policy (a game) ──▶ safety clamp ──▶ IK ──▶ rate limit ──▶ backend ◀┘   │
  └────────────────────────────────────────────────────────────────────────────┘
                                                          backend = MockArm | FeetechArm
```

Three design choices worth knowing, because they explain a lot of the
code:

- **A mailbox, not a queue.** If control falls behind a queue, it starts
  consuming stale frames. Control should always get the newest estimate
  or nothing at all.
- **Staleness is explicit.** `get_fresh(max_age)` returns `None` past a
  deadline. A policy acting confidently on a 400 ms old estimate is worse
  than one acting on nothing.
- **Sim and real are the same code path.** `MockArm` and `FeetechArm`
  satisfy one interface, so `--real` is a flag, not a different program.

### Where things live

```
src/tlod/
  types.py          Every value that crosses a module boundary. Read this
                    first — it defines JOINT_NAMES, Pose, HandObservation.
                    Observations carry the timestamp of the *physical
                    event*, which is how latency stays measurable.

  arm/
    model.py        Forward and inverse kinematics. Geometry transcribed
                    from the official URDF. IK is warm-started
                    Levenberg-Marquardt: 0.4 ms per solve.
    backend.py      The interface MockArm and FeetechArm both satisfy.
    mock.py         Simulator with a real slew rate. Does not teleport,
                    because "can the arm get there in time" is the whole
                    question.
    feetech.py      Real STS3215 servos over the Feetech SDK.
    controller.py   Safety limits, e-stop, min-jerk moves. Every command
                    passes through here.
    primitives.py   hover / strike / retract / feint / goto. Steppable,
                    so a game can abandon one mid-flight.

  vision/
    camera.py       Capture. Threaded grab-and-discard, because the usual
                    cap.read() hands you the *oldest* queued frame.
    calibration.py  Intrinsics, extrinsics, pixel <-> robot frame maths.
    calibrate_flow.py  The interactive calibration procedures.
    hands.py        MediaPipe hand landmarks, and 3D localisation.
    tracking.py     Kalman filter. This is what beats the latency.
    objects.py      Colour-blob object detection.
    scene.py        Synthetic scene for simulation.
    recording.py    Record once, replay deterministically.
    rknn.py         Orange Pi 5 NPU detector.        [unverified]

  runtime/
    signal.py       The one-slot mailbox.
    loop.py         Drift-free fixed-rate loop, latency statistics.
    app.py          Wires it all together. Defines `Policy`.

  game/
    handslap.py     The game: commit timing, feints, scoring.
    opponent.py     A simulated human, so you can test without one.
    contact.py      Did it land? geometric / proximity / piezo.
    touch.py        Visit each detected object. Good calibration check.
    base.py         State machine that is also a Policy.

  viz/              The window. Runs on the main thread, always.
  cli.py            Every command.

firmware/
  pico_sidecar.py   Hardware e-stop + piezo scoring.  [unverified]
```

### The one class you will subclass

Everything the robot *does* is a `Policy`:

```python
class Policy:
    def start(self, robot): ...
    def update(self, robot, perception, dt): ...   # called at 100 Hz
    def stop(self, robot): ...
```

`perception` is `None` when vision is stale. Handle that case
deliberately — that is the whole point of it being `None` rather than
old data.

---

## 3. Getting your hardware working

Do these in order. Each step is cheap to fail; skipping ahead is not.

### 3.1 Assemble and set motor IDs

Follow the [official assembly guide](https://huggingface.co/docs/lerobot/so101).
Motors must be given IDs 1–6 **one at a time**, before they are
daisy-chained:

```bash
pip install -e ".[robot]"
tlod ports                       # find your adapter, e.g. /dev/ttyACM0
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

On Linux you may need `sudo usermod -aG dialout $USER`, then log out and
back in.

### 3.2 Calibrate the joints

```bash
lerobot-calibrate --robot.type=so101_follower \
                  --robot.port=/dev/ttyACM0 --robot.id=my_arm
```

This teaches each joint where its zero and its limits are. Point the
config at it:

```yaml
arm:
  backend: feetech
  port: /dev/ttyACM0
  lerobot_id: my_arm
```

### 3.3 First light — the step people skip

```bash
tlod first-light
```

Moves **one joint at a time**, ±0.2 rad, slowly. This is where you find
out a direction sign is inverted, harmlessly, instead of during a strike.

**Watch each joint.** If one moves the wrong way, its `sign` is wrong in
the calibration. Fix it before continuing.

### 3.4 Check it reaches the right place

```bash
tlod move 0.22 0 0.12 --real
```

Measure where the tool tip actually ended up with a ruler. Within a few
mm is good. A consistent offset means your calibration centres are off;
a scaling error means a wrong `sign` or gear ratio.

```bash
tlod reach                       # what the workspace actually is
```

### 3.5 Replace the estimated numbers with measured ones

The simulator ships with servo figures taken from the datasheet. Measure
yours:

```bash
tlod bench all
```

Then update `arm.sim_max_speed`, `sim_accel` and `sim_latency` in your
config so simulation stays trustworthy.

### 3.6 Mount the camera and calibrate it

Fixed mount, angled down over the table. **Steeper is better** — the
error from a wrong assumed hand height scales with the tangent of the
viewing angle.

**Intrinsics** (the lens — once per camera and resolution):

```bash
tlod calibrate intrinsics --camera 0 -o calib/intrinsics.npz
```

Hold a chessboard in view and move it around: corners, edges, near, far,
tilted. It auto-captures 15 views. **Aim for an RMS below 1 px.** Above
that, reshoot with more varied views and better light.

`--pattern` counts **inner** corners, so a printed 10×7 board is `9x6`.

**Extrinsics** (where the camera is — every time either moves):

```bash
# rehearse it first, no hardware moves
tlod calibrate extrinsics --sim --intrinsics calib/intrinsics.npz \
    -o calib/extrinsics.npz

# then for real
tlod calibrate extrinsics --intrinsics calib/intrinsics.npz \
    -o calib/extrinsics.npz
```

Stick a **green marker** on the gripper first. The arm drives itself to
twelve poses and finds the marker in each; forward kinematics supplies
the 3D coordinates, so the result lands in exactly the frame the
controller commands in.

Expect RMS of a few pixels. If one point is far worse than the rest, it
is a mislocated marker — rerun.

Point the config at both files:

```yaml
camera:
  source: opencv
  index: 0
  intrinsics: calib/intrinsics.npz
  extrinsics: calib/extrinsics.npz
```

**Verify it:**

```bash
tlod touch --view
```

The drawn skeleton must land on the real arm in the picture. If it is
offset, your extrinsics are wrong — no amount of tuning downstream will
fix that.

### 3.7 Then, carefully, play

```bash
tlod play --difficulty easy --view
```

Stay out of reach for the first run. Read part 6 before putting a hand
under it.

---

## 4. How to change things

### Make the robot easier or harder

Difficulty is tuned by **how often it feints**, not by slowing the arm. A
slower arm hits softer and feels broken; a robot that feints more
genuinely gives you more chances to score.

```python
# src/tlod/game/handslap.py
"normal": cls(hover_height=0.08, strike_duration=0.21,
              feint_probability=0.45, mean_wait=1.8, settle_bonus=2.5),
```

| field | effect |
|---|---|
| `feint_probability` | ⬆ = **easier**. Feints are how you score. |
| `hover_height` | how far it strikes from. ⬆ = more warning, but harder impact |
| `strike_duration` | ⬆ = easier, but stops feeling like a slap past ~350 ms |
| `mean_wait` | average hesitation before committing |
| `settle_bonus` | how much a motionless hand tempts a strike |

Measure any change instead of guessing:

```bash
tlod eval --difficulty normal --reactions 0.18,0.25,0.35
```

Aim for ~50% against a 250 ms reaction. Calibrated result today:

| preset | vs 180 ms | vs 250 ms | vs 350 ms |
|---|---|---|---|
| easy | 21% | 0% | 0% |
| normal | 57% | 43% | 50% |
| hard | 86% | 79% | 64% |

### Change how hard it hits

In `StrikeLimits` (`src/tlod/arm/primitives.py`):

```python
max_drop: float = 0.08        # THE safety knob. Shorter = safer AND faster.
strike_speed: float = 3.5     # rad/s during a strike
torque_limit: int = 350       # of 1000, while striking
plane_margin: float = 0.005   # never command below the target plane
```

`max_drop` is unusual in improving speed *and* safety together, because a
shorter strike lands sooner and arrives slower. Reach for it first.

### Write a new game

Subclass `StateMachine` and register it:

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

Rules: `update` runs on the control thread at 100 Hz, so **never sleep or
block in it**. Express waiting as a `Hold` motion or a deadline check.

### Add a motion primitive

Subclass `Motion` in `primitives.py`. Two methods:

```python
class MyMotion(Motion):
    name = "mine"

    def _on_start(self, controller):
        self._q0 = controller.commanded.copy()   # capture the start

    def step(self, controller, dt) -> bool:      # True when finished
        controller._write(target_q, max_speed=..., dt=dt)
        return self.elapsed >= self.duration
```

If your motion changes any hardware setting (torque limit, speed), undo
it in **both** `step` when it finishes **and** `abort`. Forgetting `abort`
is a bug this codebase already had once: interrupted strikes left the arm
permanently weak.

### Swap the camera

Nothing outside `camera.py` knows what camera you have.

```yaml
camera:
  source: opencv
  index: 0
  width: 1280
  height: 720
  fps: 60
  fourcc: MJPG
```

**Frame rate is a request, not a promise.** If the camera silently gives
you 30, you will be warned at startup. Check what you actually got:

```bash
tlod bench camera --force
```

For a different camera type entirely, implement the `Camera` interface
(`start`, `stop`, `read`, `resolution`) and construct it in
`build_camera()`.

### Swap the hand detector

Implement `HandDetector.detect(frame) -> list[Hand2D]`. For a smart
camera or an NPU, see `rknn.py` as the worked example.

**Caveat:** box detectors give no finger landmarks. Fine for hand-slap,
which needs only position and apparent size. Not enough for anything
reading finger pose.

### Change the arm geometry

If you modify the arm physically, regenerate the kinematics from a URDF:

```bash
python scripts/extract_urdf.py assets/so101_new_calib.urdf
```

Paste the output into `model.py` and run `pytest tests/test_kinematics.py`.
One test pins the FK output at home, so it will fail loudly and tell you
the geometry moved — which is the point.

---

## 5. Configuration

`tlod config -o my.yaml` writes the current settings; use with `-c my.yaml`.
Unknown keys raise an error rather than being ignored, because a silently
dropped typo is how a safety limit fails to apply.

### `arm`

| field | default | notes |
|---|---|---|
| `backend` | `mock` | `mock` or `feetech`. Defaults to simulation on purpose |
| `port` | `""` | serial port; empty auto-detects if there is exactly one |
| `lerobot_id` | `""` | read calibration written by `lerobot-calibrate` |
| `goal_acceleration` | 60 | 0 = instant and harsh, 254 = very smooth |
| `torque_limit` | 800 | of 1000, normal operation |
| `sim_max_speed` | 3.5 | rad/s. **Estimate — measure yours** |
| `sim_accel` | 25.0 | rad/s². **Estimate** |

### `safety` — every command passes through these

| field | default | notes |
|---|---|---|
| `max_speed` | 2.0 | rad/s, normal motion |
| `strike_speed` | 5.0 | rad/s, allowed only in an explicit strike |
| `table_z` | 0.0 | table height in base coordinates |
| `min_height` | 0.015 | never drive the tool below this |
| `max_radius` | 0.33 | horizontal reach cap |
| `min_radius` | 0.08 | do not fold back into the base |
| `command_timeout` | 0.5 | watchdog: hold position if commands go stale |

### `camera`

| field | default | notes |
|---|---|---|
| `source` | `mock` | `mock` or `opencv` |
| `latency_offset` | `None` | `None` estimates it from the measured frame period. Do not hardcode below one frame time |
| `autofocus` / `autoexposure` | `False` | both add latency and hunt during motion |
| `intrinsics` / `extrinsics` | `""` | paths to your `.npz` files |

### `vision`

| field | default | notes |
|---|---|---|
| `depth_mode` | `auto` | `plane` (fixed height), `size` (from palm width), `auto` (size, clamped) |
| `palm_width_m` | 0.081 | knuckle span. Depth error is proportional — measure your hand |
| `process_noise` | 4.0 | Kalman responsiveness. Tuned on a *synthetic* path; refit on a recording |

### `runtime`

| field | default | notes |
|---|---|---|
| `control_hz` | 100.0 | control loop rate |
| `perception_max_age` | 0.25 | past this, policies get `None` |
| `prediction_horizon` | 0.3 | how far ahead to aim. **Set from `tlod bench loop`**, not taste |

---

## 6. When it does not work

| symptom | likely cause | what to do |
|---|---|---|
| arm reaches confidently past things | extrinsics wrong | `tlod touch --view`; skeleton must land on the real arm |
| consistent offset in one direction | extrinsics, again | recalibrate; check the marker was the only green thing in frame |
| a joint moves the wrong way | inverted `sign` | `tlod first-light`, fix the calibration |
| `IK failures` climbing | target outside the workspace | `tlod reach`; check `safety.max_radius` |
| `safety-guard hits` climbing | asking for unreachable poses | not fatal, but the game is being clamped |
| hand tracking drops out | lighting, motion blur, edge of frame | fix lighting first; it is usually lighting |
| hand detected as a red object | skin reads red to colour segmenters | already handled by hand suppression; widen the radius |
| jitter high, overruns >10% | CPU starved | lower `control_hz` or camera resolution |
| latency worse than expected | camera gave you 30 fps, not 60 | `tlod bench camera --force` |
| `no serial ports found` | power or permissions | check the power supply; `usermod -aG dialout` |
| mediapipe crashes on import (macOS) | 1.0.x aborts on arm64 | pinned to 0.10.3x already; check your install |
| a game change did nothing | you changed a preset the config overrides | `tlod config -o /tmp/x.yaml` and read what is actually in effect |

Reproduce anything odd deterministically:

```bash
tlod record -o recordings/weird --duration 20
tlod replay recordings/weird --view
```

---

## 7. Safety — read before a hand goes under it

This machine is designed to move quickly toward a human hand. That
sentence is worth re-reading.

**Non-negotiable, in addition to the software guards:**

1. **Soft end effector.** A foam paddle, never the gripper. Compliance
   matters more than speed: 1 m/s into something soft is ~15 N over
   10 ms; into something rigid it is ~150 N over 1 ms.
2. **Padded target pad.** Give on both sides. A hand on a bare table has
   nowhere to go.
3. **Hardware e-stop, tested.** A button on the Pico that cuts servo
   power in its interrupt handler. Software e-stop is a convenience — it
   stops working exactly when you need it, in a hung loop or a crashed
   process. Press it mid-strike and confirm the arm goes limp *before*
   playing.
4. **Keep `max_drop` small.** ~8 cm. It is the one knob that improves
   speed and safety at the same time.
5. **Keep `torque_limit` low for strikes.** The servo should yield on
   unexpected contact rather than push through.
6. **Start on `easy`, out of reach.** Watch a few rounds before joining.

The software guards already in place, so you know what is and is not
protecting you:

- strike depth never goes below the target plane — a wrong height
  estimate stalls rather than presses
- strike distance capped in code
- torque lowered during strikes and restored afterwards, including on
  an aborted strike
- joint limits with a margin, and a Cartesian keep-out volume
- a watchdog that holds position if commands go stale
- e-stop holds torque rather than dropping it, because a limp arm falls,
  possibly onto the hand that triggered the stop

None of that substitutes for item 3.

---

## Where to go next

- [ROADMAP.md](ROADMAP.md) — what is built, what is not, in what order
- [slap-analysis.md](slap-analysis.md) — why the robot slaps rather than
  dodges, and why dodging alone made a bad game
- [hardware.md](hardware.md) — servo control table, wiring, bring-up
- [deployment.md](deployment.md) — Orange Pi 5 and Pico, standalone
