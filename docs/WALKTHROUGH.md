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
| `main` | arm, vision, calibration **and** the game (`play`, `eval`) |
| `arm-core` | arm and vision only, no game |
| `gamification` | merged into `main`; kept for history |

Stay on `main` unless you specifically want the arm without the game.

### Install

```bash
git clone <repo> && cd thirdlegofdoom
python3 -m venv .venv && source .venv/bin/activate     # Python 3.12+
pip install -e ".[hands,dev]"
```

```bash
tlod sim --duration 5        # the whole loop, synthetic
tlod move 0.22 0 0.12        # move the tool to a point
tlod play --view             # hand slap vs a simulated human
pytest
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
    profile.py      velocity/acceleration/jerk limits on the command
                    stream, synchronised across joints. Every command
                    goes through this.
    power.py        what a motion costs the supply, and slowing down to
                    fit it. See docs/power.md.
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
    handslap.py     commit timing, feints, scoring
    opponent.py     a simulated human, so you can test without one
    contact.py      did it land? geometric / proximity / servo load
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

All of this has now been done once, on an Orange Pi 5 with the arm and
camera on the same board, and the numbers below are from that run rather
than from a datasheet. Where something is still an estimate it says so.
The end of it is `tlod touch --real` putting the tool 2.5 mm from an
object the camera found by itself.

### Just the arm — the short version

If all you want is an arm that moves to coordinates, you need **3.0
through 3.5**. No camera, no game, nothing else plugged in. Budget an
hour, most of it assembly.

```bash
tlod ports                          # 1. is the board there?
lerobot-setup-motors ...            # 2. give each motor an id
lerobot-calibrate ...               # 3. teach it zero and its limits
tlod probe --real                   # 4. read it with torque OFF
tlod first-light                    # 5. move one joint at a time
tlod move 0.22 0 0.12 --real        # 6. go to a point
tlod power -c configs/real_arm.yaml # 7. is the supply big enough?
```

Done: the arm goes where you tell it. Everything after 3.5 is the camera
and the game, and neither is needed for that.

**Before you plug anything in**

- 12 V 5 A supply for the follower arm. Not 5 V — that is the leader.
  Not 2 A either, even though some kits ship one and Seeed's own spec
  says 2 A: it browns out as soon as several joints move together, which
  looks like a software bug and is not. With the 5 A supply this arm
  peaks at 1.81 A against a 3.75 A budget and the rail does not move, so
  the problem simply does not arise — confirm yours with `tlod power`
  rather than inheriting anyone's fear of it. [power.md](power.md).
- Give it clear space. Nothing fragile, nobody's hands in range.
- Know where the power switch is. If you have not fitted an inline switch
  on the 12 V line yet, know which plug you are pulling.
- Do not skip 3.4 and 3.5. They exist so that a wrong direction sign
  turns up harmlessly rather than at speed.

### 3.0 Is it alive?

The safest possible first test. Torque off, nothing commanded, you move
the arm by hand and watch the numbers.

```bash
tlod ports                # find the adapter, e.g. /dev/ttyACM0
tlod probe --real
```

**Support the arm before this runs.** With torque off it is limp and will
fold under its own weight.

Move every joint through its range. You are checking four things:

| | what good looks like |
|---|---|
| the bus works | numbers appear at all |
| all six motors answer | every joint says `yes` under "moved?", not `NOT SEEN` |
| directions are sane | each joint's value moves the way you expect |
| nothing is unwell | temperature under ~40 °C, voltage near 12 V |

`NOT SEEN` on a joint you definitely moved means that motor is not
answering: check its 3-pin cable and that its id was actually set.

Nothing here commands motion, so nothing can lurch. If something is
wired wrong, this is where you want to find out.

### 3.1 Assemble, set motor IDs

Follow the [official assembly guide](https://huggingface.co/docs/lerobot/so101).
IDs are assigned **one motor at a time**, before daisy-chaining:

```bash
pip install pyserial feetech-servo-sdk       # what the arm needs
tlod ports                                   # find the adapter, e.g. /dev/ttyACM0

pip install -e ".[robot]"                    # only for the two lerobot- commands
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

Those are two different installs on purpose. `feetech-servo-sdk` is the
`scservo_sdk` this project actually drives the bus with, and `pyserial`
finds the port; together they are a few megabytes. `.[robot]` is
`lerobot[feetech]`, which pulls torch and **does not resolve on Python
3.13** — so on a board running 3.13 you cannot install it at all. You do
not need to: run the two `lerobot-` commands on any machine where it does
install, and point `arm.calibration` at the JSON they write. The file
format is all this project wants from lerobot.

If `tlod ports` says `no serial ports found`, check `pyserial` is
importable before you check the cable: `find_ports()` returns an empty
list when the import fails, so a missing package and a missing adapter
look identical.

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

`lerobot_id` looks the file up in lerobot's cache, so it only works on
the machine that ran the calibration. If that was a different machine,
copy the JSON over and use `calibration: calib/mine.json` instead —
`Calibration.load` reads either format.

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

What this arm does, once calibrated: **~2 mm horizontally**, and **5–20 mm
low** in z. The vertical error is not calibration and recalibrating will
not remove it — the servos yield under gravity, and yield more the
further out the arm reaches. Treat x and y as accurate and z as
approximate, command a little high when height matters, and do not spend
an evening trying to calibrate droop away. (We did.)

### 3.5 Replace estimates with measurements

*(This is the last arm-only step. If you just wanted a working arm, you
are done — 3.6 onward is the camera and the game.)*

The simulator ships with datasheet servo figures.

```bash
tlod bench all
```

Update `arm.sim_max_speed`, `sim_accel`, `sim_latency` so simulation stays
trustworthy.

The two motion numbers worth knowing, measured here with the limits in
`configs/real_arm.yaml`: an **80 mm strike drop in 0.23 s** at best, 0.27 s
at an accuracy worth having, and a **150 mm move that will not go below
0.48 s**. The game's `strike_duration` defaults are close enough to the
first of those that they are worth re-reading against your own arm rather
than assuming.

### 3.6 Camera

Fixed mount, angled down. Steeper is better — error from a wrong assumed
hand height scales with the tangent of the viewing angle.

**Find it first.** `/dev/video*` indices move between reboots and
replugs — twice in one evening here, and `/dev/video5` was a hardware
encoder on one boot and the camera on the next. `tlod cameras` lists what
opens; `python3 scripts/board_view.py 9x6 <index>` shows you what each one
is actually looking at, over HTTP, which is the only way to tell on a
headless board.

Intrinsics (once per camera and resolution):

```bash
tlod calibrate intrinsics --camera 0 -o calib/intrinsics.npz \
    --fisheye --preview 8080
```

Move a chessboard around: corners, edges, near, far, tilted. Auto-captures
15 views. Aim for RMS below 1 px. `--pattern` counts **inner** corners, so
a printed 10×7 board is `9x6`.

- `--fisheye` switches to an equidistant model. Anything much past 120° of
  field of view needs it: the default pinhole model does not merely fit
  worse, it cannot fit at all. The wide module here came out at **0.165 px
  RMS** with `--fisheye`.
- `--preview PORT` serves the annotated view while capturing. Without it,
  a run where the board is never detected prints nothing and looks exactly
  like a dead camera.
- The command prints the horizontal field of view it *recovered* next to
  the one in your config. Believe the recovered one: the module here is
  advertised at 145° diagonal / 120° horizontal and measures **74°** in the
  640x480 mode USB bandwidth allows, because that mode is a crop. Put the
  measured number in `camera.hfov_deg`; it seeds the solve.

Extrinsics (every time the camera or arm moves):

```bash
# rehearse, nothing moves
tlod calibrate extrinsics --sim --intrinsics calib/intrinsics.npz \
    -o calib/extrinsics.npz

# for real
tlod calibrate extrinsics --intrinsics calib/intrinsics.npz \
    -o calib/extrinsics.npz --marker red --gripper 0 --heights 0.06,0.14,0.22
```

Stick a coloured marker on the gripper. The arm drives to a spread of
poses and finds the marker in each; forward kinematics supplies the 3D
coordinates, so the result is in exactly the frame the controller commands
in. Expect a few px RMS. One point far worse than the rest is a mislocated
marker — rerun.

- `--marker` picks the colour. **The largest blob of that colour wins,
  whatever it belongs to**, and a wrong pick does not fail: it produces a
  confident calibration of the camera against a mug. Run
  `python3 scripts/marker_view.py red` first and watch where the crosshair
  settles.
- `--gripper` is the jaw opening held during the run. It matters because
  an open jaw puts the marker somewhere other than the tool point that
  forward kinematics is reporting, and that offset goes straight into the
  extrinsics. Which end of the travel is closed depends on the sign in
  your calibration, so check rather than assume.
- `--heights` is the set of tool heights to visit. The spread in z is what
  conditions the solve, so narrow it only if the arm hides the marker when
  raised, and narrow it as little as you can.

```yaml
camera:
  source: opencv
  index: 0
  hfov_deg: 74.0
  intrinsics: calib/intrinsics.npz
  extrinsics: calib/extrinsics.npz
```

Verify — three ways, in increasing order of how much they mean:

```bash
python3 scripts/check_extrinsics.py red     # camera vs kinematics, in mm
tlod vision-check --with-arm                # the same idea, scored and gated
tlod touch --real                           # the whole chain, end to end
```

`touch` finds coloured objects on the table and drives the tool to each
one. Watch the gap between the tool and the object: that gap is every
error in the system added up, and it is the only one of the three with
nothing standing in for anything. Here it is **2.5 mm**, against 4.5 mm
from `check_extrinsics.py` and 12.1 mm from `vision-check --with-arm` —
the last is worst because its marker sits on the gripper jaw rather than
the tool point, so it is scoring a real offset that the others do not
have.

`touch` requires `--real` and there is no simulated version. Rendering
objects through the same calibration that recovers them would agree with
itself whatever the camera's true position, which is a check that cannot
fail and therefore says nothing.

For reference, the mounted camera here solved to 445 mm forward, 35 mm
right and 451 mm above the base — a sanity check you can make with a tape
measure before trusting anything downstream.

### 3.6b Verifying vision on a headless board

If the vision board has no screen you cannot check the boxes by eye. Two
tools cover that, and the difference between them matters.

```bash
tlod vision-check --duration 30 --save-frames /tmp/frames
```

**Precision** — detection rate, jitter, teleports, depth stability.
Camera only. Tells you the pipeline is stable and self-consistent. It
does **not** tell you the answer is right: a badly calibrated camera
gives beautifully precise, consistently wrong positions.

```bash
tlod vision-check --with-arm --duration 20 --json report.json
```

**Accuracy** — drives the arm to known configurations and compares what
vision reports against forward kinematics. This is the only check that
catches a bad extrinsic, because the arm is the only ground truth
available. Green marker on the gripper, workspace clear.

Exits non-zero on failure, so it can run from cron:

```bash
tlod vision-check --with-arm --yes --json /var/log/tlod-vision.json || notify
```

Add `--fixed-distance` only if you actually held your hand at a constant
distance; otherwise depth stability is reported but not enforced, because
that precondition is an instruction to you, not something the code can
verify.

**To see what it sees**, from any other machine:

```bash
tlod vision-serve --preview 8081 --to <control board>
```

Then open `http://<vision board>:8081/` in a browser. Plain MJPEG, no
player needed. It is throttled and runs on its own thread, but it is a
diagnostic — leave it off in normal operation.

Four smaller views cover the parts of bring-up the main commands answer
badly, each serving the same kind of MJPEG stream on port 8080:

| | question it answers |
|---|---|
| `scripts/board_view.py` | is the chessboard being detected at all? |
| `scripts/marker_view.py` | which blob would the calibrator pick? |
| `scripts/calib_view.py` | is the lens model describing this lens? |
| `scripts/check_extrinsics.py` | does the camera agree with the arm, in mm? |

Each one's docstring explains the failure it was written for. More in
[headless.md](headless.md).

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

**Config files do not layer.** `Config.load` reads the one file you pass
and nothing else; every key you leave out falls back to the dataclass
defaults in `src/tlod/config.py`, *not* to `configs/default.yaml`. That
file is a dump of those defaults, not a base layer the others inherit
from — so editing it changes nothing about a run started with
`-c configs/opi.yaml`, and it can drift out of step with the code without
anything complaining. (It already has: it carries
`camera.latency_offset: 0.035` where the dataclass says `None`.) Each
config in `configs/` is a complete, standalone answer for one machine.

The files that exist: `default.yaml` (the defaults, written out),
`opi.yaml` (the single-board rig — real camera, real arm, `depth_mode:
plane`), `real_arm.yaml` (the arm on a slower control board),
`perf.yaml` (one vision tweak).

### `arm`

| field | default | |
|---|---|---|
| `backend` | `mock` | `mock` or `feetech` |
| `port` | `""` | empty auto-detects if there is exactly one |
| `lerobot_id` | `""` | look `lerobot-calibrate` output up in lerobot's cache |
| `calibration` | `""` | path to a calibration JSON — ours or lerobot's |
| `servo_accel` | 9.2 | rad/s², the servo's own ramp. **Bigger is harsher.** |
| `torque_limit` | 800 | of 1000, normal operation |
| `sim_max_speed` | 3.5 | rad/s — estimate, measure yours |
| `sim_accel` | 25.0 | rad/s² — estimate |

`servo_accel` maps to `Goal_Acceleration`, which is an acceleration
magnitude and not a smoothness dial: 0 disables the ramp for *maximum*
harshness, and 254 (~39 rad/s²) is the harshest finite setting. Given in
rad/s² here so the direction cannot be misread. See
[power.md](power.md).

### `safety`

| field | default | |
|---|---|---|
| `max_speed` | 2.0 | rad/s, normal motion |
| `strike_speed` | 5.0 | rad/s, explicit strikes only |
| `max_accel` | 8.0 | rad/s². Sets motor torque, so sets current draw. |
| `max_jerk` | 80.0 | rad/s³. Stops that current arriving as a step. |
| `table_z` | 0.0 | table height in base coordinates |
| `min_height` | 0.015 | never drive the tool below this |
| `max_radius` | 0.33 | horizontal reach cap |
| `min_radius` | 0.08 | do not fold back into the base |
| `command_timeout` | 0.5 | hold position if commands go stale |
| `max_tick_dt` | 0.05 | cap on dt, so a stalled tick cannot lurch |

### `power`

Only relevant on real hardware. See [power.md](power.md).

| field | default | |
|---|---|---|
| `governor` | `false` | slow the arm to fit the supply rather than brown out |
| `supply_current` | 5.0 | A, what the brick is rated for. Be honest. |
| `headroom` | 0.75 | fraction of that to plan for |
| `min_voltage` | 10.5 | V, below this the rail is judged to be sagging |

### `camera`

| field | default | |
|---|---|---|
| `source` | `mock` | `mock` or `opencv` |
| `latency_offset` | `None` | `None` estimates from the measured frame period. Never hardcode below one frame time. |
| `autofocus`, `autoexposure` | `False` | both add latency and hunt during motion |
| `intrinsics`, `extrinsics` | `""` | paths to your `.npz` files |
| `hfov_deg` | 145.0 | horizontal field of view, **measured**, not the box's number. Seeds the intrinsics solve. |

### `vision`

| field | default | |
|---|---|---|
| `depth_mode` | `auto` | `plane`, `size`, or `auto` (size, clamped) |
| `hand_height` | 0.06 | m above the table, used by `plane` |
| `palm_width_m` | 0.081 | knuckle span; depth error is proportional |
| `process_noise` | 4.0 | Kalman responsiveness. Fitted on a synthetic path — refit on a recording. |

One camera cannot measure depth, so `size` mode infers it from apparent
palm width — a proportional error on a quantity that varies between
people, and the largest single error source in the system. `depth_mode:
plane` with the palm **flat on the table** removes it: the height is then
known rather than guessed, and the pixel ray meets a known plane. If your
game can ask for a flat hand, ask for it, and set `hand_height` to the
thickness of a palm rather than leaving the 60 mm default that assumes a
hovering hand.

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
| arm reaches past things | extrinsics wrong | `tlod touch --real`; watch the gap between tool and object |
| the tool lands 1–2 cm low | servo droop under gravity | not calibration. Command higher; see 3.4 |
| constant offset one way | extrinsics | recalibrate; the marker must be the only green thing in frame |
| a joint moves backwards | inverted `sign` | `tlod probe --real`, move it by hand, check the sign; fix the calibration |
| a joint reads nothing | motor not answering | check its 3-pin cable and that its id was set |
| the arm folds when powered | torque is off | expected during `probe`; support it |
| `IK failures` climbing | target outside workspace | `tlod reach`; check `safety.max_radius` |
| `safety-guard hits` climbing | unreachable poses requested | not fatal, but the game is being clamped |
| tracking drops out | lighting, blur, frame edge | `tlod vision-check`; fix lighting first, it is usually lighting |
| no screen on the vision board | — | `tlod vision-check` for numbers, `--preview 8081` to watch from a browser |
| vision precise but arm reaches wrong | bad extrinsics | `tlod vision-check --with-arm`; camera-only checks cannot see this |
| hand seen as a red object | skin reads red to colour segmenters | handled by hand suppression; widen the radius |
| fine on one joint, jitters on several | supply browning out | `tlod power`; see [power.md](power.md). 12 V 5 A, not 2 A |
| a joint goes briefly limp mid-move | undervoltage, servo dropped torque | same; `tlod power` reports latched faults |
| jitter high, overruns >10% | CPU starved | lower `control_hz` or camera resolution |
| latency worse than expected | camera gave 30 fps, not 60 | `tlod bench camera --force` |
| `no serial ports found` | `pyserial` missing, power, or permissions | `python -c "import serial"` **first** — a failed import returns the same empty list; then `usermod -aG dialout` |
| camera opens, frames wrong or absent | `/dev/video*` renumbered | `tlod cameras`, `scripts/board_view.py`; indices move between boots |
| it keeps touching the wrong thing | largest blob of the colour wins | `scripts/marker_view.py`; remove the distractor or change colour |
| vision looks right but the arm acts on something else | a stale `vision-serve` still publishing | check for a leftover process; the mailbox takes whatever arrived last |
| mediapipe crashes on macOS | 1.0.x aborts on arm64 | already pinned to 0.10.3x; check your install |
| `no hand detector available` in the log | mediapipe missing or unbuildable | everything but hands still runs; `pip install -e ".[hands]"` |
| a change did nothing | your config overrides the preset, or you edited `default.yaml` expecting other configs to inherit it | configs do not layer; `tlod config -o /tmp/x.yaml` and read what is in effect |

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

- [power.md](power.md) — brownout, current budgets, motion profiles
- [headless.md](headless.md) — verifying vision on a board with no screen
- [slap-analysis.md](slap-analysis.md) — why it slaps rather than dodges
- [hardware.md](hardware.md) — servo control table, wiring
- [deployment.md](deployment.md) — Orange Pi 5, standalone
- [ROADMAP.md](ROADMAP.md) — what is and isn't built
