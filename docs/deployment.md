# Deployment: one board

Everything on a single **Orange Pi 5**: camera on USB, servo adapter on
USB-C, vision and the control loop in one process.

```
Orange Pi 5, one process

  camera ─▶ detect ─▶ 3D localise ─┐
                                   ▼
                            ┌─────────────┐
                            │   latest    │   one slot: newest, or nothing
                            └──────┬──────┘
                                   ▼
  100 Hz:  game ─▶ safety clamp ─▶ IK ─▶ rate limit ─▶ FeetechArm
                                                           │ USB
                                                           ▼
                                                      servo board ──▶ servos
```

This page used to describe a two-board split — vision here, control on a
Raspberry Pi — and said that nothing here had run on real boards. It has
now run, and the measurement went the other way:

| | overruns at 100 Hz |
|---|---|
| Orange Pi 5, control loop alone | 0.2% |
| Orange Pi 5, control loop **and** vision | 2.6% |
| Raspberry Pi, control loop alone, no vision at all | 9.9% |

The Raspberry Pi could not hold a 100 Hz tick with nothing else on it; it
needed `runtime.control_hz: 50`. The Orange Pi holds it *while* running
the camera and mediapipe. So the split was putting the loop on the slower
of the two boards and paying a network hop for the privilege. One board,
both jobs.

Everything below about the split is still true and still tested — it is
just no longer the default. Reach for it when you add the NPU, or a
policy heavy enough to genuinely contend with the control thread.

## Two boards, if you want them

```
Orange Pi 5                       Raspberry Pi
┌────────────────┐  UDP, ~150 B   ┌────────────────┐  USB   ┌────────────┐
│ camera         │───────────────▶│ latest mailbox │───────▶│ servo board│──▶ servos
│ detect (NPU)   │  60/s          │ game + IK      │        └────────────┘
│ 3D localise    │◀───────────────│ control loop   │
└────────────────┘  clock sync    └────────────────┘
```

The perception/control boundary was already a one-slot mailbox, so the
network hop drops in without the game, the controller or the IK knowing.

**Push, not pull.** Requesting an update each control tick would cost a
round trip and block the loop on the network. The vision board publishes
every detection; the control board keeps only the newest.

**UDP, not TCP.** This is a latest-value stream. TCP's ordered delivery
means one delayed packet stalls the newer ones behind it, while a dropped
datagram costs nothing because another arrives in ~16 ms.

**Base-frame coordinates cross the wire, not pixels.** The vision board
owns the calibration and does the projection, so the control board needs
no intrinsics or extrinsics, and moving the camera means recalibrating
exactly one machine. Frames never cross: a 720p stream is ~30 Mbit/s and
the control board has no use for pixels.

**Clock offset is measured.** Every timestamp means "when the shutter
opened", and the freshness gate depends on it; across two boards those
are unrelated numbers. An NTP-style exchange runs at startup and every
30 s, keeping the sample with the smallest round trip. The control board
**refuses to start** without one, because judging freshness against
nonsense fails silently.

Use **wired** ethernet, or USB-gadget ethernet. WiFi adds 1-20 ms of
jitter to the one path whose whole purpose is being timely.

On the Orange Pi:

```bash
tlod vision-serve --to 192.168.1.50        # the Pi's address
```

On the Raspberry Pi:

```bash
tlod control --vision-host 192.168.1.40 --real --policy track_hand
```

Test the link before any hardware is involved — both flags work on one
machine:

```bash
tlod vision-serve --sim --to 127.0.0.1 &
tlod control --vision-host 127.0.0.1
```

Measured across two processes on one machine: 12.8 ms shutter-to-servo,
against ~10 ms in-process. Expect wired ethernet to add well under a
millisecond.

One warning that only shows up once you have used this: a `vision-serve`
left running from an earlier session keeps publishing to the same port,
and the control side cannot tell one publisher from another. It will take
whichever datagram arrived last. Check for a stale process before
concluding anything about a run that looks subtly wrong.

## Checking it without a screen

```bash
tlod vision-check --duration 30            # precision: camera only
tlod vision-check --with-arm --json r.json # accuracy: scored against kinematics
tlod vision-serve --preview 8081 ...       # watch from a browser elsewhere
```

The camera-only checks cannot detect a bad extrinsic — they measure
consistency, not correctness. Only `--with-arm` does, because forward
kinematics is the only ground truth on the robot. Non-zero exit on
failure, so it runs from cron.

There are four more diagnostics for exactly this situation, each serving
an annotated MJPEG view over HTTP because the board has no screen. See
[headless.md](headless.md); the short version is `board_view.py` (is the
chessboard being detected at all?), `marker_view.py` (which blob would
win?), `calib_view.py` (is the lens model actually describing this lens?)
and `check_extrinsics.py` (does the camera agree with the arm, in
millimetres?).

## What you need

| | |
|---|---|
| SO-ARM101 + bus adapter | in the kit |
| 12 V 5 A supply | in the kit — check yours, some ship 2 A |
| **inline switch on the 12 V line** | the physical e-stop |
| camera | fixed mount, angled down |
| Orange Pi 5 | vision *and* control |
| a second board + wired ethernet | optional; only for the split above |

No sidecar microcontroller: the servos carry torque and overload limits
themselves, a mechanical switch is a better e-stop than any chip, and
contact is read from `Present_Load` over the bus already in use -- but
only in the one regime where that register says anything. Measured
across nothing / a book / a hand, the peak load *during* a swing read
0.330 / 0.326 / 0.350: a rigid book landed between the other two,
because the arm braking its own mass reaches the torque cap in every
run, empty table included. Held still at the bottom, the same three read
**0.001 / 0.038 / 0.037**.

So `Strike` stays down for `press_hold` (450 ms) at the strike's torque
limit, and `ServoPressContactSensor` reads only after the servo's load
filter has decayed -- `--contact press`. `Present_Current` (addr 69) was
tried alongside and is not usable: it separated the same three
conditions by 0.006 A, exactly one 6.5 mA quantisation step, and smaller
than the jitter within a single run.

**`--contact height` is the recommended one.** During that same hold the
encoders answer the question directly: the strike commands a floor below
any plausible hand, and whatever the paddle stops short by is the
thickness of what was in the way. It needs no torque model, no baseline,
and half the settling time, and it resolves 0.1 mm at the tool against a
hand worth twenty-odd millimetres.

Both need the floor to be *below* the hand, which is what
`StrikeLimits.press_depth` is for, and both need to know where the table
is. Driven to the joint angles at which the gripper rests on it, FK
reports the tool at **+0.2 mm** -- the riser is absorbed into the
calibration, so model z is height above the work surface directly, and
`safety.min_height` is that height in the same units.

The adapter's 5 V buck is specified for a Raspberry Pi, so it can power a
control board in the two-board layout. An Orange Pi 5 can draw up to 4 A
-- give it its own supply.

The arm's own draw is no longer a question mark: measured peak **1.81 A**
against a 3.75 A planning budget, no rail sag, no latched servo faults.
The arithmetic behind that budget is in [power.md](power.md).

## Can one board hold the loop?

Yes — 2.6% overruns at 100 Hz with vision running alongside, measured
above. Per tick the control thread solves IK, runs the game state
machine, clamps for safety, and talks to the servo board.

Position-only IK uses an analytic Jacobian -- one forward-kinematics pass
instead of six -- which is much of why that fits. Check it on your own
board rather than trusting this one:

```bash
tlod bench ik
tlod bench all
```

If it does not fit, lower `runtime.control_hz`. At 50 Hz a 230 ms strike
still gets ~11 command updates, and the servos run their own internal
loops between them.

## Orange Pi 5

Ubuntu or Debian arm64, Python 3.12 or 3.13. glibc 2.28 or newer, which
any current Ubuntu or Debian arm64 image has.

```bash
sudo apt install -y python3-venv python3-dev libgl1 libglib2.0-0
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[hands]"
pip install pyserial feetech-servo-sdk        # the arm
sudo usermod -aG dialout $USER                # then log out and back in
```

**Not `.[robot]`.** That extra is `lerobot[feetech]`, which drags in
torch — a very large install, and it does not resolve on Python 3.13 at
all. Nothing in the control path wants it: `feetech-servo-sdk` provides
the `scservo_sdk` that `tlod.arm.feetech` imports, and `pyserial`
provides port enumeration. lerobot is worth installing only where you run
its *calibration* tooling (`lerobot-setup-motors`, `lerobot-calibrate`),
and that need not be this board — `arm.calibration` reads the JSON
lerobot writes, so calibrate wherever it installs and copy the file
across.

### mediapipe

Pins differ here, and differ again per interpreter: upstream's aarch64
wheels stop at 0.10.18, which is cp39-cp312, and resume at 1.0. So Python
3.12 gets 0.10.18 and Python 3.13 gets 1.0. `pyproject.toml` picks; the
Tasks API is the same in both, so there is no code change either way.

The 3.13 path used to be a bet — upstream claims nothing either way about
1.0 on aarch64 Linux — and it now has a measurement behind it.
**mediapipe 1.0.1 works here**: 31 fps, 100% detection on a hand in
frame, 2.4–5 mm frame-to-frame jitter. Check which one you ended up with:

```bash
python -c "import mediapipe; print(mediapipe.__version__)"
tlod vision-check --sim --duration 3     # pipeline, no camera needed
tlod vision-check --duration 10          # real camera, real hand
```

Either version drags in `opencv-contrib-python`, which unpacks into the
same `cv2/` as the `opencv-python` this project asks for. Two
distributions, one import name, last one wins. It has always been that
way here and it works, but if `cv2` starts behaving oddly after a
reinstall, that is where to look first.

If mediapipe is missing or cannot build a landmarker, `tlod` logs
`no hand detector available` and carries on with no hands rather than
dying — object detection, the publisher and control all still run. That
is a degraded board, not a working one; fix the install.

### The camera index is not stable

`/dev/video*` numbering changes across reboots and replugs. It moved
twice in one evening here, and `/dev/video5` was a hardware video encoder
on one boot and the camera on the next. Nothing warns you: opening an
encoder node can succeed and then yield nothing, or yield something that
is not the room.

So check it rather than trusting the `camera.index` that worked last
week:

```bash
tlod cameras                              # which indices open at all
python3 scripts/board_view.py 9x6 5       # and which one is pointing at you
```

`ls -l /dev/v4l/by-id/` gives a name that survives a reboot, if you would
rather resolve it that way.

### NPU

`rknn-toolkit-lite2` ships from Rockchip, not PyPI. Model *conversion*
needs `rknn-toolkit2` on an x86_64 host; only the lite runtime goes on
the board.

```bash
pip install ./rknn_toolkit_lite2-*-cp312-cp312-linux_aarch64.whl
python -c "from rknnlite.api import RKNNLite; print('ok')"
```

Roughly YOLOv5n at 58 fps, YOLOv5s at 37 fps — vendor figures; still not
measured here, and the NPU path is still the one part of this page that
has not run. Box detectors give no finger landmarks — fine for hand-slap,
not for gesture games.

## Order

1. `tlod ports` (the servo adapter plugs into the Orange Pi)
2. motor IDs, one at a time
3. `lerobot-calibrate`, on whatever machine lerobot installs on; point
   `arm.calibration` at the JSON it writes
4. `tlod probe --real` — torque off, move it by hand, check every joint answers
5. `tlod first-light` — where an inverted sign shows up harmlessly
6. `tlod move 0.22 0 0.12 --real` — check Cartesian accuracy
7. `tlod power -c configs/real_arm.yaml` — confirm the supply, don't infer it
8. `tlod bench all` — replace estimates with measurements, retune `MockArm`
9. mount the camera, then `tlod calibrate intrinsics --fisheye --preview 8080`,
   then `tlod calibrate extrinsics --marker red --gripper 0 --heights 0.06,0.14,0.22`
10. `python3 scripts/check_extrinsics.py red` — does the camera agree with the
    arm, in millimetres, at poses the calibration never saw
11. `tlod touch --real` — the end-to-end check. It finds coloured objects on
    the table and drives the tool to each one; the gap between tool and object
    is every error in the system added up
12. test the inline power switch mid-motion, **before any game runs**
13. `tlod play --difficulty easy`, staying out of reach

Step 11 used to read `verify with tlod touch --view — drawn skeleton must
land on the real arm`. That was wrong in three ways: `touch` had never
run at all (it raised on its first line), it draws no skeleton, and it
requires `--real` because there is nothing to simulate — a synthetic
scene renders no objects, and rendering them through the same calibration
that recovers them would agree with itself whatever the camera's real
position. Judge it by watching the tool and the object, or by the
millimetres from step 10.

### What those steps measured here

| | |
|---|---|
| `tlod move --real`, horizontal | ~2 mm from the commanded point |
| `tlod move --real`, vertical | 5–20 mm low |
| intrinsics, fisheye model | 0.165 px RMS |
| extrinsics | camera 445 mm forward, 35 mm right, 451 mm above the base |
| `scripts/check_extrinsics.py` | 4.5 mm |
| `tlod vision-check --with-arm` | 12.1 mm |
| `tlod touch --real`, tool onto a detected object | **2.5 mm** |

The vertical droop is not calibration error and recalibrating will not
remove it — the servos yield under gravity, more the further out the arm
reaches. It is stiffness, and it is the reason the arm is trusted
horizontally and only approximately in z.

The three vision numbers disagree, and the disagreement is the
informative part. `vision-check --with-arm` is the worst of them because
its marker sits on the gripper *jaw* while forward kinematics reports the
tool point, so it is scoring a real offset the other two do not have.
`touch` is the number that matters: the whole chain end to end, with
nothing standing in for anything.

## Troubleshooting

**`no serial ports found`, and plugging or unplugging changes nothing.**
`find_ports()` returns an empty list when `pyserial` is not importable,
which looks exactly like a bus adapter that is not there. Check the
package before the cable:

```bash
python -c "import serial; print(serial.__version__)"
```

Then permissions (`usermod -aG dialout $USER`, log out and back in), then
the cable, then the supply.

**The camera opens but the frames are wrong or absent.** See "the camera
index is not stable" above. It is the most likely cause of a vision run
that produces nothing and does not complain.

**Calibration produced a confident, wrong answer.** Both calibration
steps take the *largest* blob of the marker colour and believe it is the
gripper. Run `python3 scripts/marker_view.py <colour>` first and watch
where the crosshair goes, before letting the arm drive a dozen poses
against a mug.

## Autostart

```ini
# /etc/systemd/system/tlod.service
[Unit]
Description=Third Leg of Doom
After=multi-user.target

[Service]
Type=simple
User=tlod
WorkingDirectory=/home/tlod/thirdlegofdoom
ExecStart=/home/tlod/thirdlegofdoom/.venv/bin/tlod play --difficulty normal
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`Restart=on-failure` is not a substitute for the power switch. A process
that restarts cleanly every five seconds while swinging at someone is
still swinging at someone.
