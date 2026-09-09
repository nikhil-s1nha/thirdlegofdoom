# Hardware

## SO-ARM101 Pro (Seeed)

Six Feetech STS3215 bus servos on one TTL serial chain, 1 Mbaud. Follower
uses ST-3215-C001 (1:345) throughout, at 12 V (30 kg·cm).

Motor IDs 1–6 along the chain, matching `tlod.types.JOINT_NAMES`:

| id | joint | axis | limit (rad) |
|---|---|---|---|
| 1 | `shoulder_pan` | base yaw | ±1.920 |
| 2 | `shoulder_lift` | pitch | ±1.745 |
| 3 | `elbow_flex` | pitch | ±1.690 |
| 4 | `wrist_flex` | pitch | ±1.658 |
| 5 | `wrist_roll` | tool roll | −2.744 … +2.841 |
| 6 | `gripper` | jaw | −0.175 … +1.745 |

Joints 2–4 are parallel pitch axes. With base yaw and tool roll that
gives five arm joints: position plus tool pitch and roll, no independent
tool yaw. This is why the arm is 5-DOF despite six motors.

Limits come from `assets/so101_new_calib.urdf`. Regenerate with
`scripts/extract_urdf.py` if the upstream model changes.

## The bus adapter board is a bridge, not a controller

The board in the kit does three things: converts USB-C to the half-duplex
TTL bus the servos speak, distributes 12 V down the chain, and offers a
5 V buck that can power a Raspberry Pi over UART. It does no kinematics,
no trajectory planning, and no coordination.

The control loops live **inside each servo**. Every STS3215 has its own
MCU, magnetic encoder and PID loop. The host sends target positions at
100 Hz; each servo closes its own loop.

What that means in practice: several safety features you might expect to
build are already there, in servo firmware.

| | register |
|---|---|
| torque cap | 48 `Torque Limit` — the driver sets this |
| overload shutdown | 34 `Protection Torque`, 36 `Overload Torque` |
| protection delay | 35 `Protection Time` |
| thermal cutout | 13 `Max Temp Limit` |
| go limp | 40 `Torque Enable` = 0 — a software e-stop over the existing bus |

What the board does *not* give you: any analog input, any e-stop input,
any spare GPIO. So it cannot read a sensor or stop the arm on its own.

**The physical e-stop is a normally-closed switch in series with the 12 V
supply.** It cuts power through copper with no software in the path,
which is better than any microcontroller and costs a few dollars.

Contact detection needs no extra hardware either: `Present_Load` (reg 60)
rises sharply when the paddle meets a hand, and the driver already reads
those bytes in the same sync transaction as position. See
`ServoLoadContactSensor`.

## STS3215 control table

4096 counts/revolution, 0.088°/count.

| addr | register | bytes | notes |
|---|---|---|---|
| 33 | Mode | 1 | 0=position, 1=speed, 2=PWM, 3=step |
| 40 | Torque Enable | 1 | 0=limp, 1=holding |
| 41 | Goal Acceleration | 1 | 0=instant, 254=smooth |
| 42 | Goal Position | 2 | 0–4095 |
| 46 | Goal Speed | 2 | 0=maximum |
| 48 | Torque Limit | 2 | 0–1000 |
| 55 | Lock | 1 | EEPROM write protect |
| 56 | Present Position | 2 | read |
| 58 | Present Speed | 2 | **sign-magnitude**, bit 15 is direction |
| 60 | Present Load | 2 | magnitude + direction |
| 62 | Present Voltage | 1 | ×0.1 V |
| 63 | Present Temperature | 1 | °C |

`Present Speed` is sign-magnitude, not two's complement. Reading it as
signed gives nonsense at negative velocities.

## Why not LeRobot for the control loop

`tlod.arm.feetech` uses the SDK directly. The loop's problem is latency,
and `GroupSyncRead`/`GroupSyncWrite` is one bus transaction per tick;
`lerobot` also pulls torch, which nothing in the control path needs.

`Calibration.from_lerobot()` reads files written by `lerobot-calibrate`,
so the standard homing and range tooling still works.

This matters for the install as well as the loop. The arm needs
`pip install pyserial feetech-servo-sdk` and nothing else —
`feetech-servo-sdk` is the `scservo_sdk` that `tlod.arm.feetech` imports.
The project's `.[robot]` extra is `lerobot[feetech]`, which pulls torch
and does not resolve on Python 3.13; install it only on a machine where
you are running lerobot's calibration commands, which need not be the
board wired to the arm. Point `arm.calibration` at the JSON afterwards.

## Camera

Fixed mount, angled down over the table. Steeper is better — error from a
wrong assumed hand height scales with the tangent of the viewing angle.

The one in use is an **Arducam B0589**, a wide-angle USB module.
Advertised as 145° diagonal / 120° horizontal — but the mode USB
bandwidth actually allows here is 640x480, and the **measured** horizontal
field of view in that mode is **74°**. That is not a small discrepancy and
it is not a defect: the advertised figure belongs to the sensor's full
frame, and the mode you can afford is a crop of it.

So `camera.hfov_deg` should hold the number you measured, not the number
on the box. It only seeds the intrinsics solve and the no-calibration
approximation, but seeding it 70° wrong is a bad start. `tlod calibrate
intrinsics` prints the field of view it recovered next to the configured
one; believe the recovered one. `configs/opi.yaml` carries 74.

The lens is wide enough that the default pinhole distortion model cannot
fit it — use `tlod calibrate intrinsics --fisheye`, which is an
equidistant model. That fit came out at 0.165 px RMS.

Mounted, the extrinsics put this camera 445 mm forward, 35 mm right and
451 mm above the arm's base.

The capture layer is source-agnostic, so a global-shutter module can be
swapped in if motion blur turns out to matter. `/dev/video*` indices move
between boots; see [headless.md](headless.md).

## Bring-up

See [WALKTHROUGH.md](WALKTHROUGH.md) section 3. Do not skip
`tlod first-light` — sign conventions cannot be checked in simulation.
