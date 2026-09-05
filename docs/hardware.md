# Hardware notes

## Arm — SO-ARM101 Pro (Seeed)

Six Feetech **STS3215** bus servos on one TTL serial chain, 1 Mbaud
default. Follower arm uses ST-3215-C001 (1:345) throughout. Pro kit
follower runs at **12 V** (30 kg·cm); the leader at 5 V.

Motor ids run 1–6 along the chain, matching `tlod.types.JOINT_NAMES`:

| id | joint | axis | limit (rad) |
|---|---|---|---|
| 1 | `shoulder_pan` | base yaw | ±1.920 |
| 2 | `shoulder_lift` | pitch | ±1.745 |
| 3 | `elbow_flex` | pitch | ±1.690 |
| 4 | `wrist_flex` | pitch | ±1.658 |
| 5 | `wrist_roll` | tool roll | −2.744 … +2.841 |
| 6 | `gripper` | jaw | −0.175 … +1.745 |

Joints 2, 3 and 4 are **parallel pitch axes**. With base yaw and tool
roll that gives five arm joints — position plus tool pitch and roll, and
no independent tool yaw. This is why the arm is 5-DOF despite six motors.

Limits are transcribed from `assets/so101_new_calib.urdf`
(TheRobotStudio/SO-ARM100, `Simulation/SO101`). Regenerate with
`scripts/extract_urdf.py` if the upstream model changes.

## STS3215 control table

Verified against the datasheet. 4096 counts/revolution, 0.088°/count.

| addr | register | bytes | notes |
|---|---|---|---|
| 33 | Mode | 1 | 0=position, 1=speed, 2=PWM, 3=step |
| 40 | Torque Enable | 1 | 0=limp, 1=holding |
| 41 | Goal Acceleration | 1 | 0=instant (harsh), 254=smooth |
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

`tlod.arm.feetech` talks to the SDK directly. Two reasons: the loop's
whole problem is latency, and direct `GroupSyncRead`/`GroupSyncWrite` is
one bus transaction per tick with nothing in between; and `lerobot`
depends on torch, which nothing in the control path needs and which is
not free to install on a Jetson or Orange Pi.

Interoperability is kept where it helps: `Calibration.from_lerobot()`
reads files written by `lerobot-calibrate`, so the standard homing and
range tooling still works.

## Bring-up order

Do not skip steps. Sign conventions cannot be checked in simulation.

1. `tlod ports` — find the bus adapter
2. Assign motor ids one at a time (`lerobot-setup-motors`, or ours)
3. `lerobot-calibrate`, or record centre/sign yourself
4. **`tlod first-light`** — moves one joint at a time, ±0.2 rad, slowly.
   This is where an inverted direction sign shows up harmlessly.
5. `tlod bench all` — replace estimated latencies with measured ones
6. Mount the camera, then `tlod calibrate extrinsics` using the arm as
   its own calibration target

## Camera

Fixed external mount, angled down over the table. Steeper is better: the
error from a wrong assumed hand height scales with `tan(viewing angle)`.

Chosen: Logitech C922 (720p60). The capture layer is source-agnostic, so
a global-shutter module can be swapped in if motion blur on a fast slap
turns out to matter — that is a tier-B question, answerable with a
laptop webcam before committing.

## Compute

Development on macOS arm64. Deployment target is an **Orange Pi 5**
(RK3588S, 6 TOPS NPU), which can run vision, IK and the servo bus over
USB as a self-contained robot. Detection models convert with
`rknn-toolkit2`.

A Raspberry Pi Pico cannot do vision — it is ~1000× short of hand
detection. Its job here is the one vision is bad at: **piezo impact
detection for scoring** (microsecond hit timestamps versus vision
guessing at 20 ms granularity) and a **hardware e-stop** that cuts servo
torque regardless of whether Python is responsive.
