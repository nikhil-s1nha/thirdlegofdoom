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

## The third leg: an Arduino on its own USB port

Separate from the arm entirely. An Arduino, USB to the Orange Pi, two
hobby servos on pins 7 and 8, running a fixed sketch. `tlod.leg` drives
it; `tlod leg` is the command.

| pin | servo | what it does | positions |
|---|---|---|---|
| 7 | `servos[0]` | jaw | 90 open, 145 shut |
| 8 | `servos[1]` | paddle | 120 up (home), 40 down |

Four commands in, one line back, plus `<3` every 500 ms unasked:

| command | does | replies |
|---|---|---|
| `open` | jaw 90, then **after 200 ms** paddle 40 | `OPEN` |
| `close` | jaw 145 | `CLOSE` |
| `home` | paddle 120 | *an empty line* |
| `slap` | paddle 40 | `s` |

9600 baud, newline-terminated, `Serial.readStringUntil('\n')` on the far
side. Send commands one at a time and wait for the reply.

**These are hobby servos, not bus servos.** No encoder, no feedback, no
`read()`. The board cannot say where the paddle is, only that it took the
word — so unlike `Present_Load` on the STS3215, there is nothing here to
detect contact with, and nothing to verify a move happened. The reply's
arrival time is the only timestamp available, and `Ack.stamp` is it.

**`slap` does not come back.** It drives the paddle to 40 and leaves it
there; a second `slap` moves nothing until something has sent `home`.
`LegLink.strike()` is the whole gesture, and its `dwell` is paddle travel
time — which, being part of the strike, wants recalibrating whenever the
strike changes. See [hit-detection.md](hit-detection.md).

**`home` acknowledges with an empty line.** `Serial.println("")`. Nothing
tells it apart from a blank line arriving for any other reason, so
replies can only be matched positionally. If you edit the sketch, make it
print `HOME`.

**`open` blocks the sketch for 200 ms.** The `delay(200)` is inside the
command handler, so `loop()` neither beats nor reads during it. An `open`
reply is never faster than 200 ms, a heartbeat can be that late, and 200
ms of incoming bytes at 9600 baud is about 192 — three times the
Arduino's 64-byte receive buffer. Waiting for each reply is what keeps
that buffer from overflowing.

**Opening the port resets the board.** DTR is asserted on open and most
Arduinos reset on it, so the first second or two after connecting belongs
to the bootloader and anything sent then is lost. `LegLink.connect()`
waits for the first heartbeat rather than guessing at a sleep — the beat
is proof that `setup()` has run.

**It is a second `/dev/ttyACM*`, and the numbering is not stable.** Two
USB devices now enumerate in whatever order they came up, so the arm and
the leg can swap between boots. `tlod leg` probes for the heartbeat when
no port is configured, the same way `tlod ports --probe` asks each port
whether six servos answer; the probe writes nothing, so pointing it at
the servo bus by mistake is harmless. Set `leg.port` once it is known.

```bash
tlod leg monitor          # listen only: is it there, is the sketch running
tlod leg strike           # slap, dwell, home
tlod leg open --repeat 3
```

## The eyes: NeoPixel rings on a XIAO SAMD21

A fourth USB device, independent of the arm, the paddle and the camera.
Two 16-pixel NeoPixel rings daisy-chained on **D10** (ring 1 `DOUT` ->
ring 2 `DIN`), so 32 pixels on one data line: `[0..15]` is the left eye,
`[16..31]` the right. Animated as three moods. `tlod.eyes` drives it,
`tlod eyes` is the command, and `scripts/eyes_link.py` streams the
tracked hand at it.

| in | means |
|---|---|
| `h` / `c` / `a` | set the mood: happy, concentrated, angry |
| `x,y,z` | a point; the **board** takes `sqrt(x²+y²+z²)` and picks the mood |

| out | when |
|---|---|
| `Distance: 24.15` | every point it parsed |
| `Emotion: ANGRY` | **only when the mood actually changed** |
| `Bad input, expected: x,y,z` | anything it could not parse |

**It never speaks first.** `setup()` prints nothing and there is no
heartbeat, so unlike the paddle board there is no passive way to know it
is alive — silence proves nothing. `?` is the probe: it parses as neither
a mood key nor a point, so `Bad input` comes back guaranteed, and it
cannot disturb what is on the LEDs. That is `EyesLink.ping()`, and it is
what `tlod eyes` uses to find the board.

**The mood line is edge-triggered.** `setEmotion()` returns early when
the mood is unchanged. So silence after `h` means *either* "already
happy" *or* "not listening", and nothing tells them apart. This is why
`tlod eyes selftest` cycles `h -> c -> a -> h`: from any starting mood
that is at least three genuine transitions, and every one must be
announced. Anything less is not evidence the pixels are being driven.

**Something has to read the replies.** The board answers every line it is
sent. A host that only writes leaves those bytes filling the kernel
receive buffer; once it is full the SAMD21's USB-CDC writes have nowhere
to go and the animation loop stalls behind them — the eyes freeze while
the sender happily reports thousands delivered. `EyesLink` runs a reader
thread, which is as much about keeping the board alive as about reading.

**`sscanf("%f")` may not work.** newlib-nano omits scanf float support
unless the core links `-u _scanf_float`, and on SAMD21 that is not a
given. When it is missing, every well-formed point comes back `Bad input`
and the mood never moves. `tlod eyes selftest` names this specifically,
because from the outside it looks identical to a wiring fault.

**Units are not obvious, and the defaults do not fit this rig.** The
sketch's thresholds are 20 near / 60 far, in whatever units arrive. This
project works in metres and the arm's reach is ~0.08–0.40 m, so raw
metres are always "nearer than 20" and the eyes are permanently angry.
`eyes.scale` defaults to 100 (metres to centimetres), which puts the
near boundary inside the workspace — but 60 cm is past the end of the
arm's reach, so **HAPPY is not reachable from a point** until the sketch's
constants are retuned. `tlod eyes point X Y Z` prints the mood a given
position should produce, which is the quickest way to choose new ones.

**Do not open this port at 1200 baud.** On SAMD21 that is the bootloader
knock, not a baud rate — it disconnects the running sketch.

```bash
tlod eyes selftest            # the one that answers "are the LEDs working"
tlod eyes point 0.25 0 0.10   # send a position, see what mood it should give
tlod eyes angry               # set a mood directly
python scripts/eyes_link.py --port /dev/ttyACM2   # stream the tracked hand
```

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
