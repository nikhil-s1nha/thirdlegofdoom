# Deployment: two boards

Vision on the **Orange Pi 5**, control on the **Raspberry Pi**, joined by
UDP over a wired link.

```
Orange Pi 5                       Raspberry Pi
┌────────────────┐  UDP, ~150 B   ┌────────────────┐  USB   ┌────────────┐
│ camera         │───────────────▶│ latest mailbox │───────▶│ servo board│──▶ servos
│ detect (NPU)   │  60/s          │ game + IK      │        └────────────┘
│ 3D localise    │◀───────────────│ control loop   │
└────────────────┘  clock sync    └────────────────┘
```

Nothing here has run on real boards.

## Why it splits this way

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

## Running it

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

## What you need

| | |
|---|---|
| SO-ARM101 + bus adapter | in the kit |
| 12 V 5 A supply | in the kit |
| **inline switch on the 12 V line** | the physical e-stop |
| camera | fixed mount, angled down |
| Orange Pi 5 | vision |
| Raspberry Pi | control |
| ethernet between them | wired, not WiFi |

No sidecar microcontroller: the servos carry torque and overload limits
themselves, a mechanical switch is a better e-stop than any chip, and
contact is read from `Present_Load` over the bus already in use.

The adapter's 5 V buck is specified for a Raspberry Pi, so it can power
the control board. An Orange Pi 5 can draw up to 4 A -- give it its own
supply.

## Can the Pi hold the loop?

Per tick it solves IK, runs the game state machine, clamps for safety,
reads a socket and talks to the servo board. No camera, no neural
network. Measured on a laptop: IK 0.17 ms, serial ~1-2 ms, everything
else under 0.1 ms.

Position-only IK uses an analytic Jacobian -- one forward-kinematics pass
instead of six -- so a board 20x slower than a laptop still solves in
3.5 ms, inside a 10 ms tick.

Check it on the actual board before trusting it:

```bash
tlod bench ik
```

If it does not fit, lower `runtime.control_hz`. At 50 Hz a 210 ms strike
still gets ~10 command updates, and the servos run their own internal
loops between them.

## Orange Pi 5

Ubuntu or Debian arm64, Python 3.12.

```bash
sudo apt install -y python3-venv python3-dev libgl1 libglib2.0-0
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[hands,robot]"
sudo usermod -aG dialout $USER      # then log out and back in
```

mediapipe pins differently here: upstream publishes aarch64 wheels only
up to 0.10.18. `pyproject.toml` handles it; same Tasks API, no code
change.

### NPU

`rknn-toolkit-lite2` ships from Rockchip, not PyPI. Model *conversion*
needs `rknn-toolkit2` on an x86_64 host; only the lite runtime goes on
the board.

```bash
pip install ./rknn_toolkit_lite2-*-cp312-cp312-linux_aarch64.whl
python -c "from rknnlite.api import RKNNLite; print('ok')"
```

Roughly YOLOv5n at 58 fps, YOLOv5s at 37 fps. Box detectors give no
finger landmarks — fine for hand-slap, not for gesture games.

Re-measure everything after porting; the A76 cores are slower than an
M-series laptop:

```bash
tlod bench all
tlod sim --duration 20
```


## Order

1. `tlod ports` (on the Pi -- the servo board plugs in there)
2. motor IDs, one at a time
3. `lerobot-calibrate`
4. `tlod first-light` — where an inverted sign shows up harmlessly
5. `tlod move 0.22 0 0.12` — check Cartesian accuracy
6. `tlod bench all` — replace estimates with measurements, retune `MockArm`
7. mount camera, `tlod calibrate intrinsics` then `extrinsics` (on the Orange Pi)
7b. start `tlod vision-serve`, then `tlod control` on the Pi; check the
    reported clock offset is small and stable
8. verify with `tlod touch --view` — drawn skeleton must land on the real arm
9. test the inline power switch mid-motion, **before any game runs**
10. `tlod play --difficulty easy`, staying out of reach

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
