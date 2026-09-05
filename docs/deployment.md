# Deployment: Orange Pi 5 + Pico sidecar

The target is a robot you switch on, with no laptop attached. The Orange
Pi 5 (RK3588S, 6 TOPS NPU) runs vision, kinematics and the servo bus over
USB. The Pico handles the two things a general-purpose computer running
Python is bad at: cutting power instantly, and timestamping an impact.

Everything here is **written but unverified** — no board has run it. The
bring-up order below is designed so that each step fails cheaply.

## Division of labour

| | Orange Pi 5 | Pico |
|---|---|---|
| hand + object detection | ✅ NPU | ✗ ~1000× short |
| kinematics, IK, control loop | ✅ | ✗ |
| servo bus (USB serial) | ✅ | possible, not useful |
| **hardware e-stop** | ✗ can hang | ✅ **the real one** |
| **impact detection** | ✗ occluded, 33 ms granularity | ✅ microseconds |
| LEDs, buzzer, score display | possible | ✅ off the critical path |

The e-stop split is the important one. A software e-stop is a
convenience; it stops working in exactly the situation you need it — a
deadlocked control loop, a crashed process, a yanked USB cable. The Pico
cuts servo power in its interrupt handler, before it tells anyone, so it
works when nothing else does.

## Orange Pi 5 setup

Ubuntu or Debian arm64. Python 3.12.

```bash
sudo apt install -y python3-venv python3-dev libgl1 libglib2.0-0
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[hands,robot]"
```

**mediapipe pins differently here.** Upstream publishes aarch64 Linux
wheels only up to 0.10.18; 0.10.2x and 1.0 are x86_64 and macOS only. The
pin in `pyproject.toml` handles this automatically. 0.10.18 has the same
Tasks API, so no code changes.

Serial port permissions for the servo bus:

```bash
sudo usermod -aG dialout $USER      # log out and back in
```

### NPU

`rknn-toolkit-lite2` ships from Rockchip, not PyPI. Model *conversion*
needs `rknn-toolkit2` on an **x86_64** host; only the lite runtime goes on
the board.

```bash
# on the board
pip install ./rknn_toolkit_lite2-*-cp312-cp312-linux_aarch64.whl
python -c "from rknnlite.api import RKNNLite; print('npu runtime ok')"
```

Expect roughly YOLOv5n at ~58 fps, YOLOv5s at ~37 fps on this NPU. Note
that a box detector gives no finger landmarks — fine for hand-slap, not
for gesture games. See the caveat at the top of `tlod/vision/rknn.py`.

**Re-measure everything after porting.** The A76 cores are slower than an
M-series laptop and the NPU path has its own latency profile. Every
number in the README was measured on macOS.

```bash
tlod bench all
tlod sim --duration 20        # check jitter and overruns on this hardware
```

## Pico sidecar

Flash MicroPython, then copy `firmware/pico_sidecar.py` as `main.py`.

```bash
mpremote connect /dev/ttyACM1 fs cp firmware/pico_sidecar.py :main.py
mpremote connect /dev/ttyACM1 reset
```

### Wiring

| pin | to |
|---|---|
| GP26 / ADC0 | piezo disc, 1 MΩ bleed to GND, clamp diodes to 3V3 and GND |
| GP15 | e-stop button to GND (internal pull-up) |
| GP14 | servo power relay / MOSFET gate, active high |
| GP16/17/18 | status LEDs: ready, hit, e-stop |
| GP19 | passive buzzer |

**The clamp diodes are not optional.** A piezo disc struck sharply
outputs tens of volts. Without clamping to the rails it will destroy the
ADC input, and quite possibly the board.

### Calibrating the piezo

`HIT_THRESHOLD` is a fraction of full scale and must be set by
observation, not taste. Watch the raw stream with the pad tapped and with
the table knocked nearby. Too low and footsteps score points; too high
and a glancing slap does not register. `HIT_DEBOUNCE_US` covers the
ringing — a piezo oscillates for milliseconds after contact and would
otherwise report one slap as a dozen.

## Bring-up order

Each step is cheap to fail. Do not skip ahead; sign conventions and
impact thresholds cannot be checked in simulation.

1. `tlod ports` — find the servo bus adapter
2. Assign motor ids one at a time (`lerobot-setup-motors`, or ours)
3. Calibrate (`lerobot-calibrate`, or record centre/sign directly)
4. **`tlod first-light`** — one joint at a time, ±0.2 rad, slowly. This
   is where an inverted direction sign shows up harmlessly.
5. `tlod move 0.22 0 0.12` — verify Cartesian accuracy
6. `tlod bench all` — replace estimated latencies with measured ones, and
   retune `MockArm` to match so tier A stays trustworthy
7. Mount the camera; `tlod calibrate extrinsics` using the arm as its own
   target. Verify with `tlod touch --view`: the drawn skeleton must land
   on the real arm, and a consistent offset on every object means the
   extrinsics are wrong.
8. Pico: confirm `READY`, then test the e-stop button **before any game
   runs with torque enabled**
9. Piezo: tune `HIT_THRESHOLD` by tapping the pad
10. `tlod play --difficulty easy` — and stay out of reach for the first
    run

## Before it plays against a person

Non-negotiable, in addition to the software guards:

- **soft end effector** — a foam paddle, never the gripper. The stopping
  distance matters more than the speed: 1 m/s into something compliant is
  ~15 N over 10 ms; into something rigid it is ~150 N over 1 ms.
- **padded target pad** — give on both sides. A hand resting on a bare
  table has nowhere to go.
- **hardware e-stop tested** — press it mid-strike and confirm the arm
  goes limp
- **`strike` torque limit lowered** (`StrikeLimits.torque_limit`), so the
  servo yields on unexpected contact rather than pushing through
- **strike distance capped** (`StrikeLimits.max_drop`), which improves
  speed and safety together

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

`Restart=on-failure` restarts a crashed process, which parks the arm on
the way out and re-homes on the way in. It does **not** substitute for
the hardware e-stop: a process that restarts cleanly every five seconds
while swinging at someone is still swinging at someone.
