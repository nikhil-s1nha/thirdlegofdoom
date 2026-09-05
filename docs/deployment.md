# Deployment: Orange Pi 5 + Pico

Target is a robot you switch on with no laptop attached. The Orange Pi 5
(RK3588S, 6 TOPS NPU) runs vision, kinematics and the servo bus over USB.
The Pico does the two things a computer running Python is bad at: cutting
power instantly, and timestamping an impact.

Nothing here has run on a board.

## Division of labour

| | Orange Pi 5 | Pico |
|---|---|---|
| hand + object detection | yes, NPU | no — ~1000× short |
| kinematics, control loop | yes | no |
| servo bus | yes | — |
| hardware e-stop | can hang | **yes, the real one** |
| impact detection | occluded, 33 ms granularity | **yes, microseconds** |
| LEDs, buzzer, display | possible | yes, off the critical path |

Software e-stop is a convenience; it stops working in exactly the case
you need it — a deadlocked loop, a crashed process, a pulled cable. The
Pico cuts servo power in its interrupt handler, before it reports.

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

## Pico

Flash MicroPython, copy `firmware/pico_sidecar.py` as `main.py`.

```bash
mpremote connect /dev/ttyACM1 fs cp firmware/pico_sidecar.py :main.py
mpremote connect /dev/ttyACM1 reset
```

| pin | to |
|---|---|
| GP26 / ADC0 | piezo disc, 1 MΩ bleed to GND, clamp diodes to 3V3 and GND |
| GP15 | e-stop button to GND (internal pull-up) |
| GP14 | servo power relay / MOSFET gate, active high |
| GP16/17/18 | LEDs: ready, hit, e-stop |
| GP19 | passive buzzer |

The clamp diodes are not optional. A piezo struck sharply outputs tens of
volts and will destroy the ADC input.

`HIT_THRESHOLD` must be set by observation: watch the raw stream with the
pad tapped and with the table knocked nearby. Too low and footsteps
score; too high and a glancing slap does not. `HIT_DEBOUNCE_US` covers
the ringing.

## Order

1. `tlod ports`
2. motor IDs, one at a time
3. `lerobot-calibrate`
4. `tlod first-light` — where an inverted sign shows up harmlessly
5. `tlod move 0.22 0 0.12` — check Cartesian accuracy
6. `tlod bench all` — replace estimates with measurements, retune `MockArm`
7. mount camera, `tlod calibrate intrinsics` then `extrinsics`
8. verify with `tlod touch --view` — drawn skeleton must land on the real arm
9. Pico: confirm `READY`, test the e-stop **before any game runs with torque on**
10. tune `HIT_THRESHOLD`
11. `tlod play --difficulty easy`, staying out of reach

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

`Restart=on-failure` is not a substitute for the hardware e-stop. A
process that restarts cleanly every five seconds while swinging at
someone is still swinging at someone.
