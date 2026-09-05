# Deployment: Orange Pi 5

Target is a robot you switch on with no laptop attached. The Orange Pi 5
(RK3588S, 6 TOPS NPU) runs everything: vision, kinematics, and the servo
bus over USB.

Nothing here has run on a board.

## What you actually need

| | |
|---|---|
| SO-ARM101 + its bus adapter | in the kit |
| 12 V 5 A supply | in the kit |
| **inline switch on the 12 V line** | the physical e-stop. A few dollars. |
| camera | fixed mount, angled down |
| Orange Pi 5 | the brain |

That is the whole bill of materials. There is no sidecar
microcontroller: the servos carry torque and overload limits themselves,
a mechanical switch is a better e-stop than any chip, and contact is
detected from servo load over the bus you already have.

The adapter's 5 V buck is specified for a Raspberry Pi. An Orange Pi 5
can draw up to 4 A, so give it its own supply rather than backfeeding.

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

1. `tlod ports`
2. motor IDs, one at a time
3. `lerobot-calibrate`
4. `tlod first-light` — where an inverted sign shows up harmlessly
5. `tlod move 0.22 0 0.12` — check Cartesian accuracy
6. `tlod bench all` — replace estimates with measurements, retune `MockArm`
7. mount camera, `tlod calibrate intrinsics` then `extrinsics`
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
