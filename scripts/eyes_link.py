"""Publish the tracked hand's xyz position to the Robot Eyes Arduino.

Runs on the Orange Pi, alongside (or instead of) `tlod vision-serve`. It
reuses the same camera/detector/locator pipeline to find the hand, then
writes plain ASCII `x,y,z\\n` lines over its own USB cable to the Seeed
XIAO SAMD21 running `xiao_neopixel_breathe_rainbow.ino` (see the Robot
Eyes handoff notes). That sketch reads at 9600 baud over USB (the SAMD21's
native USB-CDC serial, not a UART pin pair), computes
`distance = sqrt(x^2+y^2+z^2)`, and switches emotion against its
NEAR_THRESHOLD/FAR_THRESHOLD constants -- so the wire format is just a
plain CSV line per update, no framing of any kind, because that is what
the eyes sketch parses.

This link is unrelated to `tlod.net.uart_link`, which is the learning/
testing link between the Orange Pi and the Raspberry Pi control board
over real UART pins (see docs/deployment.md) -- a different transport, a
different pair of boards, and a different (framed) wire format. The
Arduino here is a third, independent board on its own USB port; this
script is deliberately its own small program rather than a `tlod`
subcommand, since it is a side accessory unrelated to the arm control
loop and has no business sharing failure modes with `vision-serve`.

The eyes sketch's thresholds (20 near / 60 far) were placeholders picked
before the real xyz source was known. `--scale` converts this rig's
metres into roughly that same range (the arm's own reach envelope is
~0.08-0.4 m, i.e. ~8-40 cm) so the defaults are a reasonable starting
point; tune NEAR_THRESHOLD/FAR_THRESHOLD in the sketch itself, or
`--scale` here, once you've watched it react.

Usage:
    python scripts/eyes_link.py --port /dev/ttyACM0
    python scripts/eyes_link.py --port /dev/ttyACM0 --sim   # no camera/hand needed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from tlod.cli import build_camera, build_detector, build_projector
from tlod.config import Config


def list_serial_ports() -> list[str]:
    from serial.tools import list_ports

    return sorted(p.device for p in list_ports.comports())


def nearest_hand(observations):
    """Pick the hand closest to the base, since the eyes react to one
    distance and multiple hands in frame would otherwise fight over it."""
    return min(observations, key=lambda o: float(np.linalg.norm(o.position)))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-c", "--config", default=None, help="YAML config path")
    p.add_argument("--port", default="", help="Arduino's serial device, e.g. /dev/ttyACM0")
    p.add_argument("--baud", type=int, default=9600,
                   help="must match the eyes sketch's Serial.begin() (default 9600)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--scale", type=float, default=100.0,
                   help="multiplies metres before sending -- see module docstring")
    p.add_argument("--rate", type=float, default=20.0,
                   help="max updates/sec sent to the Arduino (0 = as fast as detected)")
    p.add_argument("--precision", type=int, default=1, help="decimal places sent")
    p.add_argument("--sim", action="store_true",
                   help="synthetic camera + scripted hand, to test the serial link alone")
    p.add_argument("--preview", type=int, default=0, metavar="PORT",
                   help="serve the camera view over HTTP so you can watch hand tracking "
                        "from another machine, e.g. --preview 8081, then open "
                        "http://<this board>:8081/ in a browser on your laptop")
    args = p.parse_args()

    if not args.port:
        found = list_serial_ports()
        print("  --port is required. Serial ports seen on this machine:")
        print(f"    {found or 'none found'}")
        print("  (plug in the eyes Arduino and pick its device, e.g. /dev/ttyACM0)")
        return 1

    cfg = Config.load(args.config)
    if args.sim:
        cfg = cfg.with_overrides(camera={"source": "mock"}, vision={"detector": "scripted"})
    else:
        cfg = cfg.with_overrides(camera={"source": "opencv", "index": args.camera},
                                 vision={"detector": "mediapipe"})

    projector = build_projector(cfg)
    scene = None
    if cfg.vision.detector == "scripted" or cfg.camera.source == "mock":
        from tlod.vision.scene import SyntheticHandScene
        scene = SyntheticHandScene(projector)

    camera = build_camera(cfg, scene=scene)
    detector = build_detector(cfg, scene)

    from tlod.vision.hands import HandLocator
    locator = HandLocator(
        projector, depth_mode=cfg.vision.depth_mode,
        hand_height=cfg.vision.hand_height, palm_width_m=cfg.vision.palm_width_m,
    )

    import serial
    eyes = serial.Serial(args.port, args.baud, timeout=0)
    print(f"  eyes link: {args.port} @ {args.baud} baud, scale x{args.scale:g}, "
          f"rate <= {args.rate:g}/s" if args.rate > 0 else
          f"  eyes link: {args.port} @ {args.baud} baud, scale x{args.scale:g}")

    preview = None
    if args.preview:
        from tlod.vision.preview import PreviewServer
        preview = PreviewServer(port=args.preview)
        preview.start()
        print(f"  preview: open http://<this board>:{args.preview}/ from any browser "
              "on the same network (or SSH-tunnel the port from your laptop)")

    min_interval = (1.0 / args.rate) if args.rate > 0 else 0.0
    last_sent = 0.0
    last_index = -1
    sent = 0

    camera.start()
    time.sleep(1.0)  # let autoexposure/autofocus settle before the first read
    try:
        while True:
            frame = camera.read()
            if frame is None or frame.index == last_index:
                time.sleep(0.001)
                continue
            last_index = frame.index

            hands2d = detector.detect(frame)
            observations = locator.locate_all(hands2d)

            if preview is not None:
                import cv2

                vis = frame.image.copy()
                for h in hands2d:
                    cx, cy = h.palm_center
                    cv2.circle(vis, (int(cx), int(cy)), 10, (0, 220, 90), 2)
                preview.offer(vis)

            if not observations:
                continue

            now = time.perf_counter()
            if now - last_sent < min_interval:
                continue
            last_sent = now

            hand = nearest_hand(observations)
            x, y, z = (hand.position * args.scale).tolist()
            line = f"{x:.{args.precision}f},{y:.{args.precision}f},{z:.{args.precision}f}\n"
            eyes.write(line.encode("ascii"))
            sent += 1
            print(f"\r  sent {sent}: {line.strip()}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        camera.stop()
        detector.close()
        eyes.close()
        if preview is not None:
            preview.stop()
    print(f"\n  sent {sent} position updates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
