"""Check what the gripper-marker detector is actually locking onto.

`calibrate extrinsics` finds the *largest* blob of the marker colour and
believes it is the gripper. It has no way to tell a marker from a mug, a
foam roller, or a patch of sky, and it does not fail when it picks the
wrong one -- it produces a confident calibration of the camera against
that object instead. The arm has already driven twelve poses by then.

So look first. This draws every blob of the chosen colour, marks the one
that would win, and moves nothing.

    python3 scripts/marker_view.py --marker red
    python3 scripts/marker_view.py -c configs/opi.yaml --marker red

Then hold the marker where the gripper will be. If the crosshair jumps to
something else in the room, choose a different colour or remove the
distractor -- do not calibrate.
"""

import argparse
import sys

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "src")
from tlod.vision.calibrate_flow import MARKER_BANDS, _bands, find_marker  # noqa: E402

MIN_AREA = 45                        # matches find_marker's own threshold

# Positional args and flags both, because the flags are what everything
# else in this project takes and the positionals are what this script has
# always taken. `marker_view.py red --camera 1` reading the colour as a
# camera index and dying on int("red") is a worse first experience than
# either spelling deserves.
def _camera(value):
    """An index or a device path, whichever was typed."""
    return int(value) if str(value).lstrip("-").isdigit() else value


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("marker_pos", nargs="?", default=None,
                    help=f"marker colour: {', '.join(MARKER_BANDS)}")
parser.add_argument("camera_pos", nargs="?", type=_camera, default=None,
                    help="v4l2 index or device path (positional form)")
parser.add_argument("--marker", default=None, choices=sorted(MARKER_BANDS))
parser.add_argument("--camera", type=_camera, default=None,
                    help="v4l2 index or /dev/v4l/by-id/... path; defaults to "
                         "camera.index from the config")
parser.add_argument("-c", "--config", default=None, help="YAML config path")
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()

colour = args.marker or args.marker_pos or "green"
# The config, not a number baked in here. This script exists to be run
# immediately before `calibrate extrinsics`, against the same camera, and
# a default of 5 meant it opened a different device -- or nothing -- while
# the config had a by-id path pinned precisely so nobody had to think
# about indices again.
index = args.camera if args.camera is not None else args.camera_pos
if index is None:
    from tlod.config import Config  # noqa: E402

    index = Config.load(args.config).camera.index
PORT = args.port
if colour not in MARKER_BANDS:
    raise SystemExit(f"marker must be one of {', '.join(MARKER_BANDS)}, not {colour!r}")
band = MARKER_BANDS[colour]

cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"camera {index} did not open. Is another process holding it?")

print(f"  looking for {colour} blobs, {len(_bands(band))} HSV band(s)")
print(f"  camera {index}, open http://<this board>:{PORT}")
if colour == "red":
    # The one colour that regularly loses to the room. Two bands and a
    # raised saturation floor keep skin out of it most of the time, but
    # "most of the time" is not a thing to discover with the arm moving.
    print("  red fights with skin, wood and most tabletops -- put a hand in")
    print("  frame and watch whether the crosshair stays on the marker")

PAGE = (b"<!doctype html><title>marker</title>"
        b"<body style='margin:0;background:#111'>"
        b"<img src='/stream' style='width:100%'></body>")


def annotate(image):
    # Detect before drawing anything. Annotating first and detecting after
    # overwrites the blob's own edge pixels with the outline colour, which
    # shrinks it below the area threshold: the overlay then reports a
    # candidate and no detection at the same time.
    found = find_marker(image, band)
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in _bands(band):
        part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Every candidate, so a distractor is visible even when it loses.
    big = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    cv2.drawContours(image, big, -1, (0, 165, 255), 2)

    if found is None:
        biggest = max((cv2.contourArea(c) for c in contours), default=0.0)
        cv2.putText(image, f"no {colour} blob big enough "
                    f"(largest {biggest:.0f} px, need {MIN_AREA})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        u, v = int(found[0]), int(found[1])
        cv2.drawMarker(image, (u, v), (0, 220, 0), cv2.MARKER_CROSS, 40, 2)
        cv2.putText(image, f"WOULD USE THIS  ({u}, {v})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    cv2.putText(image, f"{len(big)} candidate blob(s); largest wins", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 165, 255) if len(big) > 1 else (200, 200, 200), 2)
    return image


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if not self.path.startswith("/stream"):
            self.send_response(200 if self.path in ("/", "/index.html") else 404)
            if self.path in ("/", "/index.html"):
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            else:
                self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    continue
                _, jpeg = cv2.imencode(".jpg", annotate(image))
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


try:
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
except KeyboardInterrupt:
    cap.release()
