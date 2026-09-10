"""Check what the gripper-marker detector is actually locking onto.

`calibrate extrinsics` finds the *largest* blob of the marker colour and
believes it is the gripper. It has no way to tell a marker from a mug, a
foam roller, or a patch of sky, and it does not fail when it picks the
wrong one -- it produces a confident calibration of the camera against
that object instead. The arm has already driven twelve poses by then.

So look first. This draws every blob of the chosen colour, marks the one
that would win, and moves nothing.

    python3 scripts/marker_view.py green
    python3 scripts/marker_view.py blue 5

Then hold the marker where the gripper will be. If the crosshair jumps to
something else in the room, choose a different colour or remove the
distractor -- do not calibrate.
"""

import sys

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "src")
from tlod.vision.calibrate_flow import MARKER_BANDS, find_marker  # noqa: E402

PORT = 8080
MIN_AREA = 120                       # matches find_marker's own threshold

colour = sys.argv[1] if len(sys.argv) > 1 else "green"
index = int(sys.argv[2]) if len(sys.argv) > 2 else 5
if colour not in MARKER_BANDS:
    raise SystemExit(f"colour must be one of {', '.join(MARKER_BANDS)}")
band = MARKER_BANDS[colour]

cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"camera {index} did not open. Is another process holding it?")

print(f"  looking for {colour} blobs, HSV {band[0]} to {band[1]}")
print(f"  open http://<this board>:{PORT}")

PAGE = (b"<!doctype html><title>marker</title>"
        b"<body style='margin:0;background:#111'>"
        b"<img src='/stream' style='width:100%'></body>")


def annotate(image):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(band[0]), np.array(band[1]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Every candidate, so a distractor is visible even when it loses.
    big = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    cv2.drawContours(image, big, -1, (0, 165, 255), 2)

    found = find_marker(image, band)
    if found is None:
        cv2.putText(image, f"no {colour} blob big enough", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
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
