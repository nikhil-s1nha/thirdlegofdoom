"""Look at what the intrinsics actually claim, live, over HTTP.

An RMS number says a calibration is bad without saying why, and on a
headless board there is nothing to look at. This shows the two things
that separate a poor shoot from a lens the model cannot describe:

  left    the raw frame
  right   the same frame undistorted by the calibration

If the calibration is good, lines that are straight in the room -- a door
frame, a table edge, the skirting -- are straight on the right, all the
way into the corners. If they still bow, the distortion model has not
captured the lens, and reshooting the same way will not fix it: a very
wide lens needs more coefficients than the default five.

When the board is in frame it also solves for the board's pose and draws
where each corner *should* land, with a line to where it actually did.
Those lines are the reprojection error the RMS averages, per corner, so
you can see whether the error is spread evenly (honest noise) or piled
into the frame corners (an unmodelled lens).

    python3 scripts/calib_view.py 8x5
    python3 scripts/calib_view.py 8x5 0.027102 5
"""

import sys

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "src")
from tlod.vision.calibration import Intrinsics  # noqa: E402

PORT = 8080
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

pattern = tuple(int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "8x5").split("x"))
square = float(sys.argv[2]) if len(sys.argv) > 2 else 0.025
index = int(sys.argv[3]) if len(sys.argv) > 3 else 5

intr = Intrinsics.load("calib/intrinsics.npz")
w, h = intr.resolution
print(f"  intrinsics rms {intr.rms:.3f} px, {w}x{h}")
print(f"  board {pattern[0]}x{pattern[1]} inner corners, {square * 1000:.1f} mm squares")
print(f"  open http://<this-board's-ip>:{PORT}")

objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square

cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
if not cap.isOpened():
    raise SystemExit(f"camera {index} did not open. Is another process holding it?")

BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(image, text, y, colour):
    cv2.putText(image, text, (10, y), FONT, 0.6, colour, 2)


def draw_residuals(image, grey):
    """Where each corner should be, versus where it is. Returns mean error."""
    found, corners = cv2.findChessboardCorners(grey, pattern, FLAGS)
    if not found:
        return None
    corners = cv2.cornerSubPix(
        grey, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    ok, rvec, tvec = cv2.solvePnP(objp, corners, intr.K, intr.dist)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(objp, rvec, tvec, intr.K, intr.dist)
    errors = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
    for (mx, my), (px, py) in zip(corners.reshape(-1, 2), projected.reshape(-1, 2)):
        # Exaggerated 10x, because a two-pixel error is invisible at 640x480
        # and invisible is exactly what we are trying to stop it being.
        end = (int(px + (mx - px) * 10), int(py + (my - py) * 10))
        cv2.line(image, (int(px), int(py)), end, (0, 0, 255), 1)
        cv2.circle(image, (int(px), int(py)), 2, (0, 220, 0), -1)
    return float(errors.mean())


def annotate():
    ok, raw = cap.read()
    if not ok:
        return None
    grey = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    fixed = cv2.undistort(raw, intr.K, intr.dist)

    shown = raw.copy()
    mean_err = draw_residuals(shown, grey)
    label(shown, "RAW  + reprojection error (x10)", 24, (255, 255, 255))
    if mean_err is None:
        label(shown, "no board in frame", 48, (0, 165, 255))
    else:
        colour = (0, 220, 0) if mean_err < 1.0 else (0, 0, 255)
        label(shown, "this view: %.2f px mean" % mean_err, 48, colour)
    label(shown, "sharpness %.0f" % cv2.Laplacian(grey, cv2.CV_64F).var(), 72,
          (0, 200, 255))

    label(fixed, "UNDISTORTED  (straight lines must be straight)", 24, (255, 255, 255))
    label(fixed, "calibration rms %.2f px" % intr.rms, 48,
          (0, 220, 0) if intr.rms < 1.0 else (0, 0, 255))
    return np.hstack([shown, fixed])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                pair = annotate()
                if pair is None:
                    continue
                _, jpeg = cv2.imencode(".jpg", pair)
                self.wfile.write(BOUNDARY)
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


try:
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
except KeyboardInterrupt:
    cap.release()
