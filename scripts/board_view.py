"""Watch the camera and see whether the chessboard is being detected.

`tlod calibrate intrinsics` auto-captures and prints nothing at all when
detection never fires, which is indistinguishable from a dead camera. On
a headless board there is no window to check against, so this serves the
annotated feed over HTTP instead: open it from a laptop on the same
network and the answer is immediate.

    python3 scripts/board_view.py            # default 9x6, camera 5
    python3 scripts/board_view.py 7x7 0      # other pattern, other camera

The pattern is *inner corners* -- the crossings where four squares meet,
not the squares themselves -- so a board of 10x7 squares is 9x6 here.
Getting that number wrong is the usual reason for silence, and trying a
few against the live feed settles it faster than counting twice.
"""

import sys

import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8080
FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)

pattern = tuple(int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "9x6").split("x"))
index = int(sys.argv[2]) if len(sys.argv) > 2 else 5

cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"camera {index} did not open. Is another process holding it?")

BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"


def annotate(image):
    """Draw the detected corners, and say plainly whether there were any."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(grey, pattern, FLAGS)
    cv2.drawChessboardCorners(image, pattern, corners, found)
    label = "FOUND %dx%d" % pattern if found else "no board"
    colour = (0, 200, 0) if found else (0, 0, 255)
    cv2.putText(image, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    return image


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    continue
                _, jpeg = cv2.imencode(".jpg", annotate(image))
                self.wfile.write(BOUNDARY)
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


print(f"  camera {index}, looking for {pattern[0]}x{pattern[1]} inner corners")
print(f"  open http://<this-board's-ip>:{PORT} from another machine")
try:
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
except KeyboardInterrupt:
    cap.release()
