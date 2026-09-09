"""Watch the camera and see whether the chessboard is being detected.

`tlod calibrate intrinsics` auto-captures and prints nothing at all when
detection never fires, which is indistinguishable from a dead camera. On
a headless board there is no window to check against, so this serves the
annotated feed over HTTP instead: open it from a laptop on the same
network and the answer is immediate.

    python3 scripts/board_view.py            # default 9x6, camera 5
    python3 scripts/board_view.py 7x7 0      # other pattern, other camera
    python3 scripts/board_view.py scan       # try every plausible pattern

The pattern is *inner corners* -- the crossings where four squares meet,
not the squares themselves -- so a board of 10x7 squares is 9x6 here.
Getting that number wrong is the usual reason for silence, and counting
squares off a printout is easy to do twice and still get wrong, so
`scan` asks the detector instead: it tries the common sizes against each
frame and names whichever one actually matches.
"""

import sys

import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8080
FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)

arg = sys.argv[1] if len(sys.argv) > 1 else "9x6"
scan = arg == "scan"
# Both orientations of each, because findChessboardCorners does not treat
# WxH and HxW as the same board.
CANDIDATES = [
    (9, 6), (6, 9), (7, 7), (8, 6), (6, 8), (8, 5), (5, 8),
    (7, 6), (6, 7), (7, 5), (5, 7), (9, 7), (7, 9), (6, 5), (5, 6),
    (10, 7), (7, 10), (4, 4), (5, 5), (6, 4), (4, 6), (11, 8), (8, 11),
]
pattern = (9, 6) if scan else tuple(int(v) for v in arg.split("x"))
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
    tries = CANDIDATES if scan else [pattern]
    for candidate in tries:
        found, corners = cv2.findChessboardCorners(grey, candidate, FLAGS)
        if found:
            cv2.drawChessboardCorners(image, candidate, corners, True)
            cv2.putText(image, "FOUND  --pattern %dx%d" % candidate, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
            print("  match: %dx%d" % candidate, flush=True)
            return image
    label = "scanning %d sizes..." % len(tries) if scan else "no board"
    cv2.putText(image, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)
    # Sharpness, as a number. A fixed-focus lens holding focus at room
    # distance is blurred to uselessness at arm's length, which looks
    # exactly like a wrong pattern from the far end of an SSH session.
    focus = cv2.Laplacian(grey, cv2.CV_64F).var()
    cv2.putText(image, "sharpness %.0f (want > 100)" % focus, (10, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return image



PAGE = (b"<!doctype html><title>calib</title>"
        b"<body style='margin:0;background:#111'>"
        b"<img src='/stream' style='width:100%'></body>")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        # One path streams; everything else answers immediately. A browser
        # asks for /favicon.ico too, and on a server where every path
        # blocks forever that request is enough to hang the page.
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            self.stream()
            return
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return
        self.send_response(404)
        self.end_headers()

    def stream(self):
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


print(f"  camera {index}, " + ("scanning %d patterns" % len(CANDIDATES) if scan else f"looking for {pattern[0]}x{pattern[1]} inner corners"))
print(f"  open http://<this-board's-ip>:{PORT} from another machine")
try:
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
except KeyboardInterrupt:
    cap.release()
