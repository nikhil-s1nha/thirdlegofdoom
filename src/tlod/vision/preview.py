"""MJPEG preview over HTTP, for a board with no screen.

`vision-check` answers "is it correct?" numerically, which is the part
that matters. This answers "what is it looking at?", which is the part
you want at 2am when the numbers are wrong and you cannot tell why.

Point a browser on any machine at http://<board>:8081/ and you get the
annotated camera view. Deliberately a plain multipart JPEG stream: every
browser renders it with no player, no plugin and no JavaScript, and
`curl` can save it.

It is a diagnostic, not a product. Encoding costs a few ms per frame, so
it runs at a throttled rate on its own thread and never blocks detection.
Leave it off in normal operation.
"""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

log = logging.getLogger(__name__)

PAGE = b"""<!doctype html><meta charset=utf-8><title>tlod vision</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui;text-align:center}
img{max-width:100%;height:auto}p{padding:8px}</style>
<p>tlod vision preview &mdash; annotated camera view</p><img src="/stream">
"""


class PreviewServer:
    """Holds the latest annotated frame and serves it as MJPEG."""

    def __init__(self, port: int = 8081, quality: int = 70, max_fps: float = 10.0) -> None:
        self.port = port
        self.quality = quality
        self.max_fps = max_fps
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._last_encode = 0.0

    # -- input -------------------------------------------------------------
    def offer(self, image: np.ndarray) -> None:
        """Give the server a frame. Throttled and cheap to call per frame."""
        import time

        now = time.perf_counter()
        if now - self._last_encode < 1.0 / self.max_fps:
            return
        self._last_encode = now
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()

    def latest(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    # -- server ------------------------------------------------------------
    def start(self) -> None:
        server = ThreadingHTTPServer(("0.0.0.0", self.port), _make_handler(self))
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True, name="preview")
        self._thread.start()
        log.info("preview at http://<this board>:%d/", self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _make_handler(owner: PreviewServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):    # keep the console usable
            pass

        def do_GET(self):                # noqa: N802 - http.server's API
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            import time

            try:
                while True:
                    jpeg = owner.latest()
                    if jpeg is None:
                        time.sleep(0.1)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1.0 / max(owner.max_fps, 1.0))
            except (BrokenPipeError, ConnectionResetError):
                pass   # the browser tab closed; entirely normal

    return Handler
