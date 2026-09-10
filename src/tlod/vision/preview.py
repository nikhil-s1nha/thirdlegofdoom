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

The board only ever wants one kind of HTTP server, so anything else that
needs to publish a page -- the scoreboard in viz/scoreboard.py is the
first -- registers a route here rather than standing up a second
ThreadingHTTPServer with its own port, its own lifecycle and its own
`log_message` override. A route is a callable returning
(content-type, bytes); it is invoked on the request thread, so it must be
cheap and must not touch the arm. Routes are matched before the built-in
paths, which lets a caller that has no frames to show replace `/` with
its own page instead of an index that links to an empty stream.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
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

    def __init__(self, port: int = 8081, quality: int = 70, max_fps: float = 10.0,
                 label: str = "preview") -> None:
        self.port = port
        # Only ever a name in a log line and on the thread, but the log
        # line is how you find out which of two ports you are looking at
        # from the far end of an SSH session, and "preview" on a server
        # that is serving the scoreboard sends you to the wrong one.
        self.label = label
        self.quality = quality
        self.max_fps = max_fps
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._last_encode = 0.0
        self.routes: dict[str, Callable[[], tuple[str, bytes]]] = {}

    # -- routes ------------------------------------------------------------
    def add_route(self, path: str, handler: Callable[[], tuple[str, bytes]]) -> None:
        """Publish `handler` at `path`. Returns (content-type, body).

        Deliberately not a framework: no methods other than GET, no path
        parameters, no templating. Everything on this board that wants a
        page wants a fixed URL returning either a fixed blob of HTML or a
        small JSON snapshot of some state, and the moment it wants more
        than that it should be a real service somewhere else.
        """
        self.routes[path] = handler

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
        self._thread = threading.Thread(target=server.serve_forever, daemon=True,
                                        name=self.label)
        self._thread.start()
        log.info("%s at http://<this board>:%d/", self.label, self.port)

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
            route = owner.routes.get(self.path.split("?", 1)[0])
            if route is not None:
                try:
                    content_type, body = route()
                except Exception:
                    log.debug("route %s failed", self.path, exc_info=True)
                    self.send_error(500)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                # A scoreboard polled every couple of hundred ms is
                # exactly the shape of response a browser will happily
                # serve from cache forever, and a frozen score looks
                # identical to a crashed game.
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
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
