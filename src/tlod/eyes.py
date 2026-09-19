"""The eyes: two NeoPixel rings on a Seeed XIAO SAMD21, over USB serial.

A third board, independent of both the arm and the paddle (`tlod.leg`),
on its own USB cable. 32 pixels -- two 16-pixel rings daisy-chained on
D10 -- animated as a pair of eyes with three moods, and a serial
protocol of two shapes:

    h | c | a     set the mood directly: happy, concentrated, angry
    x,y,z         a point; the *board* takes sqrt(x^2+y^2+z^2) and picks
                  the mood from its own NEAR/FAR thresholds

and three replies:

    Distance: 24.15              for every point it parsed
    Emotion: CONCENTRATED        only when the mood actually changed
    Bad input, expected: x,y,z   for anything it could not parse

Everything below follows from four properties of that sketch.

**It never speaks first.** `setup()` prints nothing and there is no
heartbeat, so unlike the paddle board there is no passive way to know it
is there. Silence is not evidence of anything. `ping()` is the fix: `?`
parses as neither a mood key nor a point, so `Bad input` comes back
guaranteed -- and, unlike sending a mood, it cannot change what is on the
LEDs. That is the one safe question to ask this board.

**The mood line is edge-triggered.** `setEmotion` returns early when the
mood is unchanged, printing nothing. So silence after `h` means *either*
"already happy" *or* "not listening", and nothing distinguishes them.
This is why `set_emotion` reports `changed` rather than pretending to
confirm, and why `selftest` cycles through all three moods: from any
starting mood, h -> c -> a -> h makes at least three real transitions,
and every one of them must be announced.

**`Distance:` is the only proof the pixels are being driven.** It is
printed for every point the board parsed, before the threshold logic, and
it is the board's *own* arithmetic on the numbers it read. Checking it
against the same sqrt computed here is what separates "the cable is
connected" from "the board understood". It is also the one test that
catches `sscanf("%f")` quietly failing, which is a real possibility on
SAMD21 -- newlib-nano omits scanf float support unless the core links
`-u _scanf_float`, and when it is missing every well-formed point comes
back `Bad input` and the mood never moves.

**Something must drain the replies.** The board answers every line it is
sent, so a host that only writes leaves those bytes to pile up in the
kernel's receive buffer; once it is full, the SAMD21's USB-CDC writes
have nowhere to go and the animation loop stalls behind them. A reader
thread is not only how the replies get read, it is what stops the eyes
freezing after a few minutes of being talked at.

Units are the caller's problem and deliberately explicit. The sketch's
thresholds (20 near, 60 far) were placeholders chosen before anything was
sending real coordinates; this project works in metres, and the arm's
reach is about 0.08-0.40 m, so metres straight down the wire is always
"nearer than 20" and the eyes are permanently angry. `SCALE_CM` is the
sane default -- and note that even in centimetres the far threshold sits
past the end of the arm's reach, so HAPPY is not reachable from a point
at all until the sketch's constants are retuned. See `emotion_for`.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

BAUDRATE = 9600

# The sketch's own constants, mirrored so the host can predict what the
# board will decide. Keep in step with the .ino if they are ever tuned.
NEAR_THRESHOLD = 20.0
FAR_THRESHOLD = 60.0

# Metres -> centimetres. The only scale that puts this rig's coordinates
# anywhere near the thresholds above; see the module docstring.
SCALE_CM = 100.0

EMOTIONS: tuple[str, ...] = ("happy", "concentrated", "angry")
_KEYS: dict[str, str] = {"happy": "h", "concentrated": "c", "angry": "a"}

PING = "?"                      # parses as nothing; replies; touches no pixels
BAD_INPUT = "Bad input, expected: x,y,z"

# The board can only answer between animation frames, and loop() ends in
# delay(20). A reply inside 0.5 s is generous even so.
REPLY_TIMEOUT = 0.5
# Faster than the sketch can consume and the backlog only grows.
MAX_RATE_HZ = 1.0 / 0.02


class EyesError(RuntimeError):
    """The eyes could not be reached, or did not answer."""


class EyesTimeout(EyesError):
    """Something was sent and the guaranteed reply never came."""


def emotion_for(distance: float) -> str:
    """What the sketch will pick for this distance. Same thresholds, same order."""
    if distance < NEAR_THRESHOLD:
        return "angry"
    if distance > FAR_THRESHOLD:
        return "happy"
    return "concentrated"


@dataclass(frozen=True, slots=True)
class Reply:
    """What the board said about one point.

    `distance` is the board's own arithmetic, not ours -- which is the
    point of keeping it. `agrees` compares it against the same sqrt
    computed here, and is the check that the numbers arrived intact.

    `emotion` is always filled in: where the board did not announce one it
    is derived from the distance the board itself reported, by the same
    rule the sketch uses. `changed` says which of those two it was.
    """

    distance: float          # as the board computed it
    expected: float          # as computed here, from what was sent
    emotion: str             # the mood the board is in now
    changed: bool            # did it announce a change -- the sketch's own word
    stamp: float             # perf_counter when the reply landed
    latency: float

    @property
    def agrees(self) -> bool:
        # println(float) prints two decimals, so half a count of rounding
        # plus a little slack for float32 on the far side.
        return abs(self.distance - self.expected) <= 0.02 + 1e-3 * self.expected


@dataclass(frozen=True, slots=True)
class EmotionAck:
    emotion: str
    changed: bool            # False means the board was already in this mood
    stamp: float


class EyesLink:
    """Talks to the XIAO: sets moods, sends points, reads what comes back."""

    def __init__(
        self,
        port: str = "",
        baudrate: int = BAUDRATE,
        scale: float = SCALE_CM,
        reply_timeout: float = REPLY_TIMEOUT,
        precision: int = 1,
        transport: object | None = None,
        on_line: Callable[[str, float], None] | None = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.scale = scale
        self.reply_timeout = reply_timeout
        self.precision = precision
        self._on_line = on_line

        self._ser = transport
        self._owns_transport = transport is None
        self._replies: queue.Queue[tuple[str, float]] = queue.Queue()
        self._cmd_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # The sketch boots into HAPPY, but opening the port does not reset
        # a XIAO -- native USB, no DTR reset -- so a board that has been
        # running keeps whatever mood it had. Unknown until it tells us.
        self._emotion: str | None = None
        self._sent = 0
        self._bad = 0

    # -- lifecycle ---------------------------------------------------------
    def connect(self, verify: bool = True) -> None:
        """Open the port and, unless told not to, prove something is there."""
        if self._thread is not None:
            return
        if self._ser is None:
            self._ser = self._open_port()
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="eyes-reader", daemon=True)
        self._thread.start()
        if verify and not self.ping():
            self.disconnect()
            raise EyesError(
                f"nothing answered on {self.port or 'the eyes port'}. It replies to "
                f"every line, so silence means the wrong port, the wrong baud rate, "
                f"or a board that is not running the eyes sketch."
            )

    def _open_port(self):
        try:
            import serial
        except ImportError as e:  # pragma: no cover - depends on the install
            raise EyesError("the eyes need pyserial: pip install -e '.[eyes]'") from e
        if not self.port:
            raise EyesError("no port given; pass port= or run `tlod eyes --port ...`")
        try:
            # Never open a XIAO at 1200 baud: on SAMD21 that is the
            # bootloader knock, and it disconnects the sketch entirely.
            ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        except Exception as e:
            raise EyesError(f"cannot open {self.port}: {e}") from e
        try:
            ser.reset_input_buffer()
        except Exception:  # pragma: no cover
            pass
        return ser

    def disconnect(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        if self._ser is not None and self._owns_transport:
            try:
                self._ser.close()
            except Exception:  # pragma: no cover
                pass
            self._ser = None

    @property
    def connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def emotion(self) -> str | None:
        """The board's mood, as last reported. None until it says."""
        with self._state_lock:
            return self._emotion

    def __enter__(self) -> EyesLink:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # -- the input side ----------------------------------------------------
    def _read_loop(self) -> None:
        buf = bytearray()
        while not self._stop.is_set():
            try:
                waiting = getattr(self._ser, "in_waiting", 0) or 1
                chunk = self._ser.read(waiting)
            except Exception:
                if not self._stop.is_set():
                    log.warning("eyes: serial read failed, reader stopping", exc_info=True)
                return
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                line = bytes(buf[:i]).decode("ascii", "replace").strip()
                del buf[: i + 1]
                if line:
                    self._on_raw_line(line)
            if len(buf) > 4096:
                log.warning("eyes: 4 kB with no line ending; discarding")
                buf.clear()

    def _on_raw_line(self, line: str) -> None:
        now = time.perf_counter()
        if self._on_line is not None:
            try:
                self._on_line(line, now)
            except Exception:  # pragma: no cover
                log.warning("eyes: on_line hook raised", exc_info=True)
        # A mood announcement is unsolicited: it rides along with whatever
        # reply is owed, so record it here and do not queue it as one.
        if line.startswith("Emotion:"):
            mood = line.split(":", 1)[1].strip().lower()
            if mood in EMOTIONS:
                with self._state_lock:
                    self._emotion = mood
            self._replies.put((line, now))
            return
        if line == BAD_INPUT:
            with self._state_lock:
                self._bad += 1
        self._replies.put((line, now))

    def _drain(self) -> None:
        while True:
            try:
                self._replies.get_nowait()
            except queue.Empty:
                return

    def _write(self, text: str) -> float:
        if self._ser is None or not self.connected:
            raise EyesError("eyes are not connected; call connect() first")
        try:
            self._ser.write(text.encode("ascii") + b"\n")
            self._ser.flush()
        except Exception as e:
            raise EyesError(f"cannot write {text!r} to {self.port}: {e}") from e
        with self._state_lock:
            self._sent += 1
        return time.perf_counter()

    def _await(self, predicate, sent: float, timeout: float) -> tuple[str, float]:
        deadline = sent + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise EyesTimeout(f"no reply within {timeout:.2f}s")
            try:
                line, stamp = self._replies.get(timeout=remaining)
            except queue.Empty:
                continue
            if predicate(line):
                return line, stamp
            # Not the reply being waited on (a mood line arriving alongside
            # a Distance, say). Keep waiting rather than treating it as one.

    # -- the output side ---------------------------------------------------
    def ping(self, timeout: float | None = None) -> bool:
        """Is the board there? Sends `?`; changes nothing on the LEDs.

        The only question this sketch answers unconditionally. Use it
        before believing a silent `set_emotion`.
        """
        timeout = self.reply_timeout if timeout is None else timeout
        with self._cmd_lock:
            self._drain()
            sent = self._write(PING)
            try:
                self._await(lambda ln: ln == BAD_INPUT, sent, timeout)
            except EyesTimeout:
                return False
        return True

    def set_emotion(self, emotion: str, timeout: float | None = None) -> EmotionAck:
        """Set the mood directly with `h`/`c`/`a`.

        `changed` is False when no announcement came back, which the
        sketch does when it is already in that mood -- and which a board
        that has stopped listening also does. `ping()` tells them apart;
        `selftest()` does it for you.
        """
        if emotion not in _KEYS:
            raise ValueError(f"unknown emotion {emotion!r}; expected one of {list(EMOTIONS)}")
        timeout = self.reply_timeout if timeout is None else timeout
        with self._cmd_lock:
            self._drain()
            sent = self._write(_KEYS[emotion])
            try:
                _, stamp = self._await(
                    lambda ln: ln.startswith("Emotion:"), sent, timeout)
                changed = True
            except EyesTimeout:
                stamp, changed = time.perf_counter(), False
        with self._state_lock:
            # Either it moved and said so, or it was already there.
            self._emotion = emotion
        return EmotionAck(emotion=emotion, changed=changed, stamp=stamp)

    def send_point(self, x: float, y: float, z: float,
                   scale: float | None = None, timeout: float | None = None) -> Reply:
        """Send one `x,y,z` and read back the distance the board computed.

        Coordinates go in this project's units -- metres -- and are scaled
        on the way out. The reply is checked against the same sqrt done
        here, so a `Reply` that does not `agree` means the board did not
        read the numbers it was sent.
        """
        scale = self.scale if scale is None else scale
        timeout = self.reply_timeout if timeout is None else timeout
        p = self.precision
        payload = f"{x * scale:.{p}f},{y * scale:.{p}f},{z * scale:.{p}f}"
        # From the rounded numbers that actually go on the wire, not the
        # full-precision ones: at precision=1 the rounding alone can move
        # the distance further than the comparison tolerance, and `agrees`
        # is meant to catch a board that misread the line, not our own
        # formatting.
        sx, sy, sz = (float(v) for v in payload.split(","))
        expected = math.sqrt(sx * sx + sy * sy + sz * sz)

        with self._cmd_lock:
            # Before the write: the reader thread records a mood change the
            # instant it arrives, and the announcement follows the distance
            # in the same burst, so reading this afterwards races it.
            mood_before = self.emotion
            self._drain()
            sent = self._write(payload)
            line, stamp = self._await(
                lambda ln: ln.startswith("Distance:") or ln == BAD_INPUT, sent, timeout)
            if line == BAD_INPUT:
                raise EyesError(
                    f"the board could not parse {payload!r}. On SAMD21 this is usually "
                    f"sscanf() built without float support -- every well-formed point "
                    f"comes back as bad input and the mood never moves."
                )
            try:
                distance = float(line.split(":", 1)[1])
            except ValueError as e:
                raise EyesError(f"unparseable distance line {line!r}") from e

            # What the sketch will have picked, from *its* number, not ours.
            # Deriving it means the mood is known even when nothing is
            # announced -- which is the usual case, since the announcement
            # only comes on a change. Waiting unconditionally would spend
            # the timeout on every point that changes nothing, and at 20 Hz
            # that is nearly all of them.
            derived = emotion_for(distance)
            announced = False
            if mood_before is None or derived != mood_before:
                try:
                    self._await(lambda ln: ln.startswith("Emotion:"),
                                time.perf_counter(), 0.15)
                    announced = True
                except EyesTimeout:
                    pass
            if not announced:
                # No line, so nothing more authoritative than the derivation.
                # An announcement, when there is one, has already been
                # recorded by the reader and is believed over this.
                with self._state_lock:
                    self._emotion = derived

        # `changed` is the board's word, not an inference: the sketch
        # announces on a change and stays silent otherwise, so the presence
        # of the line is the fact. Deriving it instead would guess wrong
        # exactly when the mirrored thresholds have drifted from the .ino.
        return Reply(distance=distance, expected=expected,
                     emotion=self.emotion or derived, changed=announced,
                     stamp=stamp, latency=stamp - sent)

    def selftest(self, point: tuple[float, float, float] = (0.25, 0.0, 0.10)) -> list[tuple]:
        """Prove the pixels are actually being driven. Returns (step, ok, note).

        Three things get checked, in the order that isolates them:
        the board answers at all, the moods really change (each transition
        announced, so the animation is running and not merely accepting
        bytes), and a point survives the trip intact.

        The mood cycle is h -> c -> a -> h because from *any* starting
        mood that makes at least three genuine transitions. Only the first
        step may legitimately be silent.
        """
        results: list[tuple[str, bool, str]] = []

        ok = self.ping()
        results.append(("answers", ok, "replied to `?`" if ok else "no reply to `?`"))
        if not ok:
            return results

        cycle = ["happy", "concentrated", "angry", "happy"]
        for i, mood in enumerate(cycle):
            ack = self.set_emotion(mood)
            # The first step is allowed to be silent: it may already be there.
            required = i > 0
            good = ack.changed or not required
            note = "changed" if ack.changed else (
                "already there" if not required else "no announcement -- mood did not change")
            results.append((f"mood {mood}", good, note))

        try:
            reply = self.send_point(*point)
        except EyesError as e:
            results.append(("point", False, str(e)))
            return results
        results.append((
            "point",
            reply.agrees,
            f"board computed {reply.distance:.2f}, expected {reply.expected:.2f}"
            + ("" if reply.agrees else "  <-- the numbers did not survive the trip"),
        ))
        return results

    def stats(self) -> dict[str, object]:
        with self._state_lock:
            return {"port": self.port, "sent": self._sent, "bad_input": self._bad,
                    "emotion": self._emotion, "connected": self.connected}


def eyes_answer(port: str, timeout: float = 2.0, baudrate: int = BAUDRATE) -> bool:
    """Is the eyes board behind this port? Sends `?`, expects `Bad input`.

    Unlike the paddle's probe this one has to write, because the sketch
    never speaks first. `?` is the safe thing to write: it sets no mood,
    so a board that *is* the eyes carries on animating undisturbed, and
    the paddle board -- which ignores anything that is not one of its four
    words -- says nothing back and is correctly not identified.
    """
    link = EyesLink(port=port, baudrate=baudrate, reply_timeout=timeout)
    try:
        link.connect(verify=False)
    except EyesError:
        return False
    try:
        return link.ping(timeout=timeout)
    finally:
        link.disconnect()


def find_eyes_port(exclude: tuple[str, ...] = (), timeout: float = 2.0,
                   baudrate: int = BAUDRATE) -> str:
    """The first port that answers `?`. Empty string if none does.

    Three USB devices now share `/dev/ttyACM*` numbering that is assigned
    in enumeration order, so pass `exclude` the ports already claimed --
    the arm's bus and the paddle -- and probe what is left.
    """
    from tlod.arm.feetech import find_ports

    for port in find_ports():
        if port in exclude:
            continue
        if eyes_answer(port, timeout=timeout, baudrate=baudrate):
            return port
    return ""
