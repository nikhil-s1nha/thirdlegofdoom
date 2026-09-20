"""The third leg: two hobby servos on an Arduino, over USB serial.

The arm is six Feetech servos on a bus that answers questions. This is
not that. It is an Arduino running a fixed sketch, wired to the Orange Pi
by USB, and the whole protocol is four lowercase words in and a line of
text back:

    open   servo 0 -> 90  (jaw open), then, after 200 ms, servo 1 -> 40
    close  servo 0 -> 145 (jaw shut)
    home   servo 1 -> 120 (paddle up)
    slap   servo 1 -> 40  (paddle down)

Plus `<3` every 500 ms, unasked, forever. That heartbeat is the only
thing here that reports anything, so this module is built around it.

Four things about that sketch shape every decision below.

**Nothing echoes state.** The servos are hobby servos on `Servo.write()`:
no encoder, no feedback, no `read()`. The board cannot tell you where the
paddle is, only that it accepted the word. So `Ack.stamp` -- when the
reply landed -- is the best evidence that exists of when the paddle began
to move, and it is what this module records. Compare `tlod.types`: every
observation carries the time of the physical event, and where the
hardware cannot supply one, the honest thing is to say which proxy you
used rather than to invent a number.

**Replies and heartbeats share one stream.** A read after a write is
overwhelmingly likely to return `<3` rather than your reply, because the
heartbeat is unsolicited and the odds favour it. Hence a reader thread
that classifies every line, rather than a blocking read per command.

**`home` acknowledges with an empty line.** `Serial.println("")` in the
sketch. Nothing distinguishes it from a blank line arriving for any other
reason, so `home` can only be matched positionally -- one reply per
command, in the order sent. If you ever edit the sketch, make it print
`HOME` and this becomes checkable; until then it is a hole in the
protocol and this module works around it rather than pretending it does
not exist.

**`open` blocks the sketch for 200 ms.** `delay(200)` sits inside the
command handler, so during it `loop()` neither beats nor reads. Two
consequences: an `open` reply is never faster than 200 ms, and a
heartbeat can arrive that late as well -- which is why liveness is judged
against `HEARTBEAT_TIMEOUT` and not against the 500 ms interval. It also
caps how fast commands may be pushed: 200 ms of silence at 9600 baud is
about 192 bytes, and the Arduino's receive buffer holds 64. Sending the
next command only after the previous one has replied keeps the buffer
from overflowing, and is why `send` is synchronous rather than queued.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

BAUDRATE = 9600
HEARTBEAT = "<3"
HEARTBEAT_INTERVAL = 0.5     # BEAT_INTERVAL in the sketch

# How long without a heartbeat before the board counts as gone. Generous
# on purpose: `open`'s delay(200) can push a beat late all by itself, and
# a false "the leg died" mid-game is worse than noticing a second later.
HEARTBEAT_TIMEOUT = 1.5

# command -> the line the sketch answers with. `home` really is "".
ACKS: dict[str, str] = {"open": "OPEN", "close": "CLOSE", "home": "", "slap": "s"}
COMMANDS: tuple[str, ...] = tuple(ACKS)


class LegError(RuntimeError):
    """The leg could not be reached, or did not answer."""


class LegTimeout(LegError):
    """A command went out and no reply came back in time."""


@dataclass(frozen=True, slots=True)
class Ack:
    """What came back from one command, and when.

    `stamp` is a `time.perf_counter()` reading, the same clock everything
    else in the project timestamps against, so a strike here can be lined
    up against a camera shutter without conversion.
    """

    command: str
    line: str        # exactly what the board said, stripped
    stamp: float     # perf_counter when the reply landed
    latency: float   # write -> reply, seconds
    expected: bool   # did the line match what the sketch should have said


@dataclass(frozen=True, slots=True)
class LegStatus:
    """A snapshot of the link, cheap enough to poll every tick."""

    connected: bool
    port: str
    alive: bool          # a heartbeat within HEARTBEAT_TIMEOUT
    age: float           # seconds since the last heartbeat; inf if never
    beats: int
    commands: int
    interval: float      # mean measured beat interval; 0 before two beats


class LegLink:
    """Talks to the Arduino: sends the four commands, watches the heartbeat.

    A reader thread owns the input side. Everything public is safe to call
    from any thread; commands are serialised, so two threads asking for a
    slap at once get two slaps in some order rather than two replies
    crossed over each other.
    """

    def __init__(
        self,
        port: str = "",
        baudrate: int = BAUDRATE,
        ack_timeout: float = 1.0,
        boot_timeout: float = 6.0,
        transport: object | None = None,
        on_line: Callable[[str, float], None] | None = None,
        exclude: tuple[str, ...] = (),
    ) -> None:
        self.port = port
        # Ports already spoken for by another board, skipped when probing.
        self.exclude = exclude
        self.baudrate = baudrate
        self.ack_timeout = ack_timeout
        # Opening the port asserts DTR, which resets most Arduino boards;
        # the bootloader then sits there for a second or two before
        # setup() runs. Waiting for the first heartbeat covers that
        # without guessing at a sleep, and covers boards that do not
        # reset at all, where it returns immediately.
        self.boot_timeout = boot_timeout
        self._on_line = on_line

        self._ser = transport
        self._owns_transport = transport is None
        self._acks: queue.Queue[tuple[str, float]] = queue.Queue()
        self._cmd_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._beat = threading.Event()
        self._thread: threading.Thread | None = None

        self._beats = 0
        self._commands = 0
        self._first_beat = 0.0
        self._last_beat = 0.0

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        """Open the port and wait until the sketch is actually running."""
        if self._thread is not None:
            return
        if self._ser is None:
            self._ser = self._open_port()
        self._stop.clear()
        self._beat.clear()
        self._thread = threading.Thread(target=self._read_loop, name="leg-reader", daemon=True)
        self._thread.start()
        if not self.wait_for_heartbeat(self.boot_timeout):
            self.disconnect()
            raise LegError(
                f"no heartbeat from {self.port or 'the leg'} in {self.boot_timeout:.0f}s. "
                f"Check it is the Arduino's port and not the servo bus, that the sketch "
                f"is the one flashed, and that nothing else has the port open."
            )

    def _open_port(self):
        try:
            import serial
        except ImportError as e:  # pragma: no cover - depends on the install
            raise LegError("the leg needs pyserial: pip install -e '.[leg]'") from e

        port = self.port or find_leg_port(exclude=self.exclude, baudrate=self.baudrate)
        if not port:
            raise LegError(
                "no Arduino answered with a heartbeat. Plug it in, check "
                "`tlod ports`, and set leg.port so a replug cannot move it."
            )
        self.port = port
        try:
            ser = serial.Serial(port, self.baudrate, timeout=0.1)
        except Exception as e:
            raise LegError(f"cannot open {port}: {e}") from e
        # Whatever the bootloader left behind is not protocol.
        try:
            ser.reset_input_buffer()
        except Exception:  # pragma: no cover - not every backend has it
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

    def __enter__(self) -> LegLink:
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
                    log.warning("leg: serial read failed, reader stopping", exc_info=True)
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
                self._on_raw_line(line)
            if len(buf) > 4096:
                # No newline in 4 kB is not this protocol -- wrong baud
                # rate, or the port belongs to something else entirely.
                log.warning("leg: 4 kB with no line ending; discarding")
                buf.clear()

    def _on_raw_line(self, line: str) -> None:
        now = time.perf_counter()
        if self._on_line is not None:
            try:
                self._on_line(line, now)
            except Exception:  # pragma: no cover - a bad hook must not kill the reader
                log.warning("leg: on_line hook raised", exc_info=True)
        if line == HEARTBEAT:
            with self._state_lock:
                self._beats += 1
                if not self._first_beat:
                    self._first_beat = now
                self._last_beat = now
            self._beat.set()
            return
        # Everything that is not a heartbeat is a reply -- including the
        # empty line `home` answers with.
        self._acks.put((line, now))

    def wait_for_heartbeat(self, timeout: float = HEARTBEAT_TIMEOUT) -> bool:
        """Block until the next beat. True if one came, False on timeout."""
        self._beat.clear()
        with self._state_lock:
            if self._beats and (time.perf_counter() - self._last_beat) < HEARTBEAT_INTERVAL:
                return True
        return self._beat.wait(timeout)

    def status(self) -> LegStatus:
        with self._state_lock:
            beats, commands = self._beats, self._commands
            first, last = self._first_beat, self._last_beat
        age = (time.perf_counter() - last) if beats else float("inf")
        interval = (last - first) / (beats - 1) if beats > 1 else 0.0
        return LegStatus(
            connected=self.connected,
            port=self.port,
            alive=self.connected and age < HEARTBEAT_TIMEOUT,
            age=age,
            beats=beats,
            commands=commands,
            interval=interval,
        )

    @property
    def alive(self) -> bool:
        """Has the board beaten recently. Read this before trusting a slap."""
        return self.status().alive

    # -- the output side ---------------------------------------------------
    def send(self, command: str, timeout: float | None = None) -> Ack:
        """Send one command and wait for its reply.

        Synchronous by design: the sketch handles one command at a time
        and stops reading during `open`, so the reply is the only signal
        that it is ready for the next word.
        """
        if command not in ACKS:
            raise ValueError(f"unknown command {command!r}; expected one of {list(COMMANDS)}")
        if self._ser is None or not self.connected:
            raise LegError("leg is not connected; call connect() first")
        timeout = self.ack_timeout if timeout is None else timeout

        with self._cmd_lock:
            # A previous command that timed out may have landed late.
            # Its reply is not this command's reply.
            while True:
                try:
                    stale, _ = self._acks.get_nowait()
                except queue.Empty:
                    break
                log.warning("leg: discarding late reply %r", stale)

            sent = time.perf_counter()
            try:
                self._ser.write(command.encode("ascii") + b"\n")
                self._ser.flush()
            except Exception as e:
                raise LegError(f"cannot write {command!r} to {self.port}: {e}") from e
            with self._state_lock:
                self._commands += 1

            deadline = sent + timeout
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise LegTimeout(
                        f"no reply to {command!r} within {timeout:.2f}s "
                        f"(the board is {'beating' if self.alive else 'silent'})"
                    )
                try:
                    line, stamp = self._acks.get(timeout=remaining)
                except queue.Empty:
                    continue
                break

        expected = ACKS[command]
        if line != expected:
            log.warning("leg: %r answered %r, expected %r", command, line, expected)
        return Ack(command=command, line=line, stamp=stamp,
                   latency=stamp - sent, expected=line == expected)

    # Named after the wire commands, except that `close` would read as
    # "close the port" on an object that also has disconnect().
    def open_hand(self) -> Ack:
        """`open`: jaw to 90, then -- 200 ms later -- the paddle down to 40."""
        return self.send("open")

    def close_hand(self) -> Ack:
        """`close`: jaw to 145. Leaves the paddle wherever it was."""
        return self.send("close")

    def home(self) -> Ack:
        """`home`: paddle up to 120. Answers with an empty line."""
        return self.send("home")

    def slap(self) -> Ack:
        """`slap`: paddle down to 40, and it stays there. See `strike`."""
        return self.send("slap")

    def strike(self, dwell: float = 0.25) -> Ack:
        """Slap, hold, and come back up.

        `slap` on its own is half a gesture: the sketch drives servo 1 to
        40 and leaves it there, so a second `slap` does nothing at all
        until something has homed it. This is the gesture you want.

        Returns the slap's `Ack`, not home's -- its `stamp` is the moment
        the board took the command, which is what a hit test wants to
        line up against. Which also means `dwell` is the paddle's *own*
        travel time and nothing else: recalibrate it if the geometry
        changes, the same way a contact threshold has to be recalibrated
        whenever the strike does.
        """
        ack = self.slap()
        time.sleep(dwell)
        self.home()
        return ack


# -- finding the thing ------------------------------------------------------
def heartbeat_answers(port: str, timeout: float = 6.0, baudrate: int = BAUDRATE) -> bool:
    """Is the leg behind this port? Listens for `<3`; writes nothing.

    Read-only, so it is safe to point at the servo bus by mistake -- the
    adapter will simply never say `<3`. It is not free of side effects
    though: opening a port resets the Arduino behind it, which is why
    `timeout` has to allow for the bootloader as well as a beat interval.
    """
    try:
        import serial
    except ImportError:
        return False
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
    except Exception:
        return False
    try:
        buf = bytearray()
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf.extend(chunk)
            if HEARTBEAT.encode() in buf:
                return True
            del buf[:-64]
        return False
    finally:
        ser.close()


def find_leg_port(exclude: tuple[str, ...] = (), timeout: float = 6.0,
                  baudrate: int = BAUDRATE) -> str:
    """The first port that beats. Empty string if none does.

    Two USB devices hang off the Orange Pi and both enumerate as
    `/dev/ttyACM*` in whatever order they came up, so the port that was
    the Arduino last week may be the servo bus today. The heartbeat
    settles it by asking, which is the same reasoning as `tlod ports
    --probe` asking each port whether six servos live behind it.
    """
    from tlod.arm.feetech import find_ports

    for port in find_ports():
        if port in exclude:
            continue
        if heartbeat_answers(port, timeout=timeout, baudrate=baudrate):
            return port
    return ""
