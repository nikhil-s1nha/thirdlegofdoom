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

# command -> the line the sketch answers with. All four are words now;
# `home` used to answer with an empty line and `slap` with a bare "s",
# which were easy to lose among the heartbeats.
ACKS: dict[str, str] = {"open": "OPEN", "close": "CLOSE",
                        "home": "HOME", "slap": "SLAP"}
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
        # Set by `connect`: did the board reboot when we opened the port,
        # and how long the first heartbeat took. See RESET_TELL.
        self.reset_on_connect = False
        self.first_beat = float("nan")
        self._first_beat = 0.0
        self._last_beat = 0.0

    # -- lifecycle ---------------------------------------------------------
    # Above this, the first heartbeat took long enough that the board must
    # have gone through a bootloader -- which means it reset, which means
    # `setup()` ran. A board that was already running answers inside its
    # own 500 ms beat interval.
    RESET_TELL: float = 1.0

    def connect(self, on_reset: Callable[[], None] | None = None) -> None:
        """Open the port and wait until the sketch is actually running.

        `on_reset` is called if the board turns out to have rebooted on
        open, *before* `connect` returns.

        This used to be a safety hook rather than a diagnostic: `setup()`
        attached both servos and wrote 90, and 90 is the door's open
        position, so merely connecting swung the hatch. The sketch no
        longer does that -- `setup()` is `Serial.begin` and nothing else,
        and the servos stay detached until a command attaches them. A
        reset now leaves them unpowered rather than driving them
        somewhere.

        Which is better but not nothing: an unpowered servo has no
        holding torque, so a door left open and a leg left out will sag
        under their own weight rather than staying put. Worth knowing
        before pulling the USB lead with the leg deployed.
        """
        if self._thread is not None:
            return
        if self._ser is None:
            self._ser = self._open_port()
        self._stop.clear()
        self._beat.clear()
        opened = time.perf_counter()
        self._thread = threading.Thread(target=self._read_loop, name="leg-reader", daemon=True)
        self._thread.start()
        got = self.wait_for_heartbeat(self.boot_timeout)
        # Measured rather than assumed. Holding DTR low stops the reset on
        # some adapters and not others -- CH340 parts in particular -- and
        # the difference is invisible from the outside until a servo moves
        # on its own. This makes it a number.
        self.first_beat = time.perf_counter() - opened if got else float("nan")
        self.reset_on_connect = bool(got and self.first_beat > self.RESET_TELL)
        if self.reset_on_connect:
            log.info(
                "leg: the board reset on connect (first beat after %.1f s). "
                "setup() attaches nothing, so the servos went limp rather than "
                "moving -- anything deployed will sag under its own weight",
                self.first_beat)
            if on_reset is not None:
                on_reset()
        if not got:
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
            ser = _open_without_resetting(serial, port, self.baudrate)
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
        """`home`: leg to 120, alone. **Bench testing only.**

        Moves servo 1 without touching the door, so it says nothing about
        whether the hatch is open and can leave the leg somewhere the
        door does not expect. `open` and `close` are the gesture; this is
        for checking that servo 1 answers at all.
        """
        return self.send("home")

    def slap(self) -> Ack:
        """`slap`: leg to 30, alone, and it stays there. **Bench testing only.**

        Not the strike, despite the name. The blow is the leg coming out
        of the hatch, which is `open` -- this drives servo 1 down with
        the door in whatever state it was already in, which at best does
        nothing (`open` leaves the leg at 30 already) and at worst drives
        it into a shut door.
        """
        return self.send("slap")

    def deploy(self) -> Ack:
        """Door open, leg out. One word: the sketch does the sequencing.

        `open` is door to 90, a 500 ms wait for it to finish swinging,
        then the leg to 30. This used to send `home` either side of it --
        once to clear the leg before the door moved, once after because
        `open` left the leg somewhere `slap` could not fall from. Both
        are the board's job now, and doing them twice only adds travel.

        Leaves the leg **down** at 30, which is where `slap` also puts it.
        So the first `strike` after a deploy homes before it can fall --
        see `strike`.
        """
        return self.open_hand()

    def retract(self) -> Ack:
        """Leg up, door shut. Also one word now.

        `close` homes the leg to 120 before driving the door to 145. It
        did not always, and a `close` sent with the leg down shut the
        door onto it and held it there -- a hobby servo stalled against a
        mechanical stop for as long as the board had power. That guard
        lives in the sketch, which is the right place for it: it holds
        however the board is driven, including from a serial terminal
        that has never heard of this file.
        """
        return self.close_hand()

    def strike(self, dwell: float = 0.8) -> Ack:
        """The whole gesture: come out and hit something, then go back in.

        `open` *is* the strike. It swings the door, waits for it, and
        drives the leg out and down in one word -- so the leg coming out
        of the hatch is the blow, not a wind-up before one. `close` puts
        it away again.

        This used to be `slap` then `home`, which was wrong about what
        the mechanism does. Those two exist for bench-testing servo 1 on
        its own and have no place in a gesture: see `slap`.

        Returns `open`'s `Ack`. Its `stamp` is the moment the board took
        the command, which is what a hit test lines up against -- though
        note the board answers *after* its own 500 ms door wait, so the
        stamp trails the door starting to move by about that much.

        `dwell` is how long the leg stays out before withdrawing, and it
        has to cover the leg's **physical travel** -- it is not
        showmanship, which is what this docstring used to claim.

        The board acks `open` when it has *taken* the command. Underneath,
        `servo.write()` sets a target and returns; there is no feedback on
        this board, so neither the sketch nor this file knows when the leg
        has actually arrived. Measured on the rig, `open` acks at ~720 ms
        while the leg only starts moving at ~700 -- so at the old default
        of 0.25 the `close` went out mid-swing and reversed the leg,
        which from outside looks like the servo doing random things.

        Typing the same two commands into a serial monitor never shows
        it, because a human takes seconds between them. That is the whole
        difference, and it is worth remembering the next time this driver
        and a hand-typed command appear to disagree: the driver is faster
        than the mechanism, and nothing on the wire says so.
        """
        ack = self.open_hand()
        time.sleep(dwell)
        self.close_hand()
        return ack


# -- finding the thing ------------------------------------------------------

# Set true to open the port exactly as a serial monitor does: the plain
# constructor, DTR left asserted, HUPCL left alone. `tlod leg --plain-open`
# flips it.
#
# It exists because those two settings are the *only* things this driver
# does to the port that the Arduino IDE does not, and "it behaves
# differently from the IDE" is a claim that deserves a controlled test
# rather than an argument. Nothing else here touches the port outside
# `send()`.
PLAIN_OPEN = False


def _open_without_resetting(serial, port: str, baudrate: int, timeout: float = 0.1):
    """Open the port without rebooting the board on the other end.

    Opening a serial port asserts DTR, and on an Arduino DTR is wired to
    RESET. So the plain `serial.Serial(port, ...)` restarts the sketch
    every time anything connects -- and this sketch's `setup()` writes 90
    to *both* servos, which swings the door open and parks the leg
    mid-travel before a single command has been sent.

    Measured on the rig, that cost about five seconds and made every
    gesture happen in two steps: the reset moved the servos, then the
    command moved them again from somewhere unexpected. The board was
    never slow -- `close` acked in 17 ms and `open` in 215, which is its
    own `delay(200)` and nothing else. All of the wait was before the
    command went out.

    Setting DTR low *before* opening leaves RESET alone. The sketch keeps
    running, its servos stay where they were, and the first heartbeat
    arrives within its own 500 ms interval instead of after a bootloader.

    Not every driver and platform honours it -- CH340 and FTDI parts
    differ, and a board wired with the reset-enable trace cut does not
    care either way -- so a failure here falls back to the ordinary open
    rather than refusing to talk to the leg at all. If the two-step
    movement comes back, that fallback is where to look; the fix in
    hardware is the usual 10 uF between RESET and GND.
    """
    if PLAIN_OPEN:
        log.info("leg: opening %s the way a serial monitor does "
                 "(DTR asserted, HUPCL untouched)", port)
        ser = serial.Serial(port, baudrate, timeout=timeout)
        _claim_exclusively(ser, port)
        return ser
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = timeout
    # Claim it before anything else can. See `_claim_exclusively`.
    try:
        ser.exclusive = True
    except Exception:                            # pragma: no cover - older pyserial
        pass
    try:
        ser.dtr = False
        ser.rts = False
    except Exception:            # pragma: no cover - backend without modem lines
        log.debug("leg: cannot hold DTR low on %s; the board may reset", port)
    try:
        ser.open()
    except Exception:
        # Some stacks refuse the pre-open setting rather than ignoring it.
        log.debug("leg: no-reset open failed on %s, falling back", port)
        ser = serial.Serial(port, baudrate, timeout=timeout)
    _keep_dtr_on_close(ser)
    return ser


def _claim_exclusively(ser, port: str) -> bool:
    """Stop anything else on this machine from opening the same port.

    pyserial does not set `TIOCEXCL` unless asked, so on Linux another
    process can open `/dev/ttyUSB0` *while we hold it* and toggle DTR or
    write bytes. Neither shows up in any trace here, because it is not
    our traffic -- from this side it looks like the board misbehaving on
    its own.

    Two services do this as a matter of course and neither exists on
    macOS, which is why a board can behave perfectly from a laptop and
    badly from a Linux SBC with the same sketch and the same wiring:

      ModemManager  probes new tty devices looking for a modem, on udev
                    events and on a timer. Opens the port, asserts and
                    drops the control lines, writes AT strings.
      brltty        claims CH340 devices believing they are braille
                    displays.

    A reset from a stray DTR toggle detaches the servos, and an attached
    servo that suddenly loses its signal is exactly the reported
    twitching.

    Best effort: an older pyserial without `exclusive`, or a platform
    without TIOCEXCL, keeps the old behaviour rather than refusing to
    run.
    """
    try:
        ser.exclusive = True
    except Exception as e:                       # pragma: no cover - platform dependent
        log.debug("leg: cannot claim %s exclusively (%s)", port, e)
        return False
    return True


def _keep_dtr_on_close(ser) -> bool:
    """Stop the board resetting when we *close* the port.

    There are two resets and they have different causes. Opening the port
    pulses DTR, which is the one a capacitor across RESET fixes. Closing
    it drops DTR because of HUPCL -- "hang up on last close" -- which is a
    terminal flag and costs nothing to clear.

    That second one is why a gesture does not stay put. Measured on the
    rig: `strike` slaps the leg to 40, homes it to 120, and then the CLI
    exits, the port closes, the board reboots and `setup()` puts it back
    to 90. From outside that looks like the leg going down, up, and then
    drifting down again on its own -- and the door reopening with it,
    since `setup()` writes 90 to servo 0 as well.

    Linux only in practice; anything without HUPCL in termios keeps its
    old behaviour rather than failing, since a board that resets on close
    still works, it just moves when it should not.
    """
    try:
        import termios
    except ImportError:                          # pragma: no cover - not POSIX
        return False
    try:
        fd = ser.fileno()
        attrs = termios.tcgetattr(fd)
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception as e:                       # pragma: no cover - backend dependent
        log.debug("leg: could not clear HUPCL (%s); the board will reset on close", e)
        return False
    return True


def heartbeat_answers(port: str, timeout: float = 6.0, baudrate: int = BAUDRATE) -> bool:
    """Is the leg behind this port? Listens for `<3`; writes nothing.

    Read-only, so it is safe to point at the servo bus by mistake -- the
    adapter will simply never say `<3`.

    Opens with DTR held low, the same as `LegLink`, so probing does not
    reboot the board it is looking for. It used to, and probing walks
    *every* candidate port -- so a single `tlod leg slap` could reset the
    Arduino more than once before the command went out, and each reset
    ran a `setup()` that writes 90 to both servos. `timeout` still allows
    for a bootloader because a board that does reset anyway needs the
    room.
    """
    try:
        import serial
    except ImportError:
        return False
    try:
        ser = _open_without_resetting(serial, port, baudrate)
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


@dataclass(slots=True)
class LegServiceStats:
    fired: int = 0
    dropped: int = 0
    blocked: int = 0
    failed: int = 0
    last_error: str = ""


class LegService:
    """Drives the leg from its own thread, so the control loop never waits.

    `LegLink.strike()` blocks: it slaps, sleeps for `dwell`, then homes.
    That is 250 ms plus two serial round trips, against a control loop
    that ticks every 10 ms. Calling it inline from a policy would stall
    the arm mid-swing -- and worse, a policy tick that raises reaches
    `RobotApp._control_loop`, which answers a failed tick by e-stopping.
    The same reasoning that keeps a second sync read out of `Strike`
    keeps a serial write out of `update()`: the control thread's job is
    to be on time, and anything that can block is not its work.

    So `fire()` hands the gesture to a worker and returns immediately.

    **One slot, and a request that arrives while the leg is busy is
    dropped rather than queued.** A queue would be worse than useless
    here: the leg would still be working through round three's slap when
    round five resolved, and a slap that lands after its round is not
    late, it is wrong. Dropping is counted so a session can say how often
    it happened -- if `dropped` is large the leg is being asked to gesture
    faster than a 250 ms gesture allows, which is a pacing decision and
    not something the driver should paper over.

    Failures are counted, never raised. The leg is decoration; the arm is
    the game. A board that has come unplugged mid-session should cost the
    gestures it was asked for and nothing else.
    """

    def __init__(self, link: LegLink, dwell: float = 0.25,
                 is_clear: Callable[[], bool] | None = None) -> None:
        self.link = link
        self.dwell = dwell
        # The interlock. The leg deploys through a hatch the arm sits in
        # front of, so the two effectors are physically exclusive and a
        # gesture asked for while the arm is in the way is a collision,
        # not a missed cue.
        #
        # It is a predicate rather than a reference to the controller so
        # that this class stays testable without an arm, and so that
        # whatever owns the sequencing -- the game, the CLI -- decides
        # what "clear" means. `ArmController.is_stowed` is the one the
        # game passes, and it reads the encoders rather than the
        # commanded pose, because a stow that was commanded and never
        # completed is exactly the case this has to catch.
        #
        # None means no interlock, for a rig where the leg has the space
        # to itself. That is a deliberate opt-out, not a default: the
        # caller has to say so.
        self.is_clear = is_clear
        self.stats = LegServiceStats()
        self._want = threading.Event()
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._gesture = "strike"
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="leg", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._want.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def fire(self, gesture: str = "strike") -> bool:
        """Ask for a gesture. Returns False if the leg was already busy.

        Safe to call from the control thread: it sets an event and
        returns. Never raises -- a caller in a policy tick cannot be
        given anything to handle.
        """
        if self._busy.is_set() or self._stop.is_set():
            with self._lock:
                self.stats.dropped += 1
            return False
        # Checked here, on the caller's thread, and not in the worker:
        # the answer has to be about the arm's position *now*, and the
        # worker may not get to the request for a tick or two. Counted
        # separately from `dropped` because they mean opposite things --
        # dropped is the leg being asked too fast, blocked is the arm
        # being where the leg needs to go.
        if self.is_clear is not None:
            try:
                clear = bool(self.is_clear())
            except Exception:                            # noqa: BLE001
                clear = False
            if not clear:
                with self._lock:
                    self.stats.blocked += 1
                return False
        with self._lock:
            self._gesture = gesture
        self._busy.set()
        self._want.set()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            self._want.wait()
            self._want.clear()
            if self._stop.is_set():
                break
            if not self._busy.is_set():
                continue
            with self._lock:
                gesture = self._gesture
            try:
                if gesture == "strike":
                    self.link.strike(dwell=self.dwell)
                else:
                    self.link.send(gesture)
            except Exception as e:                      # noqa: BLE001
                # Counted, not raised. See the class docstring: this
                # thread exists so that a serial failure costs a gesture
                # rather than the game.
                with self._lock:
                    self.stats.failed += 1
                    self.stats.last_error = f"{type(e).__name__}: {e}"
                log.warning("leg: %s failed: %s", gesture, e)
            else:
                with self._lock:
                    self.stats.fired += 1
            finally:
                self._busy.clear()

    def report(self) -> str:
        """End-of-run line, in the same shape as the other subsystems'."""
        with self._lock:
            s = self.stats
            line = f"leg: {s.fired} gesture(s)"
            if s.dropped:
                line += f", {s.dropped} dropped (asked while still moving)"
            if s.blocked:
                line += (f", {s.blocked} blocked (the arm was not stowed, so the "
                         f"hatch could not open)")
            if s.failed:
                line += f", {s.failed} failed -- last: {s.last_error}"
            return line
