"""The Arduino paddle link.

The fake below is a port of the sketch, not a stub that returns what the
tests want: it beats on its own timer whether or not anyone asked, it
blocks for 200 ms inside `open` exactly as `delay(200)` does, and it
answers `home` with an empty line. Every property worth testing here is a
property of *that* behaviour -- a reply arriving behind two heartbeats, a
blocked board, an ack that is indistinguishable from a blank line -- and
a cooperative fake would test none of them.
"""

import threading
import time

import pytest

from tlod.leg import (
    ACKS,
    COMMANDS,
    HEARTBEAT,
    Ack,
    LegError,
    LegLink,
    LegTimeout,
)


class FakeArduino:
    """The sketch, as a serial-port-shaped object.

    Its own thread pushes heartbeats into the read buffer, so the timing
    the driver has to cope with is real rather than scripted.
    """

    def __init__(self, beat_interval=0.05, open_delay=0.2, deaf=False, replies=None,
                 max_chunk=None):
        self.beat_interval = beat_interval
        self.max_chunk = max_chunk       # bytes per read(); None = whatever is asked
        self.open_delay = open_delay
        self.deaf = deaf                 # accepts commands, never answers
        self.replies = replies or ACKS   # let a test forge a wrong answer
        self.received: list[str] = []
        self._rx = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.closed = False
        self._beater = threading.Thread(target=self._beat_loop, daemon=True)
        self._beater.start()

    # -- the sketch ---------------------------------------------------------
    def _beat_loop(self):
        while not self._stop.wait(self.beat_interval):
            self._emit(HEARTBEAT)

    def _emit(self, line: str):
        with self._lock:
            self._rx.extend(line.encode() + b"\r\n")

    # -- the serial API the driver uses -------------------------------------
    @property
    def in_waiting(self):
        with self._lock:
            return len(self._rx)

    def read(self, n=1):
        if self.max_chunk is not None:
            n = min(n, self.max_chunk)
        deadline = time.perf_counter() + 0.1     # the port's timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._rx:
                    out = bytes(self._rx[:n])
                    del self._rx[:n]
                    return out
            time.sleep(0.002)
        return b""

    def write(self, data: bytes):
        cmd = data.decode().strip()
        self.received.append(cmd)
        if self.deaf:
            return len(data)
        if cmd == "open":
            # delay(200): loop() neither beats nor reads during this.
            time.sleep(self.open_delay)
        if cmd in self.replies:
            self._emit(self.replies[cmd])
        return len(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        with self._lock:
            self._rx.clear()

    def close(self):
        self._stop.set()
        self.closed = True


@pytest.fixture
def leg():
    board = FakeArduino()
    link = LegLink(port="fake", transport=board, ack_timeout=1.0, boot_timeout=2.0)
    link.connect()
    yield link, board
    link.disconnect()
    board.close()


# -- the four commands ------------------------------------------------------
def test_every_command_in_the_sketch_round_trips(leg):
    link, board = leg
    for cmd in COMMANDS:
        ack = link.send(cmd)
        assert ack.command == cmd
        assert ack.line == ACKS[cmd]
        assert ack.expected
    assert board.received == list(COMMANDS)


def test_named_methods_send_the_wire_commands(leg):
    link, board = leg
    link.open_hand()
    link.close_hand()
    link.home()
    link.slap()
    assert board.received == ["open", "close", "home", "slap"]


def test_every_command_acks_with_a_word_of_its_own(leg):
    """All four replies are distinguishable from each other and from `<3`.

    They were not always. `home` answered with `Serial.println("")` and
    `slap` with a bare `s`, and a blank line among the heartbeats is a
    reply that looks exactly like a dropped one. The sketch names them
    now, and this reads the table rather than the strings so the next
    rewording is one edit.
    """
    link, _ = leg
    for command in COMMANDS:
        ack = link.send(command)
        assert ack.line == ACKS[command]
        assert ack.expected
        assert ack.line, f"{command} answers with a blank line"
    assert len(set(ACKS.values())) == len(ACKS), "two commands share a reply"
    assert HEARTBEAT not in ACKS.values()


def test_strike_comes_out_and_goes_back_in(leg):
    """The blow is the leg coming out of the hatch, which is `open`.

    Not `slap`. That drives servo 1 with the door in whatever state it
    was already in -- at best nothing, since `open` leaves the leg at 30
    already, at worst into a shut door. `slap` and `home` are for
    bench-testing servo 1 and have no place in a gesture.
    """
    link, board = leg
    ack = link.strike(dwell=0.01)
    assert board.received == ["open", "close"]
    assert ack.command == "open"        # the stamp a hit test wants


def test_unknown_command_is_refused_before_it_reaches_the_board(leg):
    link, board = leg
    with pytest.raises(ValueError):
        link.send("wiggle")
    assert board.received == []


# -- the heartbeat sharing the stream ---------------------------------------
def test_reply_is_found_behind_heartbeats(leg):
    """The odds favour `<3` arriving between the write and the reply."""
    link, board = leg
    time.sleep(0.12)                    # let a couple of beats queue up
    ack = link.send("close")
    assert ack.line == "CLOSE"
    assert link.status().beats >= 2


def test_heartbeats_are_not_mistaken_for_replies(leg):
    link, _ = leg
    time.sleep(0.2)
    ack = link.send("slap")
    assert ack.line == ACKS["slap"]


def test_status_measures_the_beat_interval(leg):
    link, _ = leg
    time.sleep(0.3)
    st = link.status()
    assert st.alive
    assert st.beats >= 3
    assert 0.02 < st.interval < 0.2     # the fake's 50 ms, loosely
    assert st.age < 0.5


def test_a_silent_board_stops_being_alive(monkeypatch):
    """A board that beat and then stopped is worse than one that never did:
    the link is open, writes still succeed, and only the missing beat says
    the sketch is gone."""
    monkeypatch.setattr("tlod.leg.HEARTBEAT_TIMEOUT", 0.15)
    board = FakeArduino()
    link = LegLink(port="fake", transport=board, boot_timeout=2.0)
    link.connect()
    try:
        assert link.alive
        board.close()                   # the sketch stops beating
        time.sleep(0.25)
        st = link.status()
        assert not st.alive
        assert st.beats > 0             # it did beat, once
        assert st.age > 0.15
    finally:
        link.disconnect()


# -- failure ----------------------------------------------------------------
def test_no_reply_raises_rather_than_hanging():
    board = FakeArduino(deaf=True)
    link = LegLink(port="fake", transport=board, ack_timeout=0.2, boot_timeout=2.0)
    link.connect()
    try:
        with pytest.raises(LegTimeout):
            link.slap()
    finally:
        link.disconnect()
        board.close()


def test_a_late_reply_is_not_handed_to_the_next_command():
    """A timed-out command's reply must not be read as the next one's."""
    board = FakeArduino(open_delay=0.4)
    link = LegLink(port="fake", transport=board, ack_timeout=0.15, boot_timeout=2.0)
    link.connect()
    try:
        with pytest.raises(LegTimeout):
            link.open_hand()            # replies 400 ms later, too late
        time.sleep(0.4)                 # OPEN lands now, unclaimed
        ack = link.close_hand()
        assert ack.line == "CLOSE"      # not "OPEN"
    finally:
        link.disconnect()
        board.close()


def test_a_wrong_answer_is_reported_not_swallowed():
    board = FakeArduino(replies={**ACKS, "slap": "OPEN"})
    link = LegLink(port="fake", transport=board, ack_timeout=1.0, boot_timeout=2.0)
    link.connect()
    try:
        ack = link.slap()
        assert ack.line == "OPEN"
        assert not ack.expected         # the caller can see it went wrong
    finally:
        link.disconnect()
        board.close()


def test_connect_fails_when_nothing_ever_beats():
    """A port that opens but never says `<3` is the wrong port."""
    board = FakeArduino(beat_interval=100.0)
    link = LegLink(port="fake", transport=board, boot_timeout=0.3)
    with pytest.raises(LegError, match="no heartbeat"):
        link.connect()
    assert not link.connected
    board.close()


def test_commands_before_connect_are_refused():
    link = LegLink(port="fake", transport=FakeArduino())
    with pytest.raises(LegError):
        link.slap()


def test_open_is_never_faster_than_the_sketch_delay(leg):
    """delay(200) is in the command handler, so it is in the latency."""
    link, _ = leg
    ack = link.open_hand()
    assert ack.latency >= 0.2


# -- the framing ------------------------------------------------------------
def test_lines_split_across_reads_are_reassembled():
    """One byte per read, the way a port under load actually delivers.

    `<3\r\n` arriving as five separate reads must still be one heartbeat
    and not five unparseable fragments.
    """
    board = FakeArduino(max_chunk=1)
    link = LegLink(port="fake", transport=board, boot_timeout=2.0)
    link.connect()
    try:
        ack = link.slap()               # "s\r\n", a byte at a time
        assert ack.line == ACKS["slap"]
        assert ack.expected
        time.sleep(0.15)
        assert link.status().beats >= 2
    finally:
        link.disconnect()
        board.close()


def test_several_lines_in_one_read_are_all_seen():
    """The other half: a burst that arrives as a single chunk."""
    seen = []
    board = FakeArduino(beat_interval=100.0)
    link = LegLink(port="fake", transport=board, boot_timeout=1.0,
                   on_line=lambda line, t: seen.append(line))
    board._emit(HEARTBEAT)              # queued before anything reads
    board._emit("CLOSE")
    board._emit(HEARTBEAT)
    link.connect()
    try:
        time.sleep(0.1)
        assert seen == [HEARTBEAT, "CLOSE", HEARTBEAT]
        assert link.status().beats == 2
    finally:
        link.disconnect()
        board.close()


def test_on_line_hook_sees_everything_including_heartbeats():
    seen = []
    board = FakeArduino()
    link = LegLink(port="fake", transport=board, boot_timeout=2.0,
                   on_line=lambda line, t: seen.append(line))
    link.connect()
    try:
        link.slap()
        time.sleep(0.12)
        assert ACKS["slap"] in seen
        assert HEARTBEAT in seen
    finally:
        link.disconnect()
        board.close()


def test_a_hook_that_raises_does_not_kill_the_reader():
    def boom(line, t):
        raise RuntimeError("bad hook")

    board = FakeArduino()
    link = LegLink(port="fake", transport=board, boot_timeout=2.0, on_line=boom)
    link.connect()
    try:
        assert link.slap().line == ACKS["slap"]
    finally:
        link.disconnect()
        board.close()


def test_disconnect_leaves_a_caller_supplied_port_open():
    """We did not open it, so it is not ours to close."""
    board = FakeArduino()
    link = LegLink(port="fake", transport=board, boot_timeout=2.0)
    link.connect()
    link.disconnect()
    assert not board.closed
    board.close()


def test_context_manager_connects_and_disconnects():
    board = FakeArduino()
    with LegLink(port="fake", transport=board, boot_timeout=2.0) as link:
        assert link.connected
        assert link.slap().line == ACKS["slap"]
    assert not link.connected
    board.close()


def test_ack_types_are_what_the_rest_of_the_project_expects(leg):
    link, _ = leg
    ack = link.slap()
    assert isinstance(ack, Ack)
    assert ack.stamp > 0                # perf_counter, same clock as Frame.stamp
    assert 0 <= ack.latency < 1.0


class _Link:
    """Stands in for LegLink: records gestures, takes time over them."""

    def __init__(self, dwell: float = 0.03) -> None:
        self.gestures: list[str] = []
        self._dwell = dwell

    def strike(self, dwell: float = 0.25) -> None:
        self.gestures.append("strike")
        time.sleep(self._dwell)

    def send(self, command: str) -> None:
        self.gestures.append(command)


class TestLegServiceKeepsTheControlThreadMoving:
    """`fire()` is called from a policy tick, so it may never block.

    `LegLink.strike()` slaps, sleeps for `dwell`, then homes -- 250 ms
    and two serial round trips against a control loop that ticks every
    10 ms. Doing that inline would stall the arm mid-swing, and a policy
    tick that raises reaches `RobotApp._control_loop`, which answers a
    failed tick by e-stopping. The arm freezing directly above the hand
    it was aiming at is the failure this class exists to prevent.
    """

    def test_fire_returns_immediately(self):
        from tlod.leg import LegService

        svc = LegService(_Link(dwell=0.2), dwell=0.2)
        svc.start()
        try:
            t0 = time.perf_counter()
            assert svc.fire() is True
            elapsed = time.perf_counter() - t0
        finally:
            svc.stop()
        assert elapsed < 0.005, (
            f"fire() took {elapsed * 1e3:.1f} ms; the control loop ticks every 10")

    def test_a_second_request_is_dropped_not_queued(self):
        """A slap that lands two rounds late is wrong, not late.

        Queueing would have the leg still working through round three's
        gesture when round five resolved. Dropping is counted instead, so
        a session can say it is being asked to gesture faster than a
        250 ms gesture allows -- which is a pacing decision, not
        something the driver should paper over.
        """
        from tlod.leg import LegService

        link = _Link(dwell=0.15)
        svc = LegService(link, dwell=0.15)
        svc.start()
        try:
            assert svc.fire() is True
            dropped = sum(1 for _ in range(4) if svc.fire() is False)
            time.sleep(0.3)
        finally:
            svc.stop()
        assert dropped == 4
        assert link.gestures == ["strike"], "a dropped request must not arrive late"
        assert svc.stats.dropped == 4

    def test_a_dead_board_costs_a_gesture_and_not_the_game(self):
        from tlod.leg import LegService

        class Dead:
            def strike(self, dwell: float = 0.25) -> None:
                raise RuntimeError("port went away")

        svc = LegService(Dead())
        svc.start()
        try:
            svc.fire()
            time.sleep(0.15)
        finally:
            svc.stop()
        assert svc.stats.failed == 1
        assert svc.stats.fired == 0
        assert "port went away" in svc.report()


class TestTheLegWaitsForTheArmToGetOutOfTheWay:
    """The hatch the leg deploys through is behind the arm.

    So the two effectors are physically exclusive, and a gesture asked
    for while the arm is still in front of the hatch is a collision
    rather than a missed cue. `model.STOW` is what "out of the way"
    means and `ArmController.is_stowed` reads it off the encoders.
    """

    def test_nothing_moves_until_the_arm_is_stowed(self):
        from tlod.leg import LegService

        clear = {"now": False}
        link = _Link()
        svc = LegService(link, dwell=0.02, is_clear=lambda: clear["now"])
        svc.start()
        try:
            for _ in range(3):
                assert svc.fire() is False
            time.sleep(0.1)
            assert link.gestures == [], "the leg moved with the arm in the way"
            assert svc.stats.blocked == 3
            assert svc.stats.fired == 0

            clear["now"] = True
            assert svc.fire() is True
            time.sleep(0.15)
        finally:
            svc.stop()
        assert link.gestures == ["strike"]

    def test_an_unreadable_arm_counts_as_in_the_way(self):
        """Cannot prove it is clear, so it is not clear.

        The interlock guards a collision, so its failure mode has to be
        refusal. A bus read that throws must not read as permission.
        """
        from tlod.leg import LegService

        def boom() -> bool:
            raise OSError("servo bus went away")

        link = _Link()
        svc = LegService(link, is_clear=boom)
        svc.start()
        try:
            assert svc.fire() is False
            time.sleep(0.1)
        finally:
            svc.stop()
        assert link.gestures == []
        assert svc.stats.blocked == 1

    def test_blocked_and_dropped_are_reported_apart(self):
        """They mean opposite things and have opposite fixes.

        Dropped is the leg being asked faster than it can gesture.
        Blocked is the arm being where the leg needs to go. Collapsing
        them into one number would hide an interlock that never opens
        behind what looks like an over-eager caller.
        """
        from tlod.leg import LegService

        svc = LegService(_Link(), is_clear=lambda: False)
        svc.start()
        try:
            svc.fire()
        finally:
            svc.stop()
        assert "blocked" in svc.report()
        assert "not stowed" in svc.report()
        assert svc.stats.dropped == 0


class TestTheStowPose:
    def test_it_is_a_pose_the_arm_can_actually_be_sent_to(self):
        """Measured by hand with torque off, which is not the same thing.

        The rig's own measurement had `shoulder_lift` at -1.861, which is
        6.6 degrees past the URDF limit -- reachable by pushing the arm
        there, not reachable by commanding it. A stow pose that
        `clamp_to_limits` silently clips is a stow pose that does not
        happen, and the hatch would open into an arm still in the way.
        """
        import numpy as np

        from tlod.arm import model

        assert np.allclose(model.clamp_to_limits(model.STOW.copy()), model.STOW), (
            "STOW is outside the joint limits and would be silently clipped")

    def test_it_folds_the_arm_back_out_of_the_workspace(self):
        """It has to be somewhere the game never reaches, or the check is noise."""
        import numpy as np

        from tlod.arm import model

        stowed = model.tool_pose(model.STOW).xyz()
        radius = float(np.hypot(stowed[0], stowed[1]))
        assert radius < 0.20, f"stowed at {radius * 1e3:.0f} mm reach, still over the table"
        assert stowed[2] > 0.20, f"stowed at {stowed[2] * 1e3:.0f} mm, not lifted clear"


class _Recorder(LegLink):
    """A LegLink whose `send` records instead of talking to a board."""

    def __init__(self) -> None:
        super().__init__(transport=object())
        self.seen: list[str] = []

    def send(self, command: str, timeout: float | None = None) -> Ack:
        self.seen.append(command)
        return Ack(command=command, line=ACKS.get(command, ""), stamp=0.0,
                   latency=0.0, expected=True)


class TestTheSequencingLivesInTheSketch:
    """`open` and `close` are whole gestures on the board now.

    They were not always. `close` used to drive the door to 145 without
    touching the leg, so closing while the leg was down shut the door
    onto it and held it there -- a hobby servo stalled against a
    mechanical stop for as long as the board had power. And `open` waited
    only 200 ms before dropping the leg, which was not long enough for
    the door to finish swinging, so the leg hit it.

    Both are fixed in the .ino, which is the right place: the guard then
    holds however the board is driven, including from a serial terminal
    that has never heard of this file. The driver's job is to not undo it
    by sending the same moves twice.
    """

    def test_retract_is_just_close(self):
        link = _Recorder()
        link.retract()
        assert link.seen == ["close"], (
            f"retract sent {link.seen}; the sketch already homes before closing")

    def test_deploy_is_just_open(self):
        link = _Recorder()
        link.deploy()
        assert link.seen == ["open"], (
            f"deploy sent {link.seen}; the sketch already waits for the door")

    def test_strike_puts_everything_away_behind_it(self):
        """A gesture has to end somewhere the next one can start from."""
        link = _Recorder()
        link.strike(dwell=0.0)
        assert link.seen[-1] == "close", (
            f"strike ended on {link.seen[-1]!r}, leaving the hatch open")

    def test_no_sequence_reaches_for_the_bench_commands(self):
        """`slap` and `home` move servo 1 without regard for the door.

        They exist to check that servo answers at all. A gesture built
        from them can drive the leg into a shut hatch, so no sequence
        here should name one.
        """
        link = _Recorder()
        link.deploy()
        link.strike(dwell=0.0)
        link.retract()
        assert set(link.seen) <= {"open", "close"}, link.seen


class _FakeSerialModule:
    """Enough of pyserial to see how the port gets opened."""

    def __init__(self, allow_preopen_dtr: bool = True) -> None:
        self.allow = allow_preopen_dtr
        self.opened: list[dict] = []
        module = self

        class Serial:
            def __init__(self, port=None, baudrate=9600, timeout=None):
                self.port, self.baudrate, self.timeout = port, baudrate, timeout
                self._dtr = None
                self.is_open = False
                if port is not None:            # the reset-y constructor form
                    module.opened.append({"port": port, "dtr": None, "form": "direct"})
                    self.is_open = True

            @property
            def dtr(self):
                return self._dtr

            @dtr.setter
            def dtr(self, value):
                if not module.allow:
                    raise OSError("this backend will not set DTR before open")
                self._dtr = value

            rts = dtr

            def open(self):
                module.opened.append({"port": self.port, "dtr": self._dtr, "form": "deferred"})
                self.is_open = True

            def reset_input_buffer(self):
                pass

        self.Serial = Serial


class TestOpeningThePortDoesNotRebootTheBoard:
    """DTR is wired to RESET on an Arduino, and `setup()` moves both servos.

    So a plain open restarts the sketch, which writes 90 to servo 0 and
    servo 1 -- the door swings open and the leg parks mid-travel -- and
    only then does the command go out. On the rig that read as a five
    second delay and every gesture happening in two steps, from
    positions nothing had asked for.
    """

    def test_dtr_is_held_low_before_the_port_opens(self):
        from tlod.leg import _open_without_resetting

        fake = _FakeSerialModule()
        _open_without_resetting(fake, "/dev/ttyUSB0", 9600)

        assert len(fake.opened) == 1
        opened = fake.opened[0]
        assert opened["form"] == "deferred", (
            "used the constructor that asserts DTR, which resets the board")
        assert opened["dtr"] is False, "DTR was not held low, so the sketch restarts"

    def test_a_backend_that_refuses_still_gets_a_port(self):
        """Not every driver honours a pre-open DTR, and the leg still has to work.

        CH340 and FTDI parts differ, and a board with the reset-enable
        trace cut does not care either way. Refusing to talk to the leg
        because the nicety failed would trade a cosmetic problem for a
        total one.
        """
        from tlod.leg import _open_without_resetting

        fake = _FakeSerialModule(allow_preopen_dtr=False)
        ser = _open_without_resetting(fake, "/dev/ttyUSB0", 9600)
        assert ser is not None
        assert fake.opened, "fell back to nothing at all"

    def test_probing_uses_the_same_open(self):
        """`find_leg_port` walks every candidate, so a resetting probe
        reboots the board more than once before a command goes out."""
        import inspect

        from tlod import leg

        body = inspect.getsource(leg.heartbeat_answers)
        assert "_open_without_resetting" in body
        assert "serial.Serial(" not in body


class TestTheDriverIsFasterThanTheMechanism:
    """An ack is the board taking a command, not a servo arriving.

    There is no feedback on this board: `servo.write()` sets a target and
    returns, so neither the sketch nor this file knows when the leg has
    finished moving. Every wait in this protocol is a guess at travel
    time, and a guess that is too short reverses the leg mid-swing.

    Typing the same commands into a serial monitor never shows it,
    because a human takes seconds between them. That is worth a test
    rather than a comment: the failure looks like the hardware
    misbehaving and is entirely in the timing here.
    """

    def test_the_dwell_outlasts_the_boards_own_door_wait(self):
        """`open` blocks the sketch for ~700 ms before it even acks.

        attachAt's delay(100), then delay(500) for the door, then
        attachAt again -- measured as a 720 ms ack and a heartbeat
        interval stretched from 500 ms to 722. The leg only *starts*
        moving at the end of that, so a dwell shorter than the leg's own
        travel closes the door on a leg still coming out.
        """
        import inspect

        from tlod.config import LegConfig
        from tlod.leg import LegLink

        board_blocks_for = 0.7          # measured on the rig
        default = inspect.signature(LegLink.strike).parameters["dwell"].default
        assert default > board_blocks_for - 0.5, (
            f"dwell {default}s leaves no room for the leg to travel after "
            f"open acks")
        assert LegConfig().strike_dwell == default, (
            "the config default and the code default disagree, so the same "
            "gesture behaves differently depending on which one is reached")

    def test_strike_is_open_then_close_with_the_wait_between(self):
        link = _Recorder()
        link.strike(dwell=0.0)
        assert link.seen == ["open", "close"]
