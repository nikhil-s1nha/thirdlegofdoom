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


def test_home_acks_with_an_empty_line_and_that_still_counts(leg):
    """`Serial.println("")`. A blank line is the reply, not a lost one."""
    link, _ = leg
    ack = link.home()
    assert ack.line == ""
    assert ack.expected


def test_strike_slaps_then_homes(leg):
    """`slap` alone leaves the paddle down; a second one would do nothing."""
    link, board = leg
    ack = link.strike(dwell=0.01)
    assert board.received == ["slap", "home"]
    assert ack.command == "slap"        # the stamp a hit test wants


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
    assert ack.line == "s"


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
        assert ack.line == "s"
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
        assert "s" in seen
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
        assert link.slap().line == "s"
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
        assert link.slap().line == "s"
    assert not link.connected
    board.close()


def test_ack_types_are_what_the_rest_of_the_project_expects(leg):
    link, _ = leg
    ack = link.slap()
    assert isinstance(ack, Ack)
    assert ack.stamp > 0                # perf_counter, same clock as Frame.stamp
    assert 0 <= ack.latency < 1.0
