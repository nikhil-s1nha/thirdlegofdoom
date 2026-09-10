"""The NeoPixel eyes link.

The fake is a port of the .ino, including the two behaviours that make
this board awkward: it never speaks first, and it announces a mood only
when the mood actually changes. Tests that used a chatty stub would miss
both, and both are the reason the driver looks the way it does.
"""

import math
import threading
import time

import pytest

from tlod.eyes import (
    BAD_INPUT,
    EMOTIONS,
    FAR_THRESHOLD,
    NEAR_THRESHOLD,
    SCALE_CM,
    EyesError,
    EyesLink,
    EyesTimeout,
    emotion_for,
)


class FakeXiao:
    """The eyes sketch, as a serial-port-shaped object.

    `float_scanf=False` reproduces a SAMD21 whose sscanf was built
    without float support: every well-formed point comes back as bad
    input, which is exactly the failure this link has to be able to name.
    """

    def __init__(self, emotion="happy", float_scanf=True, frame_delay=0.0, mute=False):
        self.emotion = emotion
        self.float_scanf = float_scanf
        self.frame_delay = frame_delay      # the sketch's delay(20)
        self.mute = mute                    # a board that is not listening
        self.received: list[str] = []
        self._rx = bytearray()
        self._lock = threading.Lock()
        self.closed = False

    def _emit(self, line: str):
        with self._lock:
            self._rx.extend(line.encode() + b"\r\n")

    def _set_emotion(self, e):
        if self.emotion == e:
            return                          # setEmotion returns early: silence
        self.emotion = e
        self._emit(f"Emotion: {e.upper()}")

    # -- the serial API -----------------------------------------------------
    @property
    def in_waiting(self):
        with self._lock:
            return len(self._rx)

    def read(self, n=1):
        deadline = time.perf_counter() + 0.1
        while time.perf_counter() < deadline:
            with self._lock:
                if self._rx:
                    out = bytes(self._rx[:n])
                    del self._rx[:n]
                    return out
            time.sleep(0.002)
        return b""

    def write(self, data: bytes):
        line = data.decode().strip()
        self.received.append(line)
        if self.mute:
            return len(data)
        if self.frame_delay:
            # loop() gets to it on the next frame; write() itself returns
            # at once, exactly as a buffered serial write does.
            threading.Timer(self.frame_delay, self._handle, args=(line,)).start()
        else:
            self._handle(line)
        return len(data)

    def _handle(self, line: str):
        if not line:
            return
        if len(line) == 1 and line in "hca":
            self._set_emotion({"h": "happy", "c": "concentrated", "a": "angry"}[line])
            return
        parts = line.split(",")
        if len(parts) != 3 or not self.float_scanf:
            self._emit(BAD_INPUT)
            return
        try:
            x, y, z = (float(v) for v in parts)
        except ValueError:
            self._emit(BAD_INPUT)
            return
        distance = math.sqrt(x * x + y * y + z * z)
        self._emit(f"Distance: {distance:.2f}")
        self._set_emotion(emotion_for(distance))

    def flush(self):
        pass

    def reset_input_buffer(self):
        with self._lock:
            self._rx.clear()

    def close(self):
        self.closed = True


def make(board=None, **kw):
    board = board or FakeXiao()
    link = EyesLink(port="fake", transport=board, **kw)
    link.connect()
    return link, board


# -- the thresholds ---------------------------------------------------------
def test_emotion_thresholds_match_the_sketch():
    assert emotion_for(NEAR_THRESHOLD - 1) == "angry"
    assert emotion_for(FAR_THRESHOLD + 1) == "happy"
    assert emotion_for((NEAR_THRESHOLD + FAR_THRESHOLD) / 2) == "concentrated"
    # The boundaries are strict on both sides in the .ino.
    assert emotion_for(NEAR_THRESHOLD) == "concentrated"
    assert emotion_for(FAR_THRESHOLD) == "concentrated"


def test_metres_unscaled_would_pin_the_eyes_at_angry():
    """Why scale exists: the arm's whole reach is under the near threshold."""
    for reach in (0.08, 0.25, 0.40):
        assert emotion_for(reach) == "angry"


def test_centimetres_put_the_workspace_across_a_real_boundary():
    assert emotion_for(0.10 * SCALE_CM) == "angry"          # 10 cm
    assert emotion_for(0.30 * SCALE_CM) == "concentrated"   # 30 cm


# -- it never speaks first --------------------------------------------------
def test_ping_gets_an_answer_without_touching_the_pixels():
    link, board = make()
    try:
        assert link.ping()
        assert board.emotion == "happy"     # unchanged by the probe
    finally:
        link.disconnect()


def test_a_mute_board_fails_to_connect():
    board = FakeXiao(mute=True)
    link = EyesLink(port="fake", transport=board, reply_timeout=0.2)
    with pytest.raises(EyesError, match="nothing answered"):
        link.connect()
    assert not link.connected


def test_a_mute_board_pings_false():
    board = FakeXiao(mute=True)
    link = EyesLink(port="fake", transport=board, reply_timeout=0.2)
    link.connect(verify=False)
    try:
        assert not link.ping()
    finally:
        link.disconnect()


# -- moods ------------------------------------------------------------------
def test_setting_a_new_mood_is_announced():
    link, board = make(FakeXiao(emotion="happy"))
    try:
        ack = link.set_emotion("angry")
        assert ack.changed
        assert ack.emotion == "angry"
        assert link.emotion == "angry"
        assert board.emotion == "angry"
    finally:
        link.disconnect()


def test_setting_the_mood_it_is_already_in_is_silent():
    """setEmotion() returns early. Silence is the sketch working, not failing."""
    link, board = make(FakeXiao(emotion="happy"), reply_timeout=0.2)
    try:
        ack = link.set_emotion("happy")
        assert not ack.changed
        assert link.emotion == "happy"      # still known: it is there either way
    finally:
        link.disconnect()


def test_every_mood_round_trips():
    link, board = make(FakeXiao(emotion="angry"))
    try:
        for mood in EMOTIONS:
            link.set_emotion(mood)
            assert board.emotion == mood
    finally:
        link.disconnect()


def test_unknown_mood_is_refused_before_it_reaches_the_board():
    link, board = make()
    try:
        with pytest.raises(ValueError):
            link.set_emotion("smug")
        assert board.received == ["?"]      # only the connect ping
    finally:
        link.disconnect()


# -- points -----------------------------------------------------------------
def test_a_point_comes_back_with_the_boards_own_arithmetic():
    link, board = make(FakeXiao(emotion="happy"))
    try:
        reply = link.send_point(0.25, 0.0, 0.10)
        assert reply.agrees
        assert reply.distance == pytest.approx(math.sqrt(25.0**2 + 10.0**2), abs=0.02)
        assert board.received[-1] == "25.0,0.0,10.0"
    finally:
        link.disconnect()


def test_a_point_reports_the_mood_it_caused():
    link, board = make(FakeXiao(emotion="happy"))
    try:
        reply = link.send_point(0.10, 0.0, 0.0)     # 10 cm -> angry
        assert reply.emotion == "angry"
        assert reply.changed                         # the board said so
        assert board.emotion == "angry"
    finally:
        link.disconnect()


def test_a_point_that_changes_nothing_still_reports_the_mood():
    """Silence is the sketch's normal case, not a missing answer: the mood
    is derived from the distance the board itself sent back."""
    link, board = make(FakeXiao(emotion="angry"))
    try:
        reply = link.send_point(0.10, 0.0, 0.0)     # already angry
        assert reply.emotion == "angry"
        assert not reply.changed                     # nothing was announced
        assert reply.agrees                          # the distance still came back
    finally:
        link.disconnect()


def test_scale_is_applied_on_the_way_out():
    link, board = make(scale=1.0)
    try:
        link.send_point(0.25, 0.0, 0.10)
        assert board.received[-1] == "0.2,0.0,0.1"
    finally:
        link.disconnect()


def test_a_board_without_scanf_float_is_named_not_guessed():
    """The SAMD21 failure this link exists to catch."""
    link, board = make(FakeXiao(float_scanf=False))
    try:
        with pytest.raises(EyesError, match="float support"):
            link.send_point(0.25, 0.0, 0.10)
    finally:
        link.disconnect()


def test_a_disagreeing_distance_is_visible_to_the_caller():
    board = FakeXiao()
    original = board.write

    def lying_write(data):
        line = data.decode().strip()
        if "," in line:
            board.received.append(line)
            board._emit("Distance: 999.00")
            return len(data)
        return original(data)

    board.write = lying_write
    link = EyesLink(port="fake", transport=board)
    link.connect(verify=False)
    try:
        reply = link.send_point(0.25, 0.0, 0.10)
        assert not reply.agrees
    finally:
        link.disconnect()


# -- the self test ----------------------------------------------------------
def test_selftest_passes_against_a_working_board():
    link, board = make(FakeXiao(emotion="concentrated"))
    try:
        results = link.selftest()
        assert all(ok for _, ok, _ in results), results
        assert len(results) == 6            # ping + 4 moods + a point
    finally:
        link.disconnect()


def test_selftest_passes_whatever_mood_it_starts_in():
    """h -> c -> a -> h is at least three real transitions from anywhere."""
    for start in EMOTIONS:
        link, board = make(FakeXiao(emotion=start))
        try:
            assert all(ok for _, ok, _ in link.selftest()), start
        finally:
            link.disconnect()


def test_selftest_fails_a_board_that_takes_bytes_but_does_not_animate():
    """The case a write-only link cannot see: it accepts everything and
    changes nothing."""
    board = FakeXiao()

    def deaf_write(data):
        line = data.decode().strip()
        board.received.append(line)
        if line == "?":
            board._emit(BAD_INPUT)          # still answers the ping
        return len(data)                    # ... but never changes mood

    board.write = deaf_write
    link = EyesLink(port="fake", transport=board, reply_timeout=0.15)
    link.connect()
    try:
        results = link.selftest()
        assert results[0][1], "the ping should still pass"
        assert not all(ok for _, ok, _ in results)
        assert any("did not change" in note for _, _, note in results)
    finally:
        link.disconnect()


def test_selftest_stops_early_when_nothing_answers():
    board = FakeXiao(mute=True)
    link = EyesLink(port="fake", transport=board, reply_timeout=0.15)
    link.connect(verify=False)
    try:
        results = link.selftest()
        assert results == [("answers", False, "no reply to `?`")]
    finally:
        link.disconnect()


# -- housekeeping -----------------------------------------------------------
def test_replies_are_drained_so_the_board_does_not_stall():
    """Nothing may accumulate: the board blocks once the host stops reading."""
    link, board = make()
    try:
        for _ in range(20):
            link.send_point(0.25, 0.0, 0.10)
        assert link._replies.qsize() <= 1
        assert board.in_waiting == 0
    finally:
        link.disconnect()


def test_an_unchanged_mood_costs_no_waiting():
    """The common case while streaming: waiting for an announcement that is
    never coming would blow the update budget on every frame."""
    link, board = make(FakeXiao(emotion="angry"))
    try:
        link.send_point(0.10, 0.0, 0.0)          # already angry
        start = time.perf_counter()
        for _ in range(5):
            link.send_point(0.10, 0.0, 0.0)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.15, f"5 unchanged points took {elapsed:.3f}s"
    finally:
        link.disconnect()


def test_a_slow_board_is_waited_for():
    """delay(20) plus animation time, so replies are never immediate."""
    link, board = make(FakeXiao(frame_delay=0.05))
    try:
        reply = link.send_point(0.25, 0.0, 0.10)
        assert reply.agrees
        assert reply.latency >= 0.05
    finally:
        link.disconnect()


def test_a_board_that_stops_answering_points_times_out():
    board = FakeXiao()
    link = EyesLink(port="fake", transport=board, reply_timeout=0.15)
    link.connect()
    board.mute = True
    try:
        with pytest.raises(EyesTimeout):
            link.send_point(0.25, 0.0, 0.10)
    finally:
        link.disconnect()


def test_context_manager_connects_and_disconnects():
    board = FakeXiao()
    with EyesLink(port="fake", transport=board) as link:
        assert link.connected
        assert link.ping()
    assert not link.connected


def test_disconnect_leaves_a_caller_supplied_port_open():
    board = FakeXiao()
    link = EyesLink(port="fake", transport=board)
    link.connect()
    link.disconnect()
    assert not board.closed


def test_stats_report_what_went_wrong():
    link, board = make(FakeXiao(float_scanf=False))
    try:
        with pytest.raises(EyesError):
            link.send_point(0.25, 0.0, 0.10)
        st = link.stats()
        assert st["bad_input"] >= 1
        assert st["sent"] >= 2
    finally:
        link.disconnect()


def test_agreement_is_judged_on_what_went_on_the_wire():
    """At one decimal the rounding alone can move the distance further than
    the tolerance, so `agrees` must compare against the rounded line, not
    the full-precision position it came from."""
    link, board = make(FakeXiao(), precision=1)
    try:
        # 0.2567 m -> "25.7", and sqrt of the unrounded value is 0.03 away.
        reply = link.send_point(0.2567, 0.0, 0.1043)
        assert board.received[-1] == "25.7,0.0,10.4"
        assert reply.agrees, (reply.distance, reply.expected)
    finally:
        link.disconnect()
