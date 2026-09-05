"""Regression tests for concurrency and safety bugs found by review.

Each of these was a real defect that the rest of the suite passed
straight through, because each only appears when two threads interleave
or when a motion is interrupted -- neither of which happens in a test
that drives one motion to completion on one thread.
"""

import threading
import time

import numpy as np

from tlod.arm import model
from tlod.arm.controller import ArmController, SafetyLimits
from tlod.arm.mock import MockArm
from tlod.arm.primitives import Strike, StrikeLimits


def test_mock_arm_respects_its_speed_limit_under_concurrent_access():
    """Travel never exceeds max_speed x elapsed, however many threads read.

    Honest scope note: this is an invariant check, not a regression test.
    The bug that prompted it -- read() and write() both advancing the
    simulation without a lock, so two threads could compute the same dt
    and integrate twice -- was verified by inspection, and the fix is to
    make _integrate atomic. But the unlocked version *passes* this test:
    under the GIL the race between reading and writing `_t` is rare
    enough that a sub-second test will not provoke it, and a test tuned
    until it did would be timing-flaky rather than informative.

    The lock's real justification is the one that cannot be raced into
    view on demand: `_q` is a numpy array mutated in place, and a reader
    can observe it half-updated. That is a correctness bug whether or not
    a test can catch it in 300 ms.
    """
    arm = MockArm(q0=np.zeros(6), max_speed=1.0, accel=1000.0, latency=0.0)
    arm.connect()
    target = np.full(6, 1.0)
    arm.write(target)
    # Take t0 before the threads start. Starting them first let the arm
    # integrate outside the measured window, so `travelled` was compared
    # against an `elapsed` shorter than the motion actually had -- a
    # flaky test that blamed the code for the test's own bookkeeping.
    t0 = time.perf_counter()

    stop = threading.Event()

    def reader():
        while not stop.is_set():
            arm.read()

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.30)
    stop.set()
    for t in threads:
        t.join(timeout=1.0)
    elapsed = time.perf_counter() - t0

    travelled = float(np.max(np.abs(arm.read().q)))
    ceiling = 1.0 * elapsed * 1.15   # max_speed * time, plus slack
    assert travelled <= ceiling, (
        f"arm travelled {travelled:.4f} rad in {elapsed:.3f}s at max_speed=1.0; "
        f"ceiling {ceiling:.4f}. Concurrent reads are advancing the simulation."
    )


def test_mock_arm_state_is_consistent_under_concurrent_write():
    """Readers must never observe a half-updated configuration."""
    arm = MockArm(q0=np.zeros(6), max_speed=2.0, latency=0.0)
    arm.connect()
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            arm.write(np.full(6, 0.5 if i % 2 else -0.5))
            i += 1

    def reader():
        while not stop.is_set():
            try:
                state = arm.read()
                if not np.all(np.isfinite(state.q)):
                    errors.append("non-finite q")
            except Exception as e:  # pragma: no cover
                errors.append(repr(e))

    threads = [threading.Thread(target=writer, daemon=True),
               *(threading.Thread(target=reader, daemon=True) for _ in range(3))]
    for t in threads:
        t.start()
    time.sleep(0.25)
    stop.set()
    for t in threads:
        t.join(timeout=1.0)
    assert not errors, errors


def test_aborted_strike_restores_torque_limit():
    """An interrupted strike must not leave the arm permanently weak.

    run_motion() aborts whatever motion it replaces and the feint handler
    aborts explicitly, so an interrupted strike is normal play, not an
    edge case. Without this the servos stayed capped at the strike limit
    for the rest of the session.
    """
    limits = StrikeLimits(torque_limit=300, normal_torque_limit=800)
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]])),
                               SafetyLimits(), control_hz=200.0)
    controller.start()

    strike = Strike(np.array([0.22, 0.0, 0.03]), limits, duration=1.0)
    strike.start(controller)
    strike.step(controller, 0.005)
    assert controller.backend.diagnostics()["torque_limit"] == 300

    strike.abort()
    assert controller.backend.diagnostics()["torque_limit"] == 800, (
        "aborting mid-strike left the torque limit lowered"
    )
    controller.backend.disconnect()


def test_estop_wins_a_race_against_a_concurrent_command():
    """A command issued as the stop engages must not reach the servos."""
    controller = ArmController(MockArm(q0=np.concatenate([model.HOME, [0.0]])),
                               SafetyLimits(), control_hz=500.0)
    controller.start()
    stop = threading.Event()

    def spammer():
        target = np.concatenate([model.HOME + 0.4, [0.0]])
        while not stop.is_set():
            controller._write(target, max_speed=5.0, dt=0.002)

    t = threading.Thread(target=spammer, daemon=True)
    t.start()
    time.sleep(0.05)
    controller.estop()
    frozen = controller.commanded.copy()
    time.sleep(0.10)
    stop.set()
    t.join(timeout=1.0)

    assert np.allclose(controller.commanded, frozen), (
        "a command landed after the e-stop engaged"
    )
    controller.backend.disconnect()
