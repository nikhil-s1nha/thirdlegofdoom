"""Threading primitives and loop timing."""

import time

import numpy as np
import pytest

from tlod.config import Config
from tlod.runtime.loop import RateLoop, Timing
from tlod.runtime.signal import Latest


def test_latest_last_write_wins():
    s = Latest[int]()
    for i in range(5):
        s.set(i)
    assert s.get() == 4


def test_latest_is_empty_initially():
    assert Latest[int]().get() is None


def test_get_fresh_rejects_stale_values():
    s = Latest[int]()
    s.set(1)
    assert s.get_fresh(1.0) == 1
    time.sleep(0.03)
    assert s.get_fresh(0.005) is None, "stale data must not be served"


def test_wait_returns_on_write():
    import threading

    s = Latest[int]()
    threading.Timer(0.02, lambda: s.set(7)).start()
    assert s.wait(timeout=1.0) == 7


def test_wait_times_out():
    assert Latest[int]().wait(timeout=0.01) is None


def test_rate_loop_holds_its_rate():
    loop = RateLoop(200.0, "t")
    t0 = time.perf_counter()
    for _ in range(100):
        loop.tick()
    elapsed = time.perf_counter() - t0
    assert 0.45 < elapsed < 0.62, f"100 ticks at 200 Hz took {elapsed:.3f}s"


def test_rate_loop_counts_overruns():
    loop = RateLoop(1000.0, "t")
    for _ in range(5):
        loop.tick()
        time.sleep(0.004)  # deliberately overshoot the 1 ms period
    assert loop.overruns > 0
    assert loop.overrun_rate > 0


def test_rate_loop_does_not_spiral_when_far_behind():
    """After a long stall it must not burst through a queue of catch-up
    iterations, which on a robot means a sudden flurry of commands."""
    loop = RateLoop(100.0, "t")
    loop.tick()
    time.sleep(0.3)
    t0 = time.perf_counter()
    for _ in range(3):
        loop.tick()
    assert time.perf_counter() - t0 < 0.1


def test_timing_percentiles():
    t = Timing("x")
    for v in np.linspace(0.001, 0.100, 100):
        t.add(float(v))
    assert t.p50_ms < t.p95_ms <= t.max_ms
    assert 0 < t.mean_ms < 101
    assert "x" in t.summary()


def test_timing_is_empty_safe():
    t = Timing("x")
    assert t.mean_ms == 0.0 and t.p95_ms == 0.0 and t.max_ms == 0.0


def test_config_round_trip(tmp_path):
    cfg = Config()
    cfg.runtime.control_hz = 42.0
    path = tmp_path / "c.yaml"
    cfg.save(path)
    assert Config.load(path).runtime.control_hz == 42.0


def test_config_rejects_unknown_keys():
    """A silently ignored typo is how a safety limit fails to apply."""
    with pytest.raises(ValueError, match="unknown keys"):
        Config.from_dict({"safety": {"max_sped": 9.0}})


def test_config_defaults_to_simulation():
    cfg = Config()
    assert cfg.arm.backend == "mock"
    assert cfg.camera.source == "mock"


def test_config_overrides_do_not_mutate_the_original():
    cfg = Config()
    other = cfg.with_overrides(runtime={"control_hz": 5.0})
    assert other.runtime.control_hz == 5.0
    assert cfg.runtime.control_hz == 100.0
