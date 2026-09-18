"""Feetech register conventions.

These are the conversions between physical quantities and the STS3215
control table. They are pure functions, and they are pinned here because
getting one backwards is silent: the arm still moves, just with four times
the current spike you thought you had asked for.
"""

import numpy as np
import pytest

from tlod.arm.feetech import (
    AMPS_PER_CURRENT_COUNT, RAD_S2_PER_ACC_COUNT, Calibration, acc_counts, decode_errors,
)
from tlod.types import NUM_JOINTS


class TestGoalAcceleration:
    """Register 41 is an acceleration, not a smoothness dial.

    The whole point of `acc_counts` is that callers state rad/s^2 and
    cannot get the direction wrong. These tests are the record of which
    direction is right.
    """

    def test_a_bigger_number_is_a_harsher_ramp(self):
        assert acc_counts(20.0) > acc_counts(3.0)

    def test_the_register_unit_is_100_steps_per_second_squared(self):
        # 100 encoder steps/s^2, and 4096 steps to a revolution.
        assert RAD_S2_PER_ACC_COUNT == pytest.approx(100 * 2 * np.pi / 4096)
        assert RAD_S2_PER_ACC_COUNT == pytest.approx(0.1534, abs=1e-4)

    def test_round_trips_through_the_register(self):
        for rad_s2 in (1.0, 3.0, 9.2, 20.0, 35.0):
            counts = acc_counts(rad_s2)
            assert counts * RAD_S2_PER_ACC_COUNT == pytest.approx(rad_s2, abs=0.16)

    def test_the_old_default_of_60_counts_was_about_nine_rad_per_s2(self):
        assert acc_counts(9.2) == 60

    def test_maxing_the_register_asks_for_four_times_that(self):
        """The trap. Winding this to 254 to 'smooth' a brownout in fact
        quadruples the acceleration, and so the current spike, versus the
        default of 60 it replaces."""
        assert acc_counts(254 * RAD_S2_PER_ACC_COUNT) == 254
        assert 254 * RAD_S2_PER_ACC_COUNT == pytest.approx(38.97, abs=0.01)
        assert (254 * RAD_S2_PER_ACC_COUNT) / (60 * RAD_S2_PER_ACC_COUNT) > 4.0

    def test_never_emits_zero(self):
        """0 does not mean 'gentlest'. It disables the ramp entirely and
        gives maximum acceleration, so it must never be reached by asking
        for a very small one."""
        assert acc_counts(0.0) == 1
        assert acc_counts(-5.0) == 1

    def test_clamps_to_the_register_range(self):
        assert acc_counts(1e6) == 254


class TestErrorStatus:
    def test_decodes_undervoltage(self):
        assert decode_errors(0x01) == ["voltage"]

    def test_decodes_nothing_when_healthy(self):
        assert decode_errors(0x00) == []

    def test_decodes_several_at_once(self):
        assert set(decode_errors(0x01 | 0x20)) == {"voltage", "overload"}


def test_current_register_scale():
    """Present_Current counts 6.5 mA each; the servo's own over-current
    trip is at 2 A."""
    assert AMPS_PER_CURRENT_COUNT == 0.0065
    assert 308 * AMPS_PER_CURRENT_COUNT == pytest.approx(2.0, abs=0.01)


class TestCalibration:
    def test_identity_round_trip(self):
        calib = Calibration()
        q = np.linspace(-1.0, 1.0, NUM_JOINTS)
        assert np.allclose(calib.to_rad(calib.to_counts(q)), q, atol=2e-3)

    def test_sign_inverts_direction(self):
        sign = np.ones(NUM_JOINTS)
        sign[1] = -1.0
        calib = Calibration(sign=sign)
        counts = calib.to_counts(np.full(NUM_JOINTS, 0.5))
        assert counts[0] > 2048
        assert counts[1] < 2048

    def test_counts_stay_inside_the_encoder_range(self):
        counts = Calibration().to_counts(np.full(NUM_JOINTS, 100.0))
        assert counts.min() >= 0
        assert counts.max() <= 4095

    def test_reads_a_lerobot_calibration(self):
        from tlod.types import JOINT_NAMES

        data = {n: {"homing_offset": 100, "drive_mode": 0} for n in JOINT_NAMES}
        data["elbow_flex"]["drive_mode"] = 1
        calib = Calibration.from_lerobot(data)
        assert calib.center[0] == 2048 - 100
        assert calib.sign[JOINT_NAMES.index("elbow_flex")] == -1.0
        assert calib.sign[0] == 1.0


class TestConnectAsksBeforeItEnergises:
    """connect() must hear from a servo before it turns torque on.

    The failure this pins: opening the port and setting the baud rate are
    local to the USB adapter, which is powered from USB, so both succeed
    with the arm's 12 V supply switched off. connect() then wrote goal
    acceleration, speed, torque limit and torque-enable to six servos --
    discarding every return value -- and logged "connected to 6 servos",
    a count of the configured id list rather than of anything that
    answered. The first verified transaction was the sync read one layer
    up in ArmController.start(), so the traceback always pointed at
    read() rather than at the dead rail.
    """

    def _arm(self, answers, errors=None):
        from tlod.arm.feetech import FeetechArm

        arm = FeetechArm(port="/dev/null")

        class Packet:
            def read1ByteTxRx(self, port, mid, addr):
                if mid not in answers:
                    return 0, -1, 0
                return (errors or {}).get(mid, 0), 0, 0

        arm._packet_handler = Packet()
        arm._port_handler = object()
        return arm

    def test_a_silent_bus_is_reported_as_silent(self):
        arm = self._arm(answers=())
        answered, faults = arm._survey()
        assert answered == [] and faults == []

    def test_a_healthy_bus_answers_with_every_id(self):
        arm = self._arm(answers=(1, 2, 3, 4, 5, 6))
        answered, faults = arm._survey()
        assert answered == [1, 2, 3, 4, 5, 6] and faults == []

    def test_a_latched_fault_is_named_with_its_servo(self):
        arm = self._arm(answers=(1, 2, 3, 4, 5, 6), errors={3: 0x01, 5: 0x04})
        _, faults = arm._survey()
        assert faults == ["3:voltage", "5:overheat"]

    def test_a_dead_rail_says_so_and_names_the_switch(self):
        arm = self._arm(answers=())
        msg = arm._cannot_connect(list(arm.motor_ids), [])
        assert "no servo on /dev/null answered" in msg
        assert "12 V switch" in msg
        assert "power cycled" in msg

    def test_a_partial_chain_points_at_the_daisy_chain(self):
        arm = self._arm(answers=(1, 2, 3))
        msg = arm._cannot_connect([4, 5, 6], [])
        assert "servo 4, 5, 6 did not answer" in msg
        assert "daisy chain" in msg

    def test_a_survey_that_throws_does_not_replace_the_failure(self):
        """A failure to explain a failure must not become the failure."""
        from tlod.arm.feetech import FeetechArm

        arm = FeetechArm(port="/dev/null")

        class Boom:
            def read1ByteTxRx(self, *a):
                raise RuntimeError("bus is on fire")

        arm._packet_handler = Boom()
        arm._port_handler = object()
        assert arm._survey() == ([], [])


def _fake_comports(monkeypatch, entries):
    """Install a stub `serial.tools.list_ports` describing `entries`.

    pyserial is an optional extra, so the tests cannot assume it is
    importable -- and the interesting cases here are about what the port
    list *says*, which no amount of real hardware would make reproducible
    anyway.
    """
    import sys
    import types

    serial = types.ModuleType("serial")
    tools = types.ModuleType("serial.tools")
    lp = types.ModuleType("serial.tools.list_ports")
    lp.comports = lambda: entries
    tools.list_ports = lp
    serial.tools = tools
    monkeypatch.setitem(sys.modules, "serial", serial)
    monkeypatch.setitem(sys.modules, "serial.tools", tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", lp)


def _port(device, description="", vid=None, pid=None, serial_number=""):
    import types

    return types.SimpleNamespace(device=device, description=description,
                                 manufacturer="", serial_number=serial_number,
                                 vid=vid, pid=pid)


def test_describe_ports_carries_enough_to_tell_two_boards_apart(monkeypatch):
    """Two boards on USB is the normal state once the eyes are plugged in.

    `['/dev/ttyACM0', '/dev/ttyACM1']` names them without distinguishing
    them, and the numbering follows enumeration order, so it is not even
    stable across a replug. The USB ids are.
    """
    from tlod.arm import feetech

    _fake_comports(monkeypatch, [
        _port("/dev/ttyACM0", "Seeed XIAO M0", 0x2886, 0x802F, "A1B2"),
        _port("/dev/ttyACM1", "USB Single Serial", 0x1A86, 0x55D3),
    ])
    got = feetech.describe_ports()
    assert [d["device"] for d in got] == ["/dev/ttyACM0", "/dev/ttyACM1"]
    assert got[0]["usb_id"] == "2886:802f"
    assert got[1]["usb_id"] == "1a86:55d3"
    assert got[0]["usb_id"] != got[1]["usb_id"], "nothing here separates the boards"
    assert got[0]["description"] == "Seeed XIAO M0"


def test_describe_ports_survives_a_descriptor_with_nothing_in_it(monkeypatch):
    """Plenty of adapters report no vid, no serial and an empty string for
    a description. That is a thin answer, not a crash."""
    from tlod.arm import feetech

    _fake_comports(monkeypatch, [_port("/dev/ttyUSB0")])
    got = feetech.describe_ports()
    assert got == [{"device": "/dev/ttyUSB0", "description": "",
                    "manufacturer": "", "serial_number": "", "usb_id": ""}]


def test_describe_ports_ignores_ports_that_are_not_candidates(monkeypatch):
    """Bluetooth and console devices are serial ports and never the arm."""
    from tlod.arm import feetech

    _fake_comports(monkeypatch, [
        _port("/dev/ttyACM0", "arm"),
        _port("/dev/cu.Bluetooth-Incoming-Port", "bluetooth"),
    ])
    assert [d["device"] for d in feetech.describe_ports()] == ["/dev/ttyACM0"]


def test_servos_answer_is_zero_without_the_sdk(monkeypatch):
    """No SDK means no way to ask, which is not the same as an answer of
    'yes'. Autodetect keys off this, so guessing here would pick a port
    with an Arduino on it and energise nothing while claiming success."""
    import builtins

    from tlod.arm import feetech

    real = builtins.__import__

    def no_scs(name, *a, **k):
        if name == "scservo_sdk":
            raise ImportError("not installed")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_scs)
    assert feetech.servos_answer("/dev/ttyACM0") == 0
