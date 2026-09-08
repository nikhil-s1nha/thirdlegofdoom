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
