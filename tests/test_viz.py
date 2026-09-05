"""Overlay rendering.

The overlay is a calibration check as much as a display: if the projected
arm does not land on the arm in the picture, the extrinsics are wrong.
These tests keep it from silently drawing nothing.
"""

import numpy as np

from tlod.arm.controller import SafetyLimits
from tlod.arm.model import HOME, fk_all
from tlod.types import Pose
from tlod.vision.calibration import synthetic_projector
from tlod.viz.overlay import Overlay


def blank():
    return np.zeros((720, 1280, 3), np.uint8)


def ink(img):
    return int((img.sum(axis=2) > 0).sum())


def test_workspace_draws():
    img = blank()
    Overlay(synthetic_projector(), SafetyLimits()).draw_workspace(img)
    assert ink(img) > 500


def test_arm_draws_and_matches_kinematics():
    proj = synthetic_projector()
    img = blank()
    Overlay(proj, SafetyLimits()).draw_arm(img, np.concatenate([HOME, [0.0]]))
    assert ink(img) > 200
    # The rendered tip must be at the projected TCP, not somewhere plausible.
    tip = fk_all(HOME)[-1][:3, 3]
    u, v = proj.project(tip)
    patch = img[int(v) - 12:int(v) + 12, int(u) - 12:int(u) + 12]
    assert patch.sum() > 0, "no ink where the TCP projects"


def test_hand_and_prediction_draw():
    img = blank()
    o = Overlay(synthetic_projector(), SafetyLimits())
    o.draw_hand(img, np.array([0.22, 0.05, 0.12]), label="hand")
    o.draw_prediction(img, np.array([0.22, 0.05, 0.12]), np.array([0.26, 0.02, 0.14]))
    assert ink(img) > 100


def test_offscreen_geometry_does_not_crash():
    """Points behind or far outside the camera must be skipped, not drawn
    at integer-overflow coordinates."""
    img = blank()
    o = Overlay(synthetic_projector(), SafetyLimits())
    o.draw_hand(img, np.array([0.0, 0.0, 50.0]))
    o.draw_hand(img, np.array([-100.0, 0.0, 0.0]))
    o.draw_target(img, Pose(1e6, 1e6, 1e6))


def test_hud_and_banner_draw():
    img = blank()
    o = Overlay(synthetic_projector(), SafetyLimits())
    o.draw_hud(img, ["one", "two"])
    o.draw_banner(img, "READY")
    assert ink(img) > 100
