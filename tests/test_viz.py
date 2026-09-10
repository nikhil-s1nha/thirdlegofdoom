"""Overlay rendering, and the scoreboard page.

The overlay is a calibration check as much as a display: if the projected
arm does not land on the arm in the picture, the extrinsics are wrong.
These tests keep it from silently drawing nothing.

The scoreboard is the other end of the same job -- what a person watching
the game sees rather than what a person debugging it sees -- and its
tests are at the bottom of the file.
"""

import numpy as np

from tlod.arm.controller import SafetyLimits
from tlod.arm.model import HOME, fk_all
from tlod.types import Pose
from tlod.vision.calibration import synthetic_projector
from tlod.viz.overlay import Overlay
from tlod.viz.scoreboard import Scoreboard


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


# -- scoreboard ------------------------------------------------------------
#
# The score is read off a live game from an HTTP thread while the control
# loop is writing it, and the result word it flashes is erased ~600 ms
# after the round it belongs to. Both of those are easy to get subtly
# wrong in a way that only shows up as a flash of the wrong length in
# front of a person with their hand on the table, so they are pinned here.

class FakeGame:
    """The three attributes the scoreboard reads, and nothing else."""

    def __init__(self):
        from tlod.game.base import Score

        self.score = Score()
        self.last_result = ""
        self.state = "ready"
        self.running = True

    def resolve(self, result, robot=True):
        self.score.rounds += 1
        if robot:
            self.score.robot += 1
        else:
            self.score.human += 1
        self.last_result = result


def test_scoreboard_reports_the_score():
    game = FakeGame()
    game.resolve("HIT")
    snap = Scoreboard(game).snapshot()
    assert (snap["robot"], snap["human"], snap["rounds"]) == (1, 0, 1)
    assert snap["result"] == "HIT" and snap["result_round"] == 1


def test_result_is_latched_after_the_game_clears_it():
    """The whole reason the latch exists.

    `_state_settle` sets `last_result` back to "", so a page driven off
    the raw attribute would stop flashing partway through the round --
    and at an unpredictable point, because settle waits on a flourish.
    """
    game = FakeGame()
    board = Scoreboard(game)
    game.resolve("DODGED", robot=False)
    assert board.snapshot()["result"] == "DODGED"
    game.last_result = ""                      # what settle does
    snap = board.snapshot()
    assert snap["result"] == "DODGED", "the flash would have ended early"
    assert snap["result_round"] == 1


def test_the_same_result_twice_is_two_flashes():
    """The browser flashes on the round number, so it has to move."""
    game = FakeGame()
    board = Scoreboard(game)
    game.resolve("HIT")
    first = board.snapshot()["result_round"]
    game.last_result = ""
    board.snapshot()
    game.resolve("HIT")
    assert board.snapshot()["result_round"] != first


def test_a_poll_between_the_counter_and_the_word_latches_nothing():
    """`score.rounds += 1` happens one line before `last_result` is set.

    A poll that lands in between must not staple the previous round's
    verdict onto the new round's number: that would flash a stale word
    and then refuse to flash the real one, since the number has already
    been consumed.
    """
    game = FakeGame()
    board = Scoreboard(game)
    game.resolve("HIT")
    board.snapshot()
    game.last_result = ""
    game.score.rounds += 1                     # mid-update
    snap = board.snapshot()
    assert snap["result_round"] == 1, "latched a round that had no verdict yet"
    game.last_result = "DODGED"
    snap = board.snapshot()
    assert (snap["result"], snap["result_round"]) == ("DODGED", 2)


def test_age_restarts_with_each_result():
    import time

    game = FakeGame()
    board = Scoreboard(game)
    game.resolve("HIT")
    board.snapshot()
    time.sleep(0.02)
    assert board.snapshot()["age_ms"] >= 15
    game.resolve("DODGED", robot=False)
    assert board.snapshot()["age_ms"] < 15


def test_scoreboard_survives_a_policy_that_has_no_score():
    """A blank scoreboard beats a 500 on a board with no screen."""
    snap = Scoreboard(object()).snapshot()
    assert snap["rounds"] == 0 and snap["result"] == ""


def test_scoreboard_reads_a_real_game():
    """Guards the attribute names against a rename in handslap.py."""
    from tlod.game.contact import GeometricContactSensor
    from tlod.game.handslap import HandSlapGame

    game = HandSlapGame("normal", contact=GeometricContactSensor(), seed=1)
    game.score.robot, game.score.rounds = 2, 3
    game.last_result = "FLINCH"
    snap = Scoreboard(game).snapshot()
    assert snap["robot"] == 2 and snap["rounds"] == 3
    assert snap["result"] == "FLINCH" and snap["state"] == game.state


def test_page_names_every_result_word():
    """Each of the four verdicts needs a colour, or it flashes grey."""
    from tlod.viz import scoreboard

    page = scoreboard.PAGE.decode()
    for word in ("HIT", "DODGED", "FLINCH", "HELD"):
        assert word in page
    assert "score.json" in page


def test_page_makes_a_sound_for_every_result():
    """Each verdict needs its own noise, and none of them a file.

    The page is served from a board that may have no route off the LAN, so
    the sounds are synthesised: a filtered noise burst for the slap and
    oscillators for the rest. An `<audio src=...>` creeping in here would
    work on the laptop it was written on and 404 on the robot.
    """
    from tlod.viz import scoreboard

    page = scoreboard.PAGE.decode()
    assert "AudioContext" in page
    assert "createBufferSource" in page, "the slap needs noise, not a tone"
    for word in ("HIT", "DODGED", "FLINCH", "HELD"):
        assert f'=== "{word}"' in page, f"{word} has a colour but no sound"
    assert "<audio" not in page and ".mp3" not in page and ".wav" not in page
    # Silent until asked. Browsers refuse to play before a gesture anyway,
    # so a page that assumed otherwise would just be quietly broken.
    assert 'setSound(stored === "1")' in page


def test_scoreboard_serves_over_http():
    import json
    import socket
    import urllib.request

    from tlod.viz import scoreboard

    with socket.socket() as probe:              # ask the OS for a free port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    game = FakeGame()
    game.resolve("HELD", robot=False)
    server = scoreboard.serve(game, port)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as r:
            assert r.headers["Content-Type"].startswith("text/html")
            assert b"score.json" in r.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/score.json", timeout=3) as r:
            data = json.loads(r.read())
    finally:
        server.stop()
    assert data["human"] == 1 and data["result"] == "HELD"


def test_scoreboard_shares_a_port_with_the_preview():
    """Both flags on one port must not fight over the socket."""
    from tlod.vision.preview import PreviewServer
    from tlod.viz import scoreboard

    preview = PreviewServer(port=8097)
    same = scoreboard.serve(FakeGame(), 8097, server=preview)
    assert same is preview
    assert "/score.json" in preview.routes and "/" in preview.routes
