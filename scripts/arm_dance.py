"""A short arm dance -- one joint moving at a time, in small stopped steps.

Built on the same current-management pattern as home_sequential.py, for
the same reason: this rig's 12V 2A supply has been measured collapsing
to ~3V under simultaneous multi-joint load (`tlod move --home`), and
even single-joint moves need to be chunked with real dead-stop pauses
rather than one continuous fast goto (see git history on this file's
sibling script for the full story).

Two things this script is deliberately careful about:

1. Every beat moves exactly one joint. A "dance" built from combined
   poses would recreate the simultaneous-current problem the sequential
   homing script exists to avoid.
2. `shoulder_lift` -- the joint actually lifting the arm's weight, and
   the one observed sagging under gravity during unrelated joints'
   current draw -- is never an intentional target of the dance itself.
   It only moves during the opening/closing home pass, same as every
   other joint, kept as short as possible.

Usage:
    python scripts/arm_dance.py -c configs/real_arm.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tlod.arm import model
from tlod.arm.controller import ArmController, SafetyLimits
from tlod.cli import build_arm
from tlod.config import Config
from tlod.types import JOINT_NAMES

# Home first, lightest/no-gravity joints before the heavier ones -- same
# ordering rationale as home_sequential.py.
HOME_ORDER = [4, 3, 0, 2, 1]  # wrist_roll, wrist_flex, shoulder_pan, elbow_flex, shoulder_lift

# The dance itself: (joint index, offset from HOME, label). Never index 1
# (shoulder_lift). Small amplitudes throughout -- this is a wiggle, not a
# swing, both because the supply can't take a big one and because a big
# fast motion near people is exactly what SafetyLimits.strike_speed is
# supposed to be reserved for, not a party trick.
DANCE = [
    (0, -0.4, "sway left"),
    (0, +0.4, "sway right"),
    (0, +0.0, "center"),
    (4, -0.8, "spin left"),
    (4, +0.8, "spin right"),
    (4, +0.0, "spin center"),
    (3, +0.3, "nod"),
    (3, -0.2, "nod back"),
    (2, -0.3, "little bow"),
]


def step_joint(controller: ArmController, i: int, target: float,
               steps: int, step_duration: float, step_settle: float) -> None:
    """Move joint `i` to an absolute target, in `steps` small waypoints,
    each followed by a real dead stop -- the stop is what lets the rail
    recover, not the motion speed itself."""
    base = controller.commanded[:5].copy()
    before = base[i]
    for step in range(1, steps + 1):
        waypoint = base.copy()
        waypoint[i] = before + (target - before) * (step / steps)
        controller.goto_joints(waypoint, duration=step_duration)
        print(".", end="", flush=True)
        time.sleep(step_settle)


def go_home(controller: ArmController, steps: int, step_duration: float,
            step_settle: float, joint_settle: float) -> None:
    for i in HOME_ORDER:
        print(f"  home {JOINT_NAMES[i]:<14} ", end="", flush=True)
        step_joint(controller, i, model.HOME[i], steps, step_duration, step_settle)
        print()
        time.sleep(joint_settle)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default=None, help="YAML config path")
    p.add_argument("--steps", type=int, default=6, help="waypoints per beat")
    p.add_argument("--step-duration", type=float, default=0.3, dest="step_duration")
    p.add_argument("--step-settle", type=float, default=0.3, dest="step_settle")
    p.add_argument("--beat-settle", type=float, default=0.6, dest="beat_settle",
                   help="pause between dance beats, seconds")
    p.add_argument("--skip-home", action="store_true", dest="skip_home",
                   help="skip the opening return-to-HOME pass (assumes already there)")
    args = p.parse_args()

    cfg = Config.load(args.config)
    if cfg.arm.backend == "mock":
        print("  arm.backend is 'mock' -- pass a config with backend: feetech "
              "(e.g. -c configs/real_arm.yaml) to drive real hardware.")
        return 1

    limits = SafetyLimits(max_speed=cfg.safety.max_speed, strike_speed=cfg.safety.strike_speed,
                          joint_margin=cfg.safety.joint_margin)
    controller = ArmController(build_arm(cfg), limits, cfg.runtime.control_hz)
    controller.start()
    print(f"  backend {cfg.arm.backend}\n")

    try:
        if not args.skip_home:
            print("  -- opening bow --")
            go_home(controller, args.steps, args.step_duration, args.step_settle, args.beat_settle)

        print("\n  -- dance --")
        for i, offset, label in DANCE:
            target = model.HOME[i] + offset
            print(f"  {label:<14} ({JOINT_NAMES[i]}) ", end="", flush=True)
            step_joint(controller, i, target, args.steps, args.step_duration, args.step_settle)
            print()
            time.sleep(args.beat_settle)

        print("\n  -- closing bow --")
        go_home(controller, args.steps, args.step_duration, args.step_settle, args.beat_settle)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        final = controller.commanded[:5]
        print("\n  commanded final pose (software-tracked, not a fresh read):")
        for i, name in enumerate(JOINT_NAMES[:5]):
            print(f"    {name:<14} {final[i]:+.3f} rad")
        controller.stop(park=False)  # already home; park() would re-drive all 5 at once
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
