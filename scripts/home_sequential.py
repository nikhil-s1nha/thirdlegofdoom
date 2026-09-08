"""Return the arm to HOME one joint at a time, in small discrete steps.

`tlod move --home` drives all 5 joints together, which briefly asks every
servo for current at once -- fine on the rated 12V 5A supply, but enough
to collapse an undersized one (measured dropping to ~3V under a full-arm
move on this rig). Moving one joint at a time keeps peak draw down to
whatever a single servo needs.

`ArmController.goto_joints()` (what `move`/`first-light`/this script all
use) hardcodes its rate cap to `strike_speed` -- `safety.max_speed` in
config has no effect on it. So instead of one continuous goto per joint
(which for a ~2 rad swing is still a fast, current-hungry motion), each
joint's total delta is broken into small waypoints with a real dead-stop
pause between each: motion, full stop, motion, full stop. The stop is
the point -- it lets the rail recover between chunks, not just ramp
slower through one continuous move.

Order: lightest/no-gravity joints first, shoulder_lift (the one actually
lifting the arm's weight) last, so if the supply is going to give out,
it does so with the arm already mostly home rather than at the start.

Usage:
    python scripts/home_sequential.py -c configs/real_arm.yaml
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

# Index order to move in, not joint-array order: farthest from the base
# and least loaded by gravity first, shoulder_lift last.
SEQUENCE = [4, 3, 0, 2, 1]  # wrist_roll, wrist_flex, shoulder_pan, elbow_flex, shoulder_lift


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default=None, help="YAML config path")
    p.add_argument("--steps", type=int, default=8, help="waypoints per joint")
    p.add_argument("--step-duration", type=float, default=0.4, dest="step_duration",
                   help="seconds of motion per waypoint")
    p.add_argument("--step-settle", type=float, default=0.4, dest="step_settle",
                   help="dead-stop pause between waypoints, seconds -- this is what "
                        "lets the rail recover, not the motion speed itself")
    p.add_argument("--settle", type=float, default=0.8, help="pause between joints, seconds")
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
    print(f"  backend {cfg.arm.backend}, homing {len(SEQUENCE)} joints one at a time\n")

    try:
        for i in SEQUENCE:
            name = JOINT_NAMES[i]
            base = controller.commanded[:5].copy()
            before = base[i]
            after = model.HOME[i]
            print(f"  {name:<14} {before:+.3f} -> {after:+.3f} rad "
                  f"({args.steps} steps) ", end="", flush=True)
            for step in range(1, args.steps + 1):
                waypoint = base.copy()
                waypoint[i] = before + (after - before) * (step / args.steps)
                controller.goto_joints(waypoint, duration=args.step_duration)
                print(".", end="", flush=True)
                time.sleep(args.step_settle)
            try:
                measured = controller.state().q[i]
                print(f"  measured {measured:+.3f}")
            except OSError as e:
                # Same class of failure we've been chasing all session on
                # the undersized supply -- the write already happened
                # (goto_joints doesn't need a read to work), so note it
                # and keep going rather than aborting the whole sequence.
                print(f"  (read failed: {e})")
            time.sleep(args.settle)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        final = controller.commanded[:5]
        print("\n  commanded final pose (software-tracked, not a fresh read):")
        for i, name in enumerate(JOINT_NAMES[:5]):
            print(f"    {name:<14} {final[i]:+.3f} rad")
        controller.stop(park=False)  # already home; park() would just re-drive all 5 at once
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
