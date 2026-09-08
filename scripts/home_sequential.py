"""Return the arm to HOME one joint at a time.

`tlod move --home` drives all 5 joints together, which briefly asks every
servo for current at once -- fine on the rated 12V 5A supply, but enough
to collapse an undersized one (measured dropping to ~3V under a full-arm
move on this rig). Moving one joint at a time keeps peak draw down to
whatever a single servo needs, with a short pause between joints so the
rail can recover before the next one loads it.

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
    p.add_argument("--duration", type=float, default=1.2, help="seconds per joint")
    p.add_argument("--settle", type=float, default=0.5, help="pause between joints, seconds")
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
            target = controller.commanded[:5].copy()
            before = target[i]
            target[i] = model.HOME[i]
            delta = target[i] - before
            print(f"  {name:<14} {before:+.3f} -> {target[i]:+.3f} rad "
                  f"(delta {delta:+.3f}) ...", end="", flush=True)
            controller.goto_joints(target, duration=args.duration)
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
