#!/usr/bin/env python3
"""Regenerate the CHAIN table in tlod/arm/model.py from the URDF.

Run this if the upstream SO-ARM100 model changes. It prints the table;
paste it in and re-run the kinematics tests, which pin the FK output and
will fail loudly if the geometry moved.

    python scripts/extract_urdf.py assets/so101_new_calib.urdf
"""

import sys
import xml.etree.ElementTree as ET

TIP = "gripper_frame_joint"


def main(path: str) -> int:
    root = ET.parse(path).getroot()
    joints = [j for j in root if j.tag == "joint"]
    by_parent: dict[str, list] = {}
    for j in joints:
        by_parent.setdefault(j.find("parent").get("link"), []).append(j)

    links = [l.get("name") for l in root if l.tag == "link"]
    children = {j.find("child").get("link") for j in joints}
    base = next(l for l in links if l not in children)

    print("CHAIN: tuple[Link, ...] = (")
    limits = []
    link = base
    while True:
        candidates = by_parent.get(link, [])
        nxt = None
        for j in candidates:
            if j.get("type") == "revolute" or j.get("name") == TIP:
                nxt = j
                break
        if nxt is None:
            break
        origin = nxt.find("origin")
        xyz = tuple(float(v) for v in (origin.get("xyz") or "0 0 0").split())
        rpy = tuple(float(v) for v in (origin.get("rpy") or "0 0 0").split())
        axis_el = nxt.find("axis")
        name = "tcp" if nxt.get("name") == TIP else nxt.get("name")
        axis = None if nxt.get("type") == "fixed" else tuple(
            int(float(v)) for v in axis_el.get("xyz").split()
        )
        print(f'    Link("{name}", {xyz}, {rpy}, {axis}),')
        lim = nxt.find("limit")
        if lim is not None and axis is not None:
            limits.append((name, float(lim.get("lower")), float(lim.get("upper"))))
        link = nxt.find("child").get("link")
    print(")\n")

    print("JOINT_LIMITS: np.ndarray = np.array([")
    for name, lo, hi in limits[:5]:
        print(f"    [{lo:.5f}, {hi:.5f}],   # {name}")
    print("], dtype=float)")
    for name, lo, hi in limits[5:]:
        print(f"GRIPPER_LIMITS: tuple[float, float] = ({lo}, {hi})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "assets/so101_new_calib.urdf"))
