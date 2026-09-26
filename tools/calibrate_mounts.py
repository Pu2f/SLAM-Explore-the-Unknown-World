"""Measure effective sensor offsets against the real maze walls.

Put the robot in the centre of a cell with walls on its LEFT and RIGHT
(walls in front and behind as well, if the maze has such a cell), square
to the walls, then:
    python -m tools.calibrate_mounts

Sums of readings from opposite walls do not depend on where exactly the
robot stands across the cell, so the side offsets come out right even if it
is a little off-centre. The results are "effective" offsets: they also
absorb wall thickness and sensor bias, which is what the mapping needs.
Saved to calibration/mounts.json and picked up by run_robot / check_robot.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Optional, Sequence

from slam.config import DEFAULT, MOUNTS_FILE, with_calibration
from slam.robot_io import RealRobot

NEAR_WALL_M = 0.45  # raw reading below this = the adjacent wall is there


def median_tof(robot: RealRobot, gimbal_deg: float, samples: int) -> Optional[float]:
    robot.gimbal_moveto(gimbal_deg)
    vals = []
    for _ in range(samples):
        v = robot.tof()
        if v is not None:
            vals.append(v)
        time.sleep(0.03)
    return statistics.median(vals) if vals else None


def median_sharp(robot: RealRobot, samples: int):
    lefts, rights = [], []
    for _ in range(samples):
        left, right = robot.sharp()
        if left is not None:
            lefts.append(left)
        if right is not None:
            rights.append(right)
        time.sleep(0.02)
    med = lambda v: statistics.median(v) if v else None  # noqa: E731
    return med(lefts), med(rights)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--out", default=MOUNTS_FILE)
    args = p.parse_args(argv)

    cfg = with_calibration(DEFAULT)
    half = cfg.geometry.cell_size_m / 2.0
    used_offset = cfg.robot_io.tof_pivot_offset_m
    robot = RealRobot.connect(cfg)
    try:
        # Raw = what the ToF itself reports, without the offset added now.
        raw = {}
        for name, g in (("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", -90.0)):
            v = median_tof(robot, g, args.samples)
            raw[name] = None if v is None else v - used_offset
        robot.gimbal_moveto(0.0)
        sharp_l, sharp_r = median_sharp(robot, args.samples * 2)
    finally:
        robot.close()

    print("raw ToF (m):", {k: None if v is None else round(v, 3) for k, v in raw.items()})
    print(f"Sharp (m): left={sharp_l} right={sharp_r}")
    out = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            out = json.load(f)

    e, w = raw["E"], raw["W"]
    if e is None or w is None or max(e, w) > NEAR_WALL_M:
        print("Need walls on the robot's left and right (ToF at +/-90 deg). Move the robot and retry.")
        return 1
    out["tof_emitter_offset_m"] = round((2 * half - (e + w)) / 2.0, 4)

    n, s = raw["N"], raw["S"]
    if n is not None and s is not None and max(n, s) < NEAR_WALL_M:
        # With the emitter offset known, front/back asymmetry = pivot ahead of centre.
        out["tof_pivot_forward_m"] = round((s - n) / 2.0, 4)
    else:
        print("No walls in front and behind: tof_pivot_forward_m left unchanged.")

    if sharp_l is None or sharp_r is None:
        print("A Sharp gave no reading: Sharp offsets left unchanged.")
    else:
        if min(sharp_l, sharp_r) <= cfg.sensors.sharp_min_m + 1e-6:
            print("!! A Sharp is at its minimum calibrated range; add a closer point with calibrate_sharp.")
        offset = round((2 * half - (sharp_l + sharp_r)) / 2.0, 4)
        out["sharp_left_offset_m"] = offset
        out["sharp_right_offset_m"] = offset

    out["measured_at"] = datetime.now().isoformat(timespec="seconds")
    out["source"] = "python -m tools.calibrate_mounts (robot centred in a cell, walls left and right)"
    out["raw"] = {"tof": raw, "sharp_left": sharp_l, "sharp_right": sharp_r}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out}:")
    print(json.dumps({k: v for k, v in out.items() if k.endswith("_m")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
