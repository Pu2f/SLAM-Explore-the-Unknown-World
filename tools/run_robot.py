"""Run the exploration mission on the real RoboMaster EP.

Place the robot in the centre of a cell, square to the walls, then:
    python -m tools.run_robot
    python -m tools.run_robot --gt ground_truth/arena.txt     # also score the map
Ctrl+C stops the robot and still saves everything recorded so far.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from slam.config import DEFAULT
from slam.explorer import Explorer
from slam.logger import RunLogger
from slam.robot_io import RealRobot
from tools.evaluate import auto_align, evaluate, format_report, load_gt


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conn-type", choices=["ap", "sta", "rndis"], default=DEFAULT.robot_io.conn_type)
    p.add_argument("--max-time", type=float, default=DEFAULT.limits.max_mission_s, help="mission time limit (s)")
    p.add_argument("--gt", type=Path, help="ground truth to score against when the run ends")
    p.add_argument("--out", default="runs")
    args = p.parse_args(argv)

    cfg = replace(
        DEFAULT,
        robot_io=replace(DEFAULT.robot_io, conn_type=args.conn_type),
        limits=replace(DEFAULT.limits, max_mission_s=args.max_time),
    )
    out_dir = os.path.join(args.out, "robot_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    logger = RunLogger(out_dir)
    print(f"[run] output -> {out_dir}")

    robot = RealRobot.connect(cfg)
    for side, cal in (("left", robot.sharp_left_cal), ("right", robot.sharp_right_cal)):
        if cal.source == "placeholder":
            print(f"[warn] Sharp {side} is not calibrated (python -m tools.calibrate_sharp --side {side})")
    try:
        result = Explorer(robot, cfg, logger).run()
    finally:
        robot.close()
        logger.close()

    print(result.map.to_ascii(result.end_cell))
    print(json.dumps(result.summary(cfg.geometry.cell_size_m), indent=2)[:2000])

    if args.gt:
        gt, align = load_gt(args.gt)
        robot_map = result.map.to_wallmap()
        if align is None:
            align, _ = auto_align(robot_map, gt)
            print("[eval] no start arrow in the GT file; using auto-alignment")
        metrics = evaluate(align.apply(robot_map), gt)
        print(format_report(metrics, align))
        out = {"alignment": {"start": list(align.start), "heading": align.heading.name}}
        out.update(asdict(metrics))
        Path(out_dir, "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if result.reason == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
