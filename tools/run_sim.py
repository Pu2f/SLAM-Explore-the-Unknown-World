"""Run a full mission in the simulator, then score the map.

Examples:
    python -m tools.run_sim                                  # random 4x5 maze
    python -m tools.run_sim --width 6 --height 4 --loops 3 --seed 2
    python -m tools.run_sim --gt ground_truth/example_3x2.txt
    python -m tools.run_sim --perfect                        # no sensor / motion noise
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from slam.config import DEFAULT, PERFECT_SIM
from slam.explorer import Explorer
from slam.geometry import Direction
from slam.logger import RunLogger
from slam.sim import SimRobot, generate_maze
from tools.evaluate import ARROWS, Alignment, evaluate, format_report, load_gt
from tools.plot_run import plot_run

ARROW_OF = {d: a for a, d in ARROWS.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", help="ASCII maze to run in (arrow marks start); default: random maze")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--height", type=int, default=5)
    p.add_argument("--loops", type=int, default=0, help="extra openings that create cycles")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perfect", action="store_true", help="disable all simulated noise")
    p.add_argument("--out", default="runs", help="parent directory for run output")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    if args.gt:
        with open(args.gt, encoding="utf-8") as f:
            gt_text = f.read()
        gt, align = load_gt(Path(args.gt))
        if align is None:
            align = Alignment(min(gt.cells), Direction.N)
    else:
        gt = generate_maze(args.width, args.height, seed=args.seed, loops=args.loops)
        align = Alignment(rng.choice(sorted(gt.cells)), rng.choice(list(Direction)))
        gt_text = None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"sim_{stamp}_seed{args.seed}")
    logger = RunLogger(out_dir)
    if gt_text is None:
        gt_text = gt.to_ascii({align.start: ARROW_OF[align.heading]}) + "\n"
    with open(os.path.join(out_dir, "ground_truth.txt"), "w", encoding="utf-8") as f:
        f.write(gt_text)

    noise = PERFECT_SIM if args.perfect else DEFAULT.sim_noise
    robot = SimRobot(gt, align.start, align.heading, noise=noise, seed=args.seed)
    try:
        result = Explorer(robot, DEFAULT, logger).run()
    finally:
        logger.close()

    metrics = evaluate(align.apply(result.map.to_wallmap()), gt)
    print(f"Run dir        : {out_dir}")
    print(f"Stop reason    : {result.reason}")
    print(f"Mission time   : {result.duration_s:.1f} s (simulated), {result.moves} moves, "
          f"{result.blocked_moves} blocked, collisions={robot.collisions}")
    tp = robot.true_pose_map()
    ep = result.end_pose
    print(f"End pose       : EKF ({ep.x:+.3f}, {ep.y:+.3f}, {ep.heading_deg:+.1f}) "
          f"truth ({tp.x:+.3f}, {tp.y:+.3f}, {tp.heading_deg:+.1f})")
    print(format_report(metrics, align))
    for path in plot_run(Path(out_dir), Path(out_dir) / "ground_truth.txt", align):
        print(f"Figure         : {path}")
    return 0 if result.reason == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
