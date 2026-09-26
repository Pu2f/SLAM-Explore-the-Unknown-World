"""Score a robot map against a ground-truth map (DESIGN.md section 9).

Map Accuracy = cells whose 4 walls all match the GT / GT cells x 100
Coverage     = cells whose 4 walls are all known / GT cells x 100
(an UNKNOWN side counts as wrong). Per-wall numbers, a confusion table and
phantom cells (robot cells outside the GT) are reported as extras.

The robot map is in its own frame (start = (0, 0), start heading = N), so it
is first rotated / shifted onto the GT. The alignment comes from, in order:
--start/--heading, an arrow marker (^ > v <) in the GT start cell, or
--auto-align, which tries every rotation and start cell.

Usage:
    python -m tools.evaluate --gt ground_truth/maze.txt --map runs/<run>/map.json
    python -m tools.evaluate --gt maze.txt --map map.txt --start 0,4 --heading N
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from slam.geometry import Cell, Direction
from slam.maze_map import EdgeState, WallMap

ARROWS = {"^": Direction.N, ">": Direction.E, "v": Direction.S, "<": Direction.W}


@dataclass(frozen=True)
class Alignment:
    """Robot cell c maps to GT cell rotate_cell(c, heading) + start."""

    start: Cell  # GT cell where the robot started
    heading: Direction  # GT direction the robot faced at the start

    def apply(self, robot: WallMap) -> WallMap:
        return robot.transformed(int(self.heading), self.start)


@dataclass
class Metrics:
    total_cells: int
    correct_cells: int
    explored_cells: int
    map_accuracy_pct: float
    coverage_pct: float
    total_walls: int
    correct_walls: int
    known_walls: int
    wall_accuracy_pct: float
    wall_coverage_pct: float
    # confusion[gt_state][robot_state] = count of walls
    confusion: Dict[str, Dict[str, int]]
    wrong_cells: List[Cell] = field(default_factory=list)
    unexplored_cells: List[Cell] = field(default_factory=list)
    phantom_cells: List[Cell] = field(default_factory=list)


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def evaluate(robot_aligned: WallMap, gt: WallMap) -> Metrics:
    """Compare a robot map that is already in GT coordinates."""
    correct_cells = explored_cells = 0
    wrong: List[Cell] = []
    unexplored: List[Cell] = []
    for c in sorted(gt.cells):
        pred = robot_aligned.sides(c)
        truth = gt.sides(c)
        if pred == truth:
            correct_cells += 1
        else:
            wrong.append(c)
        if EdgeState.UNKNOWN not in pred:
            explored_cells += 1
        else:
            unexplored.append(c)

    states = [s.value for s in EdgeState]
    confusion = {g: {p: 0 for p in states} for g in states}
    correct_walls = known_walls = 0
    wall_keys = sorted(gt.cell_edge_keys())
    for key in wall_keys:
        cell = (key[0], key[1])
        truth = gt.get(cell, key[2])
        pred = robot_aligned.get(cell, key[2])
        confusion[truth.value][pred.value] += 1
        if pred == truth:
            correct_walls += 1
        if pred != EdgeState.UNKNOWN:
            known_walls += 1

    total = len(gt.cells)
    return Metrics(
        total_cells=total,
        correct_cells=correct_cells,
        explored_cells=explored_cells,
        map_accuracy_pct=_pct(correct_cells, total),
        coverage_pct=_pct(explored_cells, total),
        total_walls=len(wall_keys),
        correct_walls=correct_walls,
        known_walls=known_walls,
        wall_accuracy_pct=_pct(correct_walls, len(wall_keys)),
        wall_coverage_pct=_pct(known_walls, len(wall_keys)),
        confusion=confusion,
        wrong_cells=wrong,
        unexplored_cells=unexplored,
        phantom_cells=sorted(robot_aligned.cells - gt.cells),
    )


def _alignment_score(robot: WallMap, gt: WallMap, align: Alignment) -> Tuple[int, int]:
    aligned = align.apply(robot)
    correct = sum(
        1
        for key in gt.cell_edge_keys()
        if aligned.get((key[0], key[1]), key[2]) == gt.get((key[0], key[1]), key[2])
        and gt.get((key[0], key[1]), key[2]) != EdgeState.UNKNOWN
    )
    phantom = len(aligned.cells - gt.cells)
    return correct, -phantom


def auto_align(robot: WallMap, gt: WallMap) -> Tuple[Alignment, int]:
    """Best alignment over every rotation and every GT start cell.

    Returns (alignment, number_of_matching_walls).
    """
    best: Optional[Tuple[Tuple[int, int], Alignment]] = None
    for heading in Direction:
        for start in sorted(gt.cells):
            align = Alignment(start, heading)
            score = _alignment_score(robot, gt, align)
            if best is None or score > best[0]:
                best = (score, align)
    if best is None:
        raise ValueError("ground truth has no cells")
    return best[1], best[0][0]


def load_map(path: Path) -> WallMap:
    """Robot map from a run's map.json, or ASCII with the start marked 'S'."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return WallMap.from_dict(json.loads(text))
    robot, _ = WallMap.from_ascii(text, origin_marker="S")
    return robot


def load_gt(path: Path) -> Tuple[WallMap, Optional[Alignment]]:
    """GT from ASCII. An arrow marker (^ > v <) in one cell gives the start."""
    gt, markers = WallMap.from_ascii(path.read_text(encoding="utf-8"))
    arrows = [(c, ARROWS[m]) for c, m in markers.items() if m in ARROWS]
    if len(arrows) > 1:
        raise ValueError(f"{path}: more than one start arrow")
    align = Alignment(arrows[0][0], arrows[0][1]) if arrows else None
    return gt, align


def format_report(m: Metrics, align: Alignment) -> str:
    lines = [
        f"Alignment      : robot start = GT cell {align.start}, facing {align.heading.name}",
        f"Map Accuracy   : {m.map_accuracy_pct:6.2f} %  ({m.correct_cells}/{m.total_cells} cells)",
        f"Coverage       : {m.coverage_pct:6.2f} %  ({m.explored_cells}/{m.total_cells} cells)",
        f"Wall accuracy  : {m.wall_accuracy_pct:6.2f} %  ({m.correct_walls}/{m.total_walls} walls)",
        f"Wall coverage  : {m.wall_coverage_pct:6.2f} %  ({m.known_walls}/{m.total_walls} walls)",
        "Confusion (rows = GT, cols = robot):",
        "            " + "".join(f"{s.value:>9}" for s in EdgeState),
    ]
    for g in EdgeState:
        if sum(m.confusion[g.value].values()) == 0:
            continue
        lines.append(
            f"  {g.value:>8}  " + "".join(f"{m.confusion[g.value][p.value]:>9}" for p in EdgeState)
        )
    if m.wrong_cells:
        lines.append(f"Wrong cells    : {m.wrong_cells}")
    if m.phantom_cells:
        lines.append(f"Phantom cells  : {m.phantom_cells}")
    return "\n".join(lines)


def _parse_cell(text: str) -> Cell:
    x, y = (int(v) for v in text.split(","))
    return (x, y)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, required=True, help="ground-truth ASCII map")
    p.add_argument("--map", type=Path, required=True, help="robot map (.json or ASCII)")
    p.add_argument("--start", type=_parse_cell, help="GT start cell 'x,y' (x from left, y from bottom, 0-based)")
    p.add_argument("--heading", choices=[d.name for d in Direction], help="GT direction the robot faced at start")
    p.add_argument("--auto-align", action="store_true", help="search for the best alignment")
    p.add_argument("--json", type=Path, help="write metrics as JSON here")
    args = p.parse_args(argv)

    gt, gt_align = load_gt(args.gt)
    robot = load_map(args.map)

    given: Optional[Alignment] = None
    if args.start is not None or args.heading is not None:
        if args.start is None or args.heading is None:
            p.error("--start and --heading go together")
        given = Alignment(args.start, Direction[args.heading])
    elif gt_align is not None:
        given = gt_align

    auto: Optional[Alignment] = None
    if args.auto_align or given is None:
        auto, _ = auto_align(robot, gt)
        if given is not None and auto != given:
            print(
                f"WARNING: auto-align prefers start {auto.start} facing {auto.heading.name}; "
                f"using the given start {given.start} facing {given.heading.name}",
                file=sys.stderr,
            )
    align = given if given is not None else auto
    assert align is not None

    metrics = evaluate(align.apply(robot), gt)
    print(format_report(metrics, align))
    if args.json:
        out = {"alignment": {"start": list(align.start), "heading": align.heading.name}}
        out.update(asdict(metrics))
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
