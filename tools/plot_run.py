"""Draw a run's map, trajectory and (with a ground truth) the comparison.

    python -m tools.plot_run runs/<run>
    python -m tools.plot_run runs/<run> --gt ground_truth/arena.txt

Writes into the run directory:
    map.png         the robot's map (robot frame: start at (0, 0), start heading up)
    trajectory.png  EKF path, raw odometry (and simulator truth) over the map
    comparison.png  map vs ground truth, cell by cell (GT frame)
A run from tools.run_sim already has ground_truth.txt, which is used by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from slam.config import DEFAULT  # noqa: E402
from slam.geometry import Cell, Direction, cell_center, wall_segment  # noqa: E402
from slam.maze_map import EdgeState, MazeMap, WallMap  # noqa: E402
from tools.evaluate import Alignment, Metrics, auto_align, evaluate, load_gt  # noqa: E402

CELL = DEFAULT.geometry.cell_size_m

# Reference palette (dataviz skill, light mode). Series = categorical slots
# 1-3 in fixed order; cell states use the reserved status palette and always
# carry a glyph + legend label, never colour alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
VISITED_FILL = "#cde2fb"  # sequential blue step 100
SERIES_EKF = "#2a78d6"
SERIES_ODOM = "#eb6834"
SERIES_TRUTH = "#1baf7a"
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"


# ---- loading ----------------------------------------------------------------------
def load_robot_map(run_dir: Path) -> MazeMap:
    return MazeMap.from_dict(json.loads((run_dir / "map.json").read_text(encoding="utf-8")))


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_trajectory(run_dir: Path) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {}
    path = run_dir / "trajectory.csv"
    if not path.exists():
        return cols
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v not in ("", None) else float("nan"))
    return cols


# ---- drawing helpers ----------------------------------------------------------------
def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel("x (m)", color=INK_2, fontsize=9)
    ax.set_ylabel("y (m)", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(INK_MUTED)
        spine.set_linewidth(0.6)


def _frame(ax, cells) -> None:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    pad = CELL * 0.75
    ax.set_xlim(min(xs) * CELL - pad, max(xs) * CELL + pad)
    ax.set_ylim(min(ys) * CELL - pad, max(ys) * CELL + pad)


def _segment(ax, cell: Cell, d: Direction, **style) -> None:
    s = wall_segment(cell, d, CELL)
    ax.plot([s.x1, s.x2], [s.y1, s.y2], solid_capstyle="round", **style)


def draw_walls(ax, wm: WallMap, known_only: bool = False) -> None:
    """Known walls in ink; unknown sides of map cells dashed."""
    for key in sorted(wm.cell_edge_keys()):
        cell, d = (key[0], key[1]), key[2]
        state = wm.get(cell, d)
        if state == EdgeState.WALL:
            _segment(ax, cell, d, color=INK, linewidth=3.0, zorder=3)
        elif state == EdgeState.UNKNOWN and not known_only:
            _segment(ax, cell, d, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=2)


def draw_cells(ax, cells, fill: str, inset: float = 0.02) -> None:
    for c in cells:
        cx, cy = cell_center(c, CELL)
        h = CELL / 2 - inset
        ax.add_patch(Rectangle((cx - h, cy - h), 2 * h, 2 * h, facecolor=fill, edgecolor="none", zorder=1))


def draw_start_end(ax, start_xy, start_heading_deg: float, end_xy=None, labels: bool = True) -> List[Line2D]:
    """START = circle with a heading arrow, END = square (if given); both labelled."""
    import math

    dx, dy = math.sin(math.radians(start_heading_deg)), math.cos(math.radians(start_heading_deg))
    ax.annotate(
        "",
        xy=(start_xy[0] + dx * CELL * 0.35, start_xy[1] + dy * CELL * 0.35),
        xytext=start_xy,
        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.5),
        zorder=6,
    )
    ax.plot(*start_xy, marker="o", markersize=10, color=INK, markeredgecolor=SURFACE, markeredgewidth=2, zorder=7)
    if labels:
        ax.annotate("START", start_xy, xytext=(-8, -14), textcoords="offset points", color=INK, fontsize=8,
                    ha="right", zorder=8)
    handles = [
        Line2D([], [], marker="o", color=INK, linestyle="none", markersize=8, label="start (arrow = start heading)")
    ]
    if end_xy is not None:
        ax.plot(*end_xy, marker="s", markersize=10, color=INK, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=7)
        if labels:
            ax.annotate("END", end_xy, xytext=(8, -14), textcoords="offset points", color=INK, fontsize=8,
                        zorder=8)
        handles.append(Line2D([], [], marker="s", color=INK, linestyle="none", markersize=8, label="end"))
    return handles


def _legend(ax, handles, ncol: int) -> None:
    """Legend below the axes, a fixed distance under the x label."""
    below = transforms.offset_copy(ax.transAxes, fig=ax.figure, y=-38, units="points")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), bbox_transform=below,
              ncol=ncol, fontsize=8, frameon=False)


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- figures ----------------------------------------------------------------------------
def plot_map(robot: MazeMap, end_cell: Cell, out: Path) -> Path:
    wm = robot.to_wallmap()
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    _style(ax, "Robot map")
    draw_cells(ax, robot.visited, VISITED_FILL)
    draw_walls(ax, wm)
    for c in sorted(wm.cells):
        cx, cy = cell_center(c, CELL)
        ax.text(cx - CELL * 0.42, cy + CELL * 0.40, f"{c[0]},{c[1]}", color=INK_MUTED, fontsize=6,
                va="top", zorder=4)
    handles = draw_start_end(ax, (0.0, 0.0), 0.0, cell_center(end_cell, CELL))
    handles += [
        Line2D([], [], color=INK, linewidth=3, label="wall"),
        Line2D([], [], color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)), label="unknown side"),
        Patch(facecolor=VISITED_FILL, label="visited cell"),
    ]
    _frame(ax, wm.cells)
    _legend(ax, handles, ncol=3)
    return _save(fig, out)


def plot_trajectory(robot: MazeMap, traj: Dict[str, List[float]], out: Path) -> Path:
    wm = robot.to_wallmap()
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    _style(ax, "Robot trajectory")
    draw_walls(ax, wm, known_only=True)
    handles: List = []
    if traj.get("x_m"):
        ax.plot(traj["odom_x_m"], traj["odom_y_m"], color=SERIES_ODOM, linewidth=1.5,
                linestyle=(0, (4, 3)), zorder=4)
        handles.append(Line2D([], [], color=SERIES_ODOM, linewidth=1.5, linestyle=(0, (4, 3)),
                              label="wheel odometry (raw)"))
        truth = traj.get("true_x_m", [])
        if truth and any(v == v for v in truth):  # NaN != NaN: real runs have no truth
            # Drawn above the EKF line: when the estimate is good they overlap.
            ax.plot(traj["true_x_m"], traj["true_y_m"], color=SERIES_TRUTH, linewidth=2.0,
                    linestyle=(0, (1, 2)), zorder=7)
            handles.append(Line2D([], [], color=SERIES_TRUTH, linewidth=2, linestyle=(0, (1, 2)),
                                  label="true path (simulator)"))
        ax.plot(traj["x_m"], traj["y_m"], color=SERIES_EKF, linewidth=2.0, zorder=6)
        handles.insert(0, Line2D([], [], color=SERIES_EKF, linewidth=2, label="EKF estimate"))
        start_xy = (traj["x_m"][0], traj["y_m"][0])
        end_xy = (traj["x_m"][-1], traj["y_m"][-1])
    else:
        start_xy, end_xy = (0.0, 0.0), (0.0, 0.0)
    handles += draw_start_end(ax, start_xy, 0.0, end_xy)
    handles.append(Line2D([], [], color=INK, linewidth=3, label="wall"))
    _frame(ax, wm.cells)
    _legend(ax, handles, ncol=2)
    return _save(fig, out)


def plot_comparison(robot_aligned: WallMap, gt: WallMap, align: Alignment, metrics: Metrics, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6.4), facecolor=SURFACE)
    _style(ax, f"Map vs ground truth: accuracy {metrics.map_accuracy_pct:.1f} %, "
               f"coverage {metrics.coverage_pct:.1f} %")

    wrong = set(metrics.wrong_cells)
    unexplored = set(metrics.unexplored_cells)
    for c in sorted(gt.cells):
        if c in unexplored:
            fill, glyph = STATUS_WARNING, "?"
        elif c in wrong:
            fill, glyph = STATUS_CRITICAL, "✗"
        else:
            fill, glyph = STATUS_GOOD, "✓"
        cx, cy = cell_center(c, CELL)
        h = CELL / 2 - 0.02
        ax.add_patch(Rectangle((cx - h, cy - h), 2 * h, 2 * h, facecolor=fill, alpha=0.22,
                               edgecolor="none", zorder=1))
        ax.text(cx, cy, glyph, color=INK, fontsize=13, ha="center", va="center", zorder=4)

    for key in sorted(gt.cell_edge_keys()):
        cell, d = (key[0], key[1]), key[2]
        truth, pred = gt.get(cell, d), robot_aligned.get(cell, d)
        if truth == EdgeState.WALL and pred == EdgeState.WALL:
            _segment(ax, cell, d, color=INK, linewidth=3.0, zorder=3)
        elif truth == EdgeState.WALL and pred == EdgeState.OPEN:
            _segment(ax, cell, d, color=STATUS_CRITICAL, linewidth=3.5, zorder=3)
        elif truth == EdgeState.OPEN and pred == EdgeState.WALL:
            _segment(ax, cell, d, color=STATUS_SERIOUS, linewidth=3.5, linestyle=(0, (2, 1.5)), zorder=3)
        elif pred == EdgeState.UNKNOWN:
            _segment(ax, cell, d, color=INK_MUTED, linewidth=1.2 if truth == EdgeState.OPEN else 2.5,
                     linestyle=(0, (3, 3)), zorder=2)

    for c in sorted(robot_aligned.cells - gt.cells):  # phantom cells
        cx, cy = cell_center(c, CELL)
        ax.text(cx, cy, "phantom", color=INK_2, fontsize=7, ha="center", va="center")

    handles = draw_start_end(ax, cell_center(align.start, CELL), align.heading.heading_deg, labels=False)
    handles += [
        Patch(facecolor=STATUS_GOOD, alpha=0.4, label="✓ cell correct"),
        Patch(facecolor=STATUS_CRITICAL, alpha=0.4, label="✗ cell wrong"),
        Patch(facecolor=STATUS_WARNING, alpha=0.4, label="? cell not fully explored"),
        Line2D([], [], color=INK, linewidth=3, label="wall, found"),
        Line2D([], [], color=STATUS_CRITICAL, linewidth=3.5, label="wall missed (mapped open)"),
        Line2D([], [], color=STATUS_SERIOUS, linewidth=3.5, linestyle=(0, (2, 1.5)),
               label="extra wall (really open)"),
        Line2D([], [], color=INK_MUTED, linewidth=2, linestyle=(0, (3, 3)), label="not observed"),
    ]
    _frame(ax, gt.cells | robot_aligned.cells)
    _legend(ax, handles, ncol=2)
    return _save(fig, out)


# ---- entry points -------------------------------------------------------------------------
def plot_run(run_dir: Path, gt_path: Optional[Path] = None, align: Optional[Alignment] = None) -> List[Path]:
    """Write all figures for a run; returns the files written."""
    robot = load_robot_map(run_dir)
    summary = load_summary(run_dir)
    end_cell = tuple(summary.get("end_cell", [0, 0]))
    written = [
        plot_map(robot, end_cell, run_dir / "map.png"),  # type: ignore[arg-type]
        plot_trajectory(robot, load_trajectory(run_dir), run_dir / "trajectory.png"),
    ]
    if gt_path is None and (run_dir / "ground_truth.txt").exists():
        gt_path = run_dir / "ground_truth.txt"
    if gt_path is not None:
        gt, gt_align = load_gt(gt_path)
        wm = robot.to_wallmap()
        align = align or gt_align or auto_align(wm, gt)[0]
        aligned = align.apply(wm)
        written.append(plot_comparison(aligned, gt, align, evaluate(aligned, gt), run_dir / "comparison.png"))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--gt", type=Path, help="ground truth (default: <run_dir>/ground_truth.txt if present)")
    p.add_argument("--start", help="GT start cell 'x,y' (else the arrow in the GT file, else auto-align)")
    p.add_argument("--heading", choices=[d.name for d in Direction])
    args = p.parse_args(argv)
    align = None
    if args.start or args.heading:
        if not (args.start and args.heading):
            p.error("--start and --heading go together")
        x, y = (int(v) for v in args.start.split(","))
        align = Alignment((x, y), Direction[args.heading])
    for path in plot_run(args.run_dir, args.gt, align):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
