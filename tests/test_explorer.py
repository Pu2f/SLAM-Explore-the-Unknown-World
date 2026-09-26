"""End-to-end missions in the simulator."""

import json
import random
from dataclasses import replace

import pytest

from slam.config import DEFAULT, PERFECT_SIM
from slam.explorer import Explorer
from slam.geometry import Direction
from slam.logger import RunLogger
from slam.maze_map import EdgeState, WallMap
from slam.sim import SimRobot, generate_maze
from tools.evaluate import Alignment, evaluate, load_map
from tools.run_sim import main as run_sim_main


def mission(gt: WallMap, start, heading, noise=PERFECT_SIM, cfg=DEFAULT, seed=0, logger=None):
    robot = SimRobot(gt, start, heading, config=cfg, noise=noise, seed=seed)
    result = Explorer(robot, cfg, logger).run()
    metrics = evaluate(Alignment(start, heading).apply(result.map.to_wallmap()), gt)
    return robot, result, metrics


@pytest.mark.parametrize(
    "w,h,loops,seed",
    [(1, 1, 0, 0), (1, 4, 0, 1), (3, 3, 2, 2), (4, 5, 0, 3), (5, 3, 3, 4)],
)
def test_perfect_robot_maps_everything(w, h, loops, seed):
    rng = random.Random(seed)
    gt = generate_maze(w, h, seed=seed, loops=loops)
    start = rng.choice(sorted(gt.cells))
    heading = rng.choice(list(Direction))
    robot, result, m = mission(gt, start, heading)
    assert result.reason == "complete"
    assert m.map_accuracy_pct == 100.0
    assert m.coverage_pct == 100.0
    assert m.phantom_cells == []
    assert robot.collisions == 0
    assert len(result.map.visited) == w * h


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noisy_robot_maps_everything(seed):
    rng = random.Random(seed)
    gt = generate_maze(4, 5, seed=seed, loops=seed % 3)
    start = rng.choice(sorted(gt.cells))
    heading = rng.choice(list(Direction))
    robot, result, m = mission(gt, start, heading, noise=DEFAULT.sim_noise, seed=seed)
    assert result.reason == "complete"
    assert m.map_accuracy_pct == 100.0
    assert robot.collisions == 0
    assert result.localization_mismatches == 0
    true_end = robot.true_pose_map()
    assert abs(true_end.x - result.end_pose.x) < 0.05
    assert abs(true_end.y - result.end_pose.y) < 0.05


def test_mission_stops_at_cell_limit():
    cfg = replace(DEFAULT, limits=replace(DEFAULT.limits, max_cells=3))
    gt = generate_maze(4, 4, seed=5)
    _, result, m = mission(gt, (0, 0), Direction.N, cfg=cfg)
    assert result.reason == "limit_max_cells"
    assert len(result.map.visited) == 3


def test_run_outputs_are_complete(tmp_path):
    gt = generate_maze(3, 2, seed=8)
    logger = RunLogger(str(tmp_path))
    _, result, _ = mission(gt, (1, 0), Direction.E, noise=DEFAULT.sim_noise, logger=logger)
    logger.close()
    for name in ("exploration_log.csv", "trajectory.csv", "sensors.csv", "map.json", "map.txt", "summary.json"):
        assert (tmp_path / name).stat().st_size > 0, name

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["stop_reason"] == "complete"
    assert summary["start_cell"] == [0, 0]
    assert summary["estimated_size_cells"] in ([3, 2], [2, 3])
    assert summary["cell_path"][0] == [0, 0]
    assert "true_end_pose" in summary

    events = (tmp_path / "exploration_log.csv").read_text().splitlines()
    kinds = {line.split(",")[1] for line in events[1:]}
    assert {"START", "SCAN", "TURN", "DRIVE", "ARRIVED", "END"} <= kinds

    robot_map = load_map(tmp_path / "map.json")
    m = evaluate(Alignment((1, 0), Direction.E).apply(robot_map), gt)
    assert m.map_accuracy_pct == 100.0


def test_run_sim_cli(tmp_path, capsys):
    rc = run_sim_main(["--gt", "ground_truth/example_3x2.txt", "--out", str(tmp_path), "--seed", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Map Accuracy   : 100.00 %" in out
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "ground_truth.txt").exists()


ROOM = (1.33, 1.1, 0.95, 1.7)  # room walls at uneven distances, like a real room


def maze_with_exit(seed):
    gt = generate_maze(3, 3, seed=seed)
    gt.set((1, 0), Direction.S, EdgeState.OPEN)  # a gap in the outer wall
    return gt


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exit_is_found_left_and_the_rest_explored(seed):
    gt = maze_with_exit(seed)
    robot = SimRobot(gt, (1, 1), Direction.N, noise=DEFAULT.sim_noise, seed=seed, room_margin_m=ROOM)
    result = Explorer(robot, DEFAULT).run()
    assert result.reason == "complete"
    assert result.map.outside == {(0, -2)}  # GT (1, -1), just below the gap
    m = evaluate(Alignment((1, 1), Direction.N).apply(result.map.to_wallmap()), gt)
    assert m.map_accuracy_pct == 100.0
    assert m.phantom_cells == []  # nothing outside the maze left in the map
    assert robot.collisions == 0


@pytest.mark.parametrize("room", [ROOM, None])  # room far away, or nothing at all
def test_exit_behind_the_start_is_explored_last(room):
    """Real run 2026-09-27 03:05: the only open side the robot saw at the
    start was a gap behind it; it left through it before exploring anything."""
    gt = maze_with_exit(0)
    robot = SimRobot(gt, (1, 0), Direction.N, noise=DEFAULT.sim_noise, seed=0, room_margin_m=room)
    result = Explorer(robot, DEFAULT).run()
    assert result.reason == "complete"
    assert result.path[1] != (0, -1)  # first move is not out through the gap
    m = evaluate(Alignment((1, 0), Direction.N).apply(result.map.to_wallmap()), gt)
    assert m.phantom_cells == []
    assert m.map_accuracy_pct == 100.0


@pytest.mark.parametrize("seed", [3, 4])
def test_no_false_exit_in_open_closed_maze(seed):
    gt = generate_maze(4, 4, seed=seed, loops=10)  # big open areas, but closed
    robot = SimRobot(gt, (1, 1), Direction.E, noise=DEFAULT.sim_noise, seed=seed, room_margin_m=ROOM)
    result = Explorer(robot, DEFAULT).run()
    assert result.reason == "complete"
    assert result.map.outside == set()


def test_front_irs_seeing_side_walls_do_not_block_moves():
    """Real run 2026-09-27 03:32: thick walls put the corridor sides inside
    both 45 deg IRs; the robot aborted, marked open passages as walls and
    mapped 6 of 20 cells."""
    cfg = replace(DEFAULT, sensors=replace(DEFAULT.sensors, ir_range_m=0.35))  # IRs see side walls
    gt = generate_maze(4, 5, seed=3, loops=1)
    robot = SimRobot(gt, (0, 0), Direction.N, config=cfg, noise=DEFAULT.sim_noise, seed=3)
    result = Explorer(robot, cfg).run()
    m = evaluate(Alignment((0, 0), Direction.N).apply(result.map.to_wallmap()), gt)
    assert result.reason == "complete"
    assert result.blocked_moves == 0
    assert m.map_accuracy_pct == 100.0
    assert robot.collisions == 0
    assert result.ir_ticks.get("both", 0) > 0  # the IRs really did fire
