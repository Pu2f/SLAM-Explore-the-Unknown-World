import json

import pytest

from slam.geometry import Direction
from slam.maze_map import EdgeState, MazeMap, WallMap, rotate_cell
from slam.sim import generate_maze
from tools.evaluate import Alignment, auto_align, evaluate, load_gt, main

GT_3X2 = """\
+---+---+---+
|       |   |
+   +---+   +
|           |
+---+---+---+
"""

ROBOT_3X2 = """\
+---+---+---+
|       |   |
+   +   +   +
|           ?
+---+---+---+
"""


def robot_view(gt: WallMap, start, heading: Direction) -> WallMap:
    """What a perfect robot that started at ``start`` facing ``heading`` would map."""
    k = (-int(heading)) % 4
    moved = gt.transformed(0, (-start[0], -start[1]))
    return moved.transformed(k, (0, 0))


def test_readme_example_numbers():
    gt, _ = WallMap.from_ascii(GT_3X2)
    robot, _ = WallMap.from_ascii(ROBOT_3X2)
    m = evaluate(robot, gt)
    assert (m.correct_cells, m.total_cells) == (3, 6)
    assert m.map_accuracy_pct == pytest.approx(50.0)
    assert m.coverage_pct == pytest.approx(500 / 6)
    assert (m.correct_walls, m.total_walls) == (15, 17)
    assert m.confusion["wall"]["open"] == 1
    assert m.confusion["wall"]["unknown"] == 1
    assert m.wrong_cells == [(1, 0), (1, 1), (2, 0)]
    assert m.unexplored_cells == [(2, 0)]


@pytest.mark.parametrize("heading", list(Direction))
def test_perfect_map_scores_100_after_alignment(heading):
    gt = generate_maze(4, 5, seed=11, loops=2)
    start = (1, 3)
    robot = robot_view(gt, start, heading)
    assert robot.get((0, 0), Direction.N) == gt.get(start, heading)

    align = Alignment(start, heading)
    m = evaluate(align.apply(robot), gt)
    assert m.map_accuracy_pct == 100.0
    assert m.coverage_pct == 100.0
    assert m.phantom_cells == []

    found, _ = auto_align(robot, gt)
    assert found == align


def test_partial_map_and_phantom_cells():
    gt = generate_maze(3, 3, seed=4)
    robot = MazeMap()
    robot.mark_visited((0, 0))
    for d in Direction:
        robot.observe((0, 0), d, gt.get((0, 0), d) == EdgeState.WALL, 1.0)
    wm = robot.to_wallmap()
    wm.cells.add((-1, 0))  # e.g. a wall misread as open on the outer boundary
    m = evaluate(Alignment((0, 0), Direction.N).apply(wm), gt)
    assert m.explored_cells == 1
    assert m.correct_cells == 1
    assert m.phantom_cells == [(-1, 0)]


def test_gt_arrow_marker_gives_alignment(tmp_path):
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text("+---+---+\n|   | > |\n+---+---+\n")
    gt, align = load_gt(gt_file)
    assert align == Alignment((1, 0), Direction.E)
    assert rotate_cell((0, 1), int(align.heading)) == (1, 0)


def test_cli_end_to_end(tmp_path, capsys):
    gt = generate_maze(4, 3, seed=9, loops=1)
    start, heading = (2, 1), Direction.W
    markers = {start: "<"}
    (tmp_path / "gt.txt").write_text(gt.to_ascii(markers))

    robot = MazeMap()
    view = robot_view(gt, start, heading)
    for cell in view.cells:
        robot.mark_visited(cell)
        for d in Direction:
            robot.observe(cell, d, view.get(cell, d) == EdgeState.WALL, 1.0)
    (tmp_path / "map.json").write_text(json.dumps(robot.to_dict()))

    out = tmp_path / "metrics.json"
    rc = main(["--gt", str(tmp_path / "gt.txt"), "--map", str(tmp_path / "map.json"), "--json", str(out)])
    assert rc == 0
    assert "Map Accuracy   : 100.00 %" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["alignment"] == {"start": [2, 1], "heading": "W"}
    assert data["coverage_pct"] == 100.0

    # Robot ASCII map with S marker works too, with explicit start.
    (tmp_path / "map.txt").write_text(robot.to_ascii())
    rc = main(["--gt", str(tmp_path / "gt.txt"), "--map", str(tmp_path / "map.txt"),
               "--start", "2,1", "--heading", "W"])
    assert rc == 0
    assert "Map Accuracy   : 100.00 %" in capsys.readouterr().out
