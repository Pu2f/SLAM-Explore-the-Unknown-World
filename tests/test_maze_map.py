import json

import pytest

from slam.config import MapEvidence
from slam.geometry import Direction
from slam.maze_map import (
    EdgeState,
    MapFormatError,
    MazeMap,
    WallMap,
    edge_key,
    rotate_cell,
)
from slam.sim import generate_maze

N, E, S, W = Direction.N, Direction.E, Direction.S, Direction.W
WALL, OPEN, UNKNOWN = EdgeState.WALL, EdgeState.OPEN, EdgeState.UNKNOWN

EXAMPLE_3X2 = """\
+---+---+---+
|       |   |
+   +---+   +
|           |
+---+---+---+
"""


def test_neighbouring_cells_share_one_edge():
    assert edge_key((0, 0), N) == edge_key((0, 1), S)
    assert edge_key((0, 0), E) == edge_key((1, 0), W)
    m = WallMap()
    m.set((2, 2), S, WALL)
    assert m.get((2, 1), N) == WALL


def test_parse_example():
    m, markers = WallMap.from_ascii(EXAMPLE_3X2)
    assert markers == {}
    assert m.cells == {(x, y) for x in range(3) for y in range(2)}
    top_middle = (1, 1)
    assert m.sides(top_middle) == (WALL, WALL, WALL, OPEN)
    assert m.sides((0, 0)) == (OPEN, OPEN, WALL, WALL)
    assert m.get((2, 1), S) == OPEN


def test_parse_tolerates_stripped_trailing_spaces_and_legend():
    text = "legend: whatever\n+---+---+\n|    \n+---+---+\nfooter\n"
    m, _ = WallMap.from_ascii(text)
    assert m.sides((0, 0)) == (WALL, OPEN, WALL, WALL)
    assert m.get((1, 0), E) == OPEN


def test_parse_unknown_and_bad_symbols():
    m, _ = WallMap.from_ascii("+ ? +\n?   |\n+---+\n")
    assert m.sides((0, 0)) == (UNKNOWN, WALL, WALL, UNKNOWN)
    with pytest.raises(MapFormatError):
        WallMap.from_ascii("+-x-+\n|   |\n+---+\n")


def test_ascii_round_trip_on_random_mazes():
    for seed in range(5):
        maze = generate_maze(5, 4, seed=seed, loops=2)
        text = maze.to_ascii({(0, 0): "S"})
        back, markers = WallMap.from_ascii(text)
        assert markers == {(0, 0): "S"}
        assert back.to_dict() == maze.to_dict()


def test_origin_marker_moves_origin():
    text = "+---+---+\n|   | S |\n+---+---+\n"
    m, markers = WallMap.from_ascii(text, origin_marker="S")
    assert m.cells == {(-1, 0), (0, 0)}
    assert markers == {(0, 0): "S"}
    assert m.get((0, 0), W) == WALL


def test_rotate_four_times_is_identity():
    maze = generate_maze(4, 3, seed=1)
    assert maze.transformed(4, (0, 0)).to_dict() == maze.to_dict()
    assert rotate_cell((0, 1), 1) == (1, 0)  # N -> E


def test_rotation_moves_walls_with_cells():
    m = WallMap()
    m.cells = {(0, 0), (0, 1)}
    m.set((0, 1), N, WALL)
    r = m.transformed(1, (10, 0))  # N -> E, then shift
    assert r.cells == {(10, 0), (11, 0)}
    assert r.get((11, 0), E) == WALL


def test_evidence_votes_and_threshold():
    m = MazeMap(MapEvidence(wall_threshold=1.0, score_limit=3.0))
    assert m.observe((0, 0), N, True, 0.5) == UNKNOWN
    assert m.observe((0, 0), N, True, 0.5) == WALL
    assert m.observe((0, 1), S, False, 1.0) == UNKNOWN  # same wall, other side
    assert m.observe((0, 0), N, False, 1.0) == OPEN


def test_evidence_is_clamped_so_it_can_flip():
    m = MazeMap(MapEvidence(wall_threshold=1.0, score_limit=2.0))
    for _ in range(10):
        m.observe((0, 0), E, True, 1.0)
    assert m.score((0, 0), E) == 2.0
    m.observe((0, 0), E, False, 1.0)
    m.observe((0, 0), E, False, 1.0)
    m.observe((0, 0), E, False, 1.0)
    assert m.state((0, 0), E) == OPEN


def test_traversed_edge_is_locked_open_and_conflicts_counted():
    m = MazeMap()
    m.mark_traversed((0, 0), W)
    assert m.observe((-1, 0), E, True, 10.0) == OPEN
    assert m.state((0, 0), W) == OPEN
    assert m.conflicts == {edge_key((0, 0), W): 1}


def test_map_grows_in_any_direction_and_excludes_cells_behind_walls():
    m = MazeMap()
    m.mark_visited((0, 0))
    m.observe((0, 0), W, False, 1.0)
    m.observe((0, 0), S, False, 1.0)
    m.observe((0, 0), N, True, 1.0)
    assert m.cells() == {(0, 0), (-1, 0), (0, -1)}
    assert not m.is_cell_known((0, 0))
    m.observe((0, 0), E, True, 1.0)
    assert m.is_cell_known((0, 0))
    assert m.to_wallmap().bounds() == (-1, -1, 0, 0)


def test_maze_map_json_round_trip_and_wallmap_compat():
    m = MazeMap()
    m.mark_visited((0, 0))
    m.observe((0, 0), N, True, 1.0)
    m.observe((0, 0), E, False, 0.3)
    m.mark_traversed((0, 0), W)
    data = json.loads(json.dumps(m.to_dict()))
    back = MazeMap.from_dict(data)
    assert back.to_dict() == m.to_dict()
    assert WallMap.from_dict(data).to_dict() == m.to_wallmap().to_dict()


def test_robot_ascii_marks_start_and_current():
    m = MazeMap()
    m.mark_visited((0, 0))
    m.mark_traversed((0, 0), E)
    m.mark_visited((1, 0))
    text = m.to_ascii(current=(1, 0))
    parsed, markers = WallMap.from_ascii(text, origin_marker="S")
    assert markers == {(0, 0): "S", (1, 0): "R"}
    assert parsed.get((0, 0), E) == OPEN
