import pytest

from slam.config import DEFAULT, PERFECT_SIM
from slam.geometry import Direction, Line
from slam.maze_map import MazeMap, WallMap
from slam.perception import Ray, Verdict, associate_wall, classify_range, look, robust_median
from slam.sim import SimRobot

P = DEFAULT.perception
N, E, S, W = Direction.N, Direction.E, Direction.S, Direction.W


@pytest.mark.parametrize(
    "distance,expected",
    [(0.30, Verdict.WALL), (0.05, Verdict.WALL), (0.41, Verdict.WALL), (0.46, Verdict.UNSURE),
     (0.55, Verdict.OPEN), (0.90, Verdict.OPEN), (None, Verdict.UNSURE)],
)
def test_classify_tof(distance, expected):
    assert classify_range(distance, 0.30, P) == expected


def test_classify_sharp_none_means_open():
    assert classify_range(None, 0.18, P, none_means_open=True) == Verdict.OPEN


def test_robust_median_needs_enough_valid_samples():
    assert robust_median([0.3, None, 0.31, 0.29, None], 3) == pytest.approx(0.30)
    assert robust_median([0.3, None, None], 3) is None


def test_associate_wall_walks_through_open_edges():
    m = MazeMap()
    m.observe((0, 0), N, False, 1.0)
    m.observe((0, 1), N, False, 1.0)
    m.observe((0, 2), N, True, 1.0)
    a = associate_wall(m, Ray(0.0, 0.0, 3.0), 0.6, P)
    assert a is not None
    assert (a.cell, a.direction, a.line) == ((0, 2), N, Line("y", 1.5))


def test_associate_wall_refuses_unknown_and_off_axis():
    m = MazeMap()
    m.observe((0, 0), N, False, 1.0)  # (0, 1) N is unknown
    assert associate_wall(m, Ray(0.0, 0.0, 0.0), 0.6, P) is None
    m.observe((0, 0), E, True, 1.0)
    assert associate_wall(m, Ray(0.0, 0.0, 90.0), 0.6, P) is not None
    assert associate_wall(m, Ray(0.0, 0.0, 45.0), 0.6, P) is None


def test_look_points_gimbal_and_takes_median():
    maze, _ = WallMap.from_ascii("+---+---+\n|       |\n+---+---+\n")
    robot = SimRobot(maze, (0, 0), Direction.N, noise=PERFECT_SIM)
    assert look(robot, 0.0, P) == pytest.approx(0.30)
    assert look(robot, 90.0, P) == pytest.approx(0.90)
    assert robot.gimbal_yaw() == pytest.approx(90.0)
