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


@pytest.mark.parametrize("true_heading", [0.0, 4.0, -7.0, 90.0 + 3.0, 180.0 - 5.0])
def test_heading_from_two_hits_on_one_wall(true_heading):
    import math

    from slam.perception import heading_from_wall

    # Wall along x at y = 0.25 (the N side of a cell), robot at the origin.
    def hit(g):
        ang = math.radians(true_heading + g)
        return 0.25 / math.cos(ang)  # distance along the ray to y = 0.25

    g1 = -true_heading  # gimbal pointing north in the map
    g2 = g1 + 25.0
    h = heading_from_wall(g1, hit(g1), g2, hit(g2), Direction.N, believed_deg=0.0 if abs(true_heading) < 45 else true_heading)
    assert h == pytest.approx(true_heading, abs=1e-6)
