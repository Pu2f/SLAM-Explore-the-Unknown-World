import math

import pytest

from slam.geometry import (
    Direction,
    Pose,
    Segment,
    angle_diff,
    body_to_map,
    cell_of,
    heading_vector,
    point_segment_distance,
    ray_segment_distance,
    wrap_deg,
)


def test_direction_vectors_match_compass_headings():
    for d in Direction:
        vx, vy = heading_vector(d.heading_deg)
        assert (round(vx), round(vy)) == (d.dx, d.dy)


def test_direction_turns_and_opposite():
    assert Direction.N.turned(1) == Direction.E
    assert Direction.W.turned(1) == Direction.N
    assert Direction.N.turned(-1) == Direction.W
    assert Direction.E.opposite() == Direction.W
    assert Direction.from_heading(-91) == Direction.W
    assert Direction.from_heading(179) == Direction.S


def test_angles_wrap():
    assert wrap_deg(180) == -180
    assert wrap_deg(-190) == pytest.approx(170)
    assert angle_diff(10, 350) == pytest.approx(20)
    assert angle_diff(350, 10) == pytest.approx(-20)


def test_cell_of_rounds_to_nearest_center():
    assert cell_of(0.29, -0.29, 0.6) == (0, 0)
    assert cell_of(0.31, -0.31, 0.6) == (1, -1)


def test_body_to_map_right_is_clockwise_of_forward():
    # Facing E: forward = +x, right = -y.
    x, y = body_to_map(Pose(0, 0, 90), forward_m=1.0, right_m=0.5)
    assert (x, y) == (pytest.approx(1.0), pytest.approx(-0.5))


def test_ray_hits_and_misses_segment():
    wall = Segment(-0.3, 0.3, 0.3, 0.3)  # north wall of the origin cell
    assert ray_segment_distance(0, 0, 0, wall) == pytest.approx(0.3)
    assert ray_segment_distance(0, 0, 45, wall) == pytest.approx(0.3 * math.sqrt(2))
    assert ray_segment_distance(0, 0, 180, wall) is None  # behind
    assert ray_segment_distance(0, 0, 90, wall) is None  # parallel
    assert ray_segment_distance(0, 0, 60, wall) is None  # passes beyond the end


def test_point_segment_distance_uses_endpoints():
    wall = Segment(0, 0, 1, 0)
    assert point_segment_distance(0.5, 0.2, wall) == pytest.approx(0.2)
    assert point_segment_distance(1.3, 0.4, wall) == pytest.approx(0.5)
