"""Raw sensor readings -> wall verdicts, and which wall a reading belongs to."""

from __future__ import annotations

from enum import Enum
from statistics import median
from typing import Iterable, NamedTuple, Optional

from .config import Mount, Perception
from .geometry import (
    Cell,
    Direction,
    Line,
    Pose,
    angle_diff,
    cell_of,
    edge_line,
    neighbor,
    ray_line_distance,
    sensor_ray,
)
from .maze_map import EdgeState, MazeMap
from .robot_api import RobotAPI


class Verdict(str, Enum):
    WALL = "wall"
    OPEN = "open"
    UNSURE = "unsure"


class Ray(NamedTuple):
    x: float
    y: float
    heading_deg: float


def ray_of(pose: Pose, mount: Mount, extra_angle_deg: float = 0.0) -> Ray:
    return Ray(*sensor_ray(pose, mount.forward_m, mount.right_m, mount.angle_deg + extra_angle_deg))


def classify_range(
    distance: Optional[float],
    edge_distance: float,
    p: Perception,
    none_means_open: bool = False,
) -> Verdict:
    """Is there a wall on the side of the current cell this ray points at?

    ``none_means_open`` is for Sharp sensors, which report None beyond their
    maximum range (the ToF's range covers the whole maze, so for it None
    means a bad reading).
    """
    if distance is None:
        return Verdict.OPEN if none_means_open else Verdict.UNSURE
    if distance <= edge_distance + p.wall_tol_m:
        return Verdict.WALL
    if distance >= edge_distance + p.open_margin_m:
        return Verdict.OPEN
    return Verdict.UNSURE


def edge_distance(ray: Ray, cell: Cell, d: Direction, cell_size: float) -> Optional[float]:
    """Distance along the ray to side ``d`` of ``cell``."""
    return ray_line_distance(ray.x, ray.y, ray.heading_deg, edge_line(cell, d, cell_size))


def robust_median(values: Iterable[Optional[float]], min_valid: int) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if len(valid) < min_valid:
        return None
    return float(median(valid))


def look(robot: RobotAPI, gimbal_yaw_deg: float, p: Perception) -> Optional[float]:
    """Point the ToF, then take the median of several stationary readings."""
    robot.gimbal_moveto(gimbal_yaw_deg)
    samples = []
    for _ in range(p.tof_samples):
        samples.append(robot.tof())
        robot.sleep(p.tof_sample_delay_s)
    return robust_median(samples, p.tof_min_valid_samples)


class Association(NamedTuple):
    cell: Cell
    direction: Direction
    line: Line


def associate_wall(
    maze_map: MazeMap, ray: Ray, cell_size: float, p: Perception
) -> Optional[Association]:
    """The known wall this ray should hit first, or None if not sure.

    Walks from the sensor's cell along the nearest grid axis: through OPEN
    sides, stopping at the first WALL. An UNKNOWN side on the way means the
    reading cannot be trusted for localization.
    """
    d = Direction.from_heading(ray.heading_deg)
    if abs(angle_diff(d.heading_deg, ray.heading_deg)) > p.max_off_axis_deg:
        return None
    cell = cell_of(ray.x, ray.y, cell_size)
    for _ in range(p.max_association_cells):
        state = maze_map.state(cell, d)
        if state == EdgeState.WALL:
            return Association(cell, d, edge_line(cell, d, cell_size))
        if state != EdgeState.OPEN:
            return None
        cell = neighbor(cell, d)
    return None


def heading_from_wall(
    g1_deg: float, d1: float, g2_deg: float, d2: float, wall_side: Direction, believed_deg: float
) -> Optional[float]:
    """Robot heading (map frame) implied by two ToF hits on one straight wall.

    ``g`` = gimbal yaw relative to the chassis, ``d`` = distance from the
    gimbal pivot. The two hits give the wall's direction in the body frame;
    the wall on side ``wall_side`` of a cell runs perpendicular to it in the
    map, which fixes the robot's real heading (the solution nearest the
    believed one). Independent of where the robot stands in the cell and of
    Sharp / odometry errors.
    """
    import math

    p1 = (d1 * math.sin(math.radians(g1_deg)), d1 * math.cos(math.radians(g1_deg)))
    p2 = (d2 * math.sin(math.radians(g2_deg)), d2 * math.cos(math.radians(g2_deg)))
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]  # (right, forward)
    if math.hypot(vx, vy) < 0.03:
        return None
    wall_dir_body = math.degrees(math.atan2(vx, vy))  # compass angle in the body frame
    heading = wall_side.heading_deg + 90.0 - wall_dir_body
    # A wall has no direction: heading is defined modulo 180 deg.
    k = round(angle_diff(believed_deg, heading) / 180.0)
    return heading + 180.0 * k
