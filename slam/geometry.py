"""Directions, angles, cell <-> metric conversions and ray casting.

Conventions (DESIGN.md section 3): +y = N, +x = E, heading in degrees with
0 = N and clockwise positive, so the unit vector of heading h is (sin h, cos h).
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import NamedTuple, Optional, Tuple

Cell = Tuple[int, int]


class Direction(IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3

    @property
    def dx(self) -> int:
        return (0, 1, 0, -1)[self]

    @property
    def dy(self) -> int:
        return (1, 0, -1, 0)[self]

    @property
    def heading_deg(self) -> float:
        return 90.0 * int(self)

    def opposite(self) -> "Direction":
        return Direction((self + 2) % 4)

    def turned(self, quarter_turns_cw: int) -> "Direction":
        return Direction((self + quarter_turns_cw) % 4)

    @staticmethod
    def from_heading(heading_deg: float) -> "Direction":
        """Nearest cardinal direction to a heading."""
        return Direction(int(round(wrap_deg(heading_deg) / 90.0)) % 4)


def neighbor(cell: Cell, d: Direction) -> Cell:
    return (cell[0] + d.dx, cell[1] + d.dy)


def wrap_deg(deg: float) -> float:
    """Wrap to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def angle_diff(target_deg: float, current_deg: float) -> float:
    """Shortest signed rotation from current to target, in [-180, 180)."""
    return wrap_deg(target_deg - current_deg)


def heading_vector(heading_deg: float) -> Tuple[float, float]:
    rad = math.radians(heading_deg)
    return math.sin(rad), math.cos(rad)


def cell_center(cell: Cell, cell_size: float) -> Tuple[float, float]:
    return cell[0] * cell_size, cell[1] * cell_size


def cell_of(x: float, y: float, cell_size: float) -> Cell:
    return int(math.floor(x / cell_size + 0.5)), int(math.floor(y / cell_size + 0.5))


class Pose(NamedTuple):
    x: float
    y: float
    heading_deg: float


def body_to_map(pose: Pose, forward_m: float, right_m: float) -> Tuple[float, float]:
    """Map-frame position of a point given in the body frame (forward, right)."""
    fx, fy = heading_vector(pose.heading_deg)
    rx, ry = heading_vector(pose.heading_deg + 90.0)
    return pose.x + forward_m * fx + right_m * rx, pose.y + forward_m * fy + right_m * ry


class Segment(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


def wall_segment(cell: Cell, d: Direction, cell_size: float) -> Segment:
    """Segment of the wall on side ``d`` of ``cell``."""
    cx, cy = cell_center(cell, cell_size)
    h = cell_size / 2.0
    if d == Direction.N:
        return Segment(cx - h, cy + h, cx + h, cy + h)
    if d == Direction.S:
        return Segment(cx - h, cy - h, cx + h, cy - h)
    if d == Direction.E:
        return Segment(cx + h, cy - h, cx + h, cy + h)
    return Segment(cx - h, cy - h, cx - h, cy + h)


def ray_segment_distance(
    ox: float, oy: float, heading_deg: float, seg: Segment
) -> Optional[float]:
    """Distance along the ray to the segment, or None if the ray misses it."""
    dx, dy = heading_vector(heading_deg)
    sx, sy = seg.x2 - seg.x1, seg.y2 - seg.y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-12:
        return None  # parallel
    qx, qy = seg.x1 - ox, seg.y1 - oy
    t = (qx * sy - qy * sx) / denom  # along the ray
    u = (qx * dy - qy * dx) / denom  # along the segment, 0..1
    if t < 0.0 or u < -1e-9 or u > 1.0 + 1e-9:
        return None
    return t


def point_segment_distance(px: float, py: float, seg: Segment) -> float:
    sx, sy = seg.x2 - seg.x1, seg.y2 - seg.y1
    length_sq = sx * sx + sy * sy
    if length_sq == 0.0:
        return math.hypot(px - seg.x1, py - seg.y1)
    u = ((px - seg.x1) * sx + (py - seg.y1) * sy) / length_sq
    u = max(0.0, min(1.0, u))
    return math.hypot(px - (seg.x1 + u * sx), py - (seg.y1 + u * sy))


class Line(NamedTuple):
    """An infinite grid line: x = value (axis 'x') or y = value (axis 'y')."""

    axis: str
    value: float


def edge_line(cell: Cell, d: Direction, cell_size: float) -> Line:
    """The grid line on side ``d`` of ``cell``."""
    cx, cy = cell_center(cell, cell_size)
    h = cell_size / 2.0
    if d == Direction.N:
        return Line("y", cy + h)
    if d == Direction.S:
        return Line("y", cy - h)
    if d == Direction.E:
        return Line("x", cx + h)
    return Line("x", cx - h)


def ray_line_distance(ox: float, oy: float, heading_deg: float, line: Line) -> Optional[float]:
    """Distance along the ray to the line, None if parallel or behind."""
    dx, dy = heading_vector(heading_deg)
    if line.axis == "x":
        comp, origin = dx, ox
    else:
        comp, origin = dy, oy
    if abs(comp) < 1e-9:
        return None
    t = (line.value - origin) / comp
    return t if t >= 0.0 else None


def sensor_ray(
    pose: Pose, forward_m: float, right_m: float, angle_deg: float
) -> Tuple[float, float, float]:
    """(x, y, heading) of a sensor ray in the map frame."""
    ox, oy = body_to_map(pose, forward_m, right_m)
    return ox, oy, wrap_deg(pose.heading_deg + angle_deg)
