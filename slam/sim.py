"""Simulated maze + robot for testing mission logic without hardware.

``generate_maze`` builds a random ground-truth WallMap of any size.
``SimRobot`` implements RobotAPI on top of it: kinematics, wall collisions,
wheel slip, IMU drift and noisy ToF / Sharp / IR readings.

Two frames are involved:
- the *world* frame of the ground-truth map (cells as in the GT file);
- the robot's *map* frame (DESIGN.md section 3): origin at the start cell
  centre, +y = the direction the robot faced at the start.
Everything returned through RobotAPI is in the map frame, exactly like the
real robot, which does not know where in the world it started.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from .config import DEFAULT, Config, Mount, SimNoise
from .geometry import (
    Cell,
    Direction,
    Pose,
    Segment,
    body_to_map,
    cell_center,
    heading_vector,
    neighbor,
    point_segment_distance,
    ray_segment_distance,
    wall_segment,
    wrap_deg,
)
from .maze_map import EdgeState, WallMap, edge_cells
from .robot_api import RobotAPI


def generate_maze(width: int, height: int, seed: int = 0, loops: int = 0) -> WallMap:
    """Random maze with cells (0..width-1, 0..height-1) and a closed outer wall.

    A depth-first "recursive backtracker" makes a perfect maze (exactly one
    path between any two cells); ``loops`` extra inner walls are then removed
    so the maze has cycles, which DFS / Trémaux must handle.
    """
    if width < 1 or height < 1:
        raise ValueError("maze must be at least 1x1")
    rng = random.Random(seed)
    m = WallMap()
    m.cells = {(x, y) for x in range(width) for y in range(height)}
    for c in m.cells:
        for d in Direction:
            m.set(c, d, EdgeState.WALL)

    start = (0, 0)
    seen = {start}
    stack = [start]
    while stack:
        cell = stack[-1]
        options = [
            d for d in Direction if neighbor(cell, d) in m.cells and neighbor(cell, d) not in seen
        ]
        if not options:
            stack.pop()
            continue
        d = rng.choice(options)
        m.set(cell, d, EdgeState.OPEN)
        nxt = neighbor(cell, d)
        seen.add(nxt)
        stack.append(nxt)

    inner_walls = [
        (key, s)
        for key, s in m.edges()
        if s == EdgeState.WALL and all(c in m.cells for c in edge_cells(key))
    ]
    rng.shuffle(inner_walls)
    for (x, y, d), _ in inner_walls[:loops]:
        m.set((x, y), d, EdgeState.OPEN)
    return m


def wall_segments(maze: WallMap, cell_size: float) -> List[Segment]:
    return [
        wall_segment((x, y), d, cell_size)
        for (x, y, d), s in maze.edges()
        if s == EdgeState.WALL
    ]


class SimRobot(RobotAPI):
    def __init__(
        self,
        maze: WallMap,
        start_cell: Cell,
        start_heading: Direction = Direction.N,
        config: Config = DEFAULT,
        noise: Optional[SimNoise] = None,
        seed: int = 0,
    ) -> None:
        self.maze = maze
        self.cfg = config
        self.noise = noise if noise is not None else config.sim_noise
        self.rng = random.Random(seed)
        self.cell_size = config.geometry.cell_size_m
        self.segments = wall_segments(maze, self.cell_size)

        # Rotation from the map frame to the world frame (clockwise, degrees).
        self.start_cell = start_cell
        self.start_heading = start_heading
        self._frame_deg = start_heading.heading_deg
        sx, sy = cell_center(start_cell, self.cell_size)
        self._frame_origin = (sx, sy)

        # True state in the world frame.
        self.x, self.y = sx, sy
        self.heading = self._frame_deg
        self.gimbal = 0.0
        self._gimbal_target = 0.0
        self.t = 0.0

        # What the robot's own sensors believe.
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._imu_drift = 0.0

        self._cmd = (0.0, 0.0, 0.0)
        self._slip = 1.0 + self.noise.slip_bias + self.rng.gauss(0.0, self.noise.slip_std)
        self._turn_scale = 1.0 + self.rng.gauss(0.0, self.noise.turn_scale_std)
        self.collisions = 0
        self._in_contact = False

    # ---- frames -------------------------------------------------------------
    def map_to_world(self, x: float, y: float) -> Tuple[float, float]:
        a = math.radians(self._frame_deg)
        wx = x * math.cos(a) + y * math.sin(a)
        wy = -x * math.sin(a) + y * math.cos(a)
        return wx + self._frame_origin[0], wy + self._frame_origin[1]

    def world_to_map(self, x: float, y: float) -> Tuple[float, float]:
        a = math.radians(-self._frame_deg)
        dx, dy = x - self._frame_origin[0], y - self._frame_origin[1]
        return dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)

    def true_pose_world(self) -> Pose:
        return Pose(self.x, self.y, wrap_deg(self.heading))

    def true_pose_map(self) -> Pose:
        mx, my = self.world_to_map(self.x, self.y)
        return Pose(mx, my, wrap_deg(self.heading - self._frame_deg))

    # ---- RobotAPI: time ------------------------------------------------------
    def now(self) -> float:
        return self.t

    def sleep(self, dt_s: float) -> None:
        remaining = dt_s
        while remaining > 1e-9:
            step = min(self.cfg.sim_dt_s, remaining)
            self._step(step)
            remaining -= step

    # ---- RobotAPI: chassis -----------------------------------------------------
    def drive_speed(self, forward_mps: float, right_mps: float, turn_dps: float) -> None:
        self._cmd = (float(forward_mps), float(right_mps), float(turn_dps))

    def odometry(self) -> Tuple[float, float]:
        return self._odom_x, self._odom_y

    def imu_yaw(self) -> float:
        true_map_heading = self.heading - self._frame_deg
        return wrap_deg(true_map_heading + self._imu_drift + self.rng.gauss(0.0, self.noise.imu_std_deg))

    # ---- RobotAPI: gimbal + ToF ----------------------------------------------
    def gimbal_moveto(self, yaw_deg: float) -> None:
        self._gimbal_target = float(yaw_deg)
        deadline = self.t + 10.0
        while abs(self._gimbal_target - self.gimbal) > 1e-6 and self.t < deadline:
            self.sleep(self.cfg.sim_dt_s)
        # Pointing error stays until the next move, like a real gimbal.
        self.gimbal = self._gimbal_target + self.rng.gauss(0.0, self.noise.gimbal_std_deg)
        self._gimbal_target = self.gimbal

    def gimbal_yaw(self) -> float:
        return self.gimbal

    def tof(self) -> Optional[float]:
        s = self.cfg.sensors
        d = self._ray(s.tof, extra_angle_deg=self.gimbal)
        if d is None:
            return None
        d += self.rng.gauss(0.0, self.noise.tof_std_m)
        if d < s.tof_min_m or d > s.tof_max_m:
            return None
        return d

    # ---- RobotAPI: side / front IR -----------------------------------------------
    def sharp(self) -> Tuple[Optional[float], Optional[float]]:
        s = self.cfg.sensors
        out = []
        for mount in (s.sharp_left, s.sharp_right):
            d = self._ray(mount)
            if d is not None:
                d += self.rng.gauss(0.0, self.noise.sharp_std_frac * d)
            out.append(d if d is not None and s.sharp_min_m <= d <= s.sharp_max_m else None)
        return out[0], out[1]

    def ir_front(self) -> Tuple[bool, bool]:
        s = self.cfg.sensors
        hits = []
        for mount in (s.ir_front_left, s.ir_front_right):
            d = self._ray(mount)
            hits.append(d is not None and d <= s.ir_range_m)
        return hits[0], hits[1]

    # ---- internals -------------------------------------------------------------------
    def _ray(self, mount: Mount, extra_angle_deg: float = 0.0) -> Optional[float]:
        """Noise-free distance from a sensor to the nearest wall, None if none."""
        pose = Pose(self.x, self.y, self.heading)
        ox, oy = body_to_map(pose, mount.forward_m, mount.right_m)
        heading = self.heading + mount.angle_deg + extra_angle_deg
        best: Optional[float] = None
        for seg in self.segments:
            d = ray_segment_distance(ox, oy, heading, seg)
            if d is not None and (best is None or d < best):
                best = d
        return best

    def _clearance(self, x: float, y: float) -> float:
        return min(
            (point_segment_distance(x, y, seg) for seg in self.segments), default=math.inf
        )

    def _step(self, dt: float) -> None:
        fwd, right, turn = self._cmd

        # Gimbal slews toward its target.
        g_err = self._gimbal_target - self.gimbal
        g_max = self.cfg.gimbal_speed_dps * dt
        self.gimbal += max(-g_max, min(g_max, g_err))

        # True motion (with slip), in the world frame.
        true_turn = turn * self._turn_scale * dt
        mid_heading = self.heading + true_turn / 2.0
        fx, fy = heading_vector(mid_heading)
        rx, ry = heading_vector(mid_heading + 90.0)
        dist_f = fwd * self._slip * dt
        dist_r = right * self._slip * dt
        nx = self.x + dist_f * fx + dist_r * rx
        ny = self.y + dist_f * fy + dist_r * ry

        radius = self.cfg.geometry.robot_radius_m
        old_clear = self._clearance(self.x, self.y)
        new_clear = self._clearance(nx, ny)
        if new_clear < radius and new_clear < old_clear:
            if not self._in_contact:
                self.collisions += 1
            self._in_contact = True
        else:
            self._in_contact = new_clear < radius
            self.x, self.y = nx, ny
        self.heading = wrap_deg(self.heading + true_turn)

        # Wheel odometry: integrates the *commanded* wheel motion (it cannot
        # see slip) along the heading the IMU reports, in the map frame.
        self._imu_drift += self.noise.imu_drift_dps * dt
        imu_heading = self.heading - self._frame_deg + self._imu_drift
        ofx, ofy = heading_vector(imu_heading)
        orx, ory = heading_vector(imu_heading + 90.0)
        self._odom_x += fwd * dt * ofx + right * dt * orx
        self._odom_y += fwd * dt * ofy + right * dt * ory

        self.t += dt

