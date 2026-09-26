"""Motion primitives with EKF feedback (DESIGN.md section 7).

Every control tick: EKF predict from odometry + IMU, then EKF corrections
from the Sharp sensors (and the front ToF while driving) against walls that
are already in the map. The controllers steer on the EKF pose.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

from .config import Config
from .geometry import Cell, Direction, Line, Pose, angle_diff, cell_center, edge_line, heading_vector
from .localizer import EKF
from .logger import RunLogger
from .maze_map import EdgeState, MazeMap
from .perception import associate_wall, ray_of
from .robot_api import RobotAPI


class MoveResult(NamedTuple):
    ok: bool
    reason: str
    # How far the robot got along the intended direction (m, EKF estimate).
    progress_m: float = 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _speed(error: float, gain: float, lo: float, hi: float) -> float:
    """Proportional command with a minimum magnitude so it never stalls."""
    return math.copysign(_clamp(abs(error) * gain, lo, hi), error)


class Navigator:
    def __init__(
        self,
        robot: RobotAPI,
        ekf: EKF,
        maze_map: MazeMap,
        cfg: Config,
        logger: Optional[RunLogger] = None,
    ) -> None:
        self.robot = robot
        self.ekf = ekf
        self.map = maze_map
        self.cfg = cfg
        self.log = logger
        self.dt = 1.0 / cfg.motion.loop_hz
        self.ekf_updates = 0
        self.ekf_rejects = 0

    # ---- one control tick ------------------------------------------------------
    def tick(
        self, front_tof: bool, tentative_front: Optional[Line] = None
    ) -> Tuple[Pose, Optional[float]]:
        """Predict + correct. Returns (EKF pose, front ToF reading if requested).

        ``tentative_front``: a wall line not yet in the map (the far side of
        the cell being entered). A front ToF reading that cannot be tied to a
        known wall is tried against it; the EKF gate rejects it if no wall is
        really there.
        """
        r = self.robot
        t = r.now()
        self.ekf.predict(r.odometry(), r.imu_yaw(), t)

        s = self.cfg.sensors
        left, right = r.sharp()
        for name, mount, value in (("sharp_left", s.sharp_left, left), ("sharp_right", s.sharp_right, right)):
            if value is not None:
                self._range_update(name, mount, 0.0, value, self.ekf.sharp_std(value))

        tof = None
        if front_tof:
            tof = r.tof()
            if tof is not None:
                self._range_update(
                    "tof_front", s.tof, r.gimbal_yaw(), tof, self.cfg.ekf.tof_std_m, tentative_front
                )

        if self.log is not None:
            true_pose = getattr(r, "true_pose_map", None)
            self.log.trajectory(
                t,
                self.ekf.pose,
                self.ekf.std(),
                r.odometry(),
                r.imu_yaw(),
                true_pose() if true_pose else None,
            )
        return self.ekf.pose, tof

    def _range_update(
        self,
        name: str,
        mount,
        gimbal_deg: float,
        value: float,
        std: float,
        tentative: Optional[Line] = None,
    ) -> bool:
        """EKF update for one reading, if it can be tied to a known wall."""
        cell_size = self.cfg.geometry.cell_size_m
        ray = ray_of(self.ekf.pose, mount, gimbal_deg)
        assoc = associate_wall(self.map, ray, cell_size, self.cfg.perception)
        line = assoc.line if assoc is not None else tentative
        status = "no_wall"
        accepted = False
        if line is not None:
            res = self.ekf.update_range(value, mount, gimbal_deg, line, std)
            accepted = res.accepted
            status = "ok" if accepted else f"rejected({res.mahalanobis:.1f}sigma)"
            if accepted:
                self.ekf_updates += 1
            else:
                self.ekf_rejects += 1
        if self.log is not None:
            self.log.sensor(
                self.robot.now(),
                name,
                self.ekf.cell(cell_size),
                assoc.direction.name if assoc else ("tentative" if line is not None else ""),
                gimbal_deg,
                value,
                ekf=status,
            )
        return accepted

    # ---- primitives --------------------------------------------------------------
    def turn_to(self, heading_deg: float) -> MoveResult:
        m = self.cfg.motion
        r = self.robot
        t_end = r.now() + m.turn_timeout_s
        stable = 0
        try:
            while r.now() < t_end:
                pose, _ = self.tick(front_tof=False)
                err = angle_diff(heading_deg, pose.heading_deg)
                if abs(err) <= m.turn_tol_deg:
                    stable += 1
                    r.stop()
                    if stable >= m.turn_stable_ticks:
                        return MoveResult(True, "ok")
                else:
                    stable = 0
                    r.drive_speed(0.0, 0.0, _speed(err, m.k_heading, m.min_turn_dps, m.max_turn_dps))
                r.sleep(self.dt)
        finally:
            r.stop()
        return MoveResult(False, "turn_timeout")

    def drive_to(self, target: Cell, direction: Direction) -> MoveResult:
        """Drive to the centre of ``target`` along ``direction`` (forward or back)."""
        m = self.cfg.motion
        r = self.robot
        tx, ty = cell_center(target, self.cfg.geometry.cell_size_m)
        ux, uy = heading_vector(direction.heading_deg)
        rx, ry = heading_vector(direction.heading_deg + 90.0)
        start = self.ekf.pose
        far = None
        if self.map.state(target, direction) == EdgeState.UNKNOWN:
            far = edge_line(target, direction, self.cfg.geometry.cell_size_m)
        r.gimbal_moveto(0.0)
        t_end = r.now() + m.drive_timeout_s

        def progress(p: Pose) -> float:
            return (p.x - start.x) * ux + (p.y - start.y) * uy

        try:
            while True:
                pose, tof = self.tick(front_tof=True, tentative_front=far)
                remaining = (tx - pose.x) * ux + (ty - pose.y) * uy
                lateral = (pose.x - tx) * rx + (pose.y - ty) * ry
                if abs(remaining) <= m.pos_tol_m:
                    return MoveResult(True, "ok", progress(pose))
                if r.now() > t_end:
                    return MoveResult(False, "drive_timeout", progress(pose))
                forward = remaining > 0
                if forward and tof is not None and tof < m.emergency_front_m:
                    return MoveResult(False, "front_obstacle", progress(pose))
                ir_left, ir_right = r.ir_front()
                if forward and ir_left and ir_right:
                    return MoveResult(False, "ir_blocked", progress(pose))

                strafe = _clamp(-m.k_lateral * lateral, -m.max_strafe_mps, m.max_strafe_mps)
                if ir_left:
                    strafe += m.ir_avoid_mps
                if ir_right:
                    strafe -= m.ir_avoid_mps
                herr = angle_diff(direction.heading_deg, pose.heading_deg)
                turn = _clamp(m.k_heading * herr, -m.max_heading_hold_dps, m.max_heading_hold_dps)
                speed = _speed(remaining, m.k_along, m.min_mps, m.drive_mps)
                r.drive_speed(speed, strafe, turn)
                r.sleep(self.dt)
        finally:
            r.stop()
