"""Run output: the Exploration Log, trajectory, sensor readings, map, summary.

Files written to the run directory:
  exploration_log.csv  one row per event / motion command (time, cell, pose, details)
  trajectory.csv       pose at the control rate (EKF, odometry, IMU, sim truth)
  sensors.csv          every sensor reading used for mapping or localization
  map.json, map.txt    the final map (readable by tools/evaluate.py)
  summary.json         start / end, duration, stop reason, counts
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Optional

from .geometry import Cell, Pose
from .maze_map import MazeMap


class RunLogger:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._t0: Optional[float] = None
        self._events = self._open_csv(
            "exploration_log.csv",
            ["t_s", "event", "cell_x", "cell_y", "x_m", "y_m", "heading_deg", "detail"],
        )
        self._traj = self._open_csv(
            "trajectory.csv",
            [
                "t_s",
                "x_m",
                "y_m",
                "heading_deg",
                "std_x_m",
                "std_y_m",
                "std_heading_deg",
                "odom_x_m",
                "odom_y_m",
                "imu_yaw_deg",
                "true_x_m",
                "true_y_m",
                "true_heading_deg",
            ],
        )
        self._sensors = self._open_csv(
            "sensors.csv",
            ["t_s", "sensor", "cell_x", "cell_y", "direction", "gimbal_deg", "value_m", "verdict", "ekf"],
        )

    def _open_csv(self, name: str, header):
        f = open(os.path.join(self.out_dir, name), "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(header)
        return f, w

    def _rel(self, t: float) -> str:
        if self._t0 is None:
            self._t0 = t
        return f"{t - self._t0:.3f}"

    @staticmethod
    def _num(v: Optional[float], fmt: str = "{:.4f}") -> str:
        return "" if v is None else fmt.format(v)

    def event(self, t: float, name: str, cell: Cell, pose: Pose, **detail: Any) -> None:
        f, w = self._events
        w.writerow(
            [
                self._rel(t),
                name,
                cell[0],
                cell[1],
                f"{pose.x:.4f}",
                f"{pose.y:.4f}",
                f"{pose.heading_deg:.2f}",
                json.dumps(detail, default=str) if detail else "",
            ]
        )
        f.flush()

    def trajectory(
        self,
        t: float,
        pose: Pose,
        std: tuple,
        odom: tuple,
        imu_yaw: float,
        true_pose: Optional[Pose] = None,
    ) -> None:
        _, w = self._traj
        w.writerow(
            [
                self._rel(t),
                f"{pose.x:.4f}",
                f"{pose.y:.4f}",
                f"{pose.heading_deg:.2f}",
                f"{std[0]:.4f}",
                f"{std[1]:.4f}",
                f"{std[2]:.3f}",
                f"{odom[0]:.4f}",
                f"{odom[1]:.4f}",
                f"{imu_yaw:.2f}",
                self._num(true_pose.x if true_pose else None),
                self._num(true_pose.y if true_pose else None),
                self._num(true_pose.heading_deg if true_pose else None, "{:.2f}"),
            ]
        )

    def sensor(
        self,
        t: float,
        sensor: str,
        cell: Cell,
        direction: str,
        gimbal_deg: Optional[float],
        value: Optional[float],
        verdict: str = "",
        ekf: str = "",
    ) -> None:
        _, w = self._sensors
        w.writerow(
            [
                self._rel(t),
                sensor,
                cell[0],
                cell[1],
                direction,
                self._num(gimbal_deg, "{:.1f}"),
                self._num(value),
                verdict,
                ekf,
            ]
        )

    def save_map(self, maze_map: MazeMap, current: Optional[Cell] = None) -> None:
        with open(os.path.join(self.out_dir, "map.json"), "w", encoding="utf-8") as f:
            json.dump(maze_map.to_dict(), f, indent=1)
        with open(os.path.join(self.out_dir, "map.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Robot map (S = start, R = end, . = visited; start heading = up)\n"
                + maze_map.to_ascii(current)
                + "\n"
            )

    def save_summary(self, summary: Dict[str, Any]) -> None:
        with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    def close(self) -> None:
        for f, _ in (self._events, self._traj, self._sensors):
            f.close()
