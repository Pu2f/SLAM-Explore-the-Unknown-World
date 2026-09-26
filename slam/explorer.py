"""Mission loop: DFS / Trémaux exploration of a maze of unknown size.

At every new cell: stop, look in 4 directions with the ToF (plus the Sharp
sensors as a second opinion), update the map, correct the EKF. Then go
through an OPEN side to a cell never visited, preferring straight / right /
left. With nowhere new to go, step back along the DFS stack (the Trémaux
"second pass" over an edge). The mission ends when no visited cell has an
OPEN side leading to an unvisited cell, so it does not need to return home.

Trémaux marks are kept per edge (1 = passed once, 2 = passed back) for the
log. Because every cell is scanned on arrival, "never enter a visited cell
through a new edge" is the same rule as Trémaux's "treat a junction you have
seen before as a dead end", so loops in the maze are handled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import DEFAULT, Config
from .geometry import Cell, Direction, Pose, angle_diff, neighbor
from .localizer import EKF
from .logger import RunLogger
from .maze_map import EdgeKey, EdgeState, MazeMap, edge_key
from .motion import Navigator
from .perception import Verdict, classify_range, edge_distance, look, ray_of
from .robot_api import RobotAPI


@dataclass
class MissionResult:
    reason: str
    start_pose: Pose
    end_pose: Pose
    end_cell: Cell
    duration_s: float
    path: List[Cell]
    map: MazeMap
    moves: int = 0
    blocked_moves: int = 0
    localization_mismatches: int = 0
    ekf_updates: int = 0
    ekf_rejects: int = 0
    tremaux: Dict[str, int] = field(default_factory=dict)

    def summary(self, cell_size: float) -> dict:
        b = self.map.to_wallmap().bounds()
        size = None if b is None else [b[2] - b[0] + 1, b[3] - b[1] + 1]
        return {
            "stop_reason": self.reason,
            "duration_s": round(self.duration_s, 2),
            "start_cell": [0, 0],
            "end_cell": list(self.end_cell),
            "start_pose": _pose_dict(self.start_pose),
            "end_pose": _pose_dict(self.end_pose),
            "visited_cells": len(self.map.visited),
            "map_cells": len(self.map.cells()),
            "estimated_size_cells": size,
            "estimated_size_m": None if size is None else [size[0] * cell_size, size[1] * cell_size],
            "moves": self.moves,
            "blocked_moves": self.blocked_moves,
            "localization_mismatches": self.localization_mismatches,
            "ekf_updates": self.ekf_updates,
            "ekf_rejects": self.ekf_rejects,
            "cell_path": [list(c) for c in self.path],
            "tremaux_marks": self.tremaux,
        }


def _pose_dict(p: Pose) -> dict:
    return {"x_m": round(p.x, 4), "y_m": round(p.y, 4), "heading_deg": round(p.heading_deg, 2)}


class Explorer:
    def __init__(self, robot: RobotAPI, cfg: Config = DEFAULT, logger: Optional[RunLogger] = None) -> None:
        self.robot = robot
        self.cfg = cfg
        self.log = logger
        self.map = MazeMap(cfg.evidence)
        self.ekf = EKF(cfg.ekf)
        self.nav = Navigator(robot, self.ekf, self.map, cfg, logger)
        self.cell: Cell = (0, 0)
        self.heading = Direction.N
        self.stack: List[Cell] = [(0, 0)]
        self.path: List[Cell] = [(0, 0)]
        self.scans: Dict[Cell, int] = {}
        self.tremaux: Dict[EdgeKey, int] = {}
        self.moves = 0
        self.blocked = 0
        self.mismatches = 0

    # ---- logging helpers -------------------------------------------------------
    def _event(self, name: str, **detail) -> None:
        if self.log is not None:
            self.log.event(self.robot.now(), name, self.cell, self.ekf.pose, **detail)

    # ---- mission -------------------------------------------------------------------
    def run(self) -> MissionResult:
        r = self.robot
        t0 = r.now()
        self.nav.tick(front_tof=False)  # first EKF baseline
        start_pose = self.ekf.pose
        self.map.mark_visited(self.cell)
        self._event("START")
        reason = "complete"
        failures = 0
        try:
            while True:
                limit = self._limit_reason(t0)
                if limit:
                    reason = limit
                    break
                if self._needs_scan(self.cell):
                    self.scan()

                nxt = self._choose_new_cell()
                if nxt is not None:
                    d, target = nxt
                    if self._move(d, target, "EXPLORE"):
                        self.stack.append(target)
                        self.map.mark_visited(target)
                        failures = 0
                    else:
                        failures += 1
                        if failures >= self.cfg.exploration.max_consecutive_failures:
                            reason = "too_many_failed_moves"
                            break
                    continue

                if not self._frontier_exists():
                    break
                if len(self.stack) < 2:
                    reason = "stuck_at_start"  # frontier exists but no way back to it
                    break
                prev = self.stack[-2]
                d = self._direction_to(self.cell, prev)
                if not self._move(d, prev, "BACKTRACK"):
                    reason = "backtrack_failed"
                    break
                self.stack.pop()
        except KeyboardInterrupt:
            reason = "interrupted"
        finally:
            r.stop()

        self._event("END", reason=reason)
        result = MissionResult(
            reason=reason,
            start_pose=start_pose,
            end_pose=self.ekf.pose,
            end_cell=self.cell,
            duration_s=r.now() - t0,
            path=list(self.path),
            map=self.map,
            moves=self.moves,
            blocked_moves=self.blocked,
            localization_mismatches=self.mismatches,
            ekf_updates=self.nav.ekf_updates,
            ekf_rejects=self.nav.ekf_rejects,
            tremaux={f"{k[0]},{k[1]},{k[2].name}": v for k, v in sorted(self.tremaux.items())},
        )
        if self.log is not None:
            self.log.save_map(self.map, self.cell)
            summary = result.summary(self.cfg.geometry.cell_size_m)
            true_pose = getattr(r, "true_pose_map", None)
            if true_pose is not None:
                summary["true_end_pose"] = _pose_dict(true_pose())
            self.log.save_summary(summary)
        return result

    # ---- sensing ---------------------------------------------------------------------
    def _needs_scan(self, cell: Cell) -> bool:
        n = self.scans.get(cell, 0)
        if n == 0:
            return True
        return not self.map.is_cell_known(cell) and n <= self.cfg.exploration.rescan_unsure

    def scan(self) -> None:
        """Look in every direction that is not yet known (all 4 on the first visit)."""
        cfg = self.cfg
        r = self.robot
        cell_size = cfg.geometry.cell_size_m
        first = self.scans.get(self.cell, 0) == 0
        self.nav.tick(front_tof=False)
        pose = self.ekf.pose
        verdicts: Dict[str, str] = {}

        tof_readings: List[Tuple[float, float]] = []
        for k in (0, 1, 2, 3):
            d = self.heading.turned(k)
            if not first and self.map.state(self.cell, d) != EdgeState.UNKNOWN:
                continue
            gimbal = angle_diff(d.heading_deg, pose.heading_deg)
            dist = look(r, gimbal, cfg.perception)
            edge = edge_distance(ray_of(pose, cfg.sensors.tof, gimbal), self.cell, d, cell_size)
            verdict = classify_range(dist, edge if edge is not None else math.inf, cfg.perception)
            if verdict != Verdict.UNSURE:
                self.map.observe(self.cell, d, verdict == Verdict.WALL, cfg.evidence.tof_weight)
            verdicts[d.name] = verdict.value
            if dist is not None:
                tof_readings.append((gimbal, dist))
            if self.log is not None:
                self.log.sensor(r.now(), "tof_scan", self.cell, d.name, gimbal, dist, verdict.value)

        left, right = r.sharp()
        s = cfg.sensors
        for name, mount, value, d in (
            ("sharp_left", s.sharp_left, left, self.heading.turned(3)),
            ("sharp_right", s.sharp_right, right, self.heading.turned(1)),
        ):
            edge = edge_distance(ray_of(pose, mount), self.cell, d, cell_size)
            verdict = classify_range(
                value, edge if edge is not None else math.inf, cfg.perception, none_means_open=True
            )
            if verdict != Verdict.UNSURE:
                self.map.observe(self.cell, d, verdict == Verdict.WALL, cfg.evidence.sharp_weight)
            if self.log is not None:
                self.log.sensor(r.now(), name + "_scan", self.cell, d.name, None, value, verdict.value)

        r.gimbal_moveto(0.0)
        # Now that the map knows these walls, use the same readings to localize.
        for gimbal, dist in tof_readings:
            self.nav._range_update("tof_scan", s.tof, gimbal, dist, cfg.ekf.tof_std_m)

        self.scans[self.cell] = self.scans.get(self.cell, 0) + 1
        sides = {d.name: self.map.state(self.cell, d).value for d in Direction}
        self._event("SCAN", tof=verdicts, sides=sides, std=[round(v, 4) for v in self.ekf.std()])
        if self.log is not None:
            self.log.save_map(self.map, self.cell)

    # ---- decisions -------------------------------------------------------------------
    def _choose_new_cell(self) -> Optional[Tuple[Direction, Cell]]:
        for k in self.cfg.exploration.turn_preference:
            d = self.heading.turned(k)
            target = neighbor(self.cell, d)
            if self.map.state(self.cell, d) == EdgeState.OPEN and target not in self.map.visited:
                return d, target
        return None

    def _frontier_exists(self) -> bool:
        for c in self.map.visited:
            for d in Direction:
                if self.map.state(c, d) == EdgeState.OPEN and neighbor(c, d) not in self.map.visited:
                    return True
        return False

    @staticmethod
    def _direction_to(a: Cell, b: Cell) -> Direction:
        for d in Direction:
            if neighbor(a, d) == b:
                return d
        raise ValueError(f"{a} and {b} are not neighbours")

    def _limit_reason(self, t0: float) -> Optional[str]:
        lim = self.cfg.limits
        if len(self.map.visited) >= lim.max_cells:
            return "limit_max_cells"
        w, h = self.map.to_wallmap().size()
        if max(w, h) > lim.max_extent_cells:
            return "limit_max_extent"
        if self.robot.now() - t0 > lim.max_mission_s:
            return "limit_mission_time"
        return None

    # ---- moving ------------------------------------------------------------------------
    def _move(self, d: Direction, target: Cell, kind: str) -> bool:
        self._event("TURN", to=d.name, heading_deg=d.heading_deg)
        turn = self.nav.turn_to(d.heading_deg)
        if not turn.ok:
            self._event("TURN_FAILED", reason=turn.reason)
            return False
        self.heading = d

        self._event("DRIVE", kind=kind, to=list(target), dir=d.name)
        res = self.nav.drive_to(target, d)
        cell_size = self.cfg.geometry.cell_size_m
        if not res.ok and res.progress_m > cell_size / 2.0:
            # Stopped after the robot centre crossed into the target cell, so
            # the edge is open: arrive here; the next scan re-centres the EKF.
            self._event("DRIVE_STOPPED_IN_TARGET", reason=res.reason, progress_m=round(res.progress_m, 3))
        if res.ok or res.progress_m > cell_size / 2.0:
            self.map.mark_traversed(self.cell, d)
            key = edge_key(self.cell, d)
            self.tremaux[key] = self.tremaux.get(key, 0) + 1
            self.cell = target
            self.path.append(target)
            self.moves += 1
            est = self.ekf.cell(cell_size)
            if est != target:
                self.mismatches += 1
                self._event("LOCALIZATION_MISMATCH", ekf_cell=list(est))
            self._event("ARRIVED", kind=kind, progress_m=round(res.progress_m, 3))
            return True

        # Blocked: count it as wall evidence and go back to the cell centre.
        self.blocked += 1
        self._event("DRIVE_BLOCKED", reason=res.reason, progress_m=round(res.progress_m, 3))
        self.map.observe(self.cell, d, True, self.cfg.exploration.blocked_wall_weight)
        back = self.nav.drive_to(self.cell, d)
        self._event("RETURN_TO_CENTER", ok=back.ok, reason=back.reason)
        return False
