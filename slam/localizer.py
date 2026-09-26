"""EKF localization over (x, y, heading) in the map frame (DESIGN.md section 5).

Predict: wheel odometry + IMU yaw increments.
Correct: a range reading to a known wall line (ToF or Sharp). In a grid maze
walls can only lie on the lines x, y = (k + 1/2) * cell_size, so once the
map says which wall a ray hits, the expected range follows from the state.

Heading is kept in radians inside the filter and exposed in compass degrees.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import numpy as np

from .config import EKFParams, Mount
from .geometry import Line, Pose, cell_of, ray_line_distance, sensor_ray, wrap_deg


def _wrap_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class UpdateResult(NamedTuple):
    accepted: bool
    innovation_m: float
    mahalanobis: float


class EKF:
    def __init__(self, params: EKFParams, pose: Pose = Pose(0.0, 0.0, 0.0)) -> None:
        self.p = params
        self.state = np.array([pose.x, pose.y, math.radians(pose.heading_deg)], dtype=float)
        self.cov = np.diag(
            [
                params.init_std_xy_m ** 2,
                params.init_std_xy_m ** 2,
                math.radians(params.init_std_heading_deg) ** 2,
            ]
        )
        self._last_odom: Optional[Tuple[float, float]] = None
        self._last_yaw: Optional[float] = None
        self._last_t: Optional[float] = None

    # ---- accessors -----------------------------------------------------------
    @property
    def pose(self) -> Pose:
        x, y, th = self.state
        return Pose(float(x), float(y), wrap_deg(math.degrees(th)))

    def std(self) -> Tuple[float, float, float]:
        """1-sigma of (x m, y m, heading deg)."""
        sx, sy, sth = np.sqrt(np.diag(self.cov))
        return float(sx), float(sy), math.degrees(float(sth))

    def cell(self, cell_size: float):
        return cell_of(self.state[0], self.state[1], cell_size)

    # ---- predict ---------------------------------------------------------------
    def predict(self, odom: Tuple[float, float], imu_yaw_deg: float, t: float) -> None:
        """Apply the motion since the previous call."""
        if self._last_odom is None:
            self._last_odom, self._last_yaw, self._last_t = odom, imu_yaw_deg, t
            return
        assert self._last_yaw is not None and self._last_t is not None
        dox = odom[0] - self._last_odom[0]
        doy = odom[1] - self._last_odom[1]
        # Odometry is integrated along the IMU heading, so undo that rotation
        # to get the body-frame motion (forward, right).
        psi = math.radians(self._last_yaw)
        fwd = dox * math.sin(psi) + doy * math.cos(psi)
        right = dox * math.cos(psi) - doy * math.sin(psi)
        dth = math.radians(wrap_deg(imu_yaw_deg - self._last_yaw))
        dt = max(0.0, t - self._last_t)
        self._last_odom, self._last_yaw, self._last_t = odom, imu_yaw_deg, t

        x, y, th = self.state
        thm = th + dth / 2.0
        s, c = math.sin(thm), math.cos(thm)
        self.state = np.array(
            [x + fwd * s + right * c, y + fwd * c - right * s, _wrap_rad(th + dth)]
        )

        F = np.eye(3)
        F[0, 2] = fwd * c - right * s
        F[1, 2] = -fwd * s - right * c

        # Random-walk noise: variance proportional to distance / angle / time.
        dist = math.hypot(fwd, right)
        var_f = self.p.odom_along_std_per_m ** 2 * dist
        var_r = self.p.odom_lateral_std_per_m ** 2 * dist
        J = np.array([[s, c], [c, -s]])  # columns: forward, right unit vectors
        Q = np.zeros((3, 3))
        Q[:2, :2] = J @ np.diag([var_f, var_r]) @ J.T
        Q[2, 2] = (
            self.p.turn_std_per_rad ** 2 * abs(dth)
            + math.radians(self.p.gyro_std_deg_per_sqrt_s) ** 2 * dt
        )
        prev_var = self.cov[2, 2]
        self.cov = F @ self.cov @ F.T + Q
        # Growth is capped, but a larger starting uncertainty is kept until
        # a heading measurement shrinks it.
        cap = max(math.radians(self.p.heading_std_cap_deg) ** 2, prev_var)
        if self.p.heading_update_gain == 0.0 and self.cov[2, 2] > cap:
            # Rescale heading row/column so the covariance stays consistent.
            k = math.sqrt(cap / self.cov[2, 2])
            self.cov[2, :] *= k
            self.cov[:, 2] *= k

    # ---- correct ---------------------------------------------------------------
    def expected_range(
        self, state: np.ndarray, mount: Mount, extra_angle_deg: float, line: Line
    ) -> Optional[float]:
        pose = Pose(float(state[0]), float(state[1]), math.degrees(float(state[2])))
        ox, oy, heading = sensor_ray(
            pose, mount.forward_m, mount.right_m, mount.angle_deg + extra_angle_deg
        )
        return ray_line_distance(ox, oy, heading, line)

    def incidence_ok(self, mount: Mount, extra_angle_deg: float, line: Line) -> bool:
        """Reject grazing rays: the ray must hit the line steeply enough."""
        heading = math.degrees(self.state[2]) + mount.angle_deg + extra_angle_deg
        rad = math.radians(heading)
        comp = abs(math.sin(rad)) if line.axis == "x" else abs(math.cos(rad))
        return comp >= math.sin(math.radians(self.p.min_incidence_deg))

    def update_heading(self, measured_deg: float, std_deg: float, gate_deg: float) -> UpdateResult:
        """Heading measured from the walls' angle (see explorer.scan)."""
        nu = math.radians(wrap_deg(measured_deg - math.degrees(self.state[2])))
        if abs(math.degrees(nu)) > gate_deg:
            return UpdateResult(False, math.degrees(nu), math.inf)
        H = np.array([[0.0, 0.0, 1.0]])
        R = math.radians(std_deg) ** 2
        S = float(H @ self.cov @ H.T) + R
        K = (self.cov @ H.T) / S
        self.state = self.state + (K * nu).ravel()
        self.state[2] = _wrap_rad(self.state[2])
        I_KH = np.eye(3) - K @ H
        self.cov = I_KH @ self.cov @ I_KH.T + (K * R) @ K.T
        return UpdateResult(True, math.degrees(nu), abs(nu) / math.sqrt(S))

    def update_range(
        self,
        measured_m: float,
        mount: Mount,
        extra_angle_deg: float,
        line: Line,
        std_m: float,
    ) -> UpdateResult:
        if not self.incidence_ok(mount, extra_angle_deg, line):
            return UpdateResult(False, math.nan, math.inf)
        h = self.expected_range(self.state, mount, extra_angle_deg, line)
        if h is None:
            return UpdateResult(False, math.nan, math.inf)

        # Numerical Jacobian: 3 states, so this is cheap and hard to get wrong.
        H = np.zeros((1, 3))
        eps = (1e-6, 1e-6, 1e-7)
        for i in range(3):
            s2 = self.state.copy()
            s2[i] += eps[i]
            h2 = self.expected_range(s2, mount, extra_angle_deg, line)
            if h2 is None:
                return UpdateResult(False, math.nan, math.inf)
            H[0, i] = (h2 - h) / eps[i]

        nu = measured_m - h
        S = float(H @ self.cov @ H.T) + std_m ** 2
        m2 = nu * nu / S
        if m2 > self.p.gate_sigma ** 2 or abs(nu) > self.p.max_innovation_m:
            return UpdateResult(False, nu, math.sqrt(m2))

        K = (self.cov @ H.T) / S  # 3x1
        K[2, 0] *= self.p.heading_update_gain  # heading comes from the IMU
        self.state = self.state + (K * nu).ravel()
        self.state[2] = _wrap_rad(self.state[2])
        I_KH = np.eye(3) - K @ H
        self.cov = I_KH @ self.cov @ I_KH.T + (K * std_m ** 2) @ K.T
        return UpdateResult(True, nu, math.sqrt(m2))

    def sharp_std(self, distance_m: float) -> float:
        return self.p.sharp_std_m + self.p.sharp_std_frac * distance_m
