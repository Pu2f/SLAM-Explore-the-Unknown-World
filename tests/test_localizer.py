import math

import pytest

from slam.config import DEFAULT, Mount
from slam.geometry import Line, Pose
from slam.localizer import EKF

P = DEFAULT.ekf
FRONT = Mount(0.0, 0.0, 0.0)


def drive(ekf: EKF, steps: int, dist: float, yaw: float = 0.0, t0: float = 0.0) -> None:
    ekf.predict((0.0, 0.0), yaw, t0)
    for i in range(1, steps + 1):
        ekf.predict((0.0, dist * i / steps), yaw, t0 + i * 0.05)


def test_predict_moves_along_heading():
    ekf = EKF(P, Pose(0.0, 0.0, 90.0))  # facing E in the map
    # Odometry frame agrees with IMU yaw 0, so "forward" in odometry is +y.
    drive(ekf, 10, 0.6)
    assert ekf.pose.x == pytest.approx(0.6)
    assert ekf.pose.y == pytest.approx(0.0, abs=1e-9)


def test_predict_turn_changes_heading():
    ekf = EKF(P)
    ekf.predict((0.0, 0.0), 0.0, 0.0)
    ekf.predict((0.0, 0.0), 90.0, 1.0)
    assert ekf.pose.heading_deg == pytest.approx(90.0)


def test_process_noise_does_not_depend_on_control_rate():
    """Regression: per-tick noise made the filter overconfident at 20 Hz."""
    coarse, fine = EKF(P), EKF(P)
    drive(coarse, 1, 0.6)
    drive(fine, 120, 0.6)
    assert fine.std()[1] == pytest.approx(coarse.std()[1], rel=0.02)
    assert coarse.std()[1] > 0.04  # about 8 %/sqrt(m) over 0.6 m


def test_wall_update_pulls_toward_truth_and_shrinks_covariance():
    ekf = EKF(P, Pose(0.0, 0.05, 0.0))  # believes 5 cm too far north
    drive(ekf, 20, 0.0)
    wall = Line("y", 0.3)
    before = ekf.std()[1]
    res = ekf.update_range(0.30, FRONT, 0.0, wall, 0.01)  # truly at y = 0
    assert res.accepted
    assert abs(ekf.pose.y) < 0.05
    assert ekf.std()[1] < before


def test_repeated_updates_converge():
    ekf = EKF(P, Pose(0.03, -0.04, 0.0))
    for _ in range(10):
        ekf.update_range(0.30, FRONT, 0.0, Line("y", 0.3), 0.01)
        ekf.update_range(0.30, FRONT, 90.0, Line("x", 0.3), 0.01)
    assert ekf.pose.x == pytest.approx(0.0, abs=0.005)
    assert ekf.pose.y == pytest.approx(0.0, abs=0.005)


def test_gate_rejects_outlier_and_wrong_cell_wall():
    ekf = EKF(P)
    assert not ekf.update_range(0.90, FRONT, 0.0, Line("y", 0.3), 0.01).accepted
    # Even with a huge covariance, a correction larger than a quarter cell
    # could be a different wall, so it is refused.
    ekf.cov *= 400.0
    res = ekf.update_range(0.30 + 0.2, FRONT, 0.0, Line("y", 0.3), 0.01)
    assert not res.accepted
    assert ekf.pose.y == 0.0


def test_grazing_ray_is_not_used():
    ekf = EKF(P)
    # Ray at 80 deg to the N wall line hits it at a shallow angle.
    res = ekf.update_range(1.7, FRONT, 80.0, Line("y", 0.3), 0.01)
    assert not res.accepted
    assert math.isinf(res.mahalanobis)


def test_sharp_std_grows_with_distance():
    ekf = EKF(P)
    assert ekf.sharp_std(0.6) > ekf.sharp_std(0.2)


def test_range_updates_leave_heading_to_the_imu_by_default():
    ekf = EKF(P, Pose(0.0, 0.0, 3.0))
    ekf.cov[2, 2] = math.radians(5) ** 2
    ekf.cov[0, 2] = ekf.cov[2, 0] = 0.001  # correlated with x, as after driving
    for _ in range(5):
        ekf.update_range(0.30, FRONT, 90.0, Line("x", 0.35), 0.01)
    assert ekf.pose.heading_deg == pytest.approx(3.0)
    assert abs(ekf.pose.x - 0.05) < 0.01  # position still corrected


def test_heading_gain_allows_heading_updates():
    from dataclasses import replace

    ekf = EKF(replace(P, heading_update_gain=1.0), Pose(0.0, 0.0, 3.0))
    ekf.cov[2, 2] = math.radians(5) ** 2
    ekf.cov[0, 2] = ekf.cov[2, 0] = 0.001
    ekf.update_range(0.30, FRONT, 90.0, Line("x", 0.35), 0.01)
    assert ekf.pose.heading_deg != pytest.approx(3.0)
