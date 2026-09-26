"""The one interface all mission logic uses to talk to a robot.

Both ``sim.SimRobot`` and the RoboMaster ``robot_io.RealRobot`` implement it,
so perception / EKF / explorer can be tested on a laptop.

Everything is already in the DESIGN.md conventions: map frame with +y = N =
start forward and +x = E = start right, compass headings (clockwise positive),
metres. Converting RoboMaster SDK axes and signs happens only in robot_io.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class RobotStreamLost(RuntimeError):
    """The robot stopped sending sensor data; the mission cannot continue."""


class RobotAPI(ABC):
    # ---- time ------------------------------------------------------------
    @abstractmethod
    def now(self) -> float:
        """Seconds (monotonic). Simulated time for SimRobot."""

    @abstractmethod
    def sleep(self, dt_s: float) -> None:
        """Wait; the simulator advances its physics here."""

    # ---- chassis ---------------------------------------------------------
    @abstractmethod
    def drive_speed(self, forward_mps: float, right_mps: float, turn_dps: float) -> None:
        """Body-frame velocity command, held until the next command.

        ``turn_dps`` > 0 turns clockwise (heading increases).
        """

    def stop(self) -> None:
        self.drive_speed(0.0, 0.0, 0.0)

    @abstractmethod
    def odometry(self) -> Tuple[float, float]:
        """Wheel-odometry position (x, y) in the map frame; (0, 0) at start."""

    @abstractmethod
    def imu_yaw(self) -> float:
        """IMU heading in compass degrees, 0 at start, wrapped to [-180, 180)."""

    # ---- gimbal + ToF ------------------------------------------------------
    @abstractmethod
    def gimbal_moveto(self, yaw_deg: float) -> None:
        """Point the gimbal at ``yaw_deg`` relative to the chassis and wait."""

    @abstractmethod
    def gimbal_yaw(self) -> float:
        """Current gimbal yaw relative to the chassis (degrees, clockwise +)."""

    @abstractmethod
    def tof(self) -> Optional[float]:
        """ToF distance along the gimbal direction (m), None if invalid."""

    # ---- side / front IR -----------------------------------------------------
    @abstractmethod
    def sharp(self) -> Tuple[Optional[float], Optional[float]]:
        """(left, right) Sharp distances (m).

        Closer than the minimum range reads as the minimum range (a wall is
        right there); farther than the maximum range reads None.
        """

    @abstractmethod
    def ir_front(self) -> Tuple[bool, bool]:
        """(front_left, front_right) digital IR: True = obstacle detected."""
