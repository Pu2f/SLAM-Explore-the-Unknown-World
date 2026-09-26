"""All tunable constants in one place.

Conventions (see DESIGN.md section 3): metres, degrees, compass heading
(0 = N = start forward, clockwise positive), body frame = (forward, right).

Values marked MEASURE are estimates and must be measured on the real robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Geometry:
    cell_size_m: float = 0.60
    # Circle used for collision checks in the simulator (EP body is ~0.32 x 0.24 m).
    robot_radius_m: float = 0.16


@dataclass(frozen=True)
class Mount:
    """Sensor pose on the chassis: offset from the chassis centre + pointing angle."""

    forward_m: float
    right_m: float
    angle_deg: float  # 0 = straight ahead, +90 = right, -90 = left


@dataclass(frozen=True)
class Sensors:
    # ToF on the gimbal. The mount is the gimbal pivot; the gimbal yaw is added
    # to angle_deg at read time.  MEASURE
    tof: Mount = Mount(forward_m=0.0, right_m=0.0, angle_deg=0.0)
    tof_min_m: float = 0.05
    tof_max_m: float = 4.00

    # Sharp analog IR (left / right).  MEASURE
    sharp_left: Mount = Mount(forward_m=0.0, right_m=-0.12, angle_deg=-90.0)
    sharp_right: Mount = Mount(forward_m=0.0, right_m=0.12, angle_deg=90.0)
    sharp_min_m: float = 0.10
    sharp_max_m: float = 0.80

    # Digital IR (front-left / front-right), obstacle yes/no.  MEASURE
    ir_front_left: Mount = Mount(forward_m=0.16, right_m=-0.10, angle_deg=-45.0)
    ir_front_right: Mount = Mount(forward_m=0.16, right_m=0.10, angle_deg=45.0)
    ir_range_m: float = 0.15


@dataclass(frozen=True)
class MapEvidence:
    """Log-odds style evidence per wall edge (see maze_map.MazeMap)."""

    wall_threshold: float = 1.0  # |score| needed to call WALL / OPEN
    score_limit: float = 4.0  # clamp so later evidence can still flip a bad edge
    tof_weight: float = 1.0
    sharp_weight: float = 0.5


@dataclass(frozen=True)
class Limits:
    """Safety caps for exploring a maze of unknown size."""

    max_cells: int = 100
    max_extent_cells: int = 15  # width or height of the discovered bounding box
    max_mission_s: float = 900.0


@dataclass(frozen=True)
class SimNoise:
    """Noise model of the simulator. Zero everything for a perfect robot."""

    tof_std_m: float = 0.01
    sharp_std_frac: float = 0.03  # std as a fraction of the distance
    # Wheel slip: the real distance travelled = commanded * (1 + bias + N(0, std)).
    slip_bias: float = -0.02
    slip_std: float = 0.02
    # Heading actually turned vs commanded, as a fraction.
    turn_scale_std: float = 0.01
    imu_drift_dps: float = 0.02
    imu_std_deg: float = 0.1
    gimbal_std_deg: float = 0.5


PERFECT_SIM = SimNoise(
    tof_std_m=0.0,
    sharp_std_frac=0.0,
    slip_bias=0.0,
    slip_std=0.0,
    turn_scale_std=0.0,
    imu_drift_dps=0.0,
    imu_std_deg=0.0,
    gimbal_std_deg=0.0,
)


@dataclass(frozen=True)
class Config:
    geometry: Geometry = field(default_factory=Geometry)
    sensors: Sensors = field(default_factory=Sensors)
    evidence: MapEvidence = field(default_factory=MapEvidence)
    limits: Limits = field(default_factory=Limits)
    sim_noise: SimNoise = field(default_factory=SimNoise)
    # Simulator physics step. Commands are integrated in steps of this size.
    sim_dt_s: float = 0.01
    gimbal_speed_dps: float = 180.0


DEFAULT = Config()
