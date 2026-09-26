"""All tunable constants in one place.

Conventions (see DESIGN.md section 3): metres, degrees, compass heading
(0 = N = start forward, clockwise positive), body frame = (forward, right).

Values marked MEASURE are estimates and must be measured on the real robot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace


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
class Perception:
    """Turning a range into WALL / OPEN.

    ``edge`` = distance from the sensor to the side of the current cell along
    the ray (about 0.30 m from the centre). The next possible wall is a whole
    cell further, so there is a wide gap between the two cases.
    """

    wall_tol_m: float = 0.12  # reading <= edge + this -> WALL
    open_margin_m: float = 0.20  # reading >= edge + this -> OPEN, between -> unsure
    tof_samples: int = 5
    tof_sample_delay_s: float = 0.02
    tof_min_valid_samples: int = 3
    # A ToF that returns nothing at all is looked at again this far to either
    # side; still nothing -> treated as OPEN. A wall within a cell always
    # gives a reading (real run: the long corridor ahead of the start cell
    # read nothing, and calling it "unsure" sent the robot out the back).
    tof_none_retry_deg: float = 10.0
    tof_none_means_open: bool = True
    # A ray more than this far from a grid axis is not associated with a wall.
    max_off_axis_deg: float = 25.0
    max_association_cells: int = 8


@dataclass(frozen=True)
class EKFParams:
    init_std_xy_m: float = 0.02
    init_std_heading_deg: float = 1.0
    # Process noise is a random walk: variance grows with the distance / angle
    # moved, independent of the control rate. 1-sigma after 1 m of driving:
    odom_along_std_per_m: float = 0.08
    odom_lateral_std_per_m: float = 0.04
    # 1-sigma after turning 1 rad, and heading drift per sqrt(second).
    turn_std_per_rad: float = 0.03
    gyro_std_deg_per_sqrt_s: float = 0.15
    tof_std_m: float = 0.02
    sharp_std_m: float = 0.01
    sharp_std_frac: float = 0.05
    gate_sigma: float = 3.0
    # How much a wall-range update may change the heading (0 = never). The
    # IMU yaw is trusted for heading: on the first real run, wall updates
    # with wrong sensor offsets dragged the EKF heading 14-32 deg off the
    # IMU and the robot steered crooked to follow it.
    heading_update_gain: float = 0.0
    # With the heading not corrected, its variance would grow without bound
    # and inflate the position uncertainty through the motion model. Cap it
    # at what the IMU is actually good for.
    heading_std_cap_deg: float = 2.0
    # Walls repeat every cell, so a reading that disagrees by more than about
    # a quarter cell may belong to a different wall: never apply it.
    max_innovation_m: float = 0.15
    # Rays hitting a wall at a shallower angle than this are not used.
    min_incidence_deg: float = 30.0


@dataclass(frozen=True)
class Motion:
    loop_hz: float = 20.0
    drive_mps: float = 0.25
    min_mps: float = 0.04
    k_along: float = 1.5  # m/s per m of remaining distance
    k_lateral: float = 1.5  # strafe m/s per m of lateral error
    max_strafe_mps: float = 0.10
    k_heading: float = 2.5  # dps per degree
    max_turn_dps: float = 90.0
    min_turn_dps: float = 8.0
    max_heading_hold_dps: float = 30.0
    pos_tol_m: float = 0.01
    turn_tol_deg: float = 1.0
    turn_stable_ticks: int = 3
    turn_timeout_s: float = 8.0
    drive_timeout_s: float = 10.0
    # ToF straight ahead closer than this while driving -> emergency stop.
    emergency_front_m: float = 0.10
    # Front IRs (45 deg): one side hit -> side-step away at this speed
    # (instead of the centring controller); both -> drive no faster than
    # ir_both_mps. They never abort a move.
    ir_avoid_mps: float = 0.06
    ir_max_shift_m: float = 0.05
    ir_both_mps: float = 0.12


@dataclass(frozen=True)
class Exploration:
    # Order of preference relative to the current heading, in quarter turns
    # clockwise: straight, right, left, back. Directions whose ToF reading
    # looked like an exit (nothing / too far / off the grid) come after all
    # normal ones.
    turn_preference: tuple = (0, 1, 3, 2)
    # A blocked move adds this much WALL evidence to the edge.
    blocked_wall_weight: float = 2.0
    # Look again at directions that came out unsure.
    rescan_unsure: int = 2
    max_consecutive_failures: int = 5

    # Exit ("fake exit") detection. After entering a new cell: if no side has
    # a wall and at least `exit_min_abnormal` of the other sides give an
    # abnormal ToF reading (none; farther than `exit_far_m`; or more than
    # `exit_grid_tol_m` off every grid line, where maze walls must be), look
    # again `exit_verify_deg` either side of each direction. Still no wall ->
    # the robot has left the maze: it goes back in, never returns there, and
    # carries on. The mission only ends once the maze is fully explored.
    exit_detection: bool = True
    exit_far_m: float = 3.0
    exit_grid_tol_m: float = 0.10
    exit_min_abnormal: int = 2
    exit_verify_deg: float = 20.0


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
class AdapterPort:
    """A sensor-adapter input: adapter id 1..6 (set on the board), port 1..2."""

    adapter_id: int
    port: int

    @property
    def index(self) -> int:
        """Position in the 12-value lists from sensor_adaptor.sub_adapter()."""
        return (self.adapter_id - 1) * 2 + (self.port - 1)


@dataclass(frozen=True)
class RobotIO:
    """RoboMaster EP specifics used only by slam/robot_io.py.

    Signs: the SDK body frame is x forward, y right, and a positive z / yaw
    is assumed to be clockwise. Verify with `python -m tools.check_robot`
    and flip a flag here if a check fails.
    """

    conn_type: str = "ap"
    sensor_hz: int = 50  # 1, 5, 10, 20 or 50
    stream_timeout_s: float = 6.0
    stale_s: float = 0.3
    # Chassis position / attitude older than this -> the robot stopped
    # talking (Wi-Fi, battery): abort instead of steering on frozen data.
    lost_s: float = 1.0
    # The SDK stops the chassis if no new speed command arrives in this time.
    drive_cmd_timeout_s: float = 0.3

    sdk_turn_clockwise_positive: bool = True  # chassis drive_speed z
    sdk_yaw_clockwise_positive: bool = True  # chassis sub_attitude yaw
    sdk_gimbal_clockwise_positive: bool = True  # gimbal moveto / sub_angle yaw

    # ToF: which of the 4 values from sensor.sub_distance(), and the distance
    # from the gimbal pivot to the ToF emitter (added to every reading so the
    # rest of the code measures from the pivot).  MEASURE
    tof_index: int = 0
    tof_pivot_offset_m: float = 0.0

    # Slightly down, as the previous team found on this robot: level, the
    # ToF beam passes over low maze walls at long range.
    gimbal_pitch_deg: float = -4.0
    gimbal_speed_dps: float = 180.0
    gimbal_timeout_s: float = 4.0
    gimbal_settle_s: float = 0.10

    # Sensor adapter wiring (adapter id, port) as wired on the robot.
    # Find / re-check with `python -m tools.check_robot adapter`.
    sharp_left: AdapterPort = AdapterPort(2, 1)
    sharp_right: AdapterPort = AdapterPort(3, 1)
    ir_front_left: AdapterPort = AdapterPort(1, 1)
    ir_front_right: AdapterPort = AdapterPort(4, 1)
    # Most IR obstacle modules pull the output LOW when they see something.
    ir_active_low: bool = True
    # ADC -> metres tables, written by `python -m tools.calibrate_sharp`.
    sharp_left_calibration: str = "calibration/sharp_left.json"
    sharp_right_calibration: str = "calibration/sharp_right.json"


@dataclass(frozen=True)
class Config:
    geometry: Geometry = field(default_factory=Geometry)
    sensors: Sensors = field(default_factory=Sensors)
    evidence: MapEvidence = field(default_factory=MapEvidence)
    perception: Perception = field(default_factory=Perception)
    ekf: EKFParams = field(default_factory=EKFParams)
    motion: Motion = field(default_factory=Motion)
    exploration: Exploration = field(default_factory=Exploration)
    limits: Limits = field(default_factory=Limits)
    robot_io: RobotIO = field(default_factory=RobotIO)
    sim_noise: SimNoise = field(default_factory=SimNoise)
    # Simulator physics step. Commands are integrated in steps of this size.
    sim_dt_s: float = 0.01
    gimbal_speed_dps: float = 180.0


DEFAULT = Config()


MOUNTS_FILE = "calibration/mounts.json"


def with_calibration(cfg: Config = DEFAULT, path: str = MOUNTS_FILE) -> Config:
    """Apply measured sensor offsets from `tools.calibrate_mounts`, if present.

    The offsets are *effective* ones: measured against the walls of the real
    maze, so they also absorb wall thickness and small sensor biases.
    """
    if not os.path.exists(path):
        return cfg
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    s = cfg.sensors
    sensors = replace(
        s,
        tof=replace(s.tof, forward_m=float(m.get("tof_pivot_forward_m", s.tof.forward_m))),
        sharp_left=replace(s.sharp_left, right_m=-float(m.get("sharp_left_offset_m", -s.sharp_left.right_m))),
        sharp_right=replace(s.sharp_right, right_m=float(m.get("sharp_right_offset_m", s.sharp_right.right_m))),
    )
    robot_io = replace(
        cfg.robot_io,
        tof_pivot_offset_m=float(m.get("tof_emitter_offset_m", cfg.robot_io.tof_pivot_offset_m)),
    )
    return replace(cfg, sensors=sensors, robot_io=robot_io)
