"""RealRobot: RobotAPI on a RoboMaster EP through the official SDK.

This is the only file that knows SDK axes and signs. SDK facts used here
(robomaster 0.1.1.68):
- chassis.sub_position(cs=1): (x forward, y right, z) in m from the power-on pose
- chassis.sub_attitude(): (yaw, pitch, roll) in degrees
- chassis.drive_speed(x, y, z, timeout): m/s forward / right, deg/s turn
- gimbal.sub_angle(): (pitch, yaw relative to chassis, ...) in degrees
- gimbal.moveto(pitch, yaw, ...) -> action, yaw limited to [-250, 250]
- sensor.sub_distance(): 4 ToF distances in mm
- sensor_adaptor.sub_adapter(): (io[12], adc[12]) for 6 adapters x 2 ports

Every stream is subscribed once; callbacks store the latest value and the
RobotAPI getters read it, so the mission loop never blocks on the network.
The map frame (DESIGN.md section 3) is fixed by zero(): the pose when the
mission starts becomes (0, 0) facing N.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from .config import DEFAULT, Config
from .geometry import wrap_deg
from .robot_api import RobotAPI, RobotStreamLost
from .sharp import SharpCalibration

GIMBAL_YAW_LIMIT_DEG = 250.0


def _cw(value: float, clockwise_positive: bool) -> float:
    """Convert an SDK angle / rate to clockwise-positive."""
    return value if clockwise_positive else -value


class RealRobot(RobotAPI):
    def __init__(
        self,
        sdk_robot,
        cfg: Config = DEFAULT,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.sdk = sdk_robot
        self.cfg = cfg
        self.io = cfg.robot_io
        self._clock = clock
        self._sleep = sleeper
        self.sharp_left_cal = SharpCalibration.load_or_placeholder(self.io.sharp_left_calibration)
        self.sharp_right_cal = SharpCalibration.load_or_placeholder(self.io.sharp_right_calibration)

        self._lock = threading.Lock()
        self._pos: Optional[Tuple[float, float, float]] = None  # x_fwd, y_right, t
        self._yaw: Optional[Tuple[float, float]] = None  # clockwise deg, t
        self._gimbal: Optional[Tuple[float, float]] = None  # clockwise deg, t
        self._tof: Optional[Tuple[Tuple[float, ...], float]] = None  # mm, t
        self._adapter: Optional[Tuple[List[int], List[int], float]] = None  # io, adc, t
        self._origin: Optional[Tuple[float, float, float]] = None  # x, y, yaw at zero()

    # ---- setup / teardown ------------------------------------------------------
    @classmethod
    def connect(cls, cfg: Config = DEFAULT) -> "RealRobot":
        from robomaster import robot as rm_robot  # imported here so tests don't need the SDK

        ep = rm_robot.Robot()
        try:
            ok = ep.initialize(conn_type=cfg.robot_io.conn_type)
        except Exception as exc:  # SDK 0.1.1.68 raises TypeError from `raise print(...)`
            ok = False
            cause: Optional[BaseException] = exc
        else:
            cause = None
        if ok is False:
            try:
                ep.close()
            except Exception:
                pass
            raise RuntimeError(
                f"cannot connect to the robot (conn_type={cfg.robot_io.conn_type!r}). "
                "For 'ap' join the robot's WiFi first; for 'sta' the robot and this computer "
                "must be on the same network; for 'rndis' use the USB cable."
            ) from cause
        r = cls(ep, cfg)
        try:
            r.start()
        except BaseException:
            r.close()
            raise
        return r

    def start(self) -> None:
        hz = self.io.sensor_hz
        self.sdk.set_robot_mode(mode="free")  # gimbal yaw independent of the chassis
        self.sdk.gimbal.resume()
        self.sdk.chassis.sub_position(cs=1, freq=hz, callback=self._on_position)
        self.sdk.chassis.sub_attitude(freq=hz, callback=self._on_attitude)
        self.sdk.gimbal.sub_angle(freq=hz, callback=self._on_gimbal)
        self.sdk.sensor.sub_distance(freq=hz, callback=self._on_tof)
        self.sdk.sensor_adaptor.sub_adapter(freq=hz, callback=self._on_adapter)
        self.wait_for_streams()
        self.gimbal_moveto(0.0)
        self.zero()

    def wait_for_streams(self) -> None:
        deadline = self._clock() + self.io.stream_timeout_s
        while self._clock() < deadline:
            if all(self.stream_status().values()):
                return
            self._sleep(0.05)
        raise RuntimeError(f"sensor streams missing: {self.stream_status()}")

    def stream_status(self) -> Dict[str, bool]:
        with self._lock:
            stamps = {
                "chassis_position": self._pos[2] if self._pos else None,
                "chassis_attitude": self._yaw[1] if self._yaw else None,
                "gimbal_angle": self._gimbal[1] if self._gimbal else None,
                "tof": self._tof[1] if self._tof else None,
                "sensor_adapter": self._adapter[2] if self._adapter else None,
            }
        return {k: self._fresh(t) for k, t in stamps.items()}

    def zero(self) -> None:
        """Make the current pose the map origin, facing N."""
        with self._lock:
            if self._pos is None or self._yaw is None:
                raise RuntimeError("no chassis data yet")
            self._origin = (self._pos[0], self._pos[1], self._yaw[0])

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        for unsub in (
            lambda: self.sdk.chassis.unsub_position(),
            lambda: self.sdk.chassis.unsub_attitude(),
            lambda: self.sdk.gimbal.unsub_angle(),
            lambda: self.sdk.sensor.unsub_distance(),
            lambda: self.sdk.sensor_adaptor.unsub_adapter(),
            lambda: self.sdk.close(),
        ):
            try:
                unsub()
            except Exception:
                pass

    # ---- SDK callbacks (run on SDK threads) -----------------------------------
    def _on_position(self, data) -> None:
        x, y = float(data[0]), float(data[1])
        with self._lock:
            self._pos = (x, y, self._clock())

    def _on_attitude(self, data) -> None:
        yaw = _cw(float(data[0]), self.io.sdk_yaw_clockwise_positive)
        with self._lock:
            self._yaw = (yaw, self._clock())

    def _on_gimbal(self, data) -> None:
        yaw = _cw(float(data[1]), self.io.sdk_gimbal_clockwise_positive)
        with self._lock:
            self._gimbal = (yaw, self._clock())

    def _on_tof(self, data) -> None:
        with self._lock:
            self._tof = (tuple(float(v) for v in data), self._clock())

    def _on_adapter(self, data) -> None:
        io, adc = data
        with self._lock:
            self._adapter = (list(io), list(adc), self._clock())

    def _fresh(self, t: Optional[float]) -> bool:
        return t is not None and self._clock() - t <= self.io.stale_s

    # ---- RobotAPI: time -----------------------------------------------------------
    def now(self) -> float:
        return self._clock()

    def sleep(self, dt_s: float) -> None:
        self._sleep(dt_s)

    # ---- RobotAPI: chassis --------------------------------------------------------
    def drive_speed(self, forward_mps: float, right_mps: float, turn_dps: float) -> None:
        z = _cw(turn_dps, self.io.sdk_turn_clockwise_positive)
        self.sdk.chassis.drive_speed(
            x=float(forward_mps), y=float(right_mps), z=float(z), timeout=self.io.drive_cmd_timeout_s
        )

    def _check_alive(self, t: Optional[float], what: str) -> None:
        if t is not None and self._clock() - t > self.io.lost_s:
            raise RobotStreamLost(f"no {what} data for {self._clock() - t:.1f} s")

    def odometry(self) -> Tuple[float, float]:
        with self._lock:
            pos, origin = self._pos, self._origin
        if pos is None:
            return (0.0, 0.0)
        self._check_alive(pos[2], "chassis position")
        if origin is None:
            origin = (pos[0], pos[1], 0.0)
        # Power-on frame (forward, right) -> map frame (E, N) rotated by the
        # heading the robot had at zero().
        dx, dy = pos[0] - origin[0], pos[1] - origin[1]
        a = math.radians(origin[2])
        north = dx * math.cos(a) + dy * math.sin(a)
        east = dy * math.cos(a) - dx * math.sin(a)
        return (east, north)

    def imu_yaw(self) -> float:
        with self._lock:
            yaw, origin = self._yaw, self._origin
        if yaw is None:
            return 0.0
        self._check_alive(yaw[1], "chassis attitude")
        return wrap_deg(yaw[0] - (origin[2] if origin else 0.0))

    # ---- RobotAPI: gimbal + ToF ---------------------------------------------------
    def gimbal_moveto(self, yaw_deg: float) -> None:
        sdk_yaw = _cw(yaw_deg, self.io.sdk_gimbal_clockwise_positive)
        sdk_yaw = max(-GIMBAL_YAW_LIMIT_DEG, min(GIMBAL_YAW_LIMIT_DEG, sdk_yaw))
        speed = int(round(self.io.gimbal_speed_dps))
        action = self.sdk.gimbal.moveto(
            pitch=int(round(self.io.gimbal_pitch_deg)),
            yaw=int(round(sdk_yaw)),
            pitch_speed=speed,
            yaw_speed=speed,
        )
        if not action.wait_for_completed(timeout=self.io.gimbal_timeout_s):
            raise RuntimeError(f"gimbal did not reach yaw {yaw_deg:+.0f} deg")
        self._sleep(self.io.gimbal_settle_s)

    def gimbal_yaw(self) -> float:
        with self._lock:
            g = self._gimbal
        return g[0] if g is not None else 0.0

    def tof(self) -> Optional[float]:
        with self._lock:
            tof = self._tof
        if tof is None or not self._fresh(tof[1]) or len(tof[0]) <= self.io.tof_index:
            return None
        mm = tof[0][self.io.tof_index]
        if not math.isfinite(mm) or mm <= 0:
            return None
        d = mm / 1000.0 + self.io.tof_pivot_offset_m
        s = self.cfg.sensors
        return d if s.tof_min_m <= d <= s.tof_max_m else None

    # ---- RobotAPI: side / front IR ---------------------------------------------------
    def adapter_values(self) -> Optional[Tuple[List[int], List[int]]]:
        """Latest (io[12], adc[12]) for all 6 adapters x 2 ports, None if stale.

        Index = (adapter_id - 1) * 2 + (port - 1), see config.AdapterPort.
        """
        with self._lock:
            a = self._adapter
        if a is None or not self._fresh(a[2]):
            return None
        return a[0], a[1]

    def sharp_adc(self) -> Tuple[Optional[int], Optional[int]]:
        """Raw ADC values (left, right), for calibration."""
        v = self.adapter_values()
        if v is None:
            return None, None
        _, adc = v
        return adc[self.io.sharp_left.index], adc[self.io.sharp_right.index]

    def sharp(self) -> Tuple[Optional[float], Optional[float]]:
        left, right = self.sharp_adc()
        s = self.cfg.sensors
        out = []
        for cal, adc in ((self.sharp_left_cal, left), (self.sharp_right_cal, right)):
            # Above the table's highest ADC = closer than its nearest point:
            # a wall is right there, not "nothing in range".
            d = cal.min_m if adc is not None and adc > cal.adcs[-1] else cal.distance(adc)
            if d is not None:
                d = max(d, s.sharp_min_m)
            out.append(d if d is not None and d <= s.sharp_max_m else None)
        return out[0], out[1]

    def ir_front(self) -> Tuple[bool, bool]:
        v = self.adapter_values()
        if v is None:
            return False, False
        io, _ = v
        hits = []
        for port in (self.io.ir_front_left, self.io.ir_front_right):
            level = io[port.index]
            hits.append(level == 0 if self.io.ir_active_low else level != 0)
        return hits[0], hits[1]
