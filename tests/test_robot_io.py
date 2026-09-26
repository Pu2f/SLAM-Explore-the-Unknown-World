"""RealRobot against a fake SDK: axes, signs, units and staleness."""

from dataclasses import replace

import pytest

from slam.config import DEFAULT, AdapterPort
from slam.robot_io import RealRobot


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class Action:
    def __init__(self, ok=True):
        self.ok = ok

    def wait_for_completed(self, timeout=None):
        return self.ok


class FakeChassis:
    def __init__(self):
        self.speeds = []
        self.subs = {}

    def drive_speed(self, x=0.0, y=0.0, z=0.0, timeout=None):
        self.speeds.append((x, y, z, timeout))

    def sub_position(self, cs=0, freq=5, callback=None):
        assert cs == 1
        self.subs["position"] = callback
        callback((0.0, 0.0, 0.0))

    def sub_attitude(self, freq=5, callback=None):
        self.subs["attitude"] = callback
        callback((0.0, 0.0, 0.0))

    def unsub_position(self):
        self.subs.pop("position")

    def unsub_attitude(self):
        self.subs.pop("attitude")


class FakeGimbal:
    def __init__(self):
        self.moves = []
        self.cb = None
        self.ok = True

    def resume(self):
        pass

    def moveto(self, pitch=0, yaw=0, pitch_speed=30, yaw_speed=30):
        self.moves.append((pitch, yaw))
        if self.cb:
            self.cb((pitch, yaw, 0, 0))
        return Action(self.ok)

    def sub_angle(self, freq=5, callback=None):
        self.cb = callback
        callback((0.0, 0.0, 0.0, 0.0))

    def unsub_angle(self):
        self.cb = None


class FakeSensor:
    def __init__(self, emit=True):
        self.cb = None
        self.emit = emit

    def sub_distance(self, freq=5, callback=None):
        self.cb = callback
        if self.emit:
            callback([300, 0, 0, 0])

    def unsub_distance(self):
        self.cb = None


class FakeAdaptor:
    def __init__(self):
        self.cb = None

    def sub_adapter(self, freq=5, callback=None):
        self.cb = callback
        callback(([1] * 12, [0] * 12))

    def unsub_adapter(self):
        self.cb = None


class FakeSDK:
    def __init__(self, tof_emits=True):
        self.chassis = FakeChassis()
        self.gimbal = FakeGimbal()
        self.sensor = FakeSensor(tof_emits)
        self.sensor_adaptor = FakeAdaptor()
        self.mode = None
        self.closed = False

    def set_robot_mode(self, mode):
        self.mode = mode

    def close(self):
        self.closed = True


def io_cfg(**kw):
    # Never pick up real calibration files from the working directory.
    kw.setdefault("sharp_left_calibration", "/nonexistent/sharp_left.json")
    kw.setdefault("sharp_right_calibration", "/nonexistent/sharp_right.json")
    return replace(DEFAULT, robot_io=replace(DEFAULT.robot_io, **kw))


def make(cfg=None, tof_emits=True):
    cfg = cfg or io_cfg()
    sdk, clock = FakeSDK(tof_emits), Clock()
    robot = RealRobot(sdk, cfg, clock=clock, sleeper=clock.sleep)
    robot.start()
    return robot, sdk, clock


def test_start_subscribes_everything_and_zeroes():
    robot, sdk, _ = make()
    assert sdk.mode == "free"
    assert set(sdk.chassis.subs) == {"position", "attitude"}
    assert sdk.gimbal.moves[-1] == (0, 0)
    assert robot.odometry() == (0.0, 0.0)
    assert robot.imu_yaw() == 0.0
    robot.close()
    assert sdk.closed and sdk.chassis.subs == {}


def test_missing_stream_is_reported():
    with pytest.raises(RuntimeError, match="tof"):
        make(tof_emits=False)


def test_odometry_is_relative_to_the_start_pose():
    robot, sdk, _ = make()
    # Start somewhere else, turned 90 deg right of the power-on heading.
    sdk.chassis.subs["position"]((1.0, 2.0, 0.0))
    sdk.chassis.subs["attitude"]((90.0, 0.0, 0.0))
    robot.zero()
    # Drive 0.5 m along the start heading = power-on "right" (+y).
    sdk.chassis.subs["position"]((1.0, 2.5, 0.0))
    x, y = robot.odometry()
    assert (x, y) == (pytest.approx(0.0, abs=1e-9), pytest.approx(0.5))
    # Power-on forward (+x) is now the robot's left = map W.
    sdk.chassis.subs["position"]((1.3, 2.0, 0.0))
    x, y = robot.odometry()
    assert (x, y) == (pytest.approx(-0.3), pytest.approx(0.0, abs=1e-9))
    sdk.chassis.subs["attitude"]((135.0, 0.0, 0.0))
    assert robot.imu_yaw() == pytest.approx(45.0)


def test_sign_flags():
    robot, sdk, _ = make(io_cfg(sdk_turn_clockwise_positive=False, sdk_yaw_clockwise_positive=False,
                                sdk_gimbal_clockwise_positive=False))
    robot.drive_speed(0.2, -0.1, 30.0)
    assert sdk.chassis.speeds[-1] == (0.2, -0.1, -30.0, DEFAULT.robot_io.drive_cmd_timeout_s)
    sdk.chassis.subs["attitude"]((-20.0, 0.0, 0.0))
    assert robot.imu_yaw() == pytest.approx(20.0)
    robot.gimbal_moveto(90.0)
    assert sdk.gimbal.moves[-1] == (0, -90)
    assert robot.gimbal_yaw() == pytest.approx(90.0)


def test_gimbal_yaw_is_clamped_and_timeout_raises():
    robot, sdk, _ = make()
    robot.gimbal_moveto(300.0)
    assert sdk.gimbal.moves[-1][1] == 250
    sdk.gimbal.ok = False
    with pytest.raises(RuntimeError):
        robot.gimbal_moveto(10.0)


def test_tof_units_offset_and_staleness():
    robot, sdk, clock = make(io_cfg(tof_pivot_offset_m=0.05))
    sdk.sensor.cb([300, 0, 0, 0])
    assert robot.tof() == pytest.approx(0.35)
    sdk.sensor.cb([0, 0, 0, 0])
    assert robot.tof() is None
    sdk.sensor.cb([300, 0, 0, 0])
    clock.t += 1.0
    assert robot.tof() is None


def test_sharp_and_ir_from_adapter():
    cfg = io_cfg(sharp_left=AdapterPort(1, 1), sharp_right=AdapterPort(1, 2),
                 ir_front_left=AdapterPort(2, 1), ir_front_right=AdapterPort(3, 2), ir_active_low=True)
    robot, sdk, _ = make(cfg)
    io = [1] * 12
    adc = [0] * 12
    adc[0], adc[1] = 285, 50  # left 0.30 m (placeholder table), right out of range
    io[2] = 0  # adapter 2 port 1 -> front-left sees something
    sdk.sensor_adaptor.cb((io, adc))
    left, right = robot.sharp()
    assert left == pytest.approx(0.30)
    assert right is None
    assert robot.sharp_adc() == (285, 50)
    assert robot.ir_front() == (True, False)


@pytest.mark.parametrize("module", ["tools.run_robot", "tools.check_robot", "tools.calibrate_sharp"])
def test_tools_import_without_the_sdk(module):
    import importlib

    mod = importlib.import_module(module)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0


def test_check_robot_adapter_finds_the_wired_ports(capsys, monkeypatch):
    from tools import check_robot

    robot, sdk, _ = make()
    frames = iter(range(1000))

    def fake_values():
        k = next(frames)
        io = [1] * 12
        adc = [510] * 12
        adc[5] = 200 + 150 * (k % 3)  # adapter 3 port 2: a Sharp being waved at
        io[7] = k % 2  # adapter 4 port 2: an IR toggling
        return io, adc

    monkeypatch.setattr(robot, "adapter_values", fake_values)
    monkeypatch.setattr(check_robot.time, "sleep", lambda s: None)
    t = iter(range(1000))
    monkeypatch.setattr(check_robot.time, "monotonic", lambda: next(t) * 0.1)
    check_robot.adapter(robot, seconds=1.0)
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines() if line[:1] == "A" and line[2:3] == "P"}
    assert "ADC CHANGES" in lines["A3P2"]
    assert "IO TOGGLES" in lines["A4P2"]
    assert "ADC CHANGES" not in lines["A3P1"] and "config: sharp_right" in lines["A3P1"]


def test_default_wiring_matches_the_robot():
    io = DEFAULT.robot_io
    assert (io.sharp_left.index, io.sharp_right.index) == (2, 4)  # A2P1, A3P1
    assert (io.ir_front_left.index, io.ir_front_right.index) == (0, 6)  # A1P1, A4P1


@pytest.mark.parametrize("failure", ["raises", "returns_false"])
def test_connect_failure_gives_a_clear_message(monkeypatch, failure):
    import sys
    import types

    closed = []

    class Robot:
        def initialize(self, conn_type):
            if failure == "raises":
                raise TypeError("exceptions must derive from BaseException")
            return False

        def close(self):
            closed.append(True)

    pkg = types.ModuleType("robomaster")
    mod = types.ModuleType("robomaster.robot")
    mod.Robot = Robot
    pkg.robot = mod
    monkeypatch.setitem(sys.modules, "robomaster", pkg)
    monkeypatch.setitem(sys.modules, "robomaster.robot", mod)
    with pytest.raises(RuntimeError, match="cannot connect to the robot"):
        RealRobot.connect(io_cfg())
    assert closed == [True]
