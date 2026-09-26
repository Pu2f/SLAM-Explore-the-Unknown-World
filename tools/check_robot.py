"""Hardware bring-up checks. Run these before the first mission.

    python -m tools.check_robot streams     # live values of every sensor
    python -m tools.check_robot adapter     # every adapter port: find where a sensor is wired
    python -m tools.check_robot signs       # small moves; checks every sign in config

`signs` moves the robot a little (about 20 cm and 90 deg): give it room.
It prints which RobotIO flags in slam/config.py need flipping, if any.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Sequence

from slam.config import DEFAULT, with_calibration
from slam.geometry import angle_diff
from slam.robot_io import RealRobot


def ask(question: str) -> bool:
    return input(f"  {question} [y/n] ").strip().lower().startswith("y")


def streams(robot: RealRobot, seconds: float) -> None:
    print("stream status:", robot.stream_status())
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        ox, oy = robot.odometry()
        adc_l, adc_r = robot.sharp_adc()
        sl, sr = robot.sharp()
        il, ir = robot.ir_front()
        tof = robot.tof()
        print(
            f"odom=({ox:+.3f},{oy:+.3f}) yaw={robot.imu_yaw():+7.2f} gimbal={robot.gimbal_yaw():+7.2f} "
            f"tof={'---' if tof is None else f'{tof:.3f}'} "
            f"sharpL adc={adc_l} d={'---' if sl is None else f'{sl:.3f}'} "
            f"sharpR adc={adc_r} d={'---' if sr is None else f'{sr:.3f}'} "
            f"IR L={int(il)} R={int(ir)}"
        )
        time.sleep(0.25)


def adapter(robot: RealRobot, seconds: float) -> None:
    """Show all 12 adapter inputs and report which ones actually change.

    Move a hand / wall in front of one sensor at a time while this runs.
    """
    labels = [f"A{i // 2 + 1}P{i % 2 + 1}" for i in range(12)]
    print("Move something in front of ONE sensor at a time (Ctrl+C to stop early).")
    print("adc: " + " ".join(f"{name:>5}" for name in labels))
    adc_lo, adc_hi = [None] * 12, [None] * 12
    io_seen = [set() for _ in range(12)]
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            v = robot.adapter_values()
            if v is None:
                print("adapter stream stale / missing")
            else:
                io, adc = v
                for i in range(12):
                    adc_lo[i] = adc[i] if adc_lo[i] is None else min(adc_lo[i], adc[i])
                    adc_hi[i] = adc[i] if adc_hi[i] is None else max(adc_hi[i], adc[i])
                    io_seen[i].add(io[i])
                print("adc: " + " ".join(f"{a:>5}" for a in adc) + "   io: " + "".join(str(b) for b in io))
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass

    print()
    print("port    adc min..max   io levels seen   verdict")
    cfg = DEFAULT.robot_io
    wired = {
        cfg.sharp_left.index: "sharp_left",
        cfg.sharp_right.index: "sharp_right",
        cfg.ir_front_left.index: "ir_front_left",
        cfg.ir_front_right.index: "ir_front_right",
    }
    for i, name in enumerate(labels):
        if adc_lo[i] is None:
            continue
        span = adc_hi[i] - adc_lo[i]
        verdict = []
        if span >= 100:
            verdict.append("ADC CHANGES -> analog sensor here (Sharp?)")
        if len(io_seen[i]) > 1:
            verdict.append("IO TOGGLES -> digital sensor here (IR?)")
        config_name = wired.get(i)
        print(
            f"{name:5s}  {adc_lo[i]:>5}..{adc_hi[i]:<5}   {sorted(io_seen[i])!s:15s}  "
            f"{'; '.join(verdict) or '-':45s} {'<- config: ' + config_name if config_name else ''}"
        )
    print("Set RobotIO sharp_* / ir_front_* in slam/config.py to the ports that changed.")


def pulse(robot: RealRobot, fwd: float, right: float, turn: float, seconds: float) -> None:
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        robot.drive_speed(fwd, right, turn)
        time.sleep(0.05)
    robot.stop()
    time.sleep(0.5)


def signs(robot: RealRobot) -> int:
    io = DEFAULT.robot_io
    fixes = []

    print("1) Turn: the robot should turn RIGHT (clockwise, seen from above).")
    y0 = robot.imu_yaw()
    pulse(robot, 0.0, 0.0, 30.0, 1.0)
    dy = angle_diff(robot.imu_yaw(), y0)
    turned_right = ask("Did it turn RIGHT?")
    if not turned_right:
        fixes.append(f"sdk_turn_clockwise_positive = {not io.sdk_turn_clockwise_positive}")
    print(f"   IMU yaw change: {dy:+.1f} deg (should be positive for a right turn)")
    if (dy > 0) != turned_right:
        fixes.append(f"sdk_yaw_clockwise_positive = {not io.sdk_yaw_clockwise_positive}")
    pulse(robot, 0.0, 0.0, -30.0 if turned_right else 30.0, 1.0)  # turn back
    if fixes:
        print("Change in slam/config.py RobotIO:")
        for f in fixes:
            print("   ", f)
        print("then run this check again (later steps need correct turn signs).")
        return 1

    print("2) Forward 0.2 m: odometry should go to about (0, +0.2).")
    robot.zero()
    pulse(robot, 0.2, 0.0, 0.0, 1.0)
    ox, oy = robot.odometry()
    print(f"   odometry = ({ox:+.3f}, {oy:+.3f})")
    ok_fwd = oy > 0.1 and abs(ox) < 0.05

    print("3) Strafe right 0.2 m: odometry should go to about (+0.2, +0.2).")
    pulse(robot, 0.0, 0.2, 0.0, 1.0)
    ox, oy = robot.odometry()
    print(f"   odometry = ({ox:+.3f}, {oy:+.3f})")
    ok_right = ox > 0.1

    print("4) Turn right about 90 deg, then forward 0.2 m: odometry x should grow.")
    robot.zero()
    y0 = robot.imu_yaw()
    t_end = time.monotonic() + 6.0
    while time.monotonic() < t_end:
        err = angle_diff(y0 + 90.0, robot.imu_yaw())
        if abs(err) < 2.0:
            break
        robot.drive_speed(0.0, 0.0, max(-40.0, min(40.0, 1.5 * err)))
        time.sleep(0.05)
    robot.stop()
    time.sleep(0.5)
    x_before = robot.odometry()[0]
    pulse(robot, 0.2, 0.0, 0.0, 1.0)
    ox, oy = robot.odometry()
    print(f"   odometry = ({ox:+.3f}, {oy:+.3f}) (x grew by {ox - x_before:+.3f})")
    ok_frame = ox - x_before > 0.1

    print("5) Gimbal to +45 deg: it should point to the robot's RIGHT.")
    robot.gimbal_moveto(45.0)
    g = robot.gimbal_yaw()
    gimbal_right = ask("Is the gimbal pointing RIGHT?")
    print(f"   gimbal feedback = {g:+.1f} deg (should be about +45)")
    if not gimbal_right:
        fixes.append(f"sdk_gimbal_clockwise_positive = {not io.sdk_gimbal_clockwise_positive}")
    elif g < 0:
        print("   !! feedback sign disagrees with the command: check gimbal telemetry")
    robot.gimbal_moveto(0.0)

    print()
    for name, ok in (("forward", ok_fwd), ("strafe", ok_right), ("frame after turn", ok_frame)):
        print(f"odometry {name:17s}: {'OK' if ok else 'CHECK'}")
    if fixes:
        print("Change in slam/config.py RobotIO:")
        for f in fixes:
            print("   ", f)
        print("then run this check again.")
        return 1
    if not (ok_fwd and ok_right and ok_frame):
        print("Odometry axes look wrong although the turn signs are fine; see robot_io.odometry().")
        return 1
    print("All signs OK.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("check", choices=["streams", "adapter", "signs"])
    p.add_argument("--seconds", type=float, default=30.0, help="how long `streams` / `adapter` run")
    args = p.parse_args(argv)

    robot = RealRobot.connect(with_calibration(DEFAULT))
    try:
        if args.check == "streams":
            streams(robot, args.seconds)
            return 0
        if args.check == "adapter":
            adapter(robot, args.seconds)
            return 0
        return signs(robot)
    except KeyboardInterrupt:
        return 130
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
