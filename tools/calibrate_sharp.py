"""Calibrate a Sharp IR sensor: measure the ADC at known distances.

    python -m tools.calibrate_sharp --side left
    python -m tools.calibrate_sharp --side right --distances 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.8
    python -m tools.calibrate_sharp --side left --live     # print ADC / distance only

Hold a flat, matte wall square to the sensor at each distance (measured
from the sensor face) and press Enter. The table is written to the path set
in slam/config.py (RobotIO.sharp_*_calibration) and used on the next run.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Optional, Sequence

from slam.config import DEFAULT
from slam.robot_io import RealRobot
from slam.sharp import SharpCalibration

DEFAULT_DISTANCES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
# A Sharp changes by hundreds of ADC counts over 0.1-0.8 m; less than this
# across the whole run means the port is not connected to a working sensor.
FLAT_ADC_SPAN = 40


def read_adc(robot: RealRobot, side: str, samples: int) -> Optional[float]:
    values = []
    for _ in range(samples):
        left, right = robot.sharp_adc()
        v = left if side == "left" else right
        if v is not None:
            values.append(v)
        time.sleep(0.02)
    return statistics.median(values) if values else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=["left", "right"], required=True)
    p.add_argument("--distances", type=float, nargs="+", default=list(DEFAULT_DISTANCES), help="metres")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--live", action="store_true")
    args = p.parse_args(argv)

    io = DEFAULT.robot_io
    path = io.sharp_left_calibration if args.side == "left" else io.sharp_right_calibration
    robot = RealRobot.connect(DEFAULT)
    try:
        if args.live:
            cal = robot.sharp_left_cal if args.side == "left" else robot.sharp_right_cal
            print(f"calibration: {cal.source}  (Ctrl+C to stop)")
            while True:
                adc = read_adc(robot, args.side, 5)
                d = cal.distance(adc)
                print(f"adc={adc}  distance={'---' if d is None else f'{d:.3f} m'}")

        points = []
        for d in args.distances:
            input(f"Put the wall {d:.2f} m from the {args.side} Sharp, then press Enter...")
            adc = read_adc(robot, args.side, args.samples)
            print(f"  adc = {adc}")
            if adc is None:
                print("  no reading (check the adapter wiring / RobotIO ports); skipped")
                continue
            points.append((adc, d))

        adcs = [a for a, _ in points]
        if len(adcs) >= 2 and max(adcs) - min(adcs) < FLAT_ADC_SPAN:
            print(f"The ADC hardly changed ({min(adcs):.0f}..{max(adcs):.0f}) although the wall moved:")
            print("this is not reading a Sharp. Check that the sensor is powered and that")
            print(f"RobotIO.sharp_{args.side} in slam/config.py names the port it is wired to")
            print("(find it with: python -m tools.check_robot adapter).")
            return 1
        try:
            cal = SharpCalibration(points, source=path)
        except ValueError as exc:
            print(f"Calibration rejected: {exc}. Points: {points}")
            print("Readings far away are often flat or noisy; drop those distances and retry.")
            return 1
        cal.save(path)
        print(f"Saved {len(points)} points to {path}")
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
