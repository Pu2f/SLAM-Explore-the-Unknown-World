import pytest

from slam.config import DEFAULT, PERFECT_SIM, Config, SimNoise
from slam.geometry import Direction, angle_diff, neighbor
from slam.maze_map import EdgeState, WallMap, edge_cells
from slam.sim import SimRobot, generate_maze

CELL = DEFAULT.geometry.cell_size_m

# Start cell (0, 0) is open to the N only.
CORRIDOR = """\
+---+---+
|       |
+   +---+
|   |   |
+---+---+
"""


def corridor() -> WallMap:
    m, _ = WallMap.from_ascii(CORRIDOR)
    return m


def drive(robot: SimRobot, fwd: float, right: float, turn: float, seconds: float) -> None:
    robot.drive_speed(fwd, right, turn)
    robot.sleep(seconds)
    robot.stop()


@pytest.mark.parametrize("w,h,loops", [(1, 1, 0), (3, 3, 0), (4, 5, 0), (6, 4, 3)])
def test_generated_maze_is_closed_and_connected(w, h, loops):
    maze = generate_maze(w, h, seed=7, loops=loops)
    assert maze.cells == {(x, y) for x in range(w) for y in range(h)}
    open_inner = 0
    for key, state in maze.edges():
        a, b = edge_cells(key)
        inside = a in maze.cells and b in maze.cells
        if not inside:
            assert state == EdgeState.WALL  # outer boundary is always closed
        elif state == EdgeState.OPEN:
            open_inner += 1
    assert open_inner == w * h - 1 + loops  # spanning tree + extra loops

    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        c = stack.pop()
        for d in Direction:
            n = neighbor(c, d)
            if maze.get(c, d) == EdgeState.OPEN and n not in seen:
                seen.add(n)
                stack.append(n)
    assert seen == maze.cells


def test_generation_is_deterministic():
    assert generate_maze(5, 5, seed=3).to_dict() == generate_maze(5, 5, seed=3).to_dict()
    assert generate_maze(5, 5, seed=3).to_dict() != generate_maze(5, 5, seed=4).to_dict()


def test_tof_reads_walls_around_start_cell():
    robot = SimRobot(corridor(), (0, 0), Direction.N, noise=PERFECT_SIM)
    expected = {0: 2 * CELL - CELL / 2, 90: CELL / 2, 180: CELL / 2, -90: CELL / 2}
    for yaw, dist in expected.items():
        robot.gimbal_moveto(yaw)
        assert robot.tof() == pytest.approx(dist, abs=1e-6), yaw


def test_everything_is_reported_in_the_robot_frame():
    # Robot faces world E. In the robot's own frame it still starts at
    # (0, 0) facing N, and moving "forward" increases map y.
    maze = generate_maze(3, 3, seed=0)
    for d in Direction:
        maze.set((1, 1), d, EdgeState.OPEN)
    robot = SimRobot(maze, (1, 1), Direction.E, noise=PERFECT_SIM)
    assert robot.imu_yaw() == pytest.approx(0.0)
    drive(robot, 0.3, 0.0, 0.0, 1.0)
    assert robot.odometry() == (pytest.approx(0.0, abs=1e-9), pytest.approx(0.3))
    assert robot.true_pose_map().y == pytest.approx(0.3)
    wx, wy, wh = robot.true_pose_world()
    assert (wx, wy, wh) == (pytest.approx(CELL + 0.3), pytest.approx(CELL), pytest.approx(90.0))


def test_turning_clockwise_increases_yaw():
    robot = SimRobot(corridor(), (0, 0), Direction.N, noise=PERFECT_SIM)
    drive(robot, 0.0, 0.0, 45.0, 2.0)
    assert robot.imu_yaw() == pytest.approx(90.0)
    drive(robot, 0.0, 0.0, -90.0, 3.0)
    assert abs(angle_diff(180.0, robot.imu_yaw())) < 1e-6


def test_drive_one_cell_then_tof_sees_end_wall():
    robot = SimRobot(corridor(), (0, 0), Direction.N, noise=PERFECT_SIM)
    drive(robot, 0.3, 0.0, 0.0, 2.0)
    assert robot.odometry()[1] == pytest.approx(CELL)
    robot.gimbal_moveto(0)
    assert robot.tof() == pytest.approx(CELL / 2)
    robot.gimbal_moveto(90)
    assert robot.tof() == pytest.approx(CELL + CELL / 2)  # open to the E


def test_wall_blocks_motion_but_odometry_keeps_counting():
    robot = SimRobot(corridor(), (0, 0), Direction.N, noise=PERFECT_SIM)
    drive(robot, 0.0, 0.3, 0.0, 2.0)  # strafe right into the wall
    radius = DEFAULT.geometry.robot_radius_m
    assert robot.true_pose_map().x == pytest.approx(CELL / 2 - radius, abs=0.01)
    assert robot.collisions == 1
    assert robot.odometry()[0] == pytest.approx(0.6)  # wheels spun anyway


def test_sharp_and_ir():
    robot = SimRobot(corridor(), (0, 0), Direction.N, noise=PERFECT_SIM)
    s = DEFAULT.sensors
    left, right = robot.sharp()
    assert left == pytest.approx(CELL / 2 + s.sharp_left.right_m)
    assert right == pytest.approx(CELL / 2 - s.sharp_right.right_m)
    assert robot.ir_front() == (False, False)

    drive(robot, 0.0, 0.3, 0.0, 0.4)  # close to the E wall
    left, right = robot.sharp()
    assert right == pytest.approx(s.sharp_min_m)  # closer than the minimum range: reads the minimum
    assert robot.ir_front() == (False, True)

    drive(robot, 0.3, 0.0, 0.0, 2.0)  # into the open E-W corridor
    assert robot.sharp()[1] is not None


def test_noise_makes_odometry_disagree_with_truth():
    maze = generate_maze(3, 1, seed=0)
    robot = SimRobot(maze, (0, 0), Direction.E, noise=SimNoise(slip_bias=-0.05, slip_std=0.0), seed=1)
    drive(robot, 0.3, 0.0, 0.0, 2.0)
    true_y = robot.true_pose_map().y
    odom_y = robot.odometry()[1]
    assert odom_y == pytest.approx(0.6, abs=0.01)
    assert true_y == pytest.approx(0.57, abs=0.01)


def test_seeded_noise_is_reproducible():
    def run():
        robot = SimRobot(generate_maze(3, 3, seed=2), (0, 0), config=Config(), seed=5)
        drive(robot, 0.2, 0.0, 10.0, 1.0)
        return robot.odometry(), robot.imu_yaw(), robot.tof()

    assert run() == run()


def test_room_walls_are_seen_through_a_gap():
    maze = corridor()
    maze.set((0, 0), Direction.S, EdgeState.OPEN)
    no_room = SimRobot(maze, (0, 0), Direction.N, noise=PERFECT_SIM)
    no_room.gimbal_moveto(180)
    assert no_room.tof() is None
    room = SimRobot(maze, (0, 0), Direction.N, noise=PERFECT_SIM, room_margin_m=(1.0, 2.0, 2.0, 2.0))
    room.gimbal_moveto(180)
    assert room.tof() == pytest.approx(CELL / 2 + 1.0)
