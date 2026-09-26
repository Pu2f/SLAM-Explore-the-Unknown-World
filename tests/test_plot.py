import csv

from slam.config import DEFAULT
from slam.explorer import Explorer
from slam.geometry import Direction
from slam.logger import RunLogger
from slam.sim import SimRobot, generate_maze
from tools.plot_run import load_trajectory, main, plot_run


def sim_run(tmp_path):
    gt = generate_maze(3, 3, seed=4, loops=1)
    (tmp_path / "ground_truth.txt").write_text(gt.to_ascii({(1, 1): ">"}))
    logger = RunLogger(str(tmp_path))
    Explorer(SimRobot(gt, (1, 1), Direction.E, noise=DEFAULT.sim_noise, seed=4), DEFAULT, logger).run()
    logger.close()
    return tmp_path


def test_all_figures_written(tmp_path):
    run = sim_run(tmp_path)
    written = plot_run(run)
    assert [p.name for p in written] == ["map.png", "trajectory.png", "comparison.png"]
    for p in written:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_real_run_without_truth_or_gt(tmp_path):
    run = sim_run(tmp_path)
    (run / "ground_truth.txt").unlink()
    rows = list(csv.DictReader((run / "trajectory.csv").open()))
    with (run / "trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            r.update(true_x_m="", true_y_m="", true_heading_deg="")
            w.writerow(r)
    assert all(v != v for v in load_trajectory(run)["true_x_m"])  # all NaN
    assert main([str(run)]) == 0
    assert (run / "trajectory.png").exists()
    assert not (run / "comparison.png").exists()
