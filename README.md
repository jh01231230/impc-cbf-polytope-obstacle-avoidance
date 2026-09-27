# Iterative Convex Optimization with Control Barrier Functions for Obstacle Avoidance among Polytopes

Python implementation of the iterative convex MPC–DHOCBF controller for collision-free navigation of polytopic robots among polytopic obstacles.

**Shuo Liu\***, **Zhe Huang\***, and **Calin A. Belta**

\*Equal contribution.

- Paper: [arXiv:2603.05916](https://arxiv.org/abs/2603.05916)
- Video: [youtu.be/6XmuV3Gxvm0](https://youtu.be/6XmuV3Gxvm0)

At each time step the controller computes closest points between the robot and nearby obstacles, builds a supporting hyperplane from that pair, and linearizes both the robot geometry and the discrete-time dynamics. The resulting finite-horizon problem is a convex quadratic program, solved with [OSQP](https://osqp.org/). The same construction is used for a single robot in 2D, a sequential multi-robot scheme, and an L-shaped robot in a 3D maze.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. Animation export uses the FFmpeg binary installed by `imageio-ffmpeg`. Simulations that have no display automatically select the Matplotlib `Agg` backend.

## Layout

| Directory | Paper section | What it runs |
| --- | --- | --- |
| [`2d/`](2d/) | V-A | One robot (rectangle, triangle, or L-shape) in a 2D polytopic maze |
| [`multi_robot/`](multi_robot/) | V-B | Three robots navigating the same maze and avoiding one another |
| [`3d/`](3d/) | V-C | An L-shaped robot in a 3D maze of narrow openings |

Each directory is a standalone script. Run it from inside that directory so the saved trajectories land next to the code.

## One robot in 2D

```bash
cd 2d
python main.py --shape lshape
```

`--shape` is `rectangle`, `triangle`, or `lshape`. The default footprint is the L-shape. The map is the oblique maze used in the paper: the robot starts at `(0.15, 0.225)` and the goal is `(1.275, 0.975)`.

Useful flags:

```bash
python main.py --shape rectangle --horizon 12 --gamma 0.1
python main.py --shape triangle --steps 200 --headless
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--shape` | Robot footprint | `lshape` |
| `--horizon` | Prediction horizon \(N\) | `24` |
| `--gamma` | DHOCBF decay \(\gamma_1\) | `0.1` |
| `--steps` | Closed-loop steps | `600` |
| `--headless` | Skip the video and snapshot | off |
| `--verbose` | Print closest-point traces and every solver status | off |
| `--trials` | Random-start timing trials | `0` (demo only) |
| `--trial-steps` | Steps in each timing trial | `20` |
| `--seed` | Seed for those trials | `5` |

Outputs:

- `impc_run.mp4` — animation of the footprint, guiding path, and predicted horizon
- `sample_plot_L.png`, `sample_plot_tri.png`, or `sample_plot_rect.png` — snapshot in the style of Figure 3
- `trajectory.npy`, `prediction.npy` — executed state and predicted trajectories

The navigation case reported in Section V-A uses \(N = 12\) and \(\gamma_1 = 0.1\):

```bash
python main.py --shape lshape --horizon 12 --gamma 0.1
```

## Three robots in 2D

```bash
cd multi_robot
python main.py
```

The team is an L-shape, a triangle, and a rectangle, each scaled to two thirds of the single-robot size, on the same maze. Robots are solved in index order. Earlier robots broadcast their predicted trajectories, and later robots treat those predictions as moving polytopic obstacles.

```bash
python main.py --steps 80 --headless
```

Outputs: `impc_run.mp4`, `impc_multi_agent.png`, `sample_plot_multi.png`, and `multi_agent_trajectories.npz`.

## 3D maze

```bash
cd 3d
python main_3d.py
```

An extruded L-shaped body moves through a fixed five-wall maze (`seed = 48` in `env_config.py`). The prediction horizon defaults to \(N = 8\), matching Section V-C. The video shows four synchronized views (front, side, top, and an oblique view).

```bash
python main_3d.py --horizon 8 --steps 200 --headless
```

Outputs: `impc_run.mp4`, `trajectory.npy`, and `prediction.npy`.

## Timing table

Table II records the per-step solve time over random free-space starts, 20 steps each. From `2d/`:

```bash
python main.py --shape rectangle --horizon 12 --gamma 0.1 --trials 10 --trial-steps 20 --headless
python main.py --shape triangle  --horizon 12 --gamma 0.2 --trials 10 --trial-steps 20 --headless
python main.py --shape lshape    --horizon 24 --gamma 0.1 --trials 10 --trial-steps 20 --headless
```

From `3d/`:

```bash
python main_3d.py --horizon 8 --gamma 0.1 --trials 10 --trial-steps 20 --headless
```

Rows are appended to `run_summary.csv` or `run_summary_3d.csv`. Reported times are wall-clock seconds per control step and include the closest-point quadratic programs and the convex MPC solve. They depend on the CPU.

## Citation

```bibtex
@article{liu2026impcpolytope,
  title   = {Iterative Convex Optimization with Control Barrier Functions for Obstacle Avoidance among Polytopes},
  author  = {Liu, Shuo and Huang, Zhe and Belta, Calin A.},
  journal = {arXiv preprint arXiv:2603.05916},
  year    = {2026}
}
```

The convex programs are solved with OSQP:

```bibtex
@article{stellato2020osqp,
  title   = {{OSQP}: An Operator Splitting Solver for Quadratic Programs},
  author  = {Stellato, Bartolomeo and Banjac, Goran and Goulart, Paul and Bemporad, Alberto and Boyd, Stephen},
  journal = {Mathematical Programming Computation},
  volume  = {12},
  number  = {4},
  pages   = {637--672},
  year    = {2020}
}
```
