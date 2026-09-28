"""Multi-robot 2D navigation with sequential iterative convex MPC-DHOCBF.

Reproduces Section V-B of Liu, Huang, and Belta, arXiv:2603.05916.
"""
import argparse
import os
import numpy as np

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import warnings
import osqp
import time
import math
from pathlib import Path
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import imageio.v2 as imageio
from scipy.sparse import (
    block_diag as sparse_block_diag,
    coo_matrix,
    csr_matrix,
    diags,
    eye,
    hstack,
    lil_matrix,
    vstack,
)
from scipy.linalg import block_diag
from scipy.spatial import ConvexHull

from polytope_geometry import (
    _OBSTACLE_HALFSPACES,
    init_model,
    robot_halfspace,
    solve_min_distance_qp,
)
from env_config import ENVIRONMENT, OBSTACLE_REGIONS, OBSTACLE_VERTICES
from path_planner import plan_path
from config import DEFAULT_CONFIG, DualSlackConfig, NMPCConfig
from utils_interpolation import interpolate_path

_VERBOSE = False

warnings.filterwarnings('ignore', message='Converting sparse P to a CSC \(compressed sparse column\) matrix. \(It may take a while...\)', category=UserWarning, module='osqp.utils')
warnings.filterwarnings('ignore', message='Converting sparse A to a CSC \(compressed sparse column\) matrix. \(It may take a while...\)', category=UserWarning, module='osqp.utils')


@dataclass(frozen=True)
class ObstacleGeometry:
    """Static obstacle description with slack configuration."""

    A: np.ndarray
    b: np.ndarray
    slack: DualSlackConfig

    @property
    def facets(self) -> int:
        return int(self.A.shape[0])


@dataclass
class DualDistanceSolution:
    """Closest-point auxiliary variables extracted from the OSQP solution."""

    # For concave robots we represent the robot as a union of convex components.
    # All closest-point variables are indexed by (obstacle, component, horizon_step).
    y_obs: np.ndarray  # shape (m, C, N, 2)
    y_robot: np.ndarray  # shape (m, C, N, 2)
    lambda_obs: List[np.ndarray]  # list per obstacle: shape (C, N, facets_obs)
    lambda_robot: List[np.ndarray]  # list per component: shape (m, N, facets_comp)
    omega: np.ndarray  # shape (m, C, N)

    def squared_distances(self) -> np.ndarray:
        # (m, C, N)
        return np.sum((self.y_obs - self.y_robot) ** 2, axis=-1)

    def min_squared_distances(self) -> np.ndarray:
        # (m, N) minimum across robot components
        return self.squared_distances().min(axis=1)


@dataclass
class DualDistanceLayout:
    """Index bookkeeping for the enlarged OSQP decision vector."""

    state: slice
    control: slice
    y_obs: List[List[List[slice]]]  # [obs][comp][step]
    y_robot: List[List[List[slice]]]  # [obs][comp][step]
    lambda_obs: List[List[List[slice]]]  # [obs][comp][step]
    lambda_robot: List[List[List[slice]]]  # [obs][comp][step]
    omega: List[List[List[slice]]]  # [obs][comp][step]
    total_dim: int

    def omega_indices(self) -> np.ndarray:
        indices = []
        for obstacle_blocks in self.omega:
            for comp_blocks in obstacle_blocks:
                for block in comp_blocks:
                    indices.append(block.start)
        return np.asarray(indices, dtype=int)


@dataclass(frozen=True)
class RobotComponentGeometry:
    """One convex robot component in the robot frame."""

    vertices_local: np.ndarray  # (n,2) convex hull vertices (ordered)
    A_local: np.ndarray  # (m,2) half-space normals
    b_local: np.ndarray  # (m,) half-space offsets (A x <= b)


@dataclass(frozen=True)
class MultiAgentGoal:
    """Start/goal pair for one robot in the multi-agent run."""

    name: str
    robot_shape: str  # "lshape" | "triangle" | "rectangle"
    start: np.ndarray  # (x, y, theta) or (x, y, theta, v)
    goal: np.ndarray  # (x, y) or (x, y, theta, v)
    path_margin: float = 0.05  # planner obstacle clearance
    start_delay: int = 0  # steps before agent begins moving (holds position)


@dataclass
class MultiAgentState:
    """Mutable runtime state for one robot in the synchronous loop."""

    spec: MultiAgentGoal
    state: np.ndarray  # shape (4,)
    goal_state: np.ndarray  # shape (4,)
    cfg: NMPCConfig
    robot_components: List[RobotComponentGeometry]
    path_points: np.ndarray  # shape (M,2)
    u_guess_vec: np.ndarray  # shape (nu*N,1)
    u_guess_mat: np.ndarray  # shape (nu,N)
    x_guess: np.ndarray  # shape (nx,N+1)
    dual_solution: Optional[DualDistanceSolution]
    done: bool
    states_hist: List[np.ndarray]
    controls_hist: List[np.ndarray]
    probe_hist: List[np.ndarray]  # probing horizon xy per step: (N+1, 2)
    xr_hist: List[np.ndarray]  # reference trajectory (4, N+1) per step


def _convex_hull_geometry(points: np.ndarray) -> RobotComponentGeometry:
    """Return convex hull vertices + half-space representation for a 2D point set."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points for a polygon hull.")
    hull = ConvexHull(pts)
    vertices = pts[hull.vertices]
    A = hull.equations[:, :2].astype(float)
    b = (-hull.equations[:, 2]).astype(float)
    return RobotComponentGeometry(vertices_local=vertices, A_local=A, b_local=b)


def build_robot_geometry(cfg: NMPCConfig) -> List[RobotComponentGeometry]:
    """Build robot geometry components from the configured footprint (Sec. V-B)."""
    shape = str(getattr(cfg, "robot_shape", "rectangle")).strip().lower()
    if shape == "rectangle":
        length = float(getattr(cfg, "robot_rectangle_length", 0.15))
        width = float(getattr(cfg, "robot_rectangle_width", 0.06))
        hx = 0.5 * length
        hy = 0.5 * width
        rect = np.array(
            [
                [hx, hy],
                [hx, -hy],
                [-hx, -hy],
                [-hx, hy],
            ],
            dtype=float,
        )
        return [_convex_hull_geometry(rect)]
    if shape == "triangle":
        pts = np.asarray(getattr(cfg, "robot_triangle_points"), dtype=float)
        return [_convex_hull_geometry(pts)]
    if shape == "lshape":
        p1 = np.asarray(getattr(cfg, "robot_lshape_part1_points"), dtype=float)
        p2 = np.asarray(getattr(cfg, "robot_lshape_part2_points"), dtype=float)
        return [_convex_hull_geometry(p1), _convex_hull_geometry(p2)]
    raise ValueError(f"Unknown robot_shape={shape!r}. Expected 'rectangle', 'triangle', or 'lshape'.")


def _robot_geometry_world(
    state_vec: np.ndarray, components: Sequence[RobotComponentGeometry]
) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    """Return (world_vertices_list, world_halfspaces_list) for each component."""
    state = np.asarray(state_vec, dtype=float).ravel()
    center = state[:2]
    heading = float(state[2]) if state.size >= 3 else 0.0
    c = math.cos(heading)
    s = math.sin(heading)
    rotation = np.array([[c, -s], [s, c]], dtype=float)

    polys_world: List[np.ndarray] = []
    halfspaces_world: List[Tuple[np.ndarray, np.ndarray]] = []
    for comp in components:
        verts_w = comp.vertices_local @ rotation.T + center
        A_w = comp.A_local @ rotation.T
        b_w = comp.b_local + A_w @ center
        polys_world.append(verts_w)
        halfspaces_world.append((A_w, b_w))
    return polys_world, halfspaces_world


def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the closest point to p on the closed segment [a,b]."""
    p = np.asarray(p, dtype=float).reshape(2)
    a = np.asarray(a, dtype=float).reshape(2)
    b = np.asarray(b, dtype=float).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy()
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    return a + t * ab


def _closest_point_on_polygon_boundary(p: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Return the closest point to p on the boundary of a polygon defined by verts."""
    v = np.asarray(verts, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] != 2:
        return np.asarray(p, dtype=float).reshape(2)
    best_q = v[0].copy()
    best_d2 = float("inf")
    n = int(v.shape[0])
    for i in range(n):
        a = v[i]
        b = v[(i + 1) % n]
        q = _closest_point_on_segment(p, a, b)
        d2 = float(np.sum((np.asarray(p, dtype=float).reshape(2) - q) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_q = q
    return best_q


def _closest_point_on_union_boundary(p: np.ndarray, polys: Sequence[np.ndarray]) -> np.ndarray:
    """Return the closest point to p on the boundary of a union of polygons."""
    best_q = np.asarray(p, dtype=float).reshape(2)
    best_d2 = float("inf")
    for poly in polys:
        q = _closest_point_on_polygon_boundary(p, poly)
        d2 = float(np.sum((np.asarray(p, dtype=float).reshape(2) - q) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_q = q
    return best_q


def _build_layout(
    nx: int,
    nu: int,
    horizon: int,
    obstacles: Sequence[ObstacleGeometry],
    robot_component_facets: Sequence[int],
) -> DualDistanceLayout:
    """Return a map from logical variables to the flattened decision vector."""

    state = slice(0, (horizon + 1) * nx)
    control = slice(state.stop, state.stop + horizon * nu)
    cursor = control.stop

    n_components = int(len(robot_component_facets))
    if n_components <= 0:
        raise ValueError("robot_component_facets must be non-empty.")

    y_obs: List[List[List[slice]]] = [
        [[None for _ in range(horizon)] for _ in range(n_components)] for _ in obstacles  # type: ignore
    ]
    y_robot: List[List[List[slice]]] = [
        [[None for _ in range(horizon)] for _ in range(n_components)] for _ in obstacles  # type: ignore
    ]
    lambda_obs: List[List[List[slice]]] = [
        [[None for _ in range(horizon)] for _ in range(n_components)] for _ in obstacles  # type: ignore
    ]
    lambda_robot: List[List[List[slice]]] = [
        [[None for _ in range(horizon)] for _ in range(n_components)] for _ in obstacles  # type: ignore
    ]
    omega: List[List[List[slice]]] = [
        [[None for _ in range(horizon)] for _ in range(n_components)] for _ in obstacles  # type: ignore
    ]

    for obs_idx, _ in enumerate(obstacles):
        for comp_idx in range(n_components):
            for step in range(horizon):
                y_obs[obs_idx][comp_idx][step] = slice(cursor, cursor + 2)
                cursor += 2
                y_robot[obs_idx][comp_idx][step] = slice(cursor, cursor + 2)
                cursor += 2

    for obs_idx, obstacle in enumerate(obstacles):
        facet_count = obstacle.facets
        for comp_idx in range(n_components):
            for step in range(horizon):
                lambda_obs[obs_idx][comp_idx][step] = slice(cursor, cursor + facet_count)
                cursor += facet_count

    for obs_idx, _ in enumerate(obstacles):
        for comp_idx in range(n_components):
            facet_count = int(robot_component_facets[comp_idx])
            for step in range(horizon):
                lambda_robot[obs_idx][comp_idx][step] = slice(cursor, cursor + facet_count)
                cursor += facet_count

    for obs_idx, _ in enumerate(obstacles):
        for comp_idx in range(n_components):
            for step in range(horizon):
                omega[obs_idx][comp_idx][step] = slice(cursor, cursor + 1)
                cursor += 1

    return DualDistanceLayout(
        state=state,
        control=control,
        y_obs=y_obs,
        y_robot=y_robot,
        lambda_obs=lambda_obs,
        lambda_robot=lambda_robot,
        omega=omega,
        total_dim=cursor,
    )


def _load_obstacle_geometry(
    cfg: NMPCConfig,
    extra_halfspaces: Sequence[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> List[ObstacleGeometry]:
    """Return obstacle half-space data paired with slack settings.

    `extra_halfspaces` are appended after static map obstacles and are useful for
    dynamic barriers (e.g., predicted poses of other robots).
    """

    slack_configs = list(cfg.dual_slacks)
    if not slack_configs:
        slack_configs = [DualSlackConfig()]
    if len(slack_configs) < len(_OBSTACLE_HALFSPACES):
        slack_configs.extend([slack_configs[-1]] * (len(_OBSTACLE_HALFSPACES) - len(slack_configs)))
    elif len(slack_configs) > len(_OBSTACLE_HALFSPACES):
        slack_configs = slack_configs[: len(_OBSTACLE_HALFSPACES)]

    geometries: List[ObstacleGeometry] = []
    for slack_cfg, (A_obs, b_obs) in zip(slack_configs, _OBSTACLE_HALFSPACES):
        geometries.append(
            ObstacleGeometry(
                A=np.asarray(A_obs, dtype=float),
                b=np.asarray(b_obs, dtype=float),
                slack=slack_cfg,
            )
        )
    if extra_halfspaces:
        # Reuse the last configured slack policy for dynamic barriers.
        dyn_slack = slack_configs[-1] if slack_configs else DualSlackConfig()
        for A_obs, b_obs in extra_halfspaces:
            geometries.append(
                ObstacleGeometry(
                    A=np.asarray(A_obs, dtype=float),
                    b=np.asarray(b_obs, dtype=float).reshape(-1),
                    slack=dyn_slack,
                )
            )

    return geometries


def _robot_component_halfspaces_along_trajectory(
    x_traj: np.ndarray, components: Sequence[RobotComponentGeometry]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute (A,b) half-spaces for each robot component at each predicted state.

    Returns:
      - robot_As: list (len=C) of arrays with shape (N, facets_c, 2)
      - robot_bs: list (len=C) of arrays with shape (N, facets_c)
    """
    horizon = x_traj.shape[1] - 1
    centers = x_traj[:2, 1:]
    if x_traj.shape[0] >= 3:
        headings = x_traj[2, 1:]
    else:
        headings = np.zeros(horizon)

    robot_As: List[np.ndarray] = []
    robot_bs: List[np.ndarray] = []
    for comp in components:
        facet_count = int(comp.A_local.shape[0])
        A_steps = np.zeros((horizon, facet_count, 2), dtype=float)
        b_steps = np.zeros((horizon, facet_count), dtype=float)
        for step in range(horizon):
            theta = float(headings[step])
            center = centers[:, step]
            c = math.cos(theta)
            s = math.sin(theta)
            rotation = np.array([[c, -s], [s, c]], dtype=float)
            A_w = comp.A_local @ rotation.T
            b_w = comp.b_local + A_w @ center
            A_steps[step] = A_w
            b_steps[step] = b_w
        robot_As.append(A_steps)
        robot_bs.append(b_steps)
    return robot_As, robot_bs


def _initialize_dual_solution(
    x_traj: np.ndarray,
    obstacles: Sequence[ObstacleGeometry],
    robot_As: List[np.ndarray],
    robot_bs: List[np.ndarray],
) -> DualDistanceSolution:
    """Construct an initial closest-point warm start via distance queries."""

    horizon = x_traj.shape[1] - 1
    m = len(obstacles)
    n_components = int(len(robot_As))
    if n_components <= 0 or len(robot_bs) != n_components:
        raise ValueError("robot_As/robot_bs must be non-empty lists with the same length.")

    y_obs = np.zeros((m, n_components, horizon, 2))
    y_robot = np.zeros((m, n_components, horizon, 2))
    lambda_obs = [np.zeros((n_components, horizon, obstacle.facets)) for obstacle in obstacles]
    lambda_robot = [
        np.zeros((m, horizon, int(robot_As[comp_idx].shape[1]))) for comp_idx in range(n_components)
    ]
    omega = np.ones((m, n_components, horizon))

    for step in range(horizon):
        for obs_idx, obstacle in enumerate(obstacles):
            for comp_idx in range(n_components):
                A_robot = robot_As[comp_idx][step]
                b_robot = robot_bs[comp_idx][step]
                _, contact_obs, contact_robot = solve_min_distance_qp(
                    obstacle.A, obstacle.b, A_robot, b_robot
                )
                y_obs[obs_idx, comp_idx, step] = contact_obs
                y_robot[obs_idx, comp_idx, step] = contact_robot

    return DualDistanceSolution(
        y_obs=y_obs,
        y_robot=y_robot,
        lambda_obs=lambda_obs,
        lambda_robot=lambda_robot,
        omega=omega,
    )


def _shift_dual_solution(
    previous: DualDistanceSolution,
    x_traj: np.ndarray,
    obstacles: Sequence[ObstacleGeometry],
    robot_As: List[np.ndarray],
    robot_bs: List[np.ndarray],
) -> DualDistanceSolution:
    """Shift the previous dual solution forward to align with the new horizon."""

    horizon = x_traj.shape[1] - 1
    m = len(obstacles)
    n_components = int(len(robot_As))
    if n_components <= 0 or len(robot_bs) != n_components:
        return _initialize_dual_solution(x_traj, obstacles, robot_As, robot_bs)

    # If horizon/components changed, fall back to full initialization.
    if (
        previous.y_obs.ndim != 4
        or previous.y_obs.shape[0] != m
        or previous.y_obs.shape[1] != n_components
        or previous.y_obs.shape[2] != horizon
    ):
        return _initialize_dual_solution(x_traj, obstacles, robot_As, robot_bs)

    shifted_y_obs = np.zeros_like(previous.y_obs)
    shifted_y_robot = np.zeros_like(previous.y_robot)
    shifted_lambda_obs = [np.zeros_like(arr) for arr in previous.lambda_obs]
    shifted_lambda_robot = [np.zeros_like(arr) for arr in previous.lambda_robot]
    shifted_omega = np.ones_like(previous.omega)

    shifted_y_obs[:, :, :-1, :] = previous.y_obs[:, :, 1:, :]
    shifted_y_robot[:, :, :-1, :] = previous.y_robot[:, :, 1:, :]
    for obs_idx in range(m):
        shifted_lambda_obs[obs_idx][:, :-1, :] = previous.lambda_obs[obs_idx][:, 1:, :]
    for comp_idx in range(n_components):
        shifted_lambda_robot[comp_idx][:, :-1, :] = previous.lambda_robot[comp_idx][:, 1:, :]
    shifted_omega[:, :, :-1] = previous.omega[:, :, 1:]

    last_step = horizon - 1
    for obs_idx, obstacle in enumerate(obstacles):
        for comp_idx in range(n_components):
            _, obs_contact, robot_contact = solve_min_distance_qp(
                obstacle.A,
                obstacle.b,
                robot_As[comp_idx][last_step],
                robot_bs[comp_idx][last_step],
            )
            shifted_y_obs[obs_idx, comp_idx, last_step] = obs_contact
            shifted_y_robot[obs_idx, comp_idx, last_step] = robot_contact
            shifted_lambda_obs[obs_idx][comp_idx, last_step, :] = 0.0
            shifted_lambda_robot[comp_idx][obs_idx, last_step, :] = 0.0
            shifted_omega[obs_idx, comp_idx, last_step] = 1.0

    return DualDistanceSolution(
        y_obs=shifted_y_obs,
        y_robot=shifted_y_robot,
        lambda_obs=shifted_lambda_obs,
        lambda_robot=shifted_lambda_robot,
        omega=shifted_omega,
    )


def _extract_dual_solution(
    decision: np.ndarray,
    layout: DualDistanceLayout,
    obstacles: Sequence[ObstacleGeometry],
    robot_component_facets: Sequence[int],
    horizon: int,
) -> DualDistanceSolution:
    """Parse the OSQP solution vector into structured closest-point data."""

    m = len(obstacles)
    n_components = int(len(robot_component_facets))
    y_obs = np.zeros((m, n_components, horizon, 2))
    y_robot = np.zeros((m, n_components, horizon, 2))
    lambda_obs: List[np.ndarray] = []
    lambda_robot = [
        np.zeros((m, horizon, int(robot_component_facets[comp_idx]))) for comp_idx in range(n_components)
    ]
    omega = np.zeros((m, n_components, horizon))

    for obs_idx, obstacle in enumerate(obstacles):
        lam_obs = np.zeros((n_components, horizon, obstacle.facets))
        for comp_idx in range(n_components):
            for step in range(horizon):
                y_obs[obs_idx, comp_idx, step, :] = decision[
                    layout.y_obs[obs_idx][comp_idx][step]
                ]
                y_robot[obs_idx, comp_idx, step, :] = decision[
                    layout.y_robot[obs_idx][comp_idx][step]
                ]
                lam_obs[comp_idx, step, :] = decision[
                    layout.lambda_obs[obs_idx][comp_idx][step]
                ]
                lambda_robot[comp_idx][obs_idx, step, :] = decision[
                    layout.lambda_robot[obs_idx][comp_idx][step]
                ]
                omega_slice = layout.omega[obs_idx][comp_idx][step]
                omega_value = decision[omega_slice.start:omega_slice.stop]
                omega[obs_idx, comp_idx, step] = float(omega_value.reshape(-1)[0])
        lambda_obs.append(lam_obs)

    return DualDistanceSolution(
        y_obs=y_obs,
        y_robot=y_robot,
        lambda_obs=lambda_obs,
        lambda_robot=lambda_robot,
        omega=omega,
    )


def _capture_frame(fig):
    """Render the current Matplotlib figure and return it as an RGB array."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    # Matplotlib 3.8+: tostring_rgb is deprecated; use buffer_rgba then drop alpha.
    # NOTE: buffer_rgba() exposes the renderer's internal buffer. We must copy it;
    # otherwise every stored frame can alias the same memory and the saved video
    # will look like a single repeated frame.
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame_rgba = buf.reshape((height, width, 4))
    return frame_rgba[:, :, :3].copy()


def _pad_frame_to_macroblock(frame: np.ndarray, macro_block_size: int = 16) -> np.ndarray:
    """Pad a frame so its dimensions are divisible by macro_block_size."""
    if macro_block_size <= 1:
        return frame
    height, width = frame.shape[:2]
    pad_h = (macro_block_size - (height % macro_block_size)) % macro_block_size
    pad_w = (macro_block_size - (width % macro_block_size)) % macro_block_size
    if pad_h == 0 and pad_w == 0:
        return frame
    new_h = height + pad_h
    new_w = width + pad_w
    if frame.ndim == 2:
        padded = np.zeros((new_h, new_w), dtype=frame.dtype)
        padded[:height, :width] = frame
    else:
        padded = np.zeros((new_h, new_w, frame.shape[2]), dtype=frame.dtype)
        padded[:height, :width, :] = frame
    return padded


def on_segment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

def do_intersect(p1, q1, p2, q2):
    # Find the four orientations needed for the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special Cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    
    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    
    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    
    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    # Doesn't fall in any of the above cases
    return False


def _robot_vertices_global(center: np.ndarray, heading: float, half_extents: np.ndarray) -> np.ndarray:
    """Return world-frame rectangle vertices for the robot footprint."""
    c = math.cos(heading)
    s = math.sin(heading)
    rotation = np.array([[c, -s], [s, c]])
    hx, hy = half_extents
    local = np.array(
        [
            [hx, hy],
            [hx, -hy],
            [-hx, -hy],
            [-hx, hy],
        ],
        dtype=float,
    )
    return local @ rotation.T + center


def _point_in_halfspace(point: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.all(A @ point <= b.reshape(-1) + tol))


def _polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Return True if two polygons intersect (vertex containment or edge crossing)."""
    edges_a = list(zip(poly_a, np.vstack([poly_a[1:], poly_a[:1]])))
    edges_b = list(zip(poly_b, np.vstack([poly_b[1:], poly_b[:1]])))
    for p1, q1 in edges_a:
        for p2, q2 in edges_b:
            if do_intersect(p1, q1, p2, q2):
                return True
    return False


def _shrink_polygon(poly: np.ndarray, inset: float) -> np.ndarray:
    """Shrink polygon toward centroid by inset amount for tolerance in edge tests."""
    if inset <= 0.0 or poly.shape[0] < 2:
        return poly
    centroid = np.mean(poly, axis=0)
    out = []
    for v in poly:
        d = np.linalg.norm(v - centroid)
        if d < 1e-9:
            out.append(v)
        else:
            scale = max(0.01, 1.0 - inset / (d + 1e-9))
            out.append(centroid + scale * (v - centroid))
    return np.array(out, dtype=float)


def in_barrier(
    state,
    penetration_tol: float | None = None,
    robot_components: Sequence[RobotComponentGeometry] | None = None,
    cfg: NMPCConfig | None = None,
):
    """Return True if the robot footprint intersects any obstacle."""
    state_arr = np.asarray(state, dtype=float).ravel()
    if state_arr.size < 2:
        raise ValueError("State must include at least x and y coordinates.")

    if penetration_tol is None:
        # Use the *current run* config tolerance when provided (important for reproducibility
        # when cfg differs from DEFAULT_CONFIG).
        tol_cfg = cfg if cfg is not None else DEFAULT_CONFIG
        penetration_tol = float(getattr(tol_cfg, "barrier_penetration_tol", 0.0))
    penetration_tol = float(max(0.0, penetration_tol))
    # When barrier_penetration_tol=0, use a minimal numerical epsilon to avoid rejecting
    # solutions that graze obstacles due to OSQP/solver finite precision (~2-3mm).
    _NUMERICAL_EPSILON = 3e-3  # 3mm
    effective_tol = float(_NUMERICAL_EPSILON) if penetration_tol == 0.0 else penetration_tol

    # Default to config-selected robot geometry (rectangle/triangle/lshape).
    if robot_components is None:
        geom_cfg = cfg if cfg is not None else DEFAULT_CONFIG
        robot_components = build_robot_geometry(geom_cfg)

    polys_world, halfspaces_world = _robot_geometry_world(state_arr, robot_components)

    for obs_idx, (A_obs, b_obs) in enumerate(_OBSTACLE_HALFSPACES):
        obs_vertices = np.asarray(OBSTACLE_VERTICES[obs_idx], dtype=float)

        for poly_robot, (A_robot, b_robot) in zip(polys_world, halfspaces_world):
            # Any robot vertex inside obstacle
            for vertex in poly_robot:
                margin = A_obs @ vertex - b_obs
                if float(margin.max()) < -effective_tol:
                    center = state_arr[:2]
                    print(
                        f"[in_barrier] footprint vertex inside obstacle {obs_idx} at "
                        f"({center[0]:.4f},{center[1]:.4f}). "
                        f"Vertex: ({vertex[0]:.4f},{vertex[1]:.4f}). "
                        f"Max margin (should be <=0): {margin.max():.6f}"
                    )
                    return True

            # Any obstacle vertex inside this robot component
            for vertex in obs_vertices:
                margin = A_robot @ vertex - b_robot
                if float(margin.max()) < -effective_tol:
                    center = state_arr[:2]
                    print(
                        f"[in_barrier] obstacle {obs_idx} vertex inside footprint at "
                        f"({center[0]:.4f},{center[1]:.4f})"
                    )
                    return True

        # Edge intersection test. When penetration_tol > 0, use a slightly shrunk
        # robot polygon so grazing contacts (numerical or tiny overlap) are tolerated.
        for poly_robot in polys_world:
            poly_for_edge = _shrink_polygon(poly_robot, effective_tol)
            if _polygons_intersect(poly_for_edge, obs_vertices):
                center = state_arr[:2]
                print(
                    f"[in_barrier] footprint edges intersect obstacle {obs_idx} near "
                    f"({center[0]:.4f},{center[1]:.4f})"
                )
                return True

    return False


def _polyline_deviation(points: np.ndarray, path: np.ndarray) -> Tuple[float, float, float]:
    """Return mean/max and RMSE distance from points to a reference polyline."""
    seg_starts = path[:-1]
    seg_ends = path[1:]
    seg_vecs = seg_ends - seg_starts
    seg_norm_sq = (seg_vecs**2).sum(axis=1)

    def point_to_path(pt: np.ndarray) -> float:
        diffs = pt - seg_starts
        t = np.clip(np.einsum("ij,ij->i", diffs, seg_vecs) / np.maximum(seg_norm_sq, 1e-12), 0.0, 1.0)
        proj = seg_starts + seg_vecs * t[:, None]
        dists = np.linalg.norm(pt - proj, axis=1)
        return float(dists.min())

    dists = np.array([point_to_path(p) for p in points], dtype=float)
    mean = float(dists.mean())
    rms = float(np.sqrt((dists**2).mean()))
    maxd = float(dists.max())
    return mean, rms, maxd


def _integrate_one_step(state_vec: np.ndarray, control_vec: np.ndarray, dt: float) -> np.ndarray:
    """Apply one discrete dynamics step for state x=[x,y,theta,v], u=[omega,a,...]."""
    state = np.asarray(state_vec, dtype=float).reshape(-1)
    control = np.asarray(control_vec, dtype=float).reshape(-1)
    omega = float(control[0]) if control.size >= 1 else 0.0
    accel = float(control[1]) if control.size >= 2 else 0.0
    next_state = np.array(
        [
            dt * state[3] * np.cos(state[2]) + state[0],
            dt * state[3] * np.sin(state[2]) + state[1],
            dt * omega + state[2],
            dt * accel + state[3],
        ],
        dtype=float,
    )
    return next_state


def _build_reference_from_path(
    path_points: np.ndarray,
    current_state: np.ndarray,
    goal_state: np.ndarray,
    cfg: NMPCConfig,
    horizon: int,
    speed_scale: float = 1.0,
    heading_now: float | None = None,
) -> np.ndarray:
    """Build a local (4, N+1) reference by projecting onto the global path.

    Supports speed_scale for homotopy (1.0 forward, 0.5, 0.25 slower, -0.5 reverse,
    0.1 creep, 0.0 stop). Curvature-aware speed limiting for tight turns.
    """
    path = np.asarray(path_points, dtype=float).reshape(-1, 2)
    current = np.asarray(current_state, dtype=float).reshape(-1)
    goal = np.asarray(goal_state, dtype=float).reshape(-1)
    xr = np.zeros((4, horizon + 1), dtype=float)
    ref_speed = float(getattr(cfg, "reference_speed", 0.2))
    dt = float(cfg.dt)
    omega_max = float(max(abs(float(cfg.input_bounds.lower[0])), abs(float(cfg.input_bounds.upper[0]))))

    if path.shape[0] == 0:
        xr[0, :] = float(current[0])
        xr[1, :] = float(current[1])
        xr[2, :] = float(goal[2]) if len(goal) > 2 else 0.0
        xr[3, :] = 0.0
        return xr

    if path.shape[0] == 1:
        xr[0, :] = float(path[0, 0])
        xr[1, :] = float(path[0, 1])
        xr[2, :] = float(goal[2]) if len(goal) > 2 else 0.0
        xr[3, :] = 0.0
        xr[:, 0] = current[:4]
        return xr

    seg_vec = path[1:] - path[:-1]
    seg_len = np.linalg.norm(seg_vec, axis=1)
    cum_len = np.hstack([0.0, np.cumsum(seg_len)])

    # Project current position to nearest point on path.
    pos = current[:2]
    s0 = 0.0
    best_d2 = float("inf")
    seg_proj = 0
    for idx in range(seg_vec.shape[0]):
        a = path[idx]
        v = seg_vec[idx]
        denom = float(np.dot(v, v))
        if denom <= 1e-12:
            continue
        t = float(np.clip(np.dot(pos - a, v) / denom, 0.0, 1.0))
        proj = a + t * v
        d2 = float(np.sum((pos - proj) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            s0 = float(cum_len[idx] + t * seg_len[idx])
            seg_proj = idx

    total_len = float(cum_len[-1])
    speed_scale = float(speed_scale)

    # Reverse reference: move backward along path tangent.
    if speed_scale < -1e-3:
        curv_vec = seg_vec
        if curv_vec.shape[0] == 0:
            direction = np.array([math.cos(goal[2] if len(goal) > 2 else 0.0), math.sin(goal[2] if len(goal) > 2 else 0.0)])
        else:
            idx = min(seg_proj, curv_vec.shape[0] - 1)
            tangent = curv_vec[idx] / (seg_len[idx] + 1e-12)
            direction = tangent / (np.linalg.norm(tangent) + 1e-12)
        v_ref = -max(1e-6, ref_speed * abs(speed_scale))
        hd_ref = float(math.atan2(direction[1], direction[0]))
        for k in range(horizon + 1):
            t = dt * float(k)
            xr[0, k] = float(pos[0] + v_ref * t * direction[0])
            xr[1, k] = float(pos[1] + v_ref * t * direction[1])
            xr[2, k] = hd_ref
            xr[3, k] = v_ref
        xr[:, 0] = current[:4]
        return xr

    # Stop homotopy: fix reference at current position.
    if abs(speed_scale) <= 1e-3:
        hd = float(heading_now) if heading_now is not None else (
            float(goal[2]) if len(goal) > 2 else float(current[2])
        )
        xr[0, :] = float(pos[0])
        xr[1, :] = float(pos[1])
        xr[2, :] = hd
        xr[3, :] = 0.0
        xr[:, 0] = current[:4]
        return xr

    # Forward: curvature-aware speed limiting.
    trunc_path = path[seg_proj:]
    curv_vec = trunc_path[1:] - trunc_path[:-1]
    curv_length = np.linalg.norm(curv_vec, axis=1)
    if curv_length.size == 0:
        curv_length = np.array([0.0], dtype=float)
        curv_vec = np.zeros((1, 2), dtype=float)
    max_kappa = 0.0
    if curv_vec.shape[0] >= 2:
        seg_head = np.arctan2(curv_vec[:, 1], curv_vec[:, 0])
        dtheta = seg_head[1:] - seg_head[:-1]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        ds = 0.5 * (curv_length[1:] + curv_length[:-1])
        ds = np.maximum(ds, 1e-6)
        kappa = np.abs(dtheta) / ds
        max_kappa = float(np.max(kappa[:min(5, kappa.size)])) if kappa.size else 0.0

    v_nom = ref_speed * speed_scale
    if max_kappa > 1e-6 and omega_max > 0.0:
        v_limit = omega_max / max_kappa
        v_ref = max(1e-6, min(v_nom, v_limit))
    else:
        v_ref = max(1e-6, v_nom)

    for k in range(horizon + 1):
        s_k = min(total_len, s0 + v_ref * dt * float(k))
        seg_idx = int(np.searchsorted(cum_len, s_k, side="right") - 1)
        seg_idx = int(np.clip(seg_idx, 0, seg_vec.shape[0] - 1))

        if seg_len[seg_idx] > 1e-12:
            t = float((s_k - cum_len[seg_idx]) / seg_len[seg_idx])
            t = float(np.clip(t, 0.0, 1.0))
            p = path[seg_idx] + t * seg_vec[seg_idx]
            tangent = seg_vec[seg_idx] / seg_len[seg_idx]
            heading = float(math.atan2(tangent[1], tangent[0]))
        else:
            p = path[seg_idx].copy()
            heading = float(goal[2]) if len(goal) > 2 else 0.0

        xr[0, k] = float(p[0])
        xr[1, k] = float(p[1])
        xr[2, k] = heading
        xr[3, k] = 0.0 if s_k >= total_len - 1e-6 else v_ref

    xr[:, 0] = current[:4]
    return xr


def _pure_pursuit_warm_start(
    path_points: np.ndarray,
    start_state: np.ndarray,
    cfg: NMPCConfig,
    N: int,
    nu: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate x_guess and u_guess via Pure Pursuit rollout for MPC warm start."""
    x0 = start_state.ravel()
    u_guess_vec = np.zeros((nu * N, 1), dtype=float)
    x0new = np.array([x0[0], x0[1], x0[2], x0[3]]).reshape(1, -1)
    warm_path_idx = 0
    ref_speed = float(getattr(cfg, "reference_speed", 0.2))
    omega_min = float(cfg.input_bounds.lower[0])
    omega_max = float(cfg.input_bounds.upper[0])
    accel_min = float(cfg.input_bounds.lower[1])
    accel_max = float(cfg.input_bounds.upper[1])
    x_0_list = [x0new.copy()]

    for i in range(N):
        sim_pos = x0new[0, :2]
        sim_yaw = x0new[0, 2]
        sim_v = x0new[0, 3]
        if path_points.shape[0] > 0:
            while warm_path_idx < path_points.shape[0] - 1:
                if np.linalg.norm(sim_pos - path_points[warm_path_idx]) < 0.05:
                    warm_path_idx += 1
                else:
                    break
            target_pt = path_points[warm_path_idx]
            dx = target_pt[0] - sim_pos[0]
            dy = target_pt[1] - sim_pos[1]
            target_yaw = math.atan2(dy, dx)
            yaw_diff = target_yaw - sim_yaw
            while yaw_diff > math.pi:
                yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi:
                yaw_diff += 2 * math.pi
            u_omega = np.clip(2.0 * yaw_diff, omega_min, omega_max)
            v_target = ref_speed
        else:
            u_omega = 0.0
            v_target = ref_speed
        u_accel = np.clip((v_target - sim_v) / dt, accel_min, accel_max)
        u_guess_vec[i * nu, 0] = u_omega
        u_guess_vec[i * nu + 1, 0] = u_accel
        x0new = np.array(
            [
                dt * sim_v * math.cos(sim_yaw) + sim_pos[0],
                dt * sim_v * math.sin(sim_yaw) + sim_pos[1],
                dt * u_omega + sim_yaw,
                dt * u_accel + sim_v,
            ],
            dtype=float,
        ).reshape(1, -1)
        x_0_list.append(x0new.copy())

    x_0 = np.vstack(x_0_list).T  # (4, N+1)
    return x_0, u_guess_vec


def _build_predicted_robot_obstacles(
    predicted_states: Sequence[np.ndarray],
    ego_index: int,
    robot_components_per_agent: Sequence[Sequence[RobotComponentGeometry]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create dynamic obstacle halfspaces from other robots' predicted next poses.

    Each other robot component polytope is added directly as an extra barrier.
    """
    obstacles: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx, state in enumerate(predicted_states):
        if idx == ego_index:
            continue
        if idx >= len(robot_components_per_agent):
            continue
        comp = robot_components_per_agent[idx]
        _, halfspaces_world = _robot_geometry_world(np.asarray(state, dtype=float), comp)
        for A_obs, b_obs in halfspaces_world:
            obstacles.append(
                (
                    np.asarray(A_obs, dtype=float),
                    np.asarray(b_obs, dtype=float).reshape(-1),
                )
            )
    return obstacles


def _state_hits_dynamic_obstacles(
    state_vec: np.ndarray,
    obstacle_halfspaces: Sequence[Tuple[np.ndarray, np.ndarray]],
    robot_components: Sequence[RobotComponentGeometry],
    penetration_tol: float,
) -> bool:
    """Return True when robot at `state_vec` intersects any dynamic obstacle."""
    if not obstacle_halfspaces:
        return False
    _, robot_halfspaces = _robot_geometry_world(np.asarray(state_vec, dtype=float), robot_components)
    tol = float(max(0.0, penetration_tol))
    if tol == 0.0:
        tol = 3e-3  # numerical epsilon when barrier_penetration_tol=0
    for A_obs, b_obs in obstacle_halfspaces:
        for A_robot, b_robot in robot_halfspaces:
            try:
                dist, _, _ = solve_min_distance_qp(
                    np.asarray(A_obs, dtype=float),
                    np.asarray(b_obs, dtype=float).reshape(-1),
                    np.asarray(A_robot, dtype=float),
                    np.asarray(b_robot, dtype=float).reshape(-1),
                )
                if float(dist) <= tol:
                    return True
            except Exception:
                # If the auxiliary distance QP fails, skip this pair and rely on the
                # static safety gate + next iteration update.
                continue
    return False


def _expand_state(vec: Sequence[float], default_theta: float = 0.0) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).ravel()
    result = np.zeros(4, dtype=float)
    length = min(arr.size, 4)
    result[:length] = arr[:length]
    if arr.size < 3:
        result[2] = default_theta
    if arr.size < 4:
        result[3] = 0.0
    return result

def impcdcbf(
    startpoint=None,
    endpoint=None,
    cbfbutton=1,
    barrierattr=None,
    config: NMPCConfig | None = None,
):
    cfg = config or DEFAULT_CONFIG

    # Ensure the shape-aware half-space model used by OSQP matches this run's config.
    init_model(cfg)

    N = cfg.horizon
    K1 = cfg.warm_start_refine_iterations
    dt = cfg.dt
    nsim = cfg.nsim
    approx_radius = cfg.approx_radius

    # Constraints
    umin = cfg.input_bounds.lower.copy()
    umax = cfg.input_bounds.upper.copy()
    xmin = cfg.state_bounds.lower.copy()
    xmax = cfg.state_bounds.upper.copy()

    # Objective function
    Q = cfg.Q()
    QN = cfg.QN()
    if barrierattr is None:
        barrierattr = cfg.R_diagonal.copy()
    else:
        barrierattr = np.asarray(barrierattr, dtype=float)
    R = np.diag(barrierattr)
    
    env = ENVIRONMENT

    env_start_state = _expand_state(env.start, default_theta=0.0)
    env_goal_state = _expand_state(env.goal, default_theta=0.0)

    if startpoint is None:
        start_state = env_start_state
    else:
        start_state = _expand_state(startpoint)
    if endpoint is None:
        goal_state = env_goal_state
    else:
        goal_state = _expand_state(endpoint)

    # Initial and reference states
    x0 = start_state.reshape(-1, 1)
    ur = cfg.reference_input.reshape(-1, 1)

    # Dynamic system initialization
    BB = np.array([[0, 0, 0, 0,0, 0], [0, 0, 0, 0,0, 0], [dt, 0, 0, 0,0, 0], [0, dt, 0, 0,0, 0]])

    # Convex MPC frame
    nx, nu = BB.shape
    x0 = x0.T
    u_0 = np.zeros((nu, N))
    u0 = np.zeros((nu, 1))
    impc = np.zeros((nx + 2, nsim + 1))
    impc[:nx, 0] = x0
    # Note: we intentionally do not smooth/low-pass filter controls here.

    interactive_backend = "agg" not in plt.get_backend().lower()
    if interactive_backend:
        plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plotmap(ax)
    start_xy = start_state[:2]
    goal_xy = goal_state[:2]

    try:
        # Global path: grid-based A* with line-of-sight reduction (Thirugnanam et al., ICRA 2022).
        path_points = plan_path(
            start_xy,
            goal_xy,
            OBSTACLE_REGIONS,
            horizon=60,
            margin=0.05,
            bounds=ENVIRONMENT.bounds,
            cell_size=ENVIRONMENT.grid[1],
            use_grid=True,
            method="astar_los",
            quad=False,
        )

        print(f"[guiding path] Generated {path_points.shape[0]} waypoints:")
        for idx, waypoint in enumerate(path_points):
            print(f"  wp[{idx:02d}] = ({waypoint[0]:.4f}, {waypoint[1]:.4f})")
    except Exception as exc:
        # Do not silently fall back to a straight-line path: in cluttered environments this is unsafe
        # and hides planner failures. Raise explicitly so the caller can handle it.
        raise RuntimeError(f"Global path planner failed: {exc}") from exc
    path_points = np.asarray(path_points, dtype=float)

    # Keep the global guiding path as produced by the planner (grid centers + LoS reduction).

    (x_min, y_min), (x_max, y_max) = ENVIRONMENT.bounds
    padding = 0.05 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    ax.set_title('iMPC Trajectory and Probing Paths')
    ax.scatter(start_xy[0], start_xy[1], color='green', s=60, marker='o', edgecolors='black', label='Start')
    ax.scatter(goal_xy[0], goal_xy[1], color='red', s=80, marker='*', label='Goal')
    # Plot styling:
    # - Global path: grey dashed with small markers (static)
    # - Local reference window: orange (moves forward)
    guiding_line, = ax.plot(
        path_points[:, 0],
        path_points[:, 1],
        "o--",
        color="grey",
        linewidth=1.5,
        markersize=2,
        label="Global path",
    )
    reference_line, = ax.plot([], [], color="orange", linewidth=3.0, label="Reference window")
    reference_point_marker = ax.scatter(
        [], [], color="blue", s=30, marker=".", zorder=5,
        label="Reference points",
    )
    traj_line, = ax.plot([], [], color='dodgerblue', linewidth=2.0, label='Executed trajectory')
    probe_line, = ax.plot([], [], color='blue', linewidth=1.5, linestyle='--', label='Probing horizon')
    obstacle_contact_scatter = ax.scatter([], [], color='black', s=25, label='Obstacle contacts')
    robot_contact_scatter = ax.scatter([], [], color='magenta', s=25, label='Robot contacts')
    current_marker = ax.scatter([], [], color='cyan', s=35, label='Current state')
    wp_marker = ax.scatter([], [], color='pink', s=70, marker='X', edgecolors='black', linewidths=1.0, label='Guiding point')
    # Robot footprint patches (1 for rectangle/triangle, 2 for lshape).
    robot_components = build_robot_geometry(cfg)
    robot_patches: List[Polygon] = []
    for idx, comp in enumerate(robot_components):
        patch = Polygon(
            np.zeros((comp.vertices_local.shape[0], 2)),
            closed=True,
            fill=False,
            edgecolor="purple",
            linewidth=1.3,
            alpha=0.9,
            label="Robot footprint" if idx == 0 else "_nolegend_",
        )
        ax.add_patch(patch)
        robot_patches.append(patch)
    ax.legend(loc='upper right')

    # ------------------------------------------------------------------
    # Video recording
    # - Use a streaming writer to avoid storing hundreds of full-size frames in RAM.
    # - Fall back to in-memory frames if ffmpeg writer init fails.
    # ------------------------------------------------------------------
    video_dir = Path(".")
    video_dir.mkdir(exist_ok=True)
    video_path = video_dir / "impc_run.mp4"
    video_writer = None
    frames: List[np.ndarray] = []
    try:
        video_writer = imageio.get_writer(video_path, fps=10)
    except Exception as exc:
        print(f"[video] Failed to initialize writer ({exc}); falling back to in-memory frames.")

    def record_frame():
        leg = ax.get_legend()
        if leg is not None:
            leg.set_visible(False)
        title = ax.get_title()
        ax.set_title("")
        frame = _pad_frame_to_macroblock(_capture_frame(fig))
        if leg is not None:
            leg.set_visible(True)
        ax.set_title(title)
        if video_writer is not None:
            video_writer.append_data(frame)
        else:
            frames.append(frame)

    waypoint_tolerance = 0.1 # Increased tolerance for larger dt
    path_idx = 0
    last_wp_dist = float("inf")
    stall_steps = 0

    # ------------------------------------------------------------------
    # Moving-window reference guidance (Sec. V-A3):
    # - global path is a discrete grid path
    # - local reference is a constant-speed window pushed forward by projection + buffer
    # ------------------------------------------------------------------
    ref_path_index = 0
    ref_speed = float(getattr(cfg, "reference_speed", 0.2))
    proj_dist_buffer = float(getattr(cfg, "reference_proj_dist_buffer", 0.05))
    local_path_timestep = float(dt)
    num_horizon = int(N + 1)
    progress_s = 0.0
    last_progress_s = 0.0

    if path_points.shape[0] >= 2:
        _path_vecs = path_points[1:] - path_points[:-1]
        _path_lens = np.linalg.norm(_path_vecs, axis=1)
        _path_s = np.hstack([0.0, np.cumsum(_path_lens)])
    else:
        _path_lens = np.zeros(0, dtype=float)
        _path_s = np.zeros(path_points.shape[0], dtype=float)

    def reference_trajectory(
        pos_xy: np.ndarray,
        heading_cur: float | None = None,
        speed_scale: float = 1.0,
        commit: bool = True,
    ) -> np.ndarray:
        """
        Reference/cbf ConstantSpeedTrajectoryGenerator-style local trajectory.
        Returns (4, N+1): [x, y, heading, v].
        """
        nonlocal ref_path_index, progress_s
        xr_traj = np.zeros((4, num_horizon), dtype=float)
        if path_points.size == 0 or path_points.shape[0] == 0:
            xr_traj[0, :] = float(pos_xy[0])
            xr_traj[1, :] = float(pos_xy[1])
            xr_traj[2, :] = float(goal_state[2])
            xr_traj[3, :] = 0.0
            return xr_traj

        pos = np.asarray(pos_xy, dtype=float).reshape(2)
        heading_now = float(heading_cur) if heading_cur is not None else None
        # Use a local index for hypothetical references (commit=False) so inner-loop homotopy
        # doesn't advance the global reference window index.
        idx = int(np.clip(ref_path_index, 0, path_points.shape[0] - 1))

        # Reference/cbf-style moving window:
        # - Only advance the window forward when we've essentially reached the end of the current
        #   segment (projected distance near segment length).
        # - Never move the index backward. This avoids index flip-flops near corners, which can
        #   create a forward/reverse limit-cycle when reverse is enabled.
        #
        # NOTE: we make the buffer *relative* to segment length so short segments (often produced
        # by LoS reduction near obstacles) don't advance the index too early.
        proj_dist = 0.0
        while True:
            trunc_path = np.vstack([path_points[idx:, :], path_points[-1, :]])
            curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
            curv_length = np.linalg.norm(curv_vec, axis=1)
            if curv_length.size == 0:
                curv_length = np.array([0.0], dtype=float)
                curv_vec = np.zeros((1, 2), dtype=float)
            if curv_length[0] <= 1e-12:
                curv_direct = np.zeros((2,), dtype=float)
            else:
                curv_direct = curv_vec[0, :] / curv_length[0]

            proj_dist = float(np.dot(pos - trunc_path[0, :], curv_direct))
            if proj_dist < 0.0:
                proj_dist = 0.0

            seg_len0 = float(curv_length[0])
            # Do not advance too early on short segments.
            eff_buf = float(min(float(proj_dist_buffer), 0.25 * seg_len0)) if seg_len0 > 0.0 else 0.0
            if proj_dist >= float(seg_len0 - eff_buf) and idx < path_points.shape[0] - 1:
                idx += 1
                continue
            break

        trunc_path = np.vstack([path_points[idx:, :], path_points[-1, :]])
        curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
        curv_length = np.linalg.norm(curv_vec, axis=1)
        if curv_length.size == 0:
            curv_length = np.array([0.0], dtype=float)
            curv_vec = np.zeros((1, 2), dtype=float)
        if curv_length[0] <= 1e-12:
            curv_direct = np.zeros((2,), dtype=float)
        else:
            curv_direct = curv_vec[0, :] / curv_length[0]

        speed_scale = float(speed_scale)

        # Reverse reference (straight backing) to let OSQP discover backing maneuvers.
        # This does NOT constrain the actual speed; it only changes the reference the QP tracks.
        if speed_scale < -1e-3:
            v_ref = -max(1e-6, float(ref_speed) * abs(speed_scale))
            # Backing translation direction: along the *path tangent*. With v_ref < 0 this moves
            # backward along the global path, which tends to increase clearance in corridors.
            if float(np.linalg.norm(curv_direct)) > 1e-9:
                direction = curv_direct
                hd_ref = float(np.arctan2(curv_direct[1], curv_direct[0]))
            else:
                hd_ref = float(goal_state[2])
                direction = np.array([math.cos(hd_ref), math.sin(hd_ref)], dtype=float)

            for kk in range(num_horizon):
                t = local_path_timestep * float(kk)
                xr_traj[0, kk] = float(pos[0] + v_ref * t * direction[0])
                xr_traj[1, kk] = float(pos[1] + v_ref * t * direction[1])
                xr_traj[2, kk] = hd_ref
                xr_traj[3, kk] = v_ref
            if commit:
                ref_path_index = idx
                progress_s = float(_path_s[idx] + proj_dist) if _path_s.size > 0 and idx < _path_s.size else 0.0
            return xr_traj

        if abs(speed_scale) <= 1e-3:
            # "Stop" homotopy: keep the reference fixed at the current position.
            xr_traj[0, :] = float(pos[0])
            xr_traj[1, :] = float(pos[1])
            # IMPORTANT: when asking the solver to "stop", do not force an in-place rotation
            # toward the path direction. For non-circular footprints (e.g., L-shape), rotating
            # in place can itself collide in tight spaces. Keeping the heading at the current
            # value lets OSQP brake to v≈0 first; subsequent steps can re-orient if feasible.
            if heading_now is not None:
                hd = float(heading_now)
            else:
                # Fallback to path direction (or goal heading if degenerate).
                if curv_vec.shape[0] > 0 and float(np.linalg.norm(curv_vec[0])) > 1e-9:
                    hd = float(np.arctan2(curv_vec[0, 1], curv_vec[0, 0]))
                else:
                    hd = float(goal_state[2])
            xr_traj[2, :] = hd
            xr_traj[3, :] = 0.0
            if commit:
                ref_path_index = idx
                progress_s = float(_path_s[idx] + proj_dist) if _path_s.size > 0 and idx < _path_s.size else 0.0
            return xr_traj

        # --------------------------------------------------------------
        # Yaw-rate-aware reference speed (lets stop/backing emerge from OSQP)
        #
        # Our dynamics use a direct yaw-rate control with bounds ω ∈ [ωmin, ωmax].
        # To follow a curved path, a sufficient condition is:
        #   |θ_dot| = |dθ/ds| * v  <= ω_max   =>   v <= ω_max / κ
        # where κ ≈ |Δθ| / Δs from the upcoming path segments.
        #
        # By reducing the *reference* speed in high-curvature regions, the solver can
        # choose to brake/stop (and even reverse if beneficial) without fighting a
        # "must move forward at v_ref" target. This is deterministic and derived from
        # constraints (not a heuristic recovery mode).
        # --------------------------------------------------------------
        omega_max = float(max(abs(float(cfg.input_bounds.lower[0])), abs(float(cfg.input_bounds.upper[0]))))
        # Estimate curvature from the next few segment headings.
        seg_head = np.arctan2(curv_vec[:, 1], curv_vec[:, 0]).reshape(-1)
        if seg_head.size >= 2:
            dtheta = seg_head[1:] - seg_head[:-1]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi,pi]
            ds = 0.5 * (curv_length[1:] + curv_length[:-1])
            ds = np.maximum(ds, 1e-6)
            kappa = np.abs(dtheta) / ds
            # Focus on near-term curvature (first few turns dominate feasibility).
            max_kappa = float(np.max(kappa[: min(5, kappa.size)])) if kappa.size else 0.0
        else:
            max_kappa = 0.0

        v_nom = float(ref_speed) * float(speed_scale)
        if max_kappa > 1e-6 and omega_max > 0.0:
            v_limit = float(omega_max) / float(max_kappa)
            v_ref = max(1e-6, min(v_nom, v_limit))
            # If the bound is extremely restrictive, allow a "stop" reference.
            if v_ref < 0.05:
                v_ref = 1e-6
        else:
            v_ref = max(1e-6, v_nom)
        t_c = (proj_dist + proj_dist_buffer) / v_ref
        t_s = t_c + local_path_timestep * np.linspace(0, num_horizon - 1, num_horizon)

        curv_time = np.cumsum(np.hstack([0.0, curv_length / v_ref]))
        curv_time[-1] += t_c + 2 * local_path_timestep * num_horizon + proj_dist_buffer / v_ref

        path_idx = np.searchsorted(curv_time, t_s, side="right") - 1
        path_idx = np.clip(path_idx, 0, curv_vec.shape[0] - 1)

        path = np.vstack(
            [
                np.interp(t_s, curv_time, trunc_path[:, 0]),
                np.interp(t_s, curv_time, trunc_path[:, 1]),
            ]
        ).T
        heading = np.arctan2(curv_vec[path_idx, 1], curv_vec[path_idx, 0]).reshape(-1)

        xr_traj[0, :] = path[:, 0]
        xr_traj[1, :] = path[:, 1]
        xr_traj[2, :] = heading
        xr_traj[3, :] = v_ref

        # Progress metric (for stall detection): arclength up to idx + proj_dist.
        if _path_s.size > 0 and idx < _path_s.size and commit:
            ref_path_index = idx
            progress_s = float(_path_s[idx] + proj_dist)
        return xr_traj

    xr = reference_trajectory(np.asarray(start_xy, dtype=float), heading_cur=float(start_state[2]))
    reference_line.set_data(xr[0, 1:], xr[1, 1:])  # exclude robot center
    ref_pts_xy = np.column_stack([xr[0, 1:], xr[1, 1:]])
    reference_point_marker.set_offsets(ref_pts_xy)
    wp_marker.set_offsets(np.array([[xr[0, -1], xr[1, -1]]]))

    def compute_robot_polygons(state_vec: np.ndarray) -> List[np.ndarray]:
        polys_world, _ = _robot_geometry_world(np.asarray(state_vec, dtype=float), robot_components)
        return polys_world

    def update_contacts(
        solution: Optional[DualDistanceSolution],
        state_traj: np.ndarray,
        label: str = "",
        iteration: int = 0,
    ) -> None:
        if solution is None or solution.y_obs.size == 0:
            obstacle_contact_scatter.set_offsets(np.zeros((0, 2)))
            robot_contact_scatter.set_offsets(np.zeros((0, 2)))
            return

        obstacle_points = []
        robot_points = []
        n_components = int(solution.y_obs.shape[1])
        horizon = int(solution.y_obs.shape[2])
        # Display contacts at the first predicted step (x1) for a stable visual.
        step_idx = 0 if horizon > 0 else 0
        state_col = min(step_idx + 1, state_traj.shape[1] - 1)
        pred_state = state_traj[:, state_col]
        robot_polys_world, robot_halfspaces_world = _robot_geometry_world(
            np.asarray(pred_state, dtype=float), robot_components
        )
        for obs_idx, (A_obs, b_obs) in enumerate(_OBSTACLE_HALFSPACES):
            # Pick the closest component at this step for plotting.
            if n_components > 1:
                d2 = np.sum(
                    (solution.y_obs[obs_idx, :, step_idx, :] - solution.y_robot[obs_idx, :, step_idx, :]) ** 2,
                    axis=1,
                )
                comp_idx = int(np.argmin(d2))
            else:
                comp_idx = 0

            solver_obs = solution.y_obs[obs_idx, comp_idx, step_idx, :]
            solver_robot = solution.y_robot[obs_idx, comp_idx, step_idx, :]
            obs_disp = np.asarray(solver_obs, dtype=float).reshape(2)
            robot_disp = np.asarray(solver_robot, dtype=float).reshape(2)
            A_robot_pred, b_robot_pred = robot_halfspaces_world[min(comp_idx, len(robot_halfspaces_world) - 1)]
            obstacle_points.append(obs_disp)
            robot_points.append(robot_disp)

            solver_obs_res = (A_obs @ solver_obs - b_obs).max()
            solver_robot_res = (A_robot_pred @ solver_robot - b_robot_pred).max()
            display_obs_res = (A_obs @ obs_disp - b_obs).max()
            display_robot_res = (A_robot_pred @ robot_disp - b_robot_pred).max()
            obs_error = np.linalg.norm(obs_disp - solver_obs)
            robot_error = np.linalg.norm(robot_disp - solver_robot)

            if _VERBOSE:
                print(
                    f"[contacts][{label} iter={iteration}] obstacle={obs_idx} "
                    f"display_obs={obs_disp} display_robot={robot_disp} "
                    f"display_res(obs)={display_obs_res:.3e} "
                    f"display_res(robot)={display_robot_res:.3e} "
                    f"display_err(obs)={obs_error:.3e} display_err(robot)={robot_error:.3e} | "
                    f"solver_step={step_idx} solver_obs={solver_obs} "
                    f"solver_robot={solver_robot} "
                    f"solver_res(obs)={solver_obs_res:.3e} "
                    f"solver_res(robot)={solver_robot_res:.3e}"
                )

        obstacle_contact_scatter.set_offsets(np.vstack(obstacle_points))
        robot_contact_scatter.set_offsets(np.vstack(robot_points))

    init_state = np.asarray(x0).ravel()
    current_marker.set_offsets(np.array([[init_state[0], init_state[1]]]))
    traj_line.set_data([init_state[0]], [init_state[1]])

    # Initialize u_0 and x_0 using a simple Pure Pursuit-like controller
    # to generate a collision-free warm start trajectory that follows the path.
    x0new = x0.copy()
    x_0 = x0new.copy()
    
    # Track local path index for the warm start simulation
    warm_path_idx = 0
    if path_points.shape[0] > 0:
         # Start tracking the first global waypoint.
         warm_path_idx = 0

    for i in range(N + 1):
        # Current simulated state
        sim_pos = x0new[0, :2]
        sim_yaw = x0new[0, 2]
        sim_v = x0new[0, 3]
        
        # Advance waypoint index if close
        if path_points.shape[0] > 0:
            while warm_path_idx < path_points.shape[0] - 1:
                dist = np.linalg.norm(sim_pos - path_points[warm_path_idx])
                if dist < 0.05: # Lookahead/switch distance (reduced for tight tracking)
                     warm_path_idx += 1
                else:
                     break
            target_pt = path_points[warm_path_idx]
            
            # Compute control inputs
            dx = target_pt[0] - sim_pos[0]
            dy = target_pt[1] - sim_pos[1]
            target_yaw = math.atan2(dy, dx)
            
            yaw_diff = target_yaw - sim_yaw
            while yaw_diff > math.pi: yaw_diff -= 2*math.pi
            while yaw_diff < -math.pi: yaw_diff += 2*math.pi
            
            # P-control for yaw rate (gain 2.0 for faster turn)
            # Use configured yaw-rate bounds so warm-start has the same turning authority.
            omega_min = float(cfg.input_bounds.lower[0])
            omega_max = float(cfg.input_bounds.upper[0])
            u_omega = np.clip(2.0 * yaw_diff, omega_min, omega_max)
        else:
            u_omega = 0.0

        # P-control for velocity (use local reference speed to avoid overshooting sharp turns)
        v_target = float(xr[3, min(i, xr.shape[1] - 1)]) if xr.ndim == 2 else float(cfg.reference_speed)
        accel_min = float(cfg.input_bounds.lower[1])
        accel_max = float(cfg.input_bounds.upper[1])
        u_accel = np.clip((v_target - sim_v) / dt, accel_min, accel_max)
        
        # Store in u_0
        if i < N:
            u_0[0, i] = u_omega
            u_0[1, i] = u_accel
        
        # Dynamics update
        x0new = np.array([
            dt * sim_v * np.cos(sim_yaw) + sim_pos[0],
            dt * sim_v * np.sin(sim_yaw) + sim_pos[1],
            dt * u_omega + sim_yaw,
            dt * u_accel + sim_v
        ]).reshape(-1, 1).T
        
        x_0 = np.vstack((x_0, x0new))
    x_0 = x_0[1:, :]
    x_0 = x_0.T
    
    dual_solution: Optional[DualDistanceSolution] = None
    if cbfbutton == 1:
        ctrlx, ctrlu, dual_solution = lineardyn(
            ur,
            xr,
            x_0,
            x0,
            Q,
            QN,
            R,
            dt,
            N,
            nx,
            nu,
            BB,
            u_0,
            xmin,
            xmax,
            umin,
            umax,
            cfg=cfg,
            previous_solution=dual_solution,
        )
    else:
        ctrlx, ctrlu, dual_solution = lineardyn2(
            ur,
            xr,
            x_0,
            x0,
            Q,
            QN,
            R,
            dt,
            N,
            nx,
            nu,
            BB,
            u_0,
            xmin,
            xmax,
            umin,
            umax,
            cfg=cfg,
            previous_solution=dual_solution,
        )

    state_traj = trans(x0, ctrlx, nx, N + 1)
    update_contacts(dual_solution, state_traj, label="warm", iteration=0)
    for patch, poly in zip(robot_patches, compute_robot_polygons(init_state)):
        patch.set_xy(poly)
    record_frame()

    for j in range(1, K1):
        storagex = ctrlx;
        storageu = ctrlu;
        
        # Update states
        x_0 = trans(x0, ctrlx, nx, N + 1)
        u_0 = transu(ctrlu, nu, N)

        if cbfbutton == 1:
            ctrlx, ctrlu, dual_solution = lineardyn(
                ur,
                xr,
                x_0,
                x0,
                Q,
                QN,
                R,
                dt,
                N,
                nx,
                nu,
                BB,
                u_0,
                xmin,
                xmax,
                umin,
                umax,
                cfg=cfg,
                previous_solution=dual_solution,
            )
        else:
            ctrlx, ctrlu, dual_solution = lineardyn2(
                ur,
                xr,
                x_0,
                x0,
                Q,
                QN,
                R,
                dt,
                N,
                nx,
                nu,
                BB,
                u_0,
                xmin,
                xmax,
                umin,
                umax,
                cfg=cfg,
                previous_solution=dual_solution,
            )

        state_traj = trans(x0, ctrlx, nx, N + 1)
        current_state = state_traj[:, 0]
        for patch, poly in zip(robot_patches, compute_robot_polygons(current_state)):
            patch.set_xy(poly)
        update_contacts(dual_solution, state_traj, label="inner", iteration=j)
           
        testx = (storagex - ctrlx).T @ (storagex - ctrlx)
        testu = (storageu - ctrlu).T @ (storageu - ctrlu)
        test = testx / (storagex.T @ storagex)
        
        if test**0.5 <= 10**-2 and (testx / ((N + 1) * nx))**0.5 <= 10**-2:
            break
    
    ctrl = ctrlu[:nu]
    x0 = np.array([dt * x0[0, 3] * np.cos(x0[0, 2]) + x0[0, 0],
               dt * x0[0, 3] * np.sin(x0[0, 2]) + x0[0, 1],
               dt * ctrl[0, 0] + x0[0, 2],
               dt * ctrl[1, 0] + x0[0, 3]])
    ctrlu = rewrite(ctrlu, nu, N)
    x_0 = newinit(x0, ctrlu, nx, nu, N, dt)
    u_0 = transu(ctrlu, nu, N)
    impc[:nx, 1] = x0
    # Log the *applied* control (yaw-rate, accel), not the warm-start guess.
    impc[nx:, 1] = ctrl[:2, 0]
    end_idx = 0
    storagex = ctrlx.copy()
    storageu = ctrlu.copy()

    # Main optimization loop
    x_pred_all = np.zeros((0, 4 * N))
    failure_message = None
    for i in range(1, nsim):
        x0 = x0.reshape(-1, 1)
        x0 = x0.T
        # Keep a fixed copy of the *true* current state for this outer step.
        # The QP decision vector also contains an x0 variable, but we must not let it
        # overwrite the simulator state.
        x0_current = x0.copy()
        
        if i > 20:
            current_pos_flat = x0.reshape(-1)[:2]
            _, _, max_dev = _polyline_deviation(current_pos_flat.reshape(1, 2), path_points)
            if max_dev > 0.25:
                print(f"[replan] Large deviation {max_dev:.4f} > 0.25. Replanning...")
                try:
                    path_points = plan_path(
                        current_pos_flat,
                        goal_xy,
                        OBSTACLE_REGIONS,
                        horizon=60,
                        margin=0.05,
                        bounds=ENVIRONMENT.bounds,
                        cell_size=ENVIRONMENT.grid[1],
                        use_grid=True # Prefer grid search for replanning to ensure safety
                    )
                    path_idx = 0
                    # Reset moving-window reference state for new path
                    ref_path_index = 0
                    progress_s = 0.0
                    last_progress_s = 0.0
                    guiding_line.set_data(path_points[:, 0], path_points[:, 1])
                    xr = reference_trajectory(current_pos_flat, heading_cur=float(x0_current.reshape(-1)[2]))
                    reference_line.set_data(xr[0, 1:], xr[1, 1:])
                    reference_point_marker.set_offsets(np.column_stack([xr[0, 1:], xr[1, 1:]]))
                    wp_marker.set_offsets(np.array([[xr[0, -1], xr[1, -1]]]))
                    print(f"[replan] New path has {path_points.shape[0]} waypoints.")
                except Exception as exc:
                    print(f"[replan] Failed: {exc}")

        # Stall detector: if we are not making progress along the path for a while, optionally replan.
        # This helps recover from tight corners where the optimizer chooses near-zero motion.
        if cfg.enable_stall_replan and i > 5:
            # progress_s is maintained by reference_trajectory()
            prog_delta = abs(float(progress_s) - float(last_progress_s))
            if prog_delta < 1e-4:
                stall_steps += 1
            else:
                stall_steps = 0
            last_progress_s = float(progress_s)

            if stall_steps >= 30:
                print(f"[replan] Stalled for {stall_steps} steps (progress_s delta<{1e-4}). Replanning...")
                try:
                    current_pos_flat = x0.reshape(-1)[:2]
                    path_points = plan_path(
                        current_pos_flat,
                        goal_xy,
                        OBSTACLE_REGIONS,
                        horizon=60,
                        margin=0.1,
                        bounds=ENVIRONMENT.bounds,
                        cell_size=ENVIRONMENT.grid[1],
                        use_grid=True,
                    )
                    # reset reference window + waypoint index
                    ref_path_index = 0
                    stall_steps = 0
                    progress_s = 0.0
                    last_progress_s = 0.0
                    path_idx = 0
                    guiding_line.set_data(path_points[:, 0], path_points[:, 1])
                    xr = reference_trajectory(current_pos_flat, heading_cur=float(x0_current.reshape(-1)[2]))
                    reference_line.set_data(xr[0, 1:], xr[1, 1:])
                    reference_point_marker.set_offsets(np.column_stack([xr[0, 1:], xr[1, 1:]]))
                    wp_marker.set_offsets(np.array([[xr[0, -1], xr[1, -1]]]))
                    print(f"[replan] New path has {path_points.shape[0]} waypoints.")
                except Exception as exc:
                    print(f"[replan] Stall replan failed: {exc}")

        print(f"Processing {i}/{nsim-1}")
        start_time = time.time()
        
        opt_results = []
        condition_met = False
        max_step_iters = int(getattr(cfg, "max_step_iterations", 30))
        max_step_iters = max(1, min(max_step_iters, 100))
        found_viable = False
        for j in range(max_step_iters):
            # Deterministic continuation on reference speed: re-solve OSQP with a slower reference
            # window if the nominal solution collides. Executed controls still come from OSQP;
            # this only changes the reference the solver tracks so it can brake/stop/reverse.
            # Schedule order matters: if we are already in a tight turn where even 0.25× nominal
            # forward reference is infeasible, trying a tiny "creep forward" reference (0.1×) can
            # lead to a forward/reverse limit-cycle. Prefer attempting a reverse reference first,
            # then fall back to creep/stop only if reverse is also infeasible.
            if j < 10:
                speed_scale = 1.0
            elif j < 20:
                speed_scale = 0.5
            elif j < 30:
                speed_scale = 0.25
            elif j < 40:
                speed_scale = -0.5
            elif j < 50:
                speed_scale = 0.1
            else:
                speed_scale = 0.0
            xr_iter = reference_trajectory(
                x0_current.reshape(-1)[:2],
                heading_cur=float(x0_current.reshape(-1)[2]),
                speed_scale=speed_scale,
                commit=False,
            )
            if cbfbutton == 1:
                ctrlx, ctrlu, dual_solution = lineardyn(
                    ur,
                    xr_iter,
                    x_0,
                    x0,
                    Q,
                    QN,
                    R,
                    dt,
                    N,
                    nx,
                    nu,
                    BB,
                    u_0,
                    xmin,
                    xmax,
                    umin,
                    umax,
                    cfg=cfg,
                    previous_solution=dual_solution,
                )
            else:
                ctrlx, ctrlu, dual_solution = lineardyn2(
                    ur,
                    xr_iter,
                    x_0,
                    x0,
                    Q,
                    QN,
                    R,
                    dt,
                    N,
                    nx,
                    nu,
                    BB,
                    u_0,
                    xmin,
                    xmax,
                    umin,
                    umax,
                    cfg=cfg,
                    previous_solution=dual_solution,
                )
            
            # Reconstruct the candidate state trajectory using the *true* current state as
            # the first column. Do not overwrite x0_current.
            x_0 = trans(x0_current, ctrlx, nx, N + 1)
            u_0 = transu(ctrlu, nu, N)
            # Roll out dynamics from the current state using the candidate control sequence.
            x_rollout = newinit(np.asarray(x0_current).reshape(-1), ctrlu, nx, nu, N, dt)
            testx = (storagex - ctrlx).T @ (storagex - ctrlx)
            testu = (storageu - ctrlu).T @ (storageu - ctrlu)
            test = testx / (storagex.T @ storagex)
            
            # Check probing horizon collision using a dynamics rollout (not ctrlx).
            # We require the *entire* horizon to be barrier-free before accepting a candidate,
            # and we visualize the full horizon probing line accordingly.
            if_in_barrier = False
            for k in range(1, N + 1):
                if in_barrier(
                    x_rollout[:, k].flatten(),
                    robot_components=robot_components,
                    cfg=cfg,
                ):
                    if_in_barrier = True
                    break

            # Viable = barrier-free probing horizon + next executed state is barrier-free.
            # If we find one, stop iterating immediately and proceed to the next outer step.
            if not if_in_barrier:
                ctrl_tmp = ctrlu[:nu]
                x1_candidate = np.array(
                    [
                        dt * x0_current[0, 3] * np.cos(x0_current[0, 2]) + x0_current[0, 0],
                        dt * x0_current[0, 3] * np.sin(x0_current[0, 2]) + x0_current[0, 1],
                        dt * ctrl_tmp[0, 0] + x0_current[0, 2],
                        dt * ctrl_tmp[1, 0] + x0_current[0, 3],
                    ],
                    dtype=float,
                )
                if not in_barrier(
                    x1_candidate,
                    robot_components=robot_components,
                    cfg=cfg,
                ):
                    found_viable = True
                    mode = "reverse" if speed_scale < 0.0 else ("stop" if abs(speed_scale) <= 1e-6 else "forward")
                    print(
                        f"[inner-opt] viable candidate at iter {j} "
                        f"(speed_scale={float(speed_scale):.3f}, mode={mode}); proceeding to next step."
                    )
                    # Mark as condition_met so selection prefers it if we fall through (shouldn't).
                    condition_met = True
                    opt_results.append(
                        (
                            ctrlu.copy(),
                            ctrlx.copy(),
                            dual_solution,
                            False,
                            True,
                            j,
                        )
                    )
                    break
                   
            if test**0.5 <= 10**-2 and (testx / ((N + 1) * nx))**0.5 <= 10**-4 and not if_in_barrier:
                condition_met = True
                print(
                    f"[inner-opt] converged iteration {j}, if_in_barrier={if_in_barrier}, "
                    f"test={float(np.sqrt(test)):.3e}, testx={float(testx):.3e}"
                )
                opt_results.append(
                    (
                        ctrlu.copy(),
                        ctrlx.copy(),
                        dual_solution,
                        if_in_barrier,
                        condition_met,
                        j,
                    )
                )
                break
                
            opt_results.append(
                (
                    ctrlu.copy(),
                    ctrlx.copy(),
                    dual_solution,
                    if_in_barrier,
                    condition_met,
                    j,
                )
            )
            print(
                f"[inner-opt] stored iteration {j}, if_in_barrier={if_in_barrier}, "
                f"condition_met={condition_met}"
            )
            
            storagex = ctrlx.copy()
            storageu = ctrlu.copy()
        
        # Selection rule:
        # - Prefer a converged candidate with a barrier-free probing horizon.
        # - Else any barrier-free candidate.
        if_found = False
        for opt_result in reversed(opt_results):
            # tuple = (ctrlu, ctrlx, dual_solution, if_in_barrier, condition_met, j)
            if (not opt_result[3]) and bool(opt_result[4]):
                ctrlu = opt_result[0]
                ctrlx = opt_result[1]
                dual_solution = opt_result[2]
                if_found = True
                break
        if not if_found:
            for opt_result in reversed(opt_results):
                if not opt_result[3]:
                    ctrlu = opt_result[0]
                    ctrlx = opt_result[1]
                    dual_solution = opt_result[2]
                    if_found = True
                    break

        # If we couldn't find any barrier-free candidate:
        # - by default we do NOT execute an in-barrier "least violating" solution (safety-first).
        # - optionally (cfg.allow_in_barrier_fallback=True) we pick the least violating.
        if not if_found and opt_results and getattr(cfg, "allow_in_barrier_fallback", False):
            # Prefer the solution with the minimum constraint violation
            best_opt_result = None
            min_violation = float("inf")
            
            for opt_result in opt_results:
                ctrlx_cand = opt_result[1]
                
                # Evaluate max violation for this candidate
                max_violation = 0.0
                x_traj_cand = trans(x0_current, ctrlx_cand, nx, N + 1)
                for k in range(1, N + 1):
                    current_state_flat = x_traj_cand[:, k].flatten()
                    current_heading = float(current_state_flat[2])
                    current_pos = current_state_flat[:2]
                    polys_world, _ = _robot_geometry_world(
                        current_state_flat, robot_components
                    )
                    
                    for A_obs, b_obs in _OBSTACLE_HALFSPACES:
                        for poly_robot in polys_world:
                            for vertex in poly_robot:
                                margin = A_obs @ vertex - b_obs
                                max_margin = float(margin.max())
                                max_violation = max(max_violation, max(0.0, -max_margin))

                if max_violation < min_violation:
                    min_violation = max_violation
                    best_opt_result = opt_result

            if best_opt_result is not None:
                print(
                    f"[inner-opt] Warning: No valid safe solution found at step {i}. "
                    f"Using least violating solution (max viol={min_violation:.6f})."
                )
                opt_result = best_opt_result
                ctrlu = opt_result[0]
                ctrlx = opt_result[1]
                dual_solution = opt_result[2]
                if_found = True
            else:
                print(
                    f"[inner-opt] Warning: No valid safe solution found at step {i}. "
                    f"In-barrier fallback enabled but no candidate evaluated; holding."
                )

        if not if_found and opt_results and not getattr(cfg, "allow_in_barrier_fallback", False):
            print(
                f"[inner-opt] No barrier-free candidate at step {i}; "
                f"in-barrier fallback disabled => holding."
            )

        # Final safety gate: even if we selected a candidate, do not execute it if the *next executed*
        # state would be in-barrier (OSQP can be 'solved inaccurate' and slightly violate dynamics).
        if if_found:
            ctrl_tmp = ctrlu[:nu]
            x1_candidate = np.array(
                [
                    dt * x0_current[0, 3] * np.cos(x0_current[0, 2]) + x0_current[0, 0],
                    dt * x0_current[0, 3] * np.sin(x0_current[0, 2]) + x0_current[0, 1],
                    dt * ctrl_tmp[0, 0] + x0_current[0, 2],
                    dt * ctrl_tmp[1, 0] + x0_current[0, 3],
                ],
                dtype=float,
            )
            if in_barrier(
                x1_candidate,
                robot_components=robot_components,
                cfg=cfg,
            ):
                print(
                    f"[safety] Selected control would enter barrier at step {i}; holding instead."
                )
                if_found = False

        if not if_found:
            print(f"[inner-opt] all candidates intersected barriers at step {i}; applying zero-control hold.")
            ctrlu = np.zeros_like(ctrlu)
            ctrl = np.zeros((nu, 1))
            x0_flat = np.asarray(x0).reshape(-1)
            x_pred = [x0_flat.reshape(-1, 1) for _ in range(N)]
            ctrlu = rewrite(ctrlu, nu, N)
            x_0 = newinit(x0_flat, ctrlu, nx, nu, N, dt)
            u_0 = transu(ctrlu, nu, N)
            impc[:nx, i + 1] = x0_flat
            impc[nx:, i + 1] = 0.0
            executed_xy = impc[:2, :i + 1]
            traj_line.set_data(executed_xy[0], executed_xy[1])
            current_state = x0_flat
            state_traj = np.repeat(current_state[:, None], N + 1, axis=1)
            update_contacts(dual_solution, state_traj, label="outer", iteration=i)
            current_marker.set_offsets(np.array([[current_state[0], current_state[1]]]))
            for patch, poly in zip(robot_patches, compute_robot_polygons(current_state)):
                patch.set_xy(poly)
            record_frame()
            if interactive_backend:
                fig.canvas.flush_events()
                plt.pause(0.001)
            failure_message = f"[failure] step {i}: no safe candidate; stopping early."
            end_idx = i + 1
            break
          
        ctrl = ctrlu[:nu]
        
        # Probing horizon for visualization: use a dynamics rollout from ctrlu.
        x_pred = []
        x_traj_sel = newinit(np.asarray(x0_current).reshape(-1), ctrlu, nx, nu, N, dt)
        for k in range(1, N + 1):
            x_pred.append(x_traj_sel[:, k].reshape(-1, 1))
        
        # Execute ONE step from the true current state.
        x0 = np.array([dt * x0_current[0, 3] * np.cos(x0_current[0, 2]) + x0_current[0, 0],
               dt * x0_current[0, 3] * np.sin(x0_current[0, 2]) + x0_current[0, 1],
               dt * ctrl[0, 0] + x0_current[0, 2],
               dt * ctrl[1, 0] + x0_current[0, 3]])
        ctrlu = rewrite(ctrlu, nu, N)
        x_0 = newinit(x0, ctrlu, nx, nu, N, dt)
        u_0 = transu(ctrlu, nu, N)
        impc[:nx, i + 1] = x0
        # Log the *applied* control (yaw-rate, accel), not the warm-start guess.
        impc[nx:, i + 1] = ctrl[:2, 0]

        x0_flat = np.asarray(x0).reshape(-1)
        collision_detected = in_barrier(
            x0_flat, robot_components=robot_components, cfg=cfg
        )
        
        # Calculate and print deviation from guiding path
        current_pos_flat = x0_flat[:2]
        dev_mean, dev_rms, dev_max = _polyline_deviation(current_pos_flat.reshape(1, 2), path_points)
        print(
            f"[outer] iter={i} wp_idx={path_idx} wp=({path_points[path_idx,0]:.4f},{path_points[path_idx,1]:.4f}) "
            f"pos=({x0_flat[0]:.4f},{x0_flat[1]:.4f}) "
            f"dev_max={dev_max:.4f} dev_rms={dev_rms:.4f}"
        )

        # Update the moving-window reference and visualization.
        current_state_flat = x0.reshape(-1)
        current_pos = current_state_flat[:2]
        xr = reference_trajectory(current_pos, heading_cur=float(current_state_flat[2]))
        reference_line.set_data(xr[0, 1:], xr[1, 1:])
        reference_point_marker.set_offsets(np.column_stack([xr[0, 1:], xr[1, 1:]]))
        wp_marker.set_offsets(np.array([[xr[0, -1], xr[1, -1]]]))
        path_idx = int(np.clip(ref_path_index, 0, path_points.shape[0] - 1))

        end_time = time.time()
        print(f"Iteration time: {end_time - start_time} seconds")
        
        executed_xy = impc[:2, :i + 1]
        traj_line.set_data(executed_xy[0], executed_xy[1])

        if x_pred:
            pred_points = np.array([pt.flatten() for pt in x_pred])
            # Visualize the full horizon probing trajectory (N steps). We only accept candidates
            # whose *entire* horizon rollout is barrier-free.
            probe_line.set_data(pred_points[:, 0], pred_points[:, 1])
            x_pred_all = np.vstack([x_pred_all, pred_points.reshape(1, -1)])
        else:
            probe_line.set_data([], [])
            x_pred_all = np.vstack([x_pred_all, np.zeros((1, 4 * N))])

        current_state = np.asarray(x0).ravel()
        state_traj = trans(current_state.reshape(1, -1), ctrlx, nx, N + 1)
        update_contacts(dual_solution, state_traj, label="outer", iteration=i)

        current_marker.set_offsets(np.array([[current_state[0], current_state[1]]]))
        for patch, poly in zip(robot_patches, compute_robot_polygons(current_state)):
            patch.set_xy(poly)
        record_frame()
        if interactive_backend:
            fig.canvas.flush_events()
            plt.pause(0.001)
        if collision_detected:
            failure_message = (
                f"[failure] step {i}: executed state entered barrier; stopping early."
            )
            end_idx = i + 1
            break
        
        # Check if reach the destination
        if (x0[0] - goal_state[0])**2 + (x0[1] - goal_state[1])**2 <= approx_radius**2:
            end_idx = i + 1
            break
    else:
        # Only executed if the loop did not break
        failure_message = None

    if failure_message:
        print(failure_message)
            
    # Pick the final trajectory
    if end_idx != 0:
        impctra = impc[:, :end_idx + 1]  # Final trajectory
    else:
        impctra = impc

    # Evaluate deviation from baseline path if available
    if "baseline_path" in locals() and baseline_path is not None:
        executed_xy = impctra[:2, :].T
        mean_dev, rms_dev, max_dev = _polyline_deviation(executed_xy, baseline_path)
        print(
            f"[deviation] mean={mean_dev:.4f} rms={rms_dev:.4f} "
            f"max={max_dev:.4f} over {executed_xy.shape[0]} states"
        )
    try:
        if video_writer is not None:
            video_writer.close()
            print(f"Saved visualization to {video_path}")
        else:
            imageio.mimsave(video_path, frames, fps=10)
            print(f"Saved visualization to {video_path}")
    except Exception as exc:
        print(f"Failed to write video to {video_path}: {exc}")
    finally:
        # Ensure writer flushes even if mimsave/close throws.
        try:
            if video_writer is not None:
                video_writer.close()
        except Exception:
            pass

    print("Last Status: ", impctra[:, -1])
    
    if interactive_backend:
        plt.ioff()

    return impctra, x_pred_all
    
def trans(x0, vector, nxx, nyy):
    """
    Transforms a vector into a matrix.

    Args:
    x0 (numpy.ndarray): Initial state vector.
    vector (numpy.ndarray): Larger vector to be transformed.
    nxx (int): Number of rows in the resulting matrix (equivalent to nx).
    nyy (int): Number of columns in the resulting matrix (equivalent to N+1).

    Returns:
    numpy.ndarray: Resulting transformed matrix.
    """
    # Initialize the resulting matrix with zeros
    res = np.zeros((nxx, nyy))

    # Set the first column to x0
    res[:, 0] = x0.squeeze()  # Using squeeze() to ensure x0 is a 1D array

    # Loop to fill the remaining columns
    for i in range(nyy - 1):
        start_idx = (i + 1) * nxx
        end_idx = (i + 2) * nxx
        res[:, i + 1] = vector[start_idx:end_idx].squeeze()
        
    return res
    
def transu(vector, nxx, nyy):
    """
    Transforms a vector into a matrix by reshaping.

    Args:
    vector (numpy.ndarray): The vector to be transformed.
    nxx (int): Number of rows in the resulting matrix (equivalent to nu).
    nyy (int): Number of columns in the resulting matrix (equivalent to N).

    Returns:
    numpy.ndarray: The transformed matrix.
    """
    # Initialize the resulting matrix with zeros
    resu = np.zeros((nxx, nyy))

    # Loop to fill each column of the matrix
    for i in range(nyy):
        # Calculate start and end indices
        start_idx = i * nxx
        end_idx = start_idx + nxx

        resu[:, i] = vector[start_idx:end_idx].squeeze()

    return resu
    
def rewrite(vector, nu, N):
    append = vector[(N - 1) * nu:N * nu]
    vector = vector[nu:]
    reu = np.vstack((vector, append))#####################chech this

    return reu
    
def newinit(x0, ctrlu, nx, nu, N, dt):
    x_0 = np.zeros((nx, N + 1))
    x_0[:, 0] = x0.squeeze()  # Using squeeze() to ensure x0 is a 1D array

    for i in range(N):
        u0 = ctrlu[i * nu:(i + 1) * nu]
        new_x0 = [dt * x0[3] * np.cos(x0[2]) + x0[0],
                  dt * x0[3] * np.sin(x0[2]) + x0[1],
                  dt * u0[0, 0] + x0[2],
                  dt * u0[1, 0] + x0[3]]
        x0 = np.array(new_x0)
        x_0[:, i + 1] = x0

    return x_0
    
def getaa(x0, dt):
    AA = np.array([[1, 0, -x0[3] * np.sin(x0[2]) * dt, np.cos(x0[2]) * dt],
                   [0, 1, x0[3] * np.cos(x0[2]) * dt, np.sin(x0[2]) * dt],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return AA
    
def getcc(x0, x1, u0, dt):
    CC = np.array([x0[3] * np.sin(x0[2]) * x0[2] * dt - x0[3] * np.cos(x0[2]) * dt + x1[0] - x0[0],
                   -x0[3] * np.cos(x0[2]) * x0[2] * dt - x0[3] * np.sin(x0[2]) * dt + x1[1] - x0[1],
                   -u0[0] * dt + x1[2] - x0[2],
                   -u0[1] * dt + x1[3] - x0[3]])
    return CC
                   
def getax(x_0, dt, N, nx):
    x0 = x_0[:, 0]
    AA = getaa(x0, dt)
    Ax = np.kron(np.eye(N + 1), -np.eye(nx)) + np.kron(np.diag(np.ones(N), -1), AA)####################check this, address above!
    
    for i in range(N - 1):
        x0 = x_0[:, i + 1]
        AA = getaa(x0, dt)
        Ax[nx * (i + 2): nx * (i + 3), nx * (i + 1): nx * (i + 2)] = AA

    return Ax
    
def getcx(x_0, u_0, dt, N, nx):
    Cx = np.zeros(N * nx)
    for i in range(N):
        u0 = u_0[:, i]
        x0 = x_0[:, i]
        x1 = x_0[:, i + 1]
        CC = getcc(x0, x1, u0, dt)
        Cx[i * nx: i * nx + nx] = CC
    
    return Cx.reshape(-1, 1)

def lineardyn(
    ur,
    xr,
    x_traj,
    x0,
    Q,
    QN,
    R,
    dt,
    N,
    nx,
    nu,
    BB,
    u_0,
    xmin,
    xmax,
    umin,
    umax,
    cfg: NMPCConfig,
    previous_solution: Optional[DualDistanceSolution] = None,
    extra_obstacle_halfspaces: Sequence[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, DualDistanceSolution]:
    """Solve the closest-point OSQP subproblem for the current horizon."""

    obstacles = _load_obstacle_geometry(cfg, extra_halfspaces=extra_obstacle_halfspaces)
    robot_components = build_robot_geometry(cfg)
    robot_component_facets = [int(comp.A_local.shape[0]) for comp in robot_components]
    layout = _build_layout(nx, nu, N, obstacles, robot_component_facets)

    robot_As, robot_bs = _robot_component_halfspaces_along_trajectory(x_traj, robot_components)
    if previous_solution is None:
        seed_solution = _initialize_dual_solution(x_traj, obstacles, robot_As, robot_bs)
    else:
        seed_solution = _shift_dual_solution(
            previous_solution, x_traj, obstacles, robot_As, robot_bs
        )
    n_components = int(len(robot_component_facets))
    k_cbf = int(min(N, max(1, int(getattr(cfg, "probe_safety_horizon_steps", N)))))
    cbf_margin = float(getattr(cfg, "cbf_margin_dist", 0.0))

    # Precompute h_0 at current state x_t for rolled-out DCBF decay:
    #   h(x_{k+1|t}) >= gamma^{k+1} * h(x_t)
    cbf_decay_gamma = float(getattr(cfg, "cbf_decay", 1.0))
    h0_map: dict = {}
    if cbf_decay_gamma < 1.0:
        center_x0 = np.asarray(x_traj[:2, 0], dtype=float).reshape(2)
        theta_x0 = float(np.asarray(x_traj[2, 0]).reshape(()))
        cos_x0, sin_x0 = math.cos(theta_x0), math.sin(theta_x0)
        R_x0_T = np.array([[cos_x0, sin_x0], [-sin_x0, cos_x0]], dtype=float)
        for ci, comp in enumerate(robot_components):
            A_r0 = comp.A_local @ R_x0_T
            b_r0 = comp.b_local + A_r0 @ center_x0
            for oi, obs in enumerate(obstacles):
                dist_x0, _, _ = solve_min_distance_qp(obs.A, obs.b, A_r0, b_r0)
                h0_map[(oi, ci)] = float(dist_x0) - cbf_margin

    # Objective assembly
    P_base = block_diag(np.kron(np.eye(N), Q), QN, np.kron(np.eye(N), R))
    P_sparse = csr_matrix(P_base)
    base_dim = layout.control.stop
    if layout.total_dim > base_dim:
        extra_dim = layout.total_dim - base_dim
        P_sparse = sparse_block_diag(
            [P_sparse, csr_matrix((extra_dim, extra_dim))],
            format="csr",
        )

    # Add cost terms for contact point separation ||y_obs - y_robot||^2.
    contact_rows: List[int] = []
    contact_cols: List[int] = []
    contact_data: List[float] = []
    for obs_idx, _ in enumerate(obstacles):
        for comp_idx in range(n_components):
            for step in range(N):
                y_obs_slice = layout.y_obs[obs_idx][comp_idx][step]
                y_robot_slice = layout.y_robot[obs_idx][comp_idx][step]
                for dim in range(2):
                    idx_obs = y_obs_slice.start + dim
                    idx_robot = y_robot_slice.start + dim
                    contact_rows.extend([idx_obs, idx_robot, idx_obs, idx_robot])
                    contact_cols.extend([idx_obs, idx_robot, idx_robot, idx_obs])
                    contact_data.extend([2.0, 2.0, -2.0, -2.0])
    if contact_data:
        # Build contact cost with the same shape as the quadratic term to avoid shape drift
        contact_matrix = csr_matrix(
            (contact_data, (contact_rows, contact_cols)),
            shape=P_sparse.shape,
        )
        P_sparse = P_sparse + contact_matrix

    q_vec = np.zeros(layout.total_dim)
    
    # Handle both constant reference (4, 1) and trajectory reference (4, N+1)
    if xr.ndim == 2 and xr.shape[1] >= N:
        # Trajectory reference: use xr[:, 1:N+1] for states, xr[:, -1] for terminal
        # Note: lineardyn optimizes x1...xN. So we compare x1 with xr[1], etc.
        # Assuming xr covers 0...N.
        q_state_list = []
        for k in range(N):
            # State k+1 corresponds to xr[:, k+1]
            ref_k = xr[:, k+1]
            q_state_list.append((-Q @ ref_k).ravel())
        q_state = np.concatenate(q_state_list)
        q_terminal = (-QN @ xr[:, -1]).ravel()
    else:
        # Constant reference
        if xr.ndim == 2:
            xr_vec = xr.ravel()
        else:
            xr_vec = xr
        q_state = np.tile((-Q @ xr_vec).ravel(), N)
        q_terminal = (-QN @ xr_vec).ravel()

    q_control = np.tile((-R @ ur).ravel(), N)
    q_base = np.concatenate([q_state, q_terminal, q_control])
    q_vec[:base_dim] = q_base

    # ------------------------------------------------------------------
    # Unstick repulsion (lightweight, keeps iMPC/OSQP formulation)
    # ------------------------------------------------------------------
    if getattr(cfg, "enable_unstick_repulsion", False):
        dist_thr = float(getattr(cfg, "unstick_distance_threshold", 0.0))
        w_rep = float(getattr(cfg, "unstick_weight", 0.0))
        k_rep = int(getattr(cfg, "unstick_horizon_steps", 0))
        decay = float(getattr(cfg, "unstick_decay", 1.0))
        if dist_thr > 0.0 and w_rep > 0.0 and k_rep > 0:
            # Use closest obstacle normal at the CURRENT pose.
            cur_state = np.asarray(x_traj[:, 0], dtype=float).reshape(-1)
            _, robot_halfspaces_cur = _robot_geometry_world(cur_state, robot_components)

            best_dist = float("inf")
            best_normal = None
            for obstacle in obstacles:
                try:
                    for A_robot_cur, b_robot_cur in robot_halfspaces_cur:
                        dist, obs_pt, robot_pt = solve_min_distance_qp(
                            obstacle.A, obstacle.b, A_robot_cur, b_robot_cur
                        )
                        dist = float(dist)
                        if dist < best_dist:
                            # Normal points from obstacle contact to robot contact
                            denom = max(dist, 1e-6)
                            normal = (robot_pt - obs_pt) / denom
                            best_dist = dist
                            best_normal = normal
                except Exception:
                    continue

            if best_normal is not None and best_dist < dist_thr:
                # Scale repulsion by proximity (0 at threshold, max at 0 distance)
                scale = w_rep * (dist_thr - best_dist) / dist_thr
                n = np.asarray(best_normal, dtype=float).reshape(2)
                # Push x1..xK away from obstacle
                for kk in range(min(k_rep, N)):
                    step_scale = scale * (decay ** kk)
                    state_offset = layout.state.start + (kk + 1) * nx
                    q_vec[state_offset + 0] -= step_scale * float(n[0])
                    q_vec[state_offset + 1] -= step_scale * float(n[1])

    # ------------------------------------------------------------------
    # Progress reward: encourage motion along the local reference direction
    # ------------------------------------------------------------------
    if getattr(cfg, "enable_progress_reward", False):
        w_prog = float(getattr(cfg, "progress_weight", 0.0))
        k_prog = int(getattr(cfg, "progress_horizon_steps", 0))
        decay = float(getattr(cfg, "progress_decay", 1.0))
        # Only meaningful when xr is a trajectory (4, N+1)
        if w_prog > 0.0 and k_prog > 0 and xr.ndim == 2 and xr.shape[1] >= N + 1:
            for kk in range(1, min(N, k_prog) + 1):
                d = xr[:2, kk] - xr[:2, kk - 1]
                nrm = float(np.linalg.norm(d))
                if nrm < 1e-9:
                    continue
                d = d / nrm
                step_w = w_prog * (decay ** (kk - 1))
                state_offset = layout.state.start + kk * nx
                # Minimize -step_w * d^T [x_k, y_k]  => reward progress along d.
                q_vec[state_offset + 0] -= step_w * float(d[0])
                q_vec[state_offset + 1] -= step_w * float(d[1])

    diag_updates = np.zeros(layout.total_dim)
    for obs_idx, obstacle in enumerate(obstacles):
        penalty = float(obstacle.slack.penalty_weight)
        if penalty <= 0.0:
            continue
        omega_target = float(obstacle.slack.omega_lower_anchor)
        for comp_idx in range(n_components):
            for step in range(N):
                omega_idx = layout.omega[obs_idx][comp_idx][step].start
                diag_updates[omega_idx] = 2.0 * penalty
                # Penalize slack by pulling ω toward its lower anchor (near-hard constraint).
                # The previous formulation pulled ω toward 1.0, which made penetration cheap and
                # often produced probing trajectories that enter barriers.
                q_vec[omega_idx] -= 2.0 * penalty * omega_target
    if np.any(diag_updates):
        P_sparse += diags(
            diag_updates,
            offsets=0,
            shape=(layout.total_dim, layout.total_dim),
            format="csr",
        )

    # Dynamics equality constraints
    Ax = getax(x_traj, dt, N, nx)
    Bu = np.kron(np.vstack([np.zeros((1, N)), np.eye(N)]), BB)
    Cx = getcx(x_traj, u_0, dt, N, nx)
    Ax_sparse = csr_matrix(Ax)
    Bu_sparse = csr_matrix(Bu)
    extra_cols = layout.total_dim - base_dim
    if extra_cols > 0:
        zeros_extra = csr_matrix((Ax_sparse.shape[0], extra_cols))
        Aeq_dyn = hstack([Ax_sparse, Bu_sparse, zeros_extra], format="csr")
    else:
        Aeq_dyn = hstack([Ax_sparse, Bu_sparse], format="csr")
    leq_dyn = np.concatenate((-x0.T.ravel(), -Cx.ravel()))
    ueq_dyn = leq_dyn.copy()

    # KKT stationarity constraints
    kkt_rows = len(obstacles) * n_components * N * 4
    Aeq_kkt = lil_matrix((kkt_rows, layout.total_dim))
    row_cursor = 0
    for obs_idx, obstacle in enumerate(obstacles):
        A_obs = obstacle.A
        for comp_idx in range(n_components):
            facet_count_robot = int(robot_component_facets[comp_idx])
            for step in range(N):
                y_obs_slice = layout.y_obs[obs_idx][comp_idx][step]
                y_robot_slice = layout.y_robot[obs_idx][comp_idx][step]
                lambda_obs_slice = layout.lambda_obs[obs_idx][comp_idx][step]
                lambda_robot_slice = layout.lambda_robot[obs_idx][comp_idx][step]
                # Gradient wrt y_obs: 2(y_obs - y_robot) + A_obs^T λ_obs = 0
                for dim in range(2):
                    Aeq_kkt[row_cursor, y_obs_slice.start + dim] = 2.0
                    Aeq_kkt[row_cursor, y_robot_slice.start + dim] = -2.0
                    for facet in range(obstacle.facets):
                        Aeq_kkt[
                            row_cursor, lambda_obs_slice.start + facet
                        ] = A_obs[facet, dim]
                    row_cursor += 1
                # Gradient wrt y_robot: -2(y_obs - y_robot) + A_robot^T λ_robot = 0
                A_robot = robot_As[comp_idx][step]
                for dim in range(2):
                    Aeq_kkt[row_cursor, y_obs_slice.start + dim] = -2.0
                    Aeq_kkt[row_cursor, y_robot_slice.start + dim] = 2.0
                    for facet in range(facet_count_robot):
                        Aeq_kkt[
                            row_cursor, lambda_robot_slice.start + facet
                        ] = A_robot[facet, dim]
                    row_cursor += 1
    Aeq_kkt = Aeq_kkt.tocsr()
    leq_kkt = np.zeros(kkt_rows)

    Aeq = vstack([Aeq_dyn, Aeq_kkt], format="csr")
    leq = np.concatenate([leq_dyn, leq_kkt])
    ueq = leq.copy()

    # Box constraints on states/inputs
    A_box = eye(layout.total_dim, format="csr")
    lineq_box = np.full(layout.total_dim, -np.inf)
    uineq_box = np.full(layout.total_dim, np.inf)
    state_count = (N + 1) * nx
    control_count = N * nu
    lineq_box[:state_count] = np.tile(xmin, N + 1)
    uineq_box[:state_count] = np.tile(xmax, N + 1)
    lineq_box[state_count : state_count + control_count] = np.tile(umin, N)
    uineq_box[state_count : state_count + control_count] = np.tile(umax, N)

    # Additional inequalities for primal/dual feasibility and slack bounds
    ineq_row_indices: List[int] = []
    ineq_col_indices: List[int] = []
    ineq_values: List[float] = []
    lineq_extra: List[float] = []
    uineq_extra: List[float] = []

    def add_row(indices: Sequence[int], values: Sequence[float], lower: float, upper: float) -> None:
        """Append one inequality row to a COO buffer (fast, avoids per-row CSC builds)."""
        row_idx = len(lineq_extra)
        idx_list = list(indices)
        val_list = list(values)
        ineq_row_indices.extend([row_idx] * len(idx_list))
        ineq_col_indices.extend(idx_list)
        ineq_values.extend(val_list)
        lineq_extra.append(float(lower))
        uineq_extra.append(float(upper))

    for obs_idx, obstacle in enumerate(obstacles):
        A_obs = obstacle.A
        b_obs = obstacle.b
        slack_cfg = obstacle.slack
        omega_lower = float(slack_cfg.omega_lower_anchor)
        omega_upper = float(slack_cfg.omega_upper_bound)
        for comp_idx in range(n_components):
            facet_count_robot = int(robot_component_facets[comp_idx])
            for step in range(N):
                y_obs_slice = layout.y_obs[obs_idx][comp_idx][step]
                y_robot_slice = layout.y_robot[obs_idx][comp_idx][step]
                lambda_obs_slice = layout.lambda_obs[obs_idx][comp_idx][step]
                lambda_robot_slice = layout.lambda_robot[obs_idx][comp_idx][step]
                omega_slice = layout.omega[obs_idx][comp_idx][step]

                for facet in range(obstacle.facets):
                    cols = list(range(y_obs_slice.start, y_obs_slice.stop))
                    vals = A_obs[facet].tolist()
                    add_row(cols, vals, -np.inf, b_obs[facet])

                # Decouple the auxiliary closest-point variables from the MPC state so they
                # do not pull the predicted state toward obstacles.
                for facet in range(facet_count_robot):
                    cols = list(range(y_robot_slice.start, y_robot_slice.stop))
                    vals = robot_As[comp_idx][step][facet].tolist()
                    add_row(cols, vals, -np.inf, robot_bs[comp_idx][step, facet])

                for facet in range(obstacle.facets):
                    add_row(
                        [lambda_obs_slice.start + facet],
                        [1.0],
                        0.0,
                        np.inf,
                    )
                for facet in range(facet_count_robot):
                    add_row(
                        [lambda_robot_slice.start + facet],
                        [1.0],
                        0.0,
                        np.inf,
                    )

                # ----------------------------------------------------------
                # Linearized DHOCBF constraint coupled to the MPC state.
                #
                # We linearize a supporting-hyperplane constraint around the current trajectory
                # guess x_traj, producing a linear inequality in (x_k, y_k, theta_k).
                #
                # For each (obstacle, component, step) we take the closest-point pair from the
                # current warm start, define a separation normal n, then enforce:
                #     n^T (p_k + R(theta_k) q_local - y_obs) >= cbf_margin
                # with R(theta_k) q_local linearized about theta_guess.
                #
                # A rollout `in_barrier()` safety filter is still applied outside OSQP.
                # ----------------------------------------------------------
                if step < k_cbf:
                    try:
                        # Predicted state variable indices (k = step+1)
                        state_base = layout.state.start + (step + 1) * nx
                        x_idx = state_base + 0
                        y_idx = state_base + 1
                        theta_idx = state_base + 2

                        # Linearization point from the guess trajectory
                        center_guess = np.asarray(x_traj[:2, step + 1], dtype=float).reshape(2)
                        theta_guess = float(np.asarray(x_traj[2, step + 1]).reshape(()))

                        # Closest points at the guess (seeded from distance QP)
                        obs_pt = np.asarray(seed_solution.y_obs[obs_idx, comp_idx, step], dtype=float).reshape(2)
                        robot_pt = np.asarray(seed_solution.y_robot[obs_idx, comp_idx, step], dtype=float).reshape(2)

                        diff = robot_pt - obs_pt
                        dist = float(np.linalg.norm(diff))

                        # Separation normal (unit)
                        if dist > 1e-9:
                            n = diff / dist
                        else:
                            # Fallback: use the most active obstacle facet normal at obs_pt
                            residual = (A_obs @ obs_pt) - b_obs
                            facet_star = int(np.argmax(residual))
                            n = np.asarray(A_obs[facet_star], dtype=float).reshape(2)
                            nrm = float(np.linalg.norm(n))
                            if nrm > 1e-12:
                                n = n / nrm
                            else:
                                n = np.array([1.0, 0.0], dtype=float)

                        # Ensure n points from obstacle toward robot
                        if float(np.dot(n, diff)) < 0.0:
                            n = -n

                        # Local coordinates of the robot contact point (body-fixed)
                        c0 = math.cos(theta_guess)
                        s0 = math.sin(theta_guess)
                        R0_T = np.array([[c0, s0], [-s0, c0]], dtype=float)  # R(theta)^T
                        q_local = R0_T @ (robot_pt - center_guess)

                        # d/dtheta [R(theta) q] at theta_guess
                        dR0 = np.array([[-s0, -c0], [c0, -s0]], dtype=float)
                        dq_dtheta = dR0 @ q_local
                        a_theta = float(np.dot(n, dq_dtheta))

                        # Linearized inequality:
                        #   n^T p_k + a_theta * theta_k + omega >= rhs
                        # where rhs is computed so equality holds at the linearization point.
                        rhs = (
                            float(cbf_margin)
                            + float(np.dot(n, obs_pt))
                            - float(np.dot(n, robot_pt))
                            + float(np.dot(n, center_guess))
                            + a_theta * theta_guess
                        )
                        # Rolled-out DCBF decay: h_k >= gamma^k * h_0
                        if cbf_decay_gamma < 1.0:
                            h0_val = h0_map.get((obs_idx, comp_idx), 0.0)
                            if h0_val > 0.0:
                                rhs += (cbf_decay_gamma ** (step + 1)) * h0_val
                        omega_index = omega_slice.start
                        add_row(
                            [x_idx, y_idx, theta_idx, omega_index],
                            [float(n[0]), float(n[1]), a_theta, 1.0],
                            rhs,
                            np.inf,
                        )
                    except Exception:
                        # If the linearization fails for numerical reasons, skip this constraint.
                        # The outer safety filter rejects in-barrier candidates before execution.
                        pass

                dual_cols: List[int] = []
                dual_vals: List[float] = []
                for facet in range(obstacle.facets):
                    dual_cols.append(lambda_obs_slice.start + facet)
                    dual_vals.append(-b_obs[facet])
                for facet in range(facet_count_robot):
                    dual_cols.append(lambda_robot_slice.start + facet)
                    dual_vals.append(-robot_bs[comp_idx][step, facet])
                omega_index = omega_slice.start
                dual_cols.append(omega_index)
                dual_vals.append(1.0)  # Fixed coefficient for slack to ensure dist >= -omega
                add_row(dual_cols, dual_vals, 0.0, np.inf)

                add_row([omega_index], [1.0], omega_lower, np.inf)
                if np.isfinite(omega_upper):
                    add_row([omega_index], [1.0], -np.inf, omega_upper)

    if lineq_extra:
        row_count = len(lineq_extra)
        A_extra = coo_matrix(
            (ineq_values, (ineq_row_indices, ineq_col_indices)),
            shape=(row_count, layout.total_dim),
        ).tocsr()
        lineq_extra_arr = np.asarray(lineq_extra, dtype=float)
        uineq_extra_arr = np.asarray(uineq_extra, dtype=float)
    else:
        A_extra = csr_matrix((0, layout.total_dim))
        lineq_extra_arr = np.zeros(0, dtype=float)
        uineq_extra_arr = np.zeros(0, dtype=float)

    Aineq = vstack([A_box, A_extra], format="csr")
    lineq_full = np.concatenate([lineq_box, lineq_extra_arr])
    uineq_full = np.concatenate([uineq_box, uineq_extra_arr])

    A_sparse = vstack([Aeq, Aineq], format="csr")
    l_vec = np.concatenate([leq, lineq_full])
    u_vec = np.concatenate([ueq, uineq_full])

    warm = np.zeros(layout.total_dim)
    warm[layout.state] = x_traj.T.reshape(-1)
    warm[layout.control] = u_0.T.reshape(-1)
    for obs_idx in range(len(obstacles)):
        for comp_idx in range(n_components):
            for step in range(N):
                warm[layout.y_obs[obs_idx][comp_idx][step]] = seed_solution.y_obs[obs_idx, comp_idx, step]
                warm[layout.y_robot[obs_idx][comp_idx][step]] = seed_solution.y_robot[obs_idx, comp_idx, step]
                warm[layout.lambda_obs[obs_idx][comp_idx][step]] = seed_solution.lambda_obs[obs_idx][comp_idx, step]
                warm[layout.lambda_robot[obs_idx][comp_idx][step]] = seed_solution.lambda_robot[comp_idx][obs_idx, step]
                warm[layout.omega[obs_idx][comp_idx][step]] = seed_solution.omega[obs_idx, comp_idx, step]

    prob = osqp.OSQP()
    prob.setup(
        P=P_sparse.tocsc(),
        q=q_vec,
        A=A_sparse.tocsc(),
        l=l_vec,
        u=u_vec,
        warm_start=True,
        polish=True,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter=20000,
        verbose=False,
    )
    prob.warm_start(x=warm)
    res = prob.solve()

    # OSQP can report "solved inaccurate" (with a space) when constraints are tight.
    # Normalize so we reliably accept it.
    status_raw = str(res.info.status).strip().lower()
    status_norm = status_raw.replace(" ", "_")
    # With tight constraints, OSQP can hit the iteration cap but still return a usable iterate.
    # We'll evaluate it with the same safety checks; if not viable, the caller will keep iterating.
    if status_norm not in {"solved", "solved_inaccurate", "maximum_iterations_reached"}:
        raise RuntimeError(f"OSQP failed with status {res.info.status}")

    ctrlx = res.x[layout.state]
    ctrlu = res.x[layout.control]
    dual_solution = _extract_dual_solution(
        res.x,
        layout,
        obstacles,
        robot_component_facets,
        N,
    )

    if _VERBOSE or str(res.info.status) not in {"solved", "solved inaccurate"}:
        print(f"[osqp] status={res.info.status}")
    return ctrlx.reshape(-1, 1), ctrlu.reshape(-1, 1), dual_solution

def lineardyn2(
    ur,
    xr,
    x_traj,
    x0,
    Q,
    QN,
    R,
    dt,
    N,
    nx,
    nu,
    BB,
    u_0,
    xmin,
    xmax,
    umin,
    umax,
    cfg: NMPCConfig,
    previous_solution: Optional[DualDistanceSolution] = None,
    extra_obstacle_halfspaces: Sequence[Tuple[np.ndarray, np.ndarray]] | None = None,
):
    """Second-order variant currently reuses the closest-point OSQP formulation."""
    return lineardyn(
        ur,
        xr,
        x_traj,
        x0,
        Q,
        QN,
        R,
        dt,
        N,
        nx,
        nu,
        BB,
        u_0,
        xmin,
        xmax,
        umin,
        umax,
        cfg=cfg,
        previous_solution=previous_solution,
        extra_obstacle_halfspaces=extra_obstacle_halfspaces,
    )


def impcdcbf_multi_agent(
    agent_goals: Sequence[MultiAgentGoal],
    config: NMPCConfig | None = None,
    consider_robot_barriers: bool = True,
    headless: bool = False,
) -> dict[str, np.ndarray]:
    """Run synchronous iMPC for multiple robots.

    Each robot solves an iMPC QP at every outer step. When
    `consider_robot_barriers` is True, other robots' predicted next poses are
    treated as additional convex obstacles.
    """
    cfg = config or DEFAULT_CONFIG

    if len(agent_goals) < 2:
        raise ValueError("Multi-agent mode expects at least two robots.")

    N = int(cfg.horizon)
    dt = float(cfg.dt)
    nsim = int(cfg.nsim)
    nx = 4
    nu = 6
    BB = np.array(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [dt, 0, 0, 0, 0, 0], [0, dt, 0, 0, 0, 0]],
        dtype=float,
    )

    Q = cfg.Q()
    QN = cfg.QN()
    R = np.diag(cfg.R_diagonal)
    ur = cfg.reference_input.reshape(-1, 1)
    umin = cfg.input_bounds.lower.copy()
    umax = cfg.input_bounds.upper.copy()
    xmin = cfg.state_bounds.lower.copy()
    xmax = cfg.state_bounds.upper.copy()
    allowed_shapes = {"lshape", "triangle", "rectangle"}

    agents: List[MultiAgentState] = []
    for spec in agent_goals:
        shape = str(spec.robot_shape).strip().lower()
        if shape not in allowed_shapes:
            raise ValueError(
                f"Unknown robot_shape={spec.robot_shape!r} for {spec.name}. "
                f"Expected one of {sorted(allowed_shapes)}."
            )
        agent_cfg = replace(cfg, robot_shape=shape)
        init_model(agent_cfg)
        robot_components = build_robot_geometry(agent_cfg)
        start_state = _expand_state(spec.start)
        goal_state = _expand_state(spec.goal)
        path_points = plan_path(
            start_state[:2],
            goal_state[:2],
            OBSTACLE_REGIONS,
            horizon=60,
            margin=float(getattr(spec, "path_margin", 0.05)),
            bounds=ENVIRONMENT.bounds,
            cell_size=ENVIRONMENT.grid[1],
            use_grid=True,
            method="astar_los",
            quad=False,
        )
        path_points = np.asarray(path_points, dtype=float).reshape(-1, 2)
        x_guess, u_guess_vec = _pure_pursuit_warm_start(
            path_points, start_state, agent_cfg, N, nu, dt
        )
        u_guess_mat = transu(u_guess_vec, nu, N)
        agents.append(
            MultiAgentState(
                spec=spec,
                state=start_state.copy(),
                goal_state=goal_state.copy(),
                cfg=agent_cfg,
                robot_components=robot_components,
                path_points=path_points,
                u_guess_vec=u_guess_vec,
                u_guess_mat=u_guess_mat,
                x_guess=x_guess,
                dual_solution=None,
                done=False,
                states_hist=[start_state.copy()],
                controls_hist=[],
                probe_hist=[],
                xr_hist=[],
            )
        )
        print(
            f"[{spec.name}] start=({start_state[0]:.3f},{start_state[1]:.3f}) "
            f"goal=({goal_state[0]:.3f},{goal_state[1]:.3f}) "
            f"shape={shape} path_waypoints={path_points.shape[0]}"
        )

    for step in range(nsim):
        if all(agent.done for agent in agents):
            print(f"[multi-agent] All robots reached goals by step {step}.")
            break

        predicted_next: List[np.ndarray] = []
        for agent in agents:
            if agent.done:
                predicted_next.append(agent.state.copy())
                continue
            start_delay = int(getattr(agent.spec, "start_delay", 0))
            if step < start_delay:
                predicted_next.append(agent.state.copy())
                continue
            ctrl_first = (
                np.asarray(agent.u_guess_vec[:nu], dtype=float)
                if agent.u_guess_vec.shape[0] >= nu
                else np.zeros((nu, 1), dtype=float)
            )
            predicted_next.append(_integrate_one_step(agent.state, ctrl_first, dt))

        for idx, agent in enumerate(agents):
            if agent.done:
                agent.controls_hist.append(np.zeros(2, dtype=float))
                agent.states_hist.append(agent.state.copy())
                agent.probe_hist.append(np.zeros((0, 2), dtype=float))
                agent.xr_hist.append(np.zeros((4, 0), dtype=float))
                continue

            start_delay = int(getattr(agent.spec, "start_delay", 0))
            if step < start_delay:
                # Hold position until start_delay (predicted_next already set in loop above)
                agent.controls_hist.append(np.zeros(2, dtype=float))
                agent.states_hist.append(agent.state.copy())
                agent.probe_hist.append(np.zeros((0, 2), dtype=float))
                agent.xr_hist.append(np.zeros((4, 0), dtype=float))
                continue

            if consider_robot_barriers:
                dynamic_obstacles = _build_predicted_robot_obstacles(
                    predicted_next,
                    ego_index=idx,
                    robot_components_per_agent=[other.robot_components for other in agents],
                )
            else:
                dynamic_obstacles = []
            # Inner-loop homotopy: retry with different speed_scale if nominal solution collides.
            # Try reverse early (after 5 fwd) to break head-on deadlocks with robot_scale=2/3.
            max_step_iters = int(getattr(agent.cfg, "max_step_iterations", 100))
            def _speed_scale_for_iter(j: int) -> float:
                if j < 5:
                    return 1.0
                elif j < 15:
                    return -0.5   # Reverse early to yield in corridor
                elif j < 25:
                    return 0.5
                elif j < 35:
                    return -0.5
                elif j < 45:
                    return 0.25
                elif j < 55:
                    return 0.1
                elif j < 65:
                    return 0.0
                elif j < 80:
                    return -0.5   # More reverse for deadlock
                else:
                    return 0.0
            accepted_safe = False
            probing_safe = False
            ctrl_apply = np.zeros((nu, 1), dtype=float)
            next_state = agent.state.copy()
            ctrlu = np.zeros((nu * N, 1), dtype=float)
            candidate_dual = agent.dual_solution
            x_guess_iter = agent.x_guess.copy()
            u_guess_mat_iter = agent.u_guess_mat.copy()
            x_rollout = np.zeros((nx, N + 1), dtype=float)
            x_rollout[:, 0] = agent.state[:4]

            for homotopy_idx in range(max_step_iters):
                speed_scale = _speed_scale_for_iter(homotopy_idx)
                xr = _build_reference_from_path(
                    agent.path_points,
                    agent.state,
                    agent.goal_state,
                    agent.cfg,
                    N,
                    speed_scale=speed_scale,
                    heading_now=float(agent.state[2]),
                )

                x0_row = agent.state.reshape(1, -1)
                try:
                    ctrlx, ctrlu, candidate_dual = lineardyn(
                        ur,
                        xr,
                        x_guess_iter,
                        x0_row,
                        Q,
                        QN,
                        R,
                        dt,
                        N,
                        nx,
                        nu,
                        BB,
                        u_guess_mat_iter,
                        xmin,
                        xmax,
                        umin,
                        umax,
                        cfg=agent.cfg,
                        previous_solution=candidate_dual,
                        extra_obstacle_halfspaces=dynamic_obstacles,
                    )
                except Exception:
                    continue

                # Update warm start for next homotopy iteration (same step, different ref)
                x_rollout = newinit(agent.state, ctrlu, nx, nu, N, dt)
                x_guess_iter = x_rollout
                u_guess_mat_iter = transu(ctrlu, nu, N)

                probe_steps = int(getattr(agent.cfg, "probe_safety_horizon_steps", N))
                probe_steps = min(probe_steps, N)
                probing_safe = True
                for k in range(1, probe_steps + 1):
                    state_k = np.asarray(x_rollout[:, k], dtype=float)
                    if in_barrier(state_k, robot_components=agent.robot_components, cfg=agent.cfg):
                        probing_safe = False
                        break
                    if _state_hits_dynamic_obstacles(
                        state_k,
                        dynamic_obstacles,
                        agent.robot_components,
                        penetration_tol=float(getattr(agent.cfg, "barrier_penetration_tol", 0.0)),
                    ):
                        probing_safe = False
                        break

                if probing_safe:
                    ctrl_candidate = np.asarray(ctrlu[:nu], dtype=float)
                    next_candidate = _integrate_one_step(agent.state, ctrl_candidate, dt)
                    if in_barrier(
                        next_candidate, robot_components=agent.robot_components, cfg=agent.cfg
                    ) or _state_hits_dynamic_obstacles(
                        next_candidate,
                        dynamic_obstacles,
                        agent.robot_components,
                        penetration_tol=float(getattr(agent.cfg, "barrier_penetration_tol", 0.0)),
                    ):
                        probing_safe = False
                    else:
                        ctrl_apply = ctrl_candidate
                        next_state = next_candidate
                        agent.dual_solution = candidate_dual
                        accepted_safe = True
                        accepted_xr = xr.copy()
                        break

            if not accepted_safe:
                raise RuntimeError(
                    f"[{agent.spec.name}] step={step}: no barrier-free candidate found "
                    f"across {max_step_iters} homotopy attempts; exiting to avoid penetration."
                )

            agent.state = next_state.copy()
            agent.states_hist.append(agent.state.copy())
            agent.controls_hist.append(np.asarray(ctrl_apply[:2, 0], dtype=float))
            agent.probe_hist.append(x_rollout[:2, :].T.copy())  # (N+1, 2) probing horizon xy
            agent.xr_hist.append(accepted_xr.copy())

            shifted_u = rewrite(ctrlu, nu, N)
            agent.u_guess_vec = shifted_u
            agent.u_guess_mat = transu(shifted_u, nu, N)
            agent.x_guess = newinit(agent.state, shifted_u, nx, nu, N, dt)

            predicted_next[idx] = _integrate_one_step(agent.state, agent.u_guess_vec[:nu], dt)

            if np.linalg.norm(agent.state[:2] - agent.goal_state[:2]) <= float(agent.cfg.approx_radius):
                agent.done = True
                print(f"[{agent.spec.name}] reached goal at step {step}.")

        if step % 10 == 0:
            done_count = sum(1 for agent in agents if agent.done)
            print(f"[multi-agent] step={step}/{nsim - 1}, finished={done_count}/{len(agents)}")

    def _pack_trajectories() -> dict[str, np.ndarray]:
        packed: dict[str, np.ndarray] = {}
        for agent in agents:
            states_arr = np.asarray(agent.states_hist, dtype=float).T  # (4, T)
            controls_arr = np.asarray(agent.controls_hist, dtype=float).T if agent.controls_hist else np.zeros((2, 0))
            out = np.zeros((nx + 2, states_arr.shape[1]), dtype=float)
            out[:nx, :] = states_arr
            if controls_arr.shape[1] > 0:
                out[nx:, 1 : controls_arr.shape[1] + 1] = controls_arr
            packed[agent.spec.name] = out
        return packed

    if headless:
        print("[multi-agent] headless: skipped animation")
        return _pack_trajectories()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    plotmap(ax)
    (x_min, y_min), (x_max, y_max) = ENVIRONMENT.bounds
    pad = 0.05 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    ax.set_title("")
    color_cycle = ["dodgerblue", "pink", "forestgreen", "purple", "crimson"]

    trajectory_lines = []
    reference_lines = []
    probe_lines = []
    current_markers = []
    guiding_point_markers = []
    reference_point_markers = []
    robot_patches_per_agent: List[List[Polygon]] = []
    for idx, agent in enumerate(agents):
        color = color_cycle[idx % len(color_cycle)]
        path = np.asarray(agent.path_points, dtype=float)
        ax.plot(path[:, 0], path[:, 1], "--", linewidth=1.0, color=color, alpha=0.35)
        line, = ax.plot([], [], "-", linewidth=2.0, color=color, label=agent.spec.name)
        trajectory_lines.append(line)
        ref_line, = ax.plot(
            [], [], "-", linewidth=2.5, color=color, alpha=0.85,
            label=f"{agent.spec.name} ref window",
        )
        reference_lines.append(ref_line)
        probe_line, = ax.plot(
            [], [], "o--", linewidth=1.5, color=color, alpha=0.9,
            markersize=3, label=f"{agent.spec.name} probe",
        )
        probe_lines.append(probe_line)
        wp_mkr = ax.scatter(
            [], [], color="orange", s=70, marker="X", edgecolors="black",
            linewidths=1.0, zorder=5,
        )
        guiding_point_markers.append(wp_mkr)
        ref_pts_mkr = ax.scatter(
            [], [], color="orange", s=30, marker=".", zorder=5,
        )
        reference_point_markers.append(ref_pts_mkr)

        ax.scatter(
            agent.states_hist[0][0],
            agent.states_hist[0][1],
            color=color,
            marker="o",
            s=45,
            edgecolors="black",
        )
        ax.scatter(
            agent.goal_state[0],
            agent.goal_state[1],
            color=color,
            marker="*",
            s=80,
        )
        marker = ax.scatter([], [], color=color, s=30, marker="s", edgecolors="black")
        current_markers.append(marker)

        patches_this_agent: List[Polygon] = []
        for comp in agent.robot_components:
            patch = Polygon(
                np.zeros((comp.vertices_local.shape[0], 2)),
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.3,
                alpha=0.9,
            )
            ax.add_patch(patch)
            patches_this_agent.append(patch)
        robot_patches_per_agent.append(patches_this_agent)

    fig.tight_layout()

    max_frames = max(len(agent.states_hist) for agent in agents)
    video_path = Path("impc_run.mp4")
    writer = None
    fallback_frames: List[np.ndarray] = []
    try:
        writer = imageio.get_writer(video_path, fps=10)
    except Exception as exc:
        print(f"[video] Failed to initialize writer ({exc}); using fallback buffer.")

    try:
        for frame_idx in range(max_frames):
            for idx, agent in enumerate(agents):
                traj = np.asarray(agent.states_hist, dtype=float)
                step_idx = min(frame_idx, traj.shape[0] - 1)
                partial = traj[: step_idx + 1]
                trajectory_lines[idx].set_data(partial[:, 0], partial[:, 1])

                state_now = traj[step_idx]
                current_markers[idx].set_offsets(np.array([[state_now[0], state_now[1]]], dtype=float))

                xr = _build_reference_from_path(
                    agent.path_points,
                    state_now,
                    agent.goal_state,
                    agent.cfg,
                    N,
                    speed_scale=1.0,
                    heading_now=float(state_now[2]),
                )
                guiding_pt = np.array([[xr[0, -1], xr[1, -1]]], dtype=float)
                guiding_point_markers[idx].set_offsets(guiding_pt)
                reference_lines[idx].set_data(xr[0, 1:], xr[1, 1:])  # exclude robot center
                ref_pts_xy = np.column_stack([xr[0, 1:], xr[1, 1:]])
                reference_point_markers[idx].set_offsets(ref_pts_xy)

                if agent.probe_hist:
                    pi = min(frame_idx, len(agent.probe_hist) - 1)
                    probe_xy = agent.probe_hist[pi]
                    if probe_xy.size > 0:
                        probe_lines[idx].set_data(probe_xy[:, 0], probe_xy[:, 1])
                    else:
                        probe_lines[idx].set_data([], [])
                else:
                    probe_lines[idx].set_data([], [])

                polys_world, _ = _robot_geometry_world(state_now, agent.robot_components)
                for patch, poly in zip(robot_patches_per_agent[idx], polys_world):
                    patch.set_xy(poly)

            ax.set_title("")
            frame = _pad_frame_to_macroblock(_capture_frame(fig))
            if writer is not None:
                writer.append_data(frame)
            else:
                fallback_frames.append(frame)
    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass

    if writer is None and fallback_frames:
        imageio.mimsave(video_path, fallback_frames, fps=10)

    fig.savefig("impc_multi_agent.png")
    plt.close(fig)
    print("Saved visualization to impc_multi_agent.png")
    print(f"Saved visualization to {video_path}")

    _save_sample_plot_multiagent(agents)
    return _pack_trajectories()
    
def plotmap(ax=None):
    axis = ax if ax is not None else plt.gca()
    for region in OBSTACLE_REGIONS:
        patch = region.get_plot_patch()
        axis.add_patch(patch)
    axis.set_xlabel('X-axis')
    axis.set_ylabel('Y-axis')


def _rot_ccw90(points: np.ndarray) -> np.ndarray:
    """Apply 90° counter-clockwise rotation (x,y)→(-y,x)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1 and pts.size >= 2:
        return np.array([-pts[1], pts[0]], dtype=float)
    out = np.empty_like(pts)
    out[:, 0] = -pts[:, 1]
    out[:, 1] = pts[:, 0]
    return out


def _pick_sample_indices(
    trajectory_xy: np.ndarray,
    max_samples: int,
    min_dist_same: float,  # min distance between samples of this robot
    excluded_positions: np.ndarray | None = None,
    exclusivity_dist: float = 0.02,  # min distance from other robots' samples
) -> List[int]:
    """Pick up to max_samples evenly spaced indices. Drop any within min_dist_same of previous kept
    (same robot), or within exclusivity_dist of excluded_positions (other robots)."""
    T = trajectory_xy.shape[0]
    if T <= 0:
        return []
    if T <= 1:
        return [0]
    excl = np.asarray(excluded_positions, dtype=float) if excluded_positions is not None else np.zeros((0, 2))
    n_target = min(max_samples, T)
    candidates = np.linspace(0, T - 1, n_target).astype(int)
    kept: List[int] = [candidates[0]]
    for i in range(1, len(candidates)):
        idx = candidates[i]
        pos = trajectory_xy[idx]
        if np.linalg.norm(pos - trajectory_xy[kept[-1]]) < min_dist_same:
            continue
        if excl.shape[0] > 0 and np.any(np.linalg.norm(excl - pos, axis=1) < exclusivity_dist):
            continue
        kept.append(idx)
        if len(kept) >= max_samples:
            break
    return kept


def _save_sample_plot_multiagent(agents: List[MultiAgentState]) -> None:
    """Generate static PNG after iMPC: obstacles, guiding path, sampled robot footprints, ref lines, probe lines.
    Rotate 90° CCW via (x,y)→(-y,x). No legend, no axis labels, 150 dpi, tight layout."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    color_cycle = ["dodgerblue", "pink", "forestgreen"]
    for region in OBSTACLE_REGIONS:
        verts = np.asarray(region.vertices(), dtype=float)
        verts_rot = _rot_ccw90(verts)
        patch = Polygon(
            verts_rot,
            closed=True,
            facecolor="red",
            edgecolor="black",
            linewidth=1,
            alpha=0.4,
        )
        ax.add_patch(patch)

    occupied_positions: List[np.ndarray] = []
    for idx in range(len(agents) - 1, -1, -1):
        agent = agents[idx]
        color = color_cycle[idx % len(color_cycle)]
        states = np.asarray(agent.states_hist, dtype=float)
        if states.size == 0:
            continue
        traj_xy = states[:, :2]
        path = np.asarray(agent.path_points, dtype=float)
        path_rot = _rot_ccw90(path)
        ax.plot(path_rot[:, 0], path_rot[:, 1], "--", linewidth=1.0, color=color, alpha=0.35)

        excluded = np.array(occupied_positions, dtype=float) if occupied_positions else np.zeros((0, 2))
        sample_indices = _pick_sample_indices(
            traj_xy, max_samples=18, min_dist_same=0.15,
            excluded_positions=excluded, exclusivity_dist=0.02,
        )
        for si in sample_indices:
            if si >= len(agent.states_hist):
                continue
            pi = min(si, len(agent.probe_hist) - 1)
            xi = min(si, len(agent.xr_hist) - 1)
            if pi < 0 or xi < 0:
                continue
            state = agent.states_hist[si]
            probe_xy = agent.probe_hist[pi]
            xr = agent.xr_hist[xi]

            polys, _ = _robot_geometry_world(state, agent.robot_components)
            for poly in polys:
                poly_rot = _rot_ccw90(poly)
                patch = Polygon(
                    poly_rot,
                    closed=True,
                    fill=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.5,
                )
                ax.add_patch(patch)

            if xr.shape[1] >= 2:
                ref_xy = np.column_stack([xr[0, 1:], xr[1, 1:]])
                ref_rot = _rot_ccw90(ref_xy)
                ax.plot(ref_rot[:, 0], ref_rot[:, 1], "-", linewidth=2.0, color="orange", alpha=0.85)

            if probe_xy.shape[0] >= 2:
                probe_rot = _rot_ccw90(probe_xy)
                ax.plot(probe_rot[:, 0], probe_rot[:, 1], "-", linewidth=1.5, color=color, alpha=0.9)

            occupied_positions.append(traj_xy[si])

    # Start and goal markers for each robot
    (x_min, y_min), (x_max, y_max) = ENVIRONMENT.bounds
    pad = 0.05 * max(x_max - x_min, y_max - y_min)
    corners = np.array([[x_min - pad, y_min - pad], [x_max + pad, y_min - pad],
                        [x_max + pad, y_max + pad], [x_min - pad, y_max + pad]])
    corners_rot = _rot_ccw90(corners)
    rx_lo, rx_hi = corners_rot[:, 0].min(), corners_rot[:, 0].max()
    ry_lo, ry_hi = corners_rot[:, 1].min(), corners_rot[:, 1].max()
    for agent in agents:
        states = np.asarray(agent.states_hist, dtype=float)
        if states.size == 0:
            continue
        start_xy = states[0, :2]
        goal_xy = agent.goal_state[:2]
        start_rot = _rot_ccw90(np.array([start_xy]))
        goal_rot = _rot_ccw90(np.array([goal_xy]))
        rsx, rsy = float(start_rot[0, 0]), float(start_rot[0, 1])
        rgx, rgy = float(goal_rot[0, 0]), float(goal_rot[0, 1])
        ax.scatter(rsx, rsy, color="green", s=40, edgecolors="black", zorder=10)
        ax.scatter(rgx, rgy, color="red", s=80, marker="*", edgecolors="black", zorder=10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(rx_lo, rx_hi)
    ax.set_ylim(ry_lo, ry_hi)
    ax.tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    fig.savefig("sample_plot_multi.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    print("Saved sample plot to sample_plot_multi.png")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-robot polytopic navigation with sequential iMPC-DHOCBF (Section V-B)."
    )
    parser.add_argument("--steps", type=int, default=None, help="Closed-loop steps. Default: config nsim.")
    parser.add_argument("--headless", action="store_true", help="Skip the animation video and snapshot figure.")
    parser.add_argument("--verbose", action="store_true", help="Print per-obstacle contact traces and every OSQP status.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _VERBOSE = bool(args.verbose)
    # Three robots (L-shape, triangle, rectangle), each scaled to two-thirds of the single-robot size.
    robot_scale = 2.0 / 3.0
    slack_tuple = tuple(DualSlackConfig() for _ in _OBSTACLE_HALFSPACES)
    cfg = NMPCConfig(
        horizon=12,
        probe_safety_horizon_steps=12,
        approx_radius=0.08,
        reference_speed=0.12,
        reference_proj_dist_buffer=0.05,
        barrier_penetration_tol=0,
        max_step_iterations=60,
        dual_slacks=slack_tuple,
        robot_rectangle_length=DEFAULT_CONFIG.robot_rectangle_length * robot_scale,
        robot_rectangle_width=DEFAULT_CONFIG.robot_rectangle_width * robot_scale,
        robot_rectangle_rear_dist=DEFAULT_CONFIG.robot_rectangle_rear_dist * robot_scale,
        robot_triangle_points=np.asarray(DEFAULT_CONFIG.robot_triangle_points, dtype=float) * robot_scale,
        robot_lshape_part1_points=np.asarray(DEFAULT_CONFIG.robot_lshape_part1_points, dtype=float) * robot_scale,
        robot_lshape_part2_points=np.asarray(DEFAULT_CONFIG.robot_lshape_part2_points, dtype=float) * robot_scale,
    )
    if args.steps is not None:
        cfg = replace(cfg, nsim=int(args.steps))

    agents = [
        MultiAgentGoal(
            name="robot_1",
            robot_shape="lshape",
            start=np.array([0.15, 0.22, 0.0], dtype=float),
            goal=np.array([1.2, 0.97], dtype=float),
        ),
        MultiAgentGoal(
            name="robot_2",
            robot_shape="triangle",
            start=np.array([0.1, 0.62, 0.0], dtype=float),
            goal=np.array([1.4, 1.0], dtype=float),
        ),
        MultiAgentGoal(
            name="robot_3",
            robot_shape="rectangle",
            start=np.array([1.4, 0.4, 180.0 * np.pi / 180.0], dtype=float),
            goal=np.array([0.05, 0.62], dtype=float),
            path_margin=0.04,
        ),
    ]

    trajectories = impcdcbf_multi_agent(
        agents,
        config=cfg,
        consider_robot_barriers=True,
        headless=args.headless,
    )
    np.savez("multi_agent_trajectories.npz", **trajectories)
    print("Saved trajectories to multi_agent_trajectories.npz")
