import contextlib
import io
import math
import sys
from typing import Iterable, Tuple

import numpy as np
import osqp
from scipy.sparse import csc_matrix
from scipy.spatial import ConvexHull

from config import DEFAULT_CONFIG, NMPCConfig
from env_config import OBSTACLE_HALFSPACES, OBSTACLE_VERTICES

_OBSTACLE_HALFSPACES = OBSTACLE_HALFSPACES
_OBSTACLE_VERTICES = OBSTACLE_VERTICES


def _convex_hull_halfspace(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return half-space description A x <= b for a convex polygon."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    A = hull.equations[:, :2]
    b = -hull.equations[:, 2]
    return A, b


# Shape-aware robot half-space model in the robot frame (A x <= b).
# Initialized by init_model(cfg) and used by robot_halfspace().
_ROBOT_A_LOCAL: np.ndarray | None = None
_ROBOT_B_LOCAL: np.ndarray | None = None
_ROBOT_VERTICES_LOCAL: np.ndarray | None = None
_ROBOT_SHAPE_KEY: tuple | None = None


def _robot_points_from_cfg(cfg: NMPCConfig) -> np.ndarray:
    shape = str(getattr(cfg, "robot_shape", "rectangle")).strip().lower()
    if shape == "rectangle":
        length = float(getattr(cfg, "robot_rectangle_length", 0.15))
        width = float(getattr(cfg, "robot_rectangle_width", 0.06))
        rear_dist = float(getattr(cfg, "robot_rectangle_rear_dist", 0.10))
        hx = 0.5 * length
        hy = 0.5 * width
        x_center = 0.5 * rear_dist
        return np.asarray(
            [
                [x_center + hx, hy],
                [x_center + hx, -hy],
                [x_center - hx, -hy],
                [x_center - hx, hy],
            ],
            dtype=float,
        )
    if shape == "triangle":
        pts = np.asarray(getattr(cfg, "robot_triangle_points"), dtype=float).reshape(-1, 2)
        return pts
    if shape == "lshape":
        p1 = np.asarray(getattr(cfg, "robot_lshape_part1_points"), dtype=float).reshape(-1, 2)
        p2 = np.asarray(getattr(cfg, "robot_lshape_part2_points"), dtype=float).reshape(-1, 2)
        return np.vstack([p1, p2])
    raise ValueError(f"Unknown robot_shape={shape!r}. Expected 'rectangle', 'triangle', or 'lshape'.")


def init_model(cfg: NMPCConfig | None = None, mode: str = "") -> None:
    """Initialize the shape-aware robot half-space model.

    `mode` is retained for backwards compatibility with the old interface.
    """
    _ = mode
    global _ROBOT_A_LOCAL, _ROBOT_B_LOCAL, _ROBOT_VERTICES_LOCAL, _ROBOT_SHAPE_KEY

    cfg = cfg or DEFAULT_CONFIG
    pts = _robot_points_from_cfg(cfg)
    # Key: shape + rounded vertices to avoid constant recompute when cfg is unchanged.
    key = (str(getattr(cfg, "robot_shape", "rectangle")).strip().lower(), pts.round(9).tobytes())
    if _ROBOT_SHAPE_KEY == key and _ROBOT_A_LOCAL is not None and _ROBOT_B_LOCAL is not None:
        return
    _ROBOT_SHAPE_KEY = key

    hull = ConvexHull(pts)
    _ROBOT_VERTICES_LOCAL = pts[hull.vertices].astype(float)
    _ROBOT_A_LOCAL = hull.equations[:, :2].astype(float)
    _ROBOT_B_LOCAL = (-hull.equations[:, 2]).astype(float)


def robot_halfspace(
    theta: float, center: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute half-space representation of the robot polytope at a given pose."""
    if _ROBOT_A_LOCAL is None or _ROBOT_B_LOCAL is None:
        init_model(DEFAULT_CONFIG)
    c = math.cos(theta)
    s = math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    A_world = _ROBOT_A_LOCAL @ rotation.T  # type: ignore[operator]
    b_world = _ROBOT_B_LOCAL + A_world @ center  # type: ignore[operator]
    return A_world, b_world


def solve_min_distance_qp(
    A_obs: np.ndarray, b_obs: np.ndarray, A_robot: np.ndarray, b_robot: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve min ||y_obs - y_robot||^2 subject to feasibility constraints."""
    zeros_obs_robot = np.zeros((A_obs.shape[0], 2))
    zeros_robot_obs = np.zeros((A_robot.shape[0], 2))
    A_block = np.vstack(
        [np.hstack([A_obs, zeros_obs_robot]), np.hstack([zeros_robot_obs, A_robot])]
    )
    l = np.full(A_block.shape[0], -np.inf)
    u = np.concatenate([b_obs, b_robot])

    P = np.block(
        [
            [2.0 * np.eye(2), -2.0 * np.eye(2)],
            [-2.0 * np.eye(2), 2.0 * np.eye(2)],
        ]
    )

    prob = osqp.OSQP()
    prob.setup(
        P=csc_matrix(P),
        q=np.zeros(4),
        A=csc_matrix(A_block),
        l=l,
        u=u,
        verbose=False,
        polish=True,
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        res = prob.solve()
    # OSQP may report "solved inaccurate" (with a space) depending on version.
    status_raw = str(res.info.status).strip().lower()
    status_norm = status_raw.replace(" ", "_")
    if status_norm not in {"solved", "solved_inaccurate"}:
        solver_log = buffer.getvalue()
        raise RuntimeError(f"OSQP failed with status {res.info.status}\n{solver_log}")

    y = res.x
    y_obs = y[:2]
    y_robot = y[2:]
    distance = np.linalg.norm(y_robot - y_obs)
    return distance, y_obs, y_robot


def infer(state: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return distances, normals, obstacle points, and robot points for each obstacle."""
    state_array = np.asarray(list(state), dtype=float)
    if state_array.size < 2:
        raise ValueError("State must include at least x and y coordinates.")

    center = state_array[:2]
    theta = float(state_array[2]) if state_array.size >= 3 else 0.0
    A_robot, b_robot = robot_halfspace(theta, center)

    distances = []
    normals = []
    obstacle_points = []
    robot_points = []

    for (A_obs, b_obs), vertices in zip(_OBSTACLE_HALFSPACES, _OBSTACLE_VERTICES):
        try:
            dist, obs_pt, robot_pt = solve_min_distance_qp(A_obs, b_obs, A_robot, b_robot)
        except RuntimeError:
            idx = np.argmin(np.linalg.norm(vertices - center, axis=1))
            obs_pt = vertices[idx]
            robot_pt = center.copy()
            dist = np.linalg.norm(robot_pt - obs_pt)
        display_idx = np.argmin(np.linalg.norm(vertices - obs_pt, axis=1))
        display_pt = vertices[display_idx]
        dist = max(dist, 0.0)
        if dist < 1e-9:
            direction = center - vertices.mean(axis=0)
            if np.linalg.norm(direction) < 1e-9:
                direction = np.array([1.0, 0.0])
            normal = direction / np.linalg.norm(direction)
        else:
            normal = (robot_pt - obs_pt) / dist

        distances.append(dist)
        normals.append(normal)
        obstacle_points.append(display_pt)
        robot_points.append(robot_pt)

    return (
        np.asarray(distances),
        np.stack(normals, axis=1),
        np.stack(obstacle_points, axis=1),
        np.stack(robot_points, axis=1),
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: fake_infer.py x y [theta]")
    x_coord = float(sys.argv[1])
    y_coord = float(sys.argv[2])
    theta_value = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    init_model()
    distances, normals, obs_pts, robot_pts = infer((x_coord, y_coord, theta_value))
    print("distances:", distances)
    print("normals:", normals)
    print("obstacle_points:", obs_pts)
    print("robot_points:", robot_pts)
