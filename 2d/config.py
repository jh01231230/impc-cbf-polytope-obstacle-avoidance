"""Configuration for iterative convex MPC-DHOCBF experiments (Sec. V-A)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Bounds:
    """Lower/upper box bounds for state or input variables."""

    lower: np.ndarray
    upper: np.ndarray

    @staticmethod
    def from_sequences(lower: Sequence[float], upper: Sequence[float]) -> "Bounds":
        return Bounds(
            lower=np.asarray(lower, dtype=float),
            upper=np.asarray(upper, dtype=float),
        )


@dataclass(frozen=True)
class DualSlackConfig:
    """Parameters governing DHOCBF slack variables omega in the OSQP formulation."""

    # Allow very small slack to enforce near-hard distance
    omega_lower_anchor: float = 1e-4
    # Much tighter cap: probing horizons should not enter obstacles even slightly.
    omega_upper_bound: float = 1e-3
    # Stronger penalty to avoid using slack unless absolutely necessary.
    penalty_weight: float = 1000.0


@dataclass(frozen=True)
class NMPCConfig:
    """Top-level configuration for the 2D iMPC-DHOCBF controller."""

    horizon: int = 24
    # Only enforce/check safety on the first few steps.
    # Receding-horizon control will re-solve before later steps are executed, so requiring the
    # entire horizon to be collision-free can cause premature failure when only far-future
    # predicted states graze obstacles.
    probe_safety_horizon_steps: int = horizon
    dt: float = 0.1
    nsim: int = 600
    approx_radius: float = 0.1
    # Warm-start refinement iterations before entering the main simulation loop.
    warm_start_refine_iterations: int = 30
    # iMPC per-step solve attempts: keep iterating until a viable (barrier-free) solution is found,
    # but cap work per outer step to avoid worst-case slowdowns.
    max_step_iterations: int = 50
    # Enable/disable stall-triggered replanning during execution.
    # If False, the controller will keep following the original planned path.
    enable_stall_replan: bool = False
    # Unstick repulsion: when the robot is too close to an obstacle, add a temporary
    # linear cost that pushes predicted positions away from the closest obstacle.
    enable_unstick_repulsion: bool = False
    # Distance threshold (m) for optional unstick repulsion near obstacles.
    unstick_distance_threshold: float = 0.02  # meters
    unstick_weight: float = 10.0              # cost weight (higher => stronger push-off)
    unstick_horizon_steps: int = 6            # apply to first k predicted states
    unstick_decay: float = 0.8                # geometric decay across horizon steps
    # Progress reward: add a small linear term that rewards moving along the reference path
    # direction over the horizon. This helps avoid local minima where "stop" beats "move forward".
    enable_progress_reward: bool = False
    progress_weight: float = 20.0
    progress_horizon_steps: int = 15
    progress_decay: float = 0.95
    # Safety policy: if False, we will never execute an in-barrier candidate (even if it is the
    # "least violating" among infeasible inner-loop solutions). Instead we will apply a hold and
    # optionally replan.
    allow_in_barrier_fallback: bool = False
    # Clearance margin used by the linearized DHOCBF constraint inside OSQP (meters).
    # 0.0 means just avoid collision; a positive value adds clearance.
    cbf_margin_dist: float = 0.0
    # DCBF decay rate gamma in (0,1]. Rolled-out constraint: h(x_{k+1|t}) >= gamma^{k+1} h(x_t).
    # gamma = 1.0 disables decay (static h >= 0); smaller values allow h to shrink along the horizon.
    cbf_decay: float = 0.1
    # Lookahead buffer for the moving-window reference path (Sec. V-A3).
    reference_proj_dist_buffer: float = 0.03
    # Numerical tolerance for collision checks: treat penetrations smaller than this as non-colliding.
    barrier_penetration_tol: float = 0
    state_bounds: Bounds = field(
        default_factory=lambda: Bounds.from_sequences(
            # Allow small reverse speed for recovery from dead-ends (Sec. V-A1).
            lower=[-10.0, -10.0, -10.0, -10.0],
            upper=[10.0, 10.0, 10.0, 10.0],
        )
    )
    # Control bounds from Sec. V-A1: yaw-rate and acceleration in [-0.5, 0.5].
    input_bounds: Bounds = field(
        default_factory=lambda: Bounds.from_sequences(
            # Decision vector has 6 entries; dynamics use u[0]=yaw-rate and u[1]=acceleration.
            lower=[-0.5, -0.5, -np.inf, -np.inf, -np.inf, -np.inf],
            upper=[0.5, 0.5, np.inf, np.inf, np.inf, np.inf],
        )
    )
    # Tracking weights on [x, y, heading, speed].
    Q_diagonal: np.ndarray = field(
        default_factory=lambda: np.asarray([100.0, 100.0, 5.0, 20.0], dtype=float)
    )
    # Input regularization on the active control channels.
    R_diagonal: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=float
        )
    )
    reference_input: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    )
    # Desired cruising speed (m/s) for the moving-window reference (Sec. V-A3).
    reference_speed: float = 0.1

    # ------------------------------------------------------------------
    # Robot geometry (Sec. V-A2): rectangle, triangle, or L-shape footprint.
    # ------------------------------------------------------------------
    robot_shape: str = "lshape"  # "rectangle" | "triangle" | "lshape"

    # Rectangle parameters (meters). rear_dist offsets the reference point from the rear edge (+x).
    robot_rectangle_length: float = 0.15
    robot_rectangle_width: float = 0.06
    robot_rectangle_rear_dist: float = 0.10

    # Triangle vertices in the robot frame (meters), scaled by 0.75 as in Sec. V-A2.
    robot_triangle_points: np.ndarray = field(
        default_factory=lambda: 0.75
        * np.asarray(
            [
                [0.14, 0.00],
                [-0.03, 0.05],
                [-0.03, -0.05],
            ],
            dtype=float,
        )
    )

    # L-shape as the union of two convex parts (convex hull of each point set), scale 0.4.
    robot_lshape_part1_points: np.ndarray = field(
        default_factory=lambda: 0.4
        * np.asarray(
            [
                [0.00, 0.10],
                [0.02, 0.08],
                [-0.20, -0.10],
                [-0.22, -0.08],
            ],
            dtype=float,
        )
    )
    robot_lshape_part2_points: np.ndarray = field(
        default_factory=lambda: 0.4
        * np.asarray(
            [
                [0.00, 0.10],
                [-0.02, 0.08],
                [0.20, -0.10],
                [0.22, -0.08],
            ],
            dtype=float,
        )
    )

    dual_slacks: Tuple[DualSlackConfig, DualSlackConfig] = field(
        default_factory=lambda: (
            DualSlackConfig(),
            DualSlackConfig(),
        )
    )

    def Q(self) -> np.ndarray:
        return np.diag(self.Q_diagonal)

    def QN(self) -> np.ndarray:
        return np.diag(self.Q_diagonal)

    def R(self) -> np.ndarray:
        return np.diag(self.R_diagonal)


DEFAULT_CONFIG = NMPCConfig()
