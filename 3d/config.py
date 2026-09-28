"""Configuration for 3D iterative convex MPC-DHOCBF experiments (Sec. V-C)."""
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

    omega_lower_anchor: float = 1e-4
    omega_upper_bound: float = 1e-3
    penalty_weight: float = 1000.0


@dataclass(frozen=True)
class NMPCConfig:
    """Top-level configuration for the 3D iMPC-DHOCBF controller."""

    horizon: int = 8
    probe_safety_horizon_steps: int = horizon
    dt: float = 0.1
    nsim: int = 500
    approx_radius: float = 0.1
    warm_start_refine_iterations: int = 30
    max_step_iterations: int = 50
    enable_stall_replan: bool = False
    enable_unstick_repulsion: bool = False
    unstick_distance_threshold: float = 0.02
    unstick_weight: float = 10.0
    unstick_horizon_steps: int = 6
    unstick_decay: float = 0.8
    enable_progress_reward: bool = False
    progress_weight: float = 20.0
    progress_horizon_steps: int = 15
    progress_decay: float = 0.95
    allow_in_barrier_fallback: bool = False
    cbf_margin_dist: float = 0.0
    cbf_decay: float = 1.0
    reference_proj_dist_buffer: float = 0.03
    barrier_penetration_tol: float = 0.0
    # Four-element bounds reused for roll, pitch, yaw, and speed in the 7-state model.
    state_bounds: Bounds = field(
        default_factory=lambda: Bounds.from_sequences(
            lower=[-10.0, -10.0, -10.0, -10.0],
            upper=[10.0, 10.0, 10.0, 10.0],
        )
    )
    # Body rates and longitudinal acceleration (Sec. V-C).
    input_bounds: Bounds = field(
        default_factory=lambda: Bounds.from_sequences(
            lower=[-0.5, -0.5, -0.5, -0.5, -np.inf, -np.inf],
            upper=[0.5, 0.5, 0.5, 0.5, np.inf, np.inf],
        )
    )
    Q_diagonal: np.ndarray = field(
        default_factory=lambda: np.asarray([100.0, 100.0, 5.0, 20.0], dtype=float)
    )
    R_diagonal: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [0.1, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float
        )
    )
    reference_input: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    )
    reference_speed: float = 0.2

    robot_shape: str = "lshape"
    robot_rectangle_length: float = 0.15
    robot_rectangle_width: float = 0.06
    robot_rectangle_rear_dist: float = 0.10
    robot_height: float = 0.06

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
