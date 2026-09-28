"""3D L-shaped robot navigation with iterative convex MPC-DHOCBF.

Reproduces Section V-C of Liu, Huang, and Belta, arXiv:2603.05916.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Sequence, Tuple, Union

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    import matplotlib

    matplotlib.use("Agg")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import osqp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import csc_matrix, coo_matrix, diags, eye, vstack

from config import DEFAULT_CONFIG, DualSlackConfig, NMPCConfig
from env_config import ENVIRONMENT, OBSTACLE_REGIONS
from geometry3d import BoxRegion3D, HalfSpaceRegion3D, obb_intersects_aabb, obb_intersects_halfspace


def _capture_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame_rgba = buf.reshape((height, width, 4))
    return frame_rgba[:, :, :3].copy()


def _pad_frame_to_macroblock(frame: np.ndarray, macro_block_size: int = 16) -> np.ndarray:
    if macro_block_size <= 1:
        return frame
    height, width = frame.shape[:2]
    pad_h = (macro_block_size - (height % macro_block_size)) % macro_block_size
    pad_w = (macro_block_size - (width % macro_block_size)) % macro_block_size
    if pad_h == 0 and pad_w == 0:
        return frame
    new_h = height + pad_h
    new_w = width + pad_w
    padded = np.zeros((new_h, new_w, frame.shape[2]), dtype=frame.dtype)
    padded[:height, :width, :] = frame
    return padded


def _box_faces(verts: np.ndarray) -> List[np.ndarray]:
    """Return the 6 quad faces of a box from its 8 vertices.

    Vertex order follows `BoxRegion3D.vertices()`.
    """
    v = np.asarray(verts, dtype=float).reshape(8, 3)
    return [
        v[[0, 1, 2, 3]],  # bottom (z0)
        v[[4, 5, 6, 7]],  # top (z1)
        v[[0, 1, 5, 4]],  # y0 side
        v[[2, 3, 7, 6]],  # y1 side
        v[[1, 2, 6, 5]],  # x1 side
        v[[0, 3, 7, 4]],  # x0 side
    ]


@dataclass(frozen=True)
class RobotCuboidLocal:
    """One axis-aligned cuboid component in the robot local frame before body rotation."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def center_offset_local(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (float(self.x_min) + float(self.x_max)),
                0.5 * (float(self.y_min) + float(self.y_max)),
                0.5 * (float(self.z_min) + float(self.z_max)),
            ],
            dtype=float,
        )

    def half_extents(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (float(self.x_max) - float(self.x_min)),
                0.5 * (float(self.y_max) - float(self.y_min)),
                0.5 * (float(self.z_max) - float(self.z_min)),
            ],
            dtype=float,
        )

    def vertices_world(self, robot_center: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Return (8,3) vertices in world frame for this cuboid at pose (center,R)."""
        c = np.asarray(robot_center, dtype=float).reshape(3)
        Rm = np.asarray(R, dtype=float).reshape(3, 3)
        o = self.center_offset_local()
        h = self.half_extents()
        # 8 corners relative to component center in local frame
        corners = np.array(
            [
                [-h[0], -h[1], -h[2]],
                [h[0], -h[1], -h[2]],
                [h[0], h[1], -h[2]],
                [-h[0], h[1], -h[2]],
                [-h[0], -h[1], h[2]],
                [h[0], -h[1], h[2]],
                [h[0], h[1], h[2]],
                [-h[0], h[1], h[2]],
            ],
            dtype=float,
        )
        # Transform: robot_center + R*(o + corner)
        return (c[None, :] + (Rm @ (o[None, :] + corners).T).T).astype(float)

    def support_point_world(self, robot_center: np.ndarray, R: np.ndarray, direction_world: np.ndarray) -> np.ndarray:
        """Support point (argmax d^T p) on this cuboid in world direction d."""
        c = np.asarray(robot_center, dtype=float).reshape(3)
        Rm = np.asarray(R, dtype=float).reshape(3, 3)
        d = np.asarray(direction_world, dtype=float).reshape(3)
        dn = float(np.linalg.norm(d))
        if dn <= 1e-12:
            d = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            d = d / dn
        # In component local frame (robot frame)
        d_local = Rm.T @ d
        h = self.half_extents()
        o = self.center_offset_local()
        corner = np.where(d_local >= 0.0, h, -h)
        p_local = o + corner
        return (c + Rm @ p_local).astype(float)


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from roll/pitch/yaw (world z-up).

    Convention: R = Rz(yaw) @ Ry(-pitch) @ Rx(roll)
    so that pitch>0 corresponds to nose-up (positive z).
    """
    cr = math.cos(float(roll))
    sr = math.sin(float(roll))
    cp = math.cos(float(pitch))
    sp = math.sin(float(pitch))
    cy = math.cos(float(yaw))
    sy = math.sin(float(yaw))

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry_negp = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return (Rz @ Ry_negp @ Rx).astype(float)


def rpy_jacobian_times_q(roll: float, pitch: float, yaw: float, q_local: np.ndarray) -> np.ndarray:
    """Return J(q) where columns are d(R(q))/d[roll,pitch,yaw] at the given rpy.

    Uses the same convention as `rpy_to_R`: R = Rz(yaw) @ Ry(-pitch) @ Rx(roll).
    """
    q = np.asarray(q_local, dtype=float).reshape(3)
    cr = math.cos(float(roll))
    sr = math.sin(float(roll))
    cp = math.cos(float(pitch))
    sp = math.sin(float(pitch))
    cy = math.cos(float(yaw))
    sy = math.sin(float(yaw))

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    dRx = np.array([[0.0, 0.0, 0.0], [0.0, -sr, -cr], [0.0, cr, -sr]], dtype=float)

    Ry_negp = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]], dtype=float)
    # d/dpitch Ry(-pitch)
    dRy = np.array([[-sp, 0.0, -cp], [0.0, 0.0, 0.0], [cp, 0.0, -sp]], dtype=float)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    dRz = np.array([[-sy, -cy, 0.0], [cy, -sy, 0.0], [0.0, 0.0, 0.0]], dtype=float)

    dR_roll = Rz @ Ry_negp @ dRx
    dR_pitch = Rz @ dRy @ Rx
    dR_yaw = dRz @ Ry_negp @ Rx

    return np.column_stack([dR_roll @ q, dR_pitch @ q, dR_yaw @ q]).astype(float)


def velocity_to_rpy(velocity: np.ndarray, previous_rpy: np.ndarray | None = None) -> np.ndarray:
    """Compute a reasonable roll/pitch/yaw from a velocity vector.

    - yaw aligns with xy direction
    - pitch aligns with z climb angle
    - roll is set to 0 (can be extended later for banking)
    """
    v = np.asarray(velocity, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))
    if speed <= 1e-6:
        if previous_rpy is None:
            return np.zeros(3, dtype=float)
        return np.asarray(previous_rpy, dtype=float).reshape(3)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw = float(math.atan2(vy, vx))
    vxy = float(math.hypot(vx, vy))
    pitch = float(math.atan2(vz, max(vxy, 1e-12)))
    roll = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def _wrap_to_pi(angle: float) -> float:
    a = float(angle)
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _angle_diff(target: float, current: float) -> float:
    return _wrap_to_pi(float(target) - float(current))


def rate_limit_rpy(current: np.ndarray, desired: np.ndarray, omega_max: float, dt: float) -> np.ndarray:
    """Rate-limit roll/pitch/yaw changes with a symmetric bound.

    This is how we apply `input_bounds` yaw-rate to *all* rotation axes.
    """
    cur = np.asarray(current, dtype=float).reshape(3)
    des = np.asarray(desired, dtype=float).reshape(3)
    w = float(max(0.0, omega_max))
    max_delta = float(w * float(dt))
    if max_delta <= 0.0:
        return cur.copy()

    out = cur.copy()
    for i in range(3):
        d = _angle_diff(des[i], cur[i])
        d = float(np.clip(d, -max_delta, max_delta))
        out[i] = cur[i] + d
    return out


def reference_rpy_sequence(rpy0: np.ndarray, positions: np.ndarray, omega_max: float, dt: float) -> np.ndarray:
    """Predict roll/pitch/yaw along a horizon by turning toward segment directions."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    rpy = np.asarray(rpy0, dtype=float).reshape(3)
    seq = np.zeros((pos.shape[0], 3), dtype=float)
    seq[0] = rpy
    for k in range(pos.shape[0] - 1):
        d = pos[k + 1] - pos[k]
        rpy_des = velocity_to_rpy(d, previous_rpy=rpy)
        rpy = rate_limit_rpy(rpy, rpy_des, omega_max=omega_max, dt=dt)
        seq[k + 1] = rpy
    return seq


def control_rpy_sequence(
    rpy0: np.ndarray,
    u_traj: np.ndarray,
    x_ref: np.ndarray,
    omega_max: float,
    dt: float,
    speed_eps: float = 1e-4,
) -> np.ndarray:
    """Predict roll/pitch/yaw along a horizon given translational controls.

    When speed is tiny, fall back to the reference direction so we can still turn-in-place.
    """
    rpy = np.asarray(rpy0, dtype=float).reshape(3)
    u = np.asarray(u_traj, dtype=float).reshape(-1, 3)
    xr = np.asarray(x_ref, dtype=float).reshape(-1, 3)
    N = int(u.shape[0])
    seq = np.zeros((N + 1, 3), dtype=float)
    seq[0] = rpy
    for k in range(N):
        v = u[k]
        if float(np.linalg.norm(v)) > float(speed_eps):
            rpy_des = velocity_to_rpy(v, previous_rpy=rpy)
        else:
            d = xr[k + 1] - xr[k] if (k + 1) < xr.shape[0] else np.zeros(3, dtype=float)
            rpy_des = velocity_to_rpy(d, previous_rpy=rpy)
        rpy = rate_limit_rpy(rpy, rpy_des, omega_max=omega_max, dt=dt)
        seq[k + 1] = rpy
    return seq


def build_robot_cuboids(cfg: NMPCConfig) -> List[RobotCuboidLocal]:
    """Build a simple 3D robot body from cuboids.

    Starts with an **L-shape** made from two cuboids, as requested.
    """
    shape = str(getattr(cfg, "robot_shape", "lshape")).strip().lower()
    length = float(getattr(cfg, "robot_rectangle_length", 0.15))
    width = float(getattr(cfg, "robot_rectangle_width", 0.06))
    height = float(getattr(cfg, "robot_height", width))
    height = float(max(1e-6, height))

    z0 = -0.5 * height
    z1 = 0.5 * height

    if shape == "rectangle":
        hx = 0.5 * length
        hy = 0.5 * width
        return [RobotCuboidLocal(-hx, hx, -hy, hy, z0, z1)]

    if shape != "lshape":
        # Fall back to a rectangle prism if an unsupported shape is requested.
        hx = 0.5 * length
        hy = 0.5 * width
        return [RobotCuboidLocal(-hx, hx, -hy, hy, z0, z1)]

    # L-shape = union of two cuboids (arms) in the xy-plane, extruded in z.
    # Start with an "inner corner" at (0,0) and arms extending +x and +y:
    #   Arm X: x∈[0,L], y∈[0,t]
    #   Arm Y: x∈[0,t], y∈[0,L]
    # Then shift both so the union centroid is at the origin (so state x,y,z is centered).
    # L-shape arms use 2x the nominal length on both sides (Sec. V-C).
    L = float(max(1e-6, 2.0 * length))
    t = float(max(1e-6, width))
    denom = max(1e-9, 2.0 * L - t)
    c = 0.5 * (L * L + t * L - t * t) / denom

    arm_x = RobotCuboidLocal(x_min=0.0 - c, x_max=L - c, y_min=0.0 - c, y_max=t - c, z_min=z0, z_max=z1)
    arm_y = RobotCuboidLocal(x_min=0.0 - c, x_max=t - c, y_min=0.0 - c, y_max=L - c, z_min=z0, z_max=z1)
    return [arm_x, arm_y]


def robot_union_local_bounds(components: Sequence[RobotCuboidLocal]) -> Tuple[np.ndarray, np.ndarray]:
    comps = list(components)
    if not comps:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    mins = np.array([min(c.x_min for c in comps), min(c.y_min for c in comps), min(c.z_min for c in comps)], dtype=float)
    maxs = np.array([max(c.x_max for c in comps), max(c.y_max for c in comps), max(c.z_max for c in comps)], dtype=float)
    return mins, maxs


def robot_bounding_radius(components: Sequence[RobotCuboidLocal]) -> float:
    """Conservative radius of the robot body around its center (orientation-independent)."""
    r2 = 0.0
    for comp in components:
        # compute the furthest corner in local coordinates from the robot center (origin)
        xs = [float(comp.x_min), float(comp.x_max)]
        ys = [float(comp.y_min), float(comp.y_max)]
        zs = [float(comp.z_min), float(comp.z_max)]
        for x in xs:
            for y in ys:
                for z in zs:
                    r2 = max(r2, float(x * x + y * y + z * z))
    return float(math.sqrt(r2))


def _polyline_arclength(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return np.zeros((pts.shape[0],), dtype=float), np.zeros((0,), dtype=float)
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    return s, seg_len


def _interpolate_polyline(path: np.ndarray, s_query: np.ndarray, s_path: np.ndarray, seg_len: np.ndarray) -> np.ndarray:
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((len(s_query), 3), dtype=float)
    if pts.shape[0] == 1:
        return np.repeat(pts[:1], repeats=len(s_query), axis=0)

    s_q = np.asarray(s_query, dtype=float).reshape(-1)
    s_q = np.clip(s_q, s_path[0], s_path[-1])
    out = np.zeros((s_q.shape[0], 3), dtype=float)

    # For each query, locate segment index i s.t. s_path[i] <= s < s_path[i+1]
    idx = np.searchsorted(s_path, s_q, side="right") - 1
    idx = np.clip(idx, 0, pts.shape[0] - 2)
    s0 = s_path[idx]
    s1 = s_path[idx + 1]
    denom = np.maximum(s1 - s0, 1e-12)
    t = (s_q - s0) / denom
    out = pts[idx] * (1.0 - t)[:, None] + pts[idx + 1] * t[:, None]
    return out


def _project_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    p = np.asarray(p, dtype=float).reshape(3)
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy(), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    return a + t * ab, t


def _closest_s_on_polyline(p: np.ndarray, path: np.ndarray, s_path: np.ndarray) -> float:
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return 0.0
    best_s = 0.0
    best_d2 = float("inf")
    for i in range(pts.shape[0] - 1):
        q, t = _project_point_to_segment(p, pts[i], pts[i + 1])
        d2 = float(np.sum((np.asarray(p, dtype=float).reshape(3) - q) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(s_path[i] + t * (s_path[i + 1] - s_path[i]))
    return best_s


def in_barrier_3d(
    center: np.ndarray,
    obstacles: Sequence[Union[BoxRegion3D, HalfSpaceRegion3D]],
    robot_components: Sequence[RobotCuboidLocal],
    R: np.ndarray,
    penetration_tol: float,
) -> bool:
    tol = float(max(0.0, penetration_tol))
    c = np.asarray(center, dtype=float).reshape(3)
    for obs in obstacles:
        for comp in robot_components:
            # Each component is an oriented cuboid with the same R.
            obb_c = c + np.asarray(R, dtype=float).reshape(3, 3) @ comp.center_offset_local()
            if isinstance(obs, HalfSpaceRegion3D):
                if obb_intersects_halfspace(
                    obb_center=obb_c,
                    obb_R=R,
                    obb_half=comp.half_extents(),
                    n=obs.n,
                    d=obs.d,
                    penetration_tol=tol,
                    x_lo=getattr(obs, "x_lo", -float("inf")),
                    x_hi=getattr(obs, "x_hi", float("inf")),
                ):
                    return True
            else:
                if obb_intersects_aabb(
                    obb_center=obb_c,
                    obb_R=R,
                    obb_half=comp.half_extents(),
                    aabb=obs,
                    penetration_tol=tol,
                ):
                    return True
    return False


@dataclass(frozen=True)
class _Layout:
    """Index bookkeeping for the flattened OSQP decision vector."""

    N: int
    nx: int
    nu: int
    k_cbf: int
    m: int
    n_components: int
    x0: int
    u0: int
    w0: int
    dim: int

    def idx_x(self, k: int, d: int) -> int:
        return self.x0 + k * self.nx + d

    def idx_u(self, k: int, d: int) -> int:
        return self.u0 + k * self.nu + d

    def idx_w(self, obs_idx: int, comp_idx: int, step: int) -> int:
        # step is 1..k_cbf (CBF enforced on steps 1..k_cbf)
        block = (obs_idx * self.n_components + comp_idx) * self.k_cbf
        return self.w0 + block + (step - 1)


def _build_layout(N: int, nx: int, nu: int, k_cbf: int, m: int, n_components: int) -> _Layout:
    x0 = 0
    u0 = (N + 1) * nx
    w0 = u0 + N * nu
    dim = w0 + m * n_components * k_cbf
    return _Layout(
        N=N,
        nx=nx,
        nu=nu,
        k_cbf=k_cbf,
        m=m,
        n_components=n_components,
        x0=x0,
        u0=u0,
        w0=w0,
        dim=dim,
    )


def _slack_per_obstacle(cfg: NMPCConfig, m: int) -> List[DualSlackConfig]:
    slacks = list(getattr(cfg, "dual_slacks", (DualSlackConfig(),)))
    if not slacks:
        slacks = [DualSlackConfig()]
    if len(slacks) < m:
        slacks.extend([slacks[-1]] * (m - len(slacks)))
    return slacks[:m]


def _step_dynamics_3d(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """Discrete-time 3D kinematics with roll/pitch/yaw and forward speed.

    state  = [x, y, z, roll, pitch, yaw, v]
    control= [w_roll, w_pitch, w_yaw, a]
    """
    s = np.asarray(state, dtype=float).reshape(7)
    u = np.asarray(control, dtype=float).reshape(4)
    x, y, z, roll, pitch, yaw, v = (float(s[i]) for i in range(7))
    w_roll, w_pitch, w_yaw, a = (float(u[i]) for i in range(4))
    dt = float(dt)

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    x_next = x + dt * v * cy * cp
    y_next = y + dt * v * sy * cp
    z_next = z + dt * v * sp
    roll_next = roll + dt * w_roll
    pitch_next = pitch + dt * w_pitch
    yaw_next = yaw + dt * w_yaw
    v_next = v + dt * a
    return np.array([x_next, y_next, z_next, roll_next, pitch_next, yaw_next, v_next], dtype=float)


def _linearize_dynamics_3d(state_star: np.ndarray, control_star: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, B, c) for the affine approximation:

      x_{k+1} - A x_k - B u_k = c

    about (state_star, control_star).
    """
    s = np.asarray(state_star, dtype=float).reshape(7)
    u = np.asarray(control_star, dtype=float).reshape(4)
    _ = u
    dt = float(dt)

    pitch = float(s[4])
    yaw = float(s[5])
    v = float(s[6])

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    A = np.eye(7, dtype=float)
    # Position derivatives w.r.t pitch, yaw, v.
    A[0, 4] = -dt * v * cy * sp
    A[0, 5] = -dt * v * sy * cp
    A[0, 6] = dt * cy * cp

    A[1, 4] = -dt * v * sy * sp
    A[1, 5] = dt * v * cy * cp
    A[1, 6] = dt * sy * cp

    A[2, 4] = dt * v * cp
    A[2, 6] = dt * sp

    B = np.zeros((7, 4), dtype=float)
    B[3, 0] = dt  # roll rate
    B[4, 1] = dt  # pitch rate
    B[5, 2] = dt  # yaw rate
    B[6, 3] = dt  # accel

    x_next_star = _step_dynamics_3d(s, u, dt)
    c = x_next_star - (A @ s) - (B @ u)
    return A, B, c


def _solve_mpc_cbf_osqp(
    *,
    state_current: np.ndarray,
    state_ref: np.ndarray,
    state_guess: np.ndarray,
    obstacles: Sequence[Union[BoxRegion3D, HalfSpaceRegion3D]],
    robot_components: Sequence[RobotCuboidLocal],
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    cfg: NMPCConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """Solve one MPC QP with linearized dynamics + CBF constraints.

    State:   [x, y, z, roll, pitch, yaw, v]
    Control: [w_roll, w_pitch, w_yaw, a]
    Returns (state_pred, control_pred, omega, status, solve_time).
    """
    N = int(cfg.horizon)
    dt = float(cfg.dt)
    m = int(len(obstacles))
    n_components = int(len(robot_components))
    k_cbf = int(min(N, max(1, int(getattr(cfg, "probe_safety_horizon_steps", N)))))
    cbf_margin = float(getattr(cfg, "cbf_margin_dist", 0.0))
    cbf_decay_gamma = float(getattr(cfg, "cbf_decay", 1.0))
    nx = 7
    nu = 4
    layout = _build_layout(N, nx, nu, k_cbf, m, n_components)

    s_cur = np.asarray(state_current, dtype=float).reshape(nx)
    sr = np.asarray(state_ref, dtype=float).reshape(N + 1, nx)
    sg = np.asarray(state_guess, dtype=float).reshape(N + 1, nx)

    # Precompute h_0 at current state for rolled-out DCBF decay: h(x_{k+1|t}) >= gamma^{k+1} * h(x_t)
    h0_map: dict = {}
    if cbf_decay_gamma < 1.0:
        R_cur = rpy_to_R(float(s_cur[3]), float(s_cur[4]), float(s_cur[5]))
        for comp_idx, comp in enumerate(robot_components):
            comp_center = s_cur[:3] + R_cur @ comp.center_offset_local()
            for obs_idx, obs in enumerate(obstacles):
                if isinstance(obs, HalfSpaceRegion3D):
                    x_lo = float(getattr(obs, "x_lo", -float("inf")))
                    x_hi = float(getattr(obs, "x_hi", float("inf")))
                    if comp_center[0] < x_lo - 1e-6 or comp_center[0] > x_hi + 1e-6:
                        h0_map[(obs_idx, comp_idx)] = 1e6  # outside active region: treat as far
                        continue
                _q0, _n0, dist0, _inside0 = obs.closest_point_and_normal(comp_center)
                h0_map[(obs_idx, comp_idx)] = float(dist0) - cbf_margin

    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    radius = robot_bounding_radius(robot_components)
    center_min = np.array([x_min + radius, y_min + radius, z_min + radius], dtype=float)
    center_max = np.array([x_max - radius, y_max - radius, z_max - radius], dtype=float)

    # -------------------------
    # Objective (diagonal P)
    # -------------------------
    Qd = np.asarray(getattr(cfg, "Q_diagonal", DEFAULT_CONFIG.Q_diagonal), dtype=float).reshape(-1)
    q_xyz = np.array([float(Qd[0]), float(Qd[1]), float(Qd[0])], dtype=float)
    q_ang = float(Qd[2]) if Qd.size >= 3 else 1.0
    q_v = float(Qd[3]) if Qd.size >= 4 else 1.0
    q_state = np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_ang, q_ang, q_ang, q_v], dtype=float)
    Rd = np.asarray(getattr(cfg, "R_diagonal", DEFAULT_CONFIG.R_diagonal), dtype=float).reshape(-1)
    r_w = float(Rd[0])  # yaw-rate weight -> use for all angular rates
    r_a = float(Rd[1]) if Rd.size >= 2 else float(Rd[0])
    r_control = np.array([r_w, r_w, r_w, r_a], dtype=float)

    P_diag = np.zeros(layout.dim, dtype=float)
    q_vec = np.zeros(layout.dim, dtype=float)

    # State tracking
    for k in range(N + 1):
        for d in range(nx):
            idx = layout.idx_x(k, d)
            w = float(q_state[d])
            P_diag[idx] = 2.0 * w
            q_vec[idx] = -2.0 * w * float(sr[k, d])

    # Control regularization
    for k in range(N):
        for d in range(nu):
            idx = layout.idx_u(k, d)
            w = float(r_control[d])
            P_diag[idx] = 2.0 * w
            q_vec[idx] = 0.0

    # Slack penalties (per obstacle)
    slack_cfgs = _slack_per_obstacle(cfg, m)
    for obs_idx, slack in enumerate(slack_cfgs):
        penalty = float(max(0.0, slack.penalty_weight))
        omega_anchor = float(max(0.0, slack.omega_lower_anchor))
        for comp_idx in range(n_components):
            for step in range(1, k_cbf + 1):
                idx = layout.idx_w(obs_idx, comp_idx, step)
                P_diag[idx] = 2.0 * penalty
                q_vec[idx] = -2.0 * penalty * omega_anchor

    P = diags(P_diag, offsets=0, shape=(layout.dim, layout.dim), format="csc")

    # -------------------------
    # Equality constraints
    # -------------------------
    # state_0 == state_current
    # state_{k+1} - A_k state_k - B_k u_k == c_k  (linearized dynamics)
    eq_rows: List[int] = []
    eq_cols: List[int] = []
    eq_data: List[float] = []
    beq: List[float] = []
    row = 0
    for d in range(nx):
        eq_rows.append(row)
        eq_cols.append(layout.idx_x(0, d))
        eq_data.append(1.0)
        beq.append(float(s_cur[d]))
        row += 1

    for k in range(N):
        # Build a control expansion point from the guess (finite differences).
        roll0, pitch0, yaw0, v0 = (float(sg[k, 3]), float(sg[k, 4]), float(sg[k, 5]), float(sg[k, 6]))
        roll1, pitch1, yaw1, v1 = (float(sg[k + 1, 3]), float(sg[k + 1, 4]), float(sg[k + 1, 5]), float(sg[k + 1, 6]))
        u_star = np.array(
            [
                (roll1 - roll0) / max(dt, 1e-9),
                (pitch1 - pitch0) / max(dt, 1e-9),
                (yaw1 - yaw0) / max(dt, 1e-9),
                (v1 - v0) / max(dt, 1e-9),
            ],
            dtype=float,
        )
        A_k, B_k, c_k = _linearize_dynamics_3d(sg[k], u_star, dt)

        for i in range(nx):
            # x_{k+1,i}
            eq_rows.append(row)
            eq_cols.append(layout.idx_x(k + 1, i))
            eq_data.append(1.0)
            # -A_k[i,:] * x_k
            for j in range(nx):
                eq_rows.append(row)
                eq_cols.append(layout.idx_x(k, j))
                eq_data.append(-float(A_k[i, j]))
            # -B_k[i,:] * u_k
            for j in range(nu):
                eq_rows.append(row)
                eq_cols.append(layout.idx_u(k, j))
                eq_data.append(-float(B_k[i, j]))
            beq.append(float(c_k[i]))
            row += 1

    Aeq = coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(row, layout.dim)).tocsc()
    leq = np.asarray(beq, dtype=float)
    ueq = leq.copy()

    # -------------------------
    # Variable bounds (identity)
    # -------------------------
    Abounds = eye(layout.dim, format="csc")
    lb = np.full(layout.dim, -np.inf, dtype=float)
    ub = np.full(layout.dim, np.inf, dtype=float)

    # State bounds: constrain center position so the full body stays within env bounds.
    theta_lo = float(cfg.state_bounds.lower[2]) if getattr(cfg, "state_bounds", None) is not None else -10.0
    theta_hi = float(cfg.state_bounds.upper[2]) if getattr(cfg, "state_bounds", None) is not None else 10.0
    v_lo = float(cfg.state_bounds.lower[3]) if getattr(cfg, "state_bounds", None) is not None else -10.0
    v_hi = float(cfg.state_bounds.upper[3]) if getattr(cfg, "state_bounds", None) is not None else 10.0
    for k in range(N + 1):
        lb[layout.idx_x(k, 0)] = float(center_min[0])
        ub[layout.idx_x(k, 0)] = float(center_max[0])
        lb[layout.idx_x(k, 1)] = float(center_min[1])
        ub[layout.idx_x(k, 1)] = float(center_max[1])
        lb[layout.idx_x(k, 2)] = float(center_min[2])
        ub[layout.idx_x(k, 2)] = float(center_max[2])
        # roll/pitch/yaw
        lb[layout.idx_x(k, 3)] = theta_lo
        ub[layout.idx_x(k, 3)] = theta_hi
        lb[layout.idx_x(k, 4)] = theta_lo
        ub[layout.idx_x(k, 4)] = theta_hi
        lb[layout.idx_x(k, 5)] = theta_lo
        ub[layout.idx_x(k, 5)] = theta_hi
        # speed
        lb[layout.idx_x(k, 6)] = v_lo
        ub[layout.idx_x(k, 6)] = v_hi

    # Control bounds: apply yaw-rate bound to roll/pitch/yaw rates; accel bound to a.
    umin = np.asarray(cfg.input_bounds.lower, dtype=float).reshape(-1)
    umax = np.asarray(cfg.input_bounds.upper, dtype=float).reshape(-1)
    yaw_lo = float(umin[0])
    yaw_hi = float(umax[0])
    # Optional: allow separate pitch/roll rate bounds using spare input slots if finite.
    pitch_lo, pitch_hi = yaw_lo, yaw_hi
    roll_lo, roll_hi = yaw_lo, yaw_hi
    if umin.size >= 4 and umax.size >= 4:
        if np.isfinite(umin[2]) and np.isfinite(umax[2]):
            pitch_lo, pitch_hi = float(umin[2]), float(umax[2])
        if np.isfinite(umin[3]) and np.isfinite(umax[3]):
            roll_lo, roll_hi = float(umin[3]), float(umax[3])
    accel_lo = float(umin[1]) if umin.size >= 2 else float(umin[0])
    accel_hi = float(umax[1]) if umax.size >= 2 else float(umax[0])
    for k in range(N):
        lb[layout.idx_u(k, 0)] = roll_lo
        ub[layout.idx_u(k, 0)] = roll_hi
        lb[layout.idx_u(k, 1)] = pitch_lo
        ub[layout.idx_u(k, 1)] = pitch_hi
        lb[layout.idx_u(k, 2)] = yaw_lo
        ub[layout.idx_u(k, 2)] = yaw_hi
        lb[layout.idx_u(k, 3)] = accel_lo
        ub[layout.idx_u(k, 3)] = accel_hi

    # Slack bounds
    for obs_idx, slack in enumerate(slack_cfgs):
        omega_lower = float(max(0.0, slack.omega_lower_anchor))
        omega_upper = float(max(omega_lower, float(slack.omega_upper_bound)))
        for comp_idx in range(n_components):
            for step in range(1, k_cbf + 1):
                idx = layout.idx_w(obs_idx, comp_idx, step)
                lb[idx] = omega_lower
                ub[idx] = omega_upper

    # -------------------------
    # Linearized CBF constraints
    # -------------------------
    cbf_rows: List[int] = []
    cbf_cols: List[int] = []
    cbf_data: List[float] = []
    l_cbf: List[float] = []
    u_cbf: List[float] = []
    row_cbf = 0
    for obs_idx, obs in enumerate(obstacles):
        for comp_idx, comp in enumerate(robot_components):
            for step in range(1, k_cbf + 1):
                # Linearize around guess pose at this step.
                p_star = np.asarray(sg[step, :3], dtype=float).reshape(3)
                rpy_star = np.asarray(sg[step, 3:6], dtype=float).reshape(3)
                R_star = rpy_to_R(float(rpy_star[0]), float(rpy_star[1]), float(rpy_star[2]))

                # Slope-plane tunnel halfspaces are only active on their x-segment.
                # Outside [x_lo, x_hi], skip this CBF row to avoid over-constraining the QP.
                if isinstance(obs, HalfSpaceRegion3D):
                    x_lo = float(getattr(obs, "x_lo", -float("inf")))
                    x_hi = float(getattr(obs, "x_hi", float("inf")))
                    if p_star[0] < x_lo - 1e-6 or p_star[0] > x_hi + 1e-6:
                        continue

                # Initial normal from obstacle to component center.
                comp_center = p_star + R_star @ comp.center_offset_local()
                _q0, n0, _dist0, _inside0 = obs.closest_point_and_normal(comp_center)
                n = np.asarray(n0, dtype=float).reshape(3)
                nn = float(np.linalg.norm(n))
                if nn <= 1e-12:
                    n = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    n = n / nn

                # Fixed-point refine (2 iterations): support point on robot, then closest point on obstacle.
                q_local = comp.center_offset_local()
                q_obs = obs.closest_point(p_star)
                for _ in range(2):
                    # Contact on robot component: support in direction -n.
                    d_local = R_star.T @ (-n)
                    h = comp.half_extents()
                    o = comp.center_offset_local()
                    corner = np.where(d_local >= 0.0, h, -h)
                    q_local = o + corner
                    p_robot = p_star + R_star @ q_local
                    q_obs = obs.closest_point(p_robot)
                    diff = p_robot - q_obs
                    dist = float(np.linalg.norm(diff))
                    if dist > 1e-9:
                        n = diff / dist
                    else:
                        _q_tmp, n_ref, _d2, _inside2 = obs.closest_point_and_normal(p_robot)
                        n = np.asarray(n_ref, dtype=float).reshape(3)
                        nn = float(np.linalg.norm(n))
                        if nn <= 1e-12:
                            n = np.array([1.0, 0.0, 0.0], dtype=float)
                        else:
                            n = n / nn

                # Freeze the body-fixed contact point at the guess (world vector from center).
                Jq = rpy_jacobian_times_q(float(rpy_star[0]), float(rpy_star[1]), float(rpy_star[2]), q_local)
                a = (Jq.T @ n).reshape(3)
                Rq = R_star @ q_local
                rhs = float(np.dot(n, q_obs) + cbf_margin - float(np.dot(n, Rq)) + float(np.dot(a, rpy_star)))
                # Rolled-out DCBF decay: h_k >= gamma^k * h_0
                if cbf_decay_gamma < 1.0:
                    h0_val = h0_map.get((obs_idx, comp_idx), 0.0)
                    if h0_val > 0.0:
                        rhs += (cbf_decay_gamma ** (step + 1)) * h0_val

                # Row: n^T p + a_roll*roll + a_pitch*pitch + a_yaw*yaw + omega >= rhs
                for d in range(3):
                    cbf_rows.append(row_cbf)
                    cbf_cols.append(layout.idx_x(step, d))
                    cbf_data.append(float(n[d]))
                for j in range(3):
                    cbf_rows.append(row_cbf)
                    cbf_cols.append(layout.idx_x(step, 3 + j))
                    cbf_data.append(float(a[j]))

                w_idx = layout.idx_w(obs_idx, comp_idx, step)
                cbf_rows.append(row_cbf)
                cbf_cols.append(w_idx)
                cbf_data.append(1.0)

                l_cbf.append(rhs)
                u_cbf.append(np.inf)
                row_cbf += 1

    Acbf = coo_matrix((cbf_data, (cbf_rows, cbf_cols)), shape=(row_cbf, layout.dim)).tocsc()

    A = vstack([Aeq, Abounds, Acbf], format="csc")
    l = np.concatenate([leq, lb, np.asarray(l_cbf, dtype=float)])
    u = np.concatenate([ueq, ub, np.asarray(u_cbf, dtype=float)])

    prob = osqp.OSQP()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        prob.setup(
            P=P,
            q=q_vec,
            A=A,
            l=l,
            u=u,
            verbose=False,
            polish=True,
            warm_start=True,
        )
        # Warm start from guess
        warm = np.zeros(layout.dim, dtype=float)
        warm[: (N + 1) * nx] = sg.reshape(-1)
        # Finite-difference guess for control from angle/speed differences.
        u_guess = np.zeros((N, nu), dtype=float)
        u_guess[:, 0] = (sg[1:, 3] - sg[:-1, 3]) / max(dt, 1e-9)
        u_guess[:, 1] = (sg[1:, 4] - sg[:-1, 4]) / max(dt, 1e-9)
        u_guess[:, 2] = (sg[1:, 5] - sg[:-1, 5]) / max(dt, 1e-9)
        u_guess[:, 3] = (sg[1:, 6] - sg[:-1, 6]) / max(dt, 1e-9)
        warm[layout.u0 : layout.u0 + N * nu] = u_guess.reshape(-1)
        # Slacks warm-start at lower anchor
        for obs_idx, slack in enumerate(slack_cfgs):
            omega_lower = float(max(0.0, slack.omega_lower_anchor))
            for comp_idx in range(n_components):
                for step in range(1, k_cbf + 1):
                    warm[layout.idx_w(obs_idx, comp_idx, step)] = omega_lower
        prob.warm_start(x=warm)
        res = prob.solve()

    # run_time = setup+solve+polish (preferred); solve_time = QP solve only (fallback when run_time is 0)
    solve_time = float(getattr(res.info, "run_time", 0.0) or getattr(res.info, "solve_time", 0.0) or 0.0)
    status = str(res.info.status).strip()
    status_norm = status.lower().replace(" ", "_")
    if status_norm not in {"solved", "solved_inaccurate"}:
        omega_fallback = np.zeros((m, n_components, k_cbf), dtype=float)
        return sg.copy(), np.zeros((N, nu), dtype=float), omega_fallback, status, solve_time

    z = np.asarray(res.x, dtype=float).reshape(-1)
    state_pred = z[: (N + 1) * nx].reshape(N + 1, nx)
    control_pred = z[layout.u0 : layout.u0 + N * nu].reshape(N, nu)
    omega_flat = z[layout.w0 :]
    omega = (
        omega_flat.reshape(m, n_components, k_cbf)
        if omega_flat.size
        else np.zeros((m, n_components, k_cbf), dtype=float)
    )
    return state_pred, control_pred, omega, status, solve_time


def _save_sample_plot_3d(
    *,
    impctra: np.ndarray,
    path_points: np.ndarray,
    xr_history: List[np.ndarray],
    x_pred_all: np.ndarray,
    obstacles: List[Union[BoxRegion3D, HalfSpaceRegion3D]],
    robot_components: List[RobotCuboidLocal],
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    state_history: List[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    filename: str = "sample_plot_3d_L.png",
    dpi: int = 300,
) -> None:
    """Generate a static PNG with 4 perspective views, obstacles, guiding path, and up to 12 samples."""
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    dx, dy, dz = float(x_max - x_min), float(y_max - y_min), float(z_max - z_min)
    n_pts = impctra.shape[0]
    n_samples = min(12, max(1, n_pts))
    sample_indices = np.linspace(0, n_pts - 1, n_samples, dtype=int)

    obstacle_faces = [_box_faces(obs.vertices()) for obs in obstacles]

    view_specs = [
        ("Front", dict(elev=0, azim=-90), "y"),  # y into screen
        ("Side", dict(elev=0, azim=0), "x"),    # x into screen
        ("Top", dict(elev=90, azim=-90), "z"),  # z into screen
        ("Tilt", dict(elev=35, azim=-45), "z"), # z foreshortened in tilted view
    ]

    fig = plt.figure(figsize=(16, 14), dpi=dpi)
    axs = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]

    for ax_idx, (ax, (_title, view, hide_axis)) in enumerate(zip(axs, view_specs)):
        ax.view_init(**view)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        try:
            ax.set_box_aspect((dx, dy, dz))
        except Exception:
            pass
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
        if hide_axis == "x":
            ax.set_xticks([])
        elif hide_axis == "y":
            ax.set_yticks([])
        elif hide_axis == "z":
            ax.set_zticks([])
        ax.tick_params(axis='both', labelsize=14)
        labelpad = 30
        ax.set_xlabel("" if ax_idx == 1 else "x", fontsize=18, labelpad=labelpad)
        ax.set_ylabel("y", fontsize=18, labelpad=labelpad)
        ax.set_zlabel("z", fontsize=18, labelpad=labelpad)

        # Add axis labels at user-marked locations (right of axis / upper-left)
        pad = 0.06 * max(dx, dy, dz)
        if ax_idx == 0:
            # Top-left (Front view): 'z' to the right of the vertical z axis
            ax.set_xlim(x_min, x_max + pad)
            ax.text(x_max - 0.5 * pad + 0.6, (y_min + y_max) / 2, (z_min + z_max) / 2, "z", fontsize=18, color="black")
        elif ax_idx == 2:
            # Bottom-left (Top view): 'y' to the right of the vertical y axis
            ax.set_xlim(x_min, x_max + pad)
            ax.text(x_max - 0.5 * pad + 0.6, (y_min + y_max) / 2, z_min, "y", fontsize=18, color="black")
        elif ax_idx == 3:
            # Bottom-right (Tilt view): 'z' in upper-left near top of z axis
            ax.set_zlim(z_min, z_max + pad)
            ax.text(x_min, y_max, z_max - 0.5 * pad + 0.6, "z", fontsize=18, color="black")

        for faces in obstacle_faces:
            coll = Poly3DCollection(
                faces,
                facecolors=(0.8, 0.1, 0.1, 0.18),
                edgecolors=(0.0, 0.0, 0.0, 0.25),
                linewidths=0.4,
            )
            ax.add_collection3d(coll)

        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], "--", color="grey", linewidth=1.2)
        ax.scatter([start[0]], [start[1]], [start[2]], color="green", s=60, edgecolors="black")
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="red", s=80, marker="*", edgecolors="black")

        for idx in sample_indices:
            if idx >= len(state_history) or idx >= impctra.shape[0]:
                continue
            pos = impctra[idx]
            st = state_history[idx]
            R = rpy_to_R(float(st[3]), float(st[4]), float(st[5]))
            for comp in robot_components:
                faces = _box_faces(comp.vertices_world(pos, R))
                coll = Poly3DCollection(
                    faces,
                    facecolors=(0.1, 0.35, 0.9, 0.35),
                    edgecolors=(0.0, 0.0, 0.0, 0.45),
                    linewidths=0.6,
                )
                ax.add_collection3d(coll)
            xr_idx = min(idx, len(xr_history) - 1)
            if xr_idx >= 0:
                xr = xr_history[xr_idx]
                ax.plot(xr[:, 0], xr[:, 1], xr[:, 2], color="orange", linewidth=2.0, alpha=0.8)
            pred_idx = min(idx, x_pred_all.shape[0] - 1)
            if pred_idx >= 0:
                pred = x_pred_all[pred_idx]
                pred_xyz = pred[1:, :3] if pred.shape[0] > 1 else np.zeros((0, 3))
                if pred_xyz.shape[0] > 0:
                    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color="blue", linewidth=1.2, linestyle="--", alpha=0.9)

        ax.plot(impctra[:, 0], impctra[:, 1], impctra[:, 2], color="dodgerblue", linewidth=2.0)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_3d(
    config: NMPCConfig | None = None,
    *,
    render: bool = True,
    nsim_override: int | None = None,
    headless: bool = False,
    start: np.ndarray | None = None,
    goal: np.ndarray | None = None,
    obstacles: List[Union[BoxRegion3D, HalfSpaceRegion3D]] | None = None,
    use_straight_path: bool = False,
    path_points_override: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[np.ndarray]]:
    cfg = config or DEFAULT_CONFIG
    env = ENVIRONMENT
    _obstacles = obstacles if obstacles is not None else list(OBSTACLE_REGIONS)
    obstacles = list(_obstacles)

    robot_components = build_robot_cuboids(cfg)

    start = np.asarray(start if start is not None else env.start, dtype=float).reshape(3)
    goal = np.asarray(goal if goal is not None else env.goal, dtype=float).reshape(3)
    bounds = env.bounds

    N = int(cfg.horizon)
    dt = float(cfg.dt)
    nsim = int(nsim_override) if nsim_override is not None else int(cfg.nsim)
    approx_radius = float(cfg.approx_radius)
    k_cbf = int(min(N, max(1, int(getattr(cfg, "probe_safety_horizon_steps", N)))))
    penetration_tol = float(getattr(cfg, "barrier_penetration_tol", 0.0))
    ref_speed = float(getattr(cfg, "reference_speed", 0.2))

    # Use the environment's built-in guiding waypoints (holes) as the global path.
    # When path_points_override is provided, use it. Else when use_straight_path, use straight line start->goal.
    if path_points_override is not None and path_points_override.size >= 2:
        path_points = np.asarray(path_points_override, dtype=float).reshape(-1, 3)
    elif use_straight_path:
        path_points = np.vstack([start, goal]).astype(float)
    else:
        path_points = np.asarray(getattr(env, "guiding_waypoints", np.vstack([start, goal])), dtype=float).reshape(-1, 3)
    s_path, seg_len = _polyline_arclength(path_points)
    total_len = float(s_path[-1]) if s_path.size else 0.0
    # Initial orientation from the first path segment.
    rpy0 = velocity_to_rpy(
        path_points[1] - path_points[0] if path_points.shape[0] > 1 else np.array([1.0, 0.0, 0.0], dtype=float)
    )

    # -------------------------
    # 3D multi-view visualization (optional)
    # -------------------------
    if headless:
        render = False
    interactive_backend = False
    fig = None
    axs = []
    view_specs = []
    if render:
        interactive_backend = "agg" not in plt.get_backend().lower()
        if interactive_backend:
            plt.ion()

        fig = plt.figure(figsize=(16, 14), dpi=120)
        axs = [
            fig.add_subplot(2, 2, 1, projection="3d"),
            fig.add_subplot(2, 2, 2, projection="3d"),
            fig.add_subplot(2, 2, 3, projection="3d"),
            fig.add_subplot(2, 2, 4, projection="3d"),
        ]
        view_specs = [
            ("Front", dict(elev=0, azim=-90), "y"),   # y into screen
            ("Side", dict(elev=0, azim=0), "x"),      # x into screen
            ("Top", dict(elev=90, azim=-90), "z"),    # z into screen
            ("Tilt", dict(elev=35, azim=-45), "z"),   # z foreshortened
        ]

    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    radius = robot_bounding_radius(robot_components)
    center_min = np.array([x_min + radius, y_min + radius, z_min + radius], dtype=float)
    center_max = np.array([x_max - radius, y_max - radius, z_max - radius], dtype=float)

    # Ensure initial/goal centers keep the full body inside the env bounds.
    start = np.clip(start, center_min, center_max)
    goal = np.clip(goal, center_min, center_max)
    dx, dy, dz = float(x_max - x_min), float(y_max - y_min), float(z_max - z_min)

    obstacle_faces = []
    robot_colls: List[List[Poly3DCollection]] = []
    traj_lines = []
    ref_lines = []
    pred_lines = []
    cur_scats = []
    if render:
        obstacle_faces = []
        for obs in obstacles:
            obstacle_faces.append(_box_faces(obs.vertices()))

        robot_colls = []
        labelpad = 30
        pad = 0.06 * max(dx, dy, dz)
        for ax_idx, (ax, (title, view, hide_axis)) in enumerate(zip(axs, view_specs)):
            ax.set_title(title, fontsize=14)
            ax.view_init(**view)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            try:
                ax.set_box_aspect((dx, dy, dz))
            except Exception:
                pass
            if hide_axis == "x":
                ax.set_xticks([])
            elif hide_axis == "y":
                ax.set_yticks([])
            elif hide_axis == "z":
                ax.set_zticks([])
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlabel("" if ax_idx == 1 else "x", fontsize=18, labelpad=labelpad)
            ax.set_ylabel("", fontsize=18, labelpad=labelpad)
            ax.set_zlabel("", fontsize=18, labelpad=labelpad)
            if ax_idx == 0:
                ax.set_xlim(x_min, x_max + pad)
                ax.text(x_max - 0.5 * pad + 0.6, (y_min + y_max) / 2, (z_min + z_max) / 2, "z", fontsize=18, color="black")
            elif ax_idx == 2:
                ax.set_xlim(x_min, x_max + pad)
                ax.text(x_max - 0.5 * pad + 0.6, (y_min + y_max) / 2, z_min, "y", fontsize=18, color="black")
            elif ax_idx == 3:
                ax.set_zlim(z_min, z_max + pad)
                ax.text(x_min, y_max, z_max - 0.5 * pad + 0.6, "z", fontsize=18, color="black")
            ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

            # Obstacles (translucent)
            for faces in obstacle_faces:
                coll = Poly3DCollection(
                    faces,
                    facecolors=(0.8, 0.1, 0.1, 0.18),
                    edgecolors=(0.0, 0.0, 0.0, 0.25),
                    linewidths=0.4,
                )
                ax.add_collection3d(coll)

            # Global path
            ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], "--", color="grey", linewidth=1.2)
            ax.scatter([start[0]], [start[1]], [start[2]], color="green", s=60, edgecolors="black")
            ax.scatter([goal[0]], [goal[1]], [goal[2]], color="red", s=80, marker="*", edgecolors="black")

            # Robot body (L-shape) at start pose
            axis_robot: List[Poly3DCollection] = []
            R0 = rpy_to_R(*rpy0)
            for comp in robot_components:
                faces = _box_faces(comp.vertices_world(start, R0))
                coll = Poly3DCollection(
                    faces,
                    facecolors=(0.1, 0.35, 0.9, 0.35),
                    edgecolors=(0.0, 0.0, 0.0, 0.45),
                    linewidths=0.6,
                )
                ax.add_collection3d(coll)
                axis_robot.append(coll)
            robot_colls.append(axis_robot)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)

        # Dynamic artists (per axis)
        traj_lines = []
        ref_lines = []
        pred_lines = []
        cur_scats = []
        for ax in axs:
            (traj_line,) = ax.plot([], [], [], color="dodgerblue", linewidth=2.0)
            (ref_line,) = ax.plot([], [], [], color="orange", linewidth=3.0, alpha=0.8)
            (pred_line,) = ax.plot([], [], [], color="blue", linewidth=1.4, linestyle="--", alpha=0.9)
            cur_sc = ax.scatter([], [], [], color="cyan", s=28)
            traj_lines.append(traj_line)
            ref_lines.append(ref_line)
            pred_lines.append(pred_line)
            cur_scats.append(cur_sc)

    # -------------------------
    # Video recording (optional)
    # -------------------------
    video_path = Path("impc_run.mp4")
    video_writer = None
    frames: List[np.ndarray] = []
    if render:
        video_dir = Path(".")
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir / "impc_run.mp4"
        try:
            video_writer = imageio.get_writer(video_path, fps=10)
        except Exception as exc:
            print(f"[video] Failed to initialize writer ({exc}); falling back to in-memory frames.")

    def record_frame():
        if not render or fig is None:
            return
        frame = _pad_frame_to_macroblock(_capture_frame(fig))
        if video_writer is not None:
            video_writer.append_data(frame)
        else:
            frames.append(frame)

    # -------------------------
    # MPC loop
    # -------------------------
    state = np.zeros(7, dtype=float)
    state[:3] = start.copy()
    state[3:6] = rpy0.copy()
    state[6] = 0.0  # start from rest; MPC will accelerate as needed

    executed_pos: List[np.ndarray] = [state[:3].copy()]
    predictions_pos: List[np.ndarray] = []
    xr_history: List[np.ndarray] = []
    state_history: List[np.ndarray] = [state.copy()]

    progress_s = 0.0
    last_progress_s = 0.0

    # Initial guess: follow the global path arc-length reference.
    def reference_positions(progress: float) -> np.ndarray:
        s_query = progress + (np.arange(N + 1, dtype=float) * ref_speed * dt)
        return _interpolate_polyline(path_points, s_query, s_path, seg_len)

    def reference_state(progress: float, rpy_seed: np.ndarray) -> np.ndarray:
        pos = reference_positions(progress)
        rpy_ref = np.zeros((pos.shape[0], 3), dtype=float)
        rpy_ref[0] = np.asarray(rpy_seed, dtype=float).reshape(3)
        for k in range(pos.shape[0] - 1):
            d = pos[k + 1] - pos[k]
            rpy_ref[k + 1] = velocity_to_rpy(d, previous_rpy=rpy_ref[k])
        # Unwrap yaw/pitch to avoid -pi/pi discontinuities in the optimizer.
        rpy_ref[:, 2] = np.unwrap(rpy_ref[:, 2])
        rpy_ref[:, 1] = np.unwrap(rpy_ref[:, 1])
        # Keep roll at 0 (but still present in the state).
        rpy_ref[:, 0] = 0.0
        v_ref = float(ref_speed) * np.ones((pos.shape[0], 1), dtype=float)
        return np.hstack([pos, rpy_ref, v_ref])

    state_guess = reference_state(progress_s, rpy_seed=state[3:6])
    state_guess[0, :] = state

    accel_brake = float(cfg.input_bounds.lower[1]) if getattr(cfg, "input_bounds", None) is not None else 0.0

    solve_times: List[float] = []

    for t in range(nsim):
        # Update progress by projecting current position to the global path arc-length.
        s_closest = _closest_s_on_polyline(state[:3], path_points, s_path)
        progress_s = max(progress_s, s_closest)
        progress_s = min(progress_s, total_len)

        state_ref = reference_state(progress_s, rpy_seed=state[3:6])

        accepted = False
        state_pred = state_guess.copy()
        control_pred = np.zeros((N, 4), dtype=float)
        omega = np.zeros((len(obstacles), len(robot_components), k_cbf), dtype=float)
        status = "not_solved"
        probe_statuses: List[str] = []

        # Sequential convexification loop: update linearizations around the latest guess.
        max_inner = int(max(1, int(getattr(cfg, "max_step_iterations", 50))))
        for _inner in range(max_inner):
            state_pred, control_pred, omega, status, solve_time = _solve_mpc_cbf_osqp(
                state_current=state,
                state_ref=state_ref,
                state_guess=state_pred,
                obstacles=obstacles,
                robot_components=robot_components,
                bounds=bounds,
                cfg=cfg,
            )
            status_norm = str(status).strip().lower().replace(" ", "_")
            probe_statuses.append(status_norm)
            if status_norm not in {"solved", "solved_inaccurate"}:
                continue

            # Safety filter: reject if any early predicted step is in barrier.
            collides = False
            for kk in range(1, k_cbf + 1):
                Rk = rpy_to_R(float(state_pred[kk, 3]), float(state_pred[kk, 4]), float(state_pred[kk, 5]))
                if in_barrier_3d(state_pred[kk, :3], obstacles, robot_components, Rk, penetration_tol=penetration_tol):
                    collides = True
                    break
            if not collides:
                accepted = True
                solve_times.append(solve_time)
                break

        if (not accepted) and probe_statuses and all("infeasible" in s for s in probe_statuses):
            if not headless:
                status_set = ", ".join(sorted(set(probe_statuses)))
                print(
                    f"[iMPC] Infeasible at step {t}: all {len(probe_statuses)} probing iterations "
                    f"returned infeasible ({status_set}). Breaking out of loop and completing wrap-up."
                )
            break

        if accepted:
            u0 = np.asarray(control_pred[0], dtype=float).reshape(4)
        else:
            # Fallback: stop turning/accelerating; gently brake.
            u0 = np.array([0.0, 0.0, 0.0, accel_brake], dtype=float)

        state = _step_dynamics_3d(state, u0, dt)
        # Keep within center bounds (so full body stays inside env bounds)
        state[0] = float(np.clip(state[0], center_min[0], center_max[0]))
        state[1] = float(np.clip(state[1], center_min[1], center_max[1]))
        state[2] = float(np.clip(state[2], center_min[2], center_max[2]))

        R_now = rpy_to_R(float(state[3]), float(state[4]), float(state[5]))

        executed_pos.append(state[:3].copy())
        predictions_pos.append(state_pred[:, :3].copy())
        xr_history.append(state_ref.copy())
        state_history.append(state.copy())
        state_guess = state_pred.copy()
        state_guess[0, :] = state

        # Update visuals (optional)
        if render:
            traj = np.asarray(executed_pos, dtype=float)
            for ax_idx in range(len(axs)):
                traj_lines[ax_idx].set_data(traj[:, 0], traj[:, 1])
                traj_lines[ax_idx].set_3d_properties(traj[:, 2])

                ref_lines[ax_idx].set_data(state_ref[:, 0], state_ref[:, 1])
                ref_lines[ax_idx].set_3d_properties(state_ref[:, 2])

                # Visualize the probing horizon *from the post-step state*.
                #
                # At this point in the loop we have already advanced `state` by one dynamics step
                # (executed control u0). Plotting the full `state_pred` would start at the *pre-step*
                # state_pred[0], which can make the probing line appear "ahead" of the robot when the
                # executed step goes backward. Using state_pred[1:] matches the 2D visualization
                # convention (probing horizon excludes the current state) and keeps the line aligned
                # with the rendered robot pose.
                pred_xyz = state_pred[1:, :3]
                pred_lines[ax_idx].set_data(pred_xyz[:, 0], pred_xyz[:, 1])
                pred_lines[ax_idx].set_3d_properties(pred_xyz[:, 2])

                # Current marker: re-create scatter offsets for 3D
                cur_scats[ax_idx]._offsets3d = ([state[0]], [state[1]], [state[2]])  # type: ignore[attr-defined]

                # Robot body: update cuboid faces
                for comp_idx, comp in enumerate(robot_components):
                    faces = _box_faces(comp.vertices_world(state[:3], R_now))
                    robot_colls[ax_idx][comp_idx].set_verts(faces)

            if interactive_backend:
                plt.pause(0.001)
            record_frame()

        # Goal check
        if float(np.linalg.norm(state[:3] - goal)) <= approx_radius:
            break

        # Simple stall detection (optional)
        if float(np.linalg.norm(executed_pos[-1] - executed_pos[-2])) < 1e-6:
            # don't let reference run away if we're stuck
            progress_s = last_progress_s
        else:
            last_progress_s = progress_s

    executed_arr = np.asarray(executed_pos, dtype=float)
    pred_arr = (
        np.asarray(predictions_pos, dtype=float)
        if predictions_pos
        else np.zeros((0, N + 1, 3), dtype=float)
    )
    if render:
        if video_writer is not None:
            video_writer.close()
        else:
            imageio.mimsave(video_path, frames, fps=10)
        _save_sample_plot_3d(
            impctra=executed_arr,
            path_points=path_points,
            xr_history=xr_history,
            x_pred_all=pred_arr,
            obstacles=obstacles,
            robot_components=robot_components,
            bounds=bounds,
            state_history=state_history,
            start=start,
            goal=goal,
            filename="sample_plot_3d_L.png",
            dpi=300,
        )
    return executed_arr, pred_arr, solve_times, state_history


def _sample_start_goal_pairs_from_guiding_path_3d(
    n_pairs: int,
    rng: np.random.Generator,
    min_sep_frac: float = 0.1,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample n_pairs of (start, goal, path_segment) from the main run's guiding path.

    Returns (start_xyz, goal_xyz, path_segment) for each pair. path_segment is the
    segment of the guiding path between start and goal for use as the trial path.
    """
    path_points = np.asarray(getattr(ENVIRONMENT, "guiding_waypoints", None), dtype=float)
    if path_points is None or path_points.size == 0:
        return []
    path_points = path_points.reshape(-1, 3)
    if path_points.shape[0] < 2:
        return []

    s_path, seg_len = _polyline_arclength(path_points)
    total_len = float(s_path[-1])
    min_sep = max(0.02, min_sep_frac * total_len)
    if total_len <= min_sep:
        return []

    pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    max_attempts = n_pairs * 50
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        s1 = rng.uniform(0.0, total_len - min_sep)
        s2 = rng.uniform(s1 + min_sep, total_len)
        if s2 - s1 < min_sep:
            continue
        pt1 = _interpolate_polyline(path_points, np.array([s1]), s_path, seg_len)[0]
        pt2 = _interpolate_polyline(path_points, np.array([s2]), s_path, seg_len)[0]
        start = np.asarray(pt1, dtype=float).reshape(3)
        goal = np.asarray(pt2, dtype=float).reshape(3)
        # Build path segment: interpolate at n_pts along [s1, s2]
        n_pts = max(5, int((s2 - s1) / 0.05))
        s_query = np.linspace(s1, s2, n_pts)
        path_segment = _interpolate_polyline(path_points, s_query, s_path, seg_len)
        pairs.append((start, goal, path_segment))
    return pairs


def run_success_rate_trials_3d(
    config: NMPCConfig,
    num_trials: int = 20,
    num_steps: int = 20,
    seed: int = 42,
) -> Tuple[int, int, float, List[float]]:
    """
    Run num_trials with random (start, goal) pairs for num_steps each.
    A trial is successful if all num_steps are feasible (no collision).
    Returns (success_count, fail_count, success_rate, all_solve_times).
    """
    rng = np.random.default_rng(seed)
    obstacles = list(OBSTACLE_REGIONS)
    pairs = _sample_start_goal_pairs_from_guiding_path_3d(num_trials, rng)
    if len(pairs) < num_trials:
        print(f"[trials] Warning: only sampled {len(pairs)}/{num_trials} valid start-goal pairs")
    success_count = 0
    fail_count = 0
    all_solve_times: List[float] = []
    robot_components = build_robot_cuboids(config)
    penetration_tol = float(getattr(config, "barrier_penetration_tol", 0.0))
    for idx, (start_xyz, goal_xyz, path_segment) in enumerate(pairs):
        try:
            executed_arr, _, solve_times, state_history = run_3d(
                config=config,
                render=False,
                headless=True,
                nsim_override=num_steps + 1,
                start=start_xyz,
                goal=goal_xyz,
                path_points_override=path_segment,
            )
            all_solve_times.extend(solve_times)
            n_executed = len(state_history) - 1
            if n_executed < num_steps:
                fail_count += 1
                continue
            all_feasible = True
            for step in range(1, min(num_steps + 1, len(state_history))):
                state_full = np.asarray(state_history[step], dtype=float).reshape(7)
                center = state_full[:3]
                R = rpy_to_R(float(state_full[3]), float(state_full[4]), float(state_full[5]))
                if in_barrier_3d(center, obstacles, robot_components, R, penetration_tol=penetration_tol):
                    all_feasible = False
                    break
            if all_feasible:
                success_count += 1
            else:
                fail_count += 1
        except Exception as exc:
            fail_count += 1
            print(f"[trials] Trial {idx + 1} failed: {exc}")
    total = success_count + fail_count
    success_rate = success_count / total if total > 0 else 0.0
    return success_count, fail_count, success_rate, all_solve_times


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D polytopic maze navigation with iterative convex MPC-DHOCBF (Section V-C)."
    )
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon N. Default: 8.")
    parser.add_argument("--gamma", type=float, default=None, help="DHOCBF decay rate gamma_1. Default: 1.0.")
    parser.add_argument("--steps", type=int, default=None, help="Closed-loop steps. Default: config nsim.")
    parser.add_argument(
        "--trials",
        type=int,
        default=0,
        help="Number of random-start timing trials. 0 runs only the navigation demo.",
    )
    parser.add_argument("--trial-steps", type=int, default=20, help="Steps per timing trial.")
    parser.add_argument("--seed", type=int, default=5, help="Seed for the timing trials.")
    parser.add_argument("--headless", action="store_true", help="Skip the multi-view animation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    slack_tuple = tuple(DualSlackConfig() for _ in OBSTACLE_REGIONS)
    cfg = NMPCConfig(dual_slacks=slack_tuple)
    updates = {}
    if args.horizon is not None:
        updates["horizon"] = int(args.horizon)
        updates["probe_safety_horizon_steps"] = int(args.horizon)
    if args.gamma is not None:
        updates["cbf_decay"] = float(args.gamma)
    if args.steps is not None:
        updates["nsim"] = int(args.steps)
    if updates:
        cfg = replace(cfg, **updates)
    goal = np.asarray(ENVIRONMENT.goal, dtype=float).reshape(3)
    approx_radius = float(cfg.approx_radius)

    traj, preds, solve_times, state_history = None, None, None, None
    try:
        traj, preds, solve_times, state_history = run_3d(
            cfg,
            nsim_override=args.steps,
            headless=args.headless,
        )
        if traj is not None:
            np.save("trajectory.npy", traj)
        if preds is not None:
            np.save("prediction.npy", preds)
        if solve_times:
            print(
                f"[timing] steps={len(solve_times)} "
                f"mean={float(np.mean(solve_times)) * 1e3:.2f} ms "
                f"std={float(np.std(solve_times)) * 1e3:.2f} ms"
            )
    except Exception as exc:
        print(f"[main] Run failed: {exc}")
        import traceback
        traceback.print_exc()
        traj = np.zeros((0, 3), dtype=float)
        preds = np.zeros((0, int(cfg.horizon) + 1, 3), dtype=float)
        solve_times = []
        state_history = []

    try:
        n_steps = max(0, (traj.shape[0] - 1)) if traj is not None and traj.size > 0 else 0
        goal_reached = (
            n_steps > 0
            and traj is not None
            and float(np.linalg.norm(traj[-1] - goal)) <= approx_radius
        )
        print(f"[summary] Main run: {n_steps} steps executed, goal_reached={goal_reached}")

        if args.trials > 0:
            success_count, fail_count, success_rate, trial_solve_times = run_success_rate_trials_3d(
                config=cfg,
                num_trials=int(args.trials),
                num_steps=int(args.trial_steps),
                seed=int(args.seed),
            )
            print(f"[trials] success={success_count}, fail={fail_count}, rate={success_rate:.2%}")

            robot_shape = str(getattr(cfg, "robot_shape", "lshape"))
            horizon = int(cfg.horizon)
            cbf_decay = float(getattr(cfg, "cbf_decay", 1.0))
            all_times = list(trial_solve_times or [])
            step_time_avg = float(np.mean(all_times)) if all_times else np.nan
            step_time_std = float(np.std(all_times)) if len(all_times) > 1 else (0.0 if len(all_times) == 1 else np.nan)
            csv_path = Path("run_summary_3d.csv")
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "robot_shape", "horizon", "cbf_decay", "step_time_average", "step_time_std",
                        "success_count", "fail_count", "success_rate",
                    ])
                writer.writerow([
                    robot_shape, horizon, f"{cbf_decay:.4f}", f"{step_time_avg:.6f}", f"{step_time_std:.6f}",
                    success_count, fail_count, f"{success_rate:.4f}",
                ])
            print(
                f"Appended to {csv_path}: cbf_decay={cbf_decay}, "
                f"step_time_avg={step_time_avg:.4f}s, success_rate={success_rate:.2%}"
            )
    except Exception as exc:
        print(f"[main] Failed during summary/trials/CSV: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close("all")

