from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import math
import os
import numpy as np

from config import DEFAULT_CONFIG
from geometry3d import (
    BoxRegion3D,
    HalfSpaceRegion3D,
    obb_intersects_aabb,
    obb_intersects_halfspace,
)


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    seed: int
    start: np.ndarray
    goal: np.ndarray
    obstacles: List[Union[BoxRegion3D, HalfSpaceRegion3D]]
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    grid: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], float]
    guiding_waypoints: np.ndarray


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Match main_3d.rpy_to_R convention: R = Rz(yaw) @ Ry(-pitch) @ Rx(roll)."""
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


def _velocity_to_yaw_pitch(direction: np.ndarray) -> Tuple[float, float]:
    d = np.asarray(direction, dtype=float).reshape(3)
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    yaw = float(math.atan2(dy, dx))
    dxy = float(math.hypot(dx, dy))
    pitch = float(math.atan2(dz, max(dxy, 1e-12)))
    return yaw, pitch


def _robot_components_lshape_local(L: float, t: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (offsets, half_extents) for two cuboids in local frame."""
    denom = max(1e-9, 2.0 * L - t)
    c = 0.5 * (L * L + t * L - t * t) / denom
    z0 = -0.5 * h
    z1 = 0.5 * h
    # arm_x: x∈[0,L], y∈[0,t] then shift by -c in both x,y
    arm_x = (-c, L - c, -c, t - c, z0, z1)
    # arm_y: x∈[0,t], y∈[0,L] then shift by -c in both x,y
    arm_y = (-c, t - c, -c, L - c, z0, z1)
    comps = [arm_x, arm_y]
    offsets = []
    halves = []
    for (xmin, xmax, ymin, ymax, zmin, zmax) in comps:
        offsets.append([0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)])
        halves.append([0.5 * (xmax - xmin), 0.5 * (ymax - ymin), 0.5 * (zmax - zmin)])
    return np.asarray(offsets, dtype=float), np.asarray(halves, dtype=float)


def _collides_robot_aabbs(
    center: np.ndarray,
    R: np.ndarray,
    comp_offsets: np.ndarray,
    comp_halves: np.ndarray,
    obstacles: List[Union[BoxRegion3D, HalfSpaceRegion3D]],
    penetration_tol: float,
) -> bool:
    c = np.asarray(center, dtype=float).reshape(3)
    Rm = np.asarray(R, dtype=float).reshape(3, 3)
    tol = float(max(0.0, penetration_tol))
    for i in range(int(comp_offsets.shape[0])):
        obb_c = c + Rm @ comp_offsets[i]
        obb_half = comp_halves[i]
        for obs in obstacles:
            if isinstance(obs, HalfSpaceRegion3D):
                if obb_intersects_halfspace(
                    obb_center=obb_c,
                    obb_R=Rm,
                    obb_half=obb_half,
                    n=obs.n,
                    d=obs.d,
                    penetration_tol=tol,
                    x_lo=getattr(obs, "x_lo", -np.inf),
                    x_hi=getattr(obs, "x_hi", np.inf),
                ):
                    return True
            else:
                if obb_intersects_aabb(
                    obb_center=obb_c,
                    obb_R=Rm,
                    obb_half=obb_half,
                    aabb=obs,
                    penetration_tol=tol,
                ):
                    return True
    return False

def _gate_wall_x(
    *,
    x_center: float,
    thickness: float,
    hole_y: Tuple[float, float],
    hole_z: Tuple[float, float],
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
) -> List[BoxRegion3D]:
    """Wall slab normal to +x with a rectangular hole (y,z)."""
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    x0 = float(np.clip(x_center - 0.5 * thickness, x_min, x_max))
    x1 = float(np.clip(x_center + 0.5 * thickness, x_min, x_max))
    if x1 - x0 <= 1e-9:
        return []

    y0, y1 = float(hole_y[0]), float(hole_y[1])
    z0, z1 = float(hole_z[0]), float(hole_z[1])
    y0 = float(np.clip(y0, y_min, y_max))
    y1 = float(np.clip(y1, y_min, y_max))
    z0 = float(np.clip(z0, z_min, z_max))
    z1 = float(np.clip(z1, z_min, z_max))
    if y1 - y0 <= 1e-9 or z1 - z0 <= 1e-9:
        # Degenerate hole: treat as a solid wall.
        return [BoxRegion3D(x0, x1, y_min, y_max, z_min, z_max)]

    pieces: List[BoxRegion3D] = []
    # Left/right of hole (in y)
    if y0 - y_min > 1e-9:
        pieces.append(BoxRegion3D(x0, x1, y_min, y0, z_min, z_max))
    if y_max - y1 > 1e-9:
        pieces.append(BoxRegion3D(x0, x1, y1, y_max, z_min, z_max))
    # Below/above hole (in z), within hole's y-span
    if z0 - z_min > 1e-9:
        pieces.append(BoxRegion3D(x0, x1, y0, y1, z_min, z0))
    if z_max - z1 > 1e-9:
        pieces.append(BoxRegion3D(x0, x1, y0, y1, z1, z_max))
    return pieces


def _plane_from_three_points(P: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (n, d) for plane n·p = d through P, Q, R."""
    P = np.asarray(P, dtype=float).reshape(3)
    Q = np.asarray(Q, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3)
    n = np.cross(Q - P, R - P)
    nn = float(np.linalg.norm(n))
    if nn <= 1e-12:
        n = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        n = n / nn
    d = float(np.dot(n, P))
    return n, d


def _connect_windows_with_slope_planes(
    *,
    wall_x0: List[float],
    wall_x1: List[float],
    holes_y: List[Tuple[float, float]],
    holes_z: List[Tuple[float, float]],
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    margin: float = 0.0,
) -> List[HalfSpaceRegion3D]:
    """Build smooth tunnels by connecting consecutive windows with 4 sloped planes."""
    m = int(min(len(wall_x0), len(wall_x1), len(holes_y), len(holes_z)))
    if m <= 1:
        return []

    (x_min, _, _), (x_max, _, _) = bounds
    pieces: List[HalfSpaceRegion3D] = []
    mar = float(max(0.0, margin))

    for i in range(m - 1):
        x_a = float(np.clip(float(wall_x1[i]), x_min, x_max))
        x_b = float(np.clip(float(wall_x0[i + 1]), x_min, x_max))
        if x_b - x_a <= 1e-6:
            continue

        y0_i, y1_i = float(holes_y[i][0]) - mar, float(holes_y[i][1]) + mar
        y0_j, y1_j = float(holes_y[i + 1][0]) - mar, float(holes_y[i + 1][1]) + mar
        z0_i, z1_i = float(holes_z[i][0]) - mar, float(holes_z[i][1]) + mar
        z0_j, z1_j = float(holes_z[i + 1][0]) - mar, float(holes_z[i + 1][1]) + mar

        # Left plane (y-min boundary): obstacle is "too far left", normal points +y.
        P = np.array([x_a, y0_i, z0_i], dtype=float)
        Q = np.array([x_a, y0_i, z1_i], dtype=float)
        R = np.array([x_b, y0_j, z0_j], dtype=float)
        n, d = _plane_from_three_points(P, Q, R)
        if np.dot(n, np.array([0.0, 1.0, 0.0], dtype=float)) < 0.0:
            n, d = -n, -d
        viz = np.array(
            [[x_a, y0_i, z0_i], [x_a, y0_i, z1_i], [x_b, y0_j, z1_j], [x_b, y0_j, z0_j]],
            dtype=float,
        )
        pieces.append(HalfSpaceRegion3D(n=n, d=d, viz_quad=viz, x_lo=x_a, x_hi=x_b))

        # Right plane (y-max boundary): obstacle is "too far right", normal points -y.
        P = np.array([x_a, y1_i, z0_i], dtype=float)
        Q = np.array([x_a, y1_i, z1_i], dtype=float)
        R = np.array([x_b, y1_j, z0_j], dtype=float)
        n, d = _plane_from_three_points(P, Q, R)
        if np.dot(n, np.array([0.0, -1.0, 0.0], dtype=float)) < 0.0:
            n, d = -n, -d
        viz = np.array(
            [[x_a, y1_i, z0_i], [x_b, y1_j, z0_j], [x_b, y1_j, z1_j], [x_a, y1_i, z1_i]],
            dtype=float,
        )
        pieces.append(HalfSpaceRegion3D(n=n, d=d, viz_quad=viz, x_lo=x_a, x_hi=x_b))

        # Bottom plane (z-min boundary): obstacle is "too low", normal points +z.
        P = np.array([x_a, y0_i, z0_i], dtype=float)
        Q = np.array([x_a, y1_i, z0_i], dtype=float)
        R = np.array([x_b, y0_j, z0_j], dtype=float)
        n, d = _plane_from_three_points(P, Q, R)
        if np.dot(n, np.array([0.0, 0.0, 1.0], dtype=float)) < 0.0:
            n, d = -n, -d
        viz = np.array(
            [[x_a, y0_i, z0_i], [x_b, y0_j, z0_j], [x_b, y1_j, z0_j], [x_a, y1_i, z0_i]],
            dtype=float,
        )
        pieces.append(HalfSpaceRegion3D(n=n, d=d, viz_quad=viz, x_lo=x_a, x_hi=x_b))

        # Top plane (z-max boundary): obstacle is "too high", normal points -z.
        P = np.array([x_a, y0_i, z1_i], dtype=float)
        Q = np.array([x_a, y1_i, z1_i], dtype=float)
        R = np.array([x_b, y0_j, z1_j], dtype=float)
        n, d = _plane_from_three_points(P, Q, R)
        if np.dot(n, np.array([0.0, 0.0, -1.0], dtype=float)) < 0.0:
            n, d = -n, -d
        viz = np.array(
            [[x_a, y0_i, z1_i], [x_a, y1_i, z1_i], [x_b, y1_j, z1_j], [x_b, y0_j, z1_j]],
            dtype=float,
        )
        pieces.append(HalfSpaceRegion3D(n=n, d=d, viz_quad=viz, x_lo=x_a, x_hi=x_b))

    return pieces


def _create_random_3d_gate_maze(seed: int = 0) -> EnvironmentSpec:
    """Random 3D maze built from polytopic (box) walls with holes.

    Design goals:
    - reproducible randomness (seed)
    - guaranteed start→goal feasibility (sequential x-walls with pass-through holes)
    - moderate obstacle count (keeps OSQP fast)
    """
    rng = np.random.default_rng(int(seed))
    # Base unit length (m) for maze dimensions. Scaled 1.5× for extra wall-to-wall spacing.
    base_unit_m = 0.15
    maneuver_multiplier = 1.5
    env_scale = base_unit_m * maneuver_multiplier

    # Keep y/z scale the same, but extend x so goal x=3.0 is reachable after clipping.
    bounds = ((0.0, 0.0, 0.0), (16.0 * env_scale, 7.0 * env_scale, 5.0 * env_scale))
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    cell_size = 0.2 * env_scale

    start = np.array([1.0 * env_scale, 1.5 * env_scale, 0.6 * env_scale], dtype=float)
    goal = np.array([13.33 * env_scale, 6.5 * env_scale, 4.4 * env_scale], dtype=float)

    # Gate wall parameters
    #
    # IMPORTANT for feasibility with a finite-size robot body:
    # While passing through wall i's hole, the robot body should not overlap wall i+1,
    # otherwise the fixed (y,z) alignment for wall i can collide with wall i+1.
    #
    # With the default L-shape cuboid robot (~0.15m span in x), using too many walls
    # makes the inter-wall free space too short. We therefore use fewer walls with
    # larger gaps.
    # With the longer L-shape robot (Sec. V-C), we keep fewer walls so there is enough
    # free space between slabs to maneuver.
    n_walls = 5
    thickness = 0.22 * env_scale
    y_span = float(y_max - y_min)
    z_span = float(z_max - z_min)

    # Robot x-extents (used to place pre/post waypoints so the body clears each wall
    # before changing (y,z) toward the next gate).
    shape = str(getattr(DEFAULT_CONFIG, "robot_shape", "lshape")).strip().lower()
    robot_len = float(getattr(DEFAULT_CONFIG, "robot_rectangle_length", 0.15))
    robot_w = float(getattr(DEFAULT_CONFIG, "robot_rectangle_width", 0.06))
    robot_h = float(getattr(DEFAULT_CONFIG, "robot_height", robot_w))
    if shape == "lshape":
        # Keep consistent with main_3d.build_robot_cuboids: L-arms are 2× cfg length.
        L = float(max(1e-6, 2.0 * robot_len))
        t = float(max(1e-6, robot_w))
        denom = max(1e-9, 2.0 * L - t)
        c = 0.5 * (L * L + t * L - t * t) / denom
        robot_front_extent = float(L - c)
        robot_back_extent = float(c)
        # Conservative bounding radius (for clipping start/goal inside bounds).
        z0_r = -0.5 * float(max(1e-6, robot_h))
        z1_r = 0.5 * float(max(1e-6, robot_h))
        corners = []
        for x in (-c, L - c):
            for y in (-c, t - c):
                for z in (z0_r, z1_r):
                    corners.append((x, y, z))
        for x in (-c, t - c):
            for y in (-c, L - c):
                for z in (z0_r, z1_r):
                    corners.append((x, y, z))
        corners = np.asarray(corners, dtype=float)
        robot_radius = float(np.sqrt((corners**2).sum(axis=1).max()))
    else:
        robot_front_extent = 0.5 * float(max(1e-6, robot_len))
        robot_back_extent = robot_front_extent
        hx = robot_front_extent
        hy = 0.5 * float(max(1e-6, robot_w))
        hz = 0.5 * float(max(1e-6, robot_h))
        robot_radius = float(math.sqrt(hx * hx + hy * hy + hz * hz))
    # Extra clearance so we don't graze due to numerical tolerances.
    x_clear = 0.005

    # Clip start/goal so the *entire body* (any orientation) stays within the env bounds.
    center_min = np.array([x_min + robot_radius, y_min + robot_radius, z_min + robot_radius], dtype=float)
    center_max = np.array([x_max - robot_radius, y_max - robot_radius, z_max - robot_radius], dtype=float)
    start = np.clip(start, center_min, center_max)
    goal = np.clip(goal, center_min, center_max)

    # Walls positioned between start.x and goal.x with fixed spacing of 0.5
    x_lo = float(min(start[0], goal[0]) + 0.18 * (max(start[0], goal[0]) - min(start[0], goal[0])))
    x_hi = float(max(start[0], goal[0]) - 0.18 * (max(start[0], goal[0]) - min(start[0], goal[0])))
    wall_spacing = 0.5
    total_span = wall_spacing * (n_walls - 1)
    first_x = np.clip((x_lo + x_hi - total_span) / 2, x_lo, max(x_lo, x_hi - total_span))
    x_positions = first_x + np.arange(n_walls, dtype=float) * wall_spacing

    # Revert window (hole) sizes + locations to the original design:
    # - hole size is a fixed fraction of span (0.28..0.36)
    # - hole center follows a correlated random-walk
    y_c = float(np.clip(start[1], y_min + 0.2 * y_span, y_max - 0.2 * y_span))
    z_c = float(np.clip(start[2], z_min + 0.2 * z_span, z_max - 0.2 * z_span))

    obstacles: List[Union[BoxRegion3D, HalfSpaceRegion3D]] = []
    # First pass: build wall holes (windows) with the original random-walk logic.
    wall_x0: List[float] = []
    wall_x1: List[float] = []
    holes_y: List[Tuple[float, float]] = []
    holes_z: List[Tuple[float, float]] = []
    hole_centers: List[Tuple[float, float]] = []
    wall_pieces_per_wall: List[List[BoxRegion3D]] = []

    # Make the maze more challenging:
    # - windows (holes) are slightly smaller than the (doubled) L-arm length
    # - consecutive windows are spread farther apart on the wall (y,z), but not so far that
    #   the nonholonomic robot cannot realistically re-align between slabs.
    arm_len = float(L) if shape == "lshape" else float(max(1e-6, robot_len))
    min_center_sep = float(0.35 * arm_len)
    max_step = np.array([0.16 * y_span, 0.16 * z_span], dtype=float)

    prev_center = np.array([float(y_c), float(z_c)], dtype=float)
    for wall_idx, x_c in enumerate(x_positions):
        # Window size: a bit smaller than arm length (but not too extreme).
        # Challenging windows:
        # - y-window is smaller than before, but still wide enough to pass without extreme tilt
        # - z-window is slightly smaller than the L-arm length to force more careful attitude control
        hole_y = float(rng.uniform(1.40, 1.55) * arm_len * 0.9)
        hole_z = float(rng.uniform(0.75, 0.85) * arm_len)
        # Clamp to be within bounds (and always larger than body thickness).
        hole_y = float(np.clip(hole_y, 2.5 * robot_w, 0.90 * y_span))
        hole_z = float(np.clip(hole_z, 2.5 * robot_h, 0.90 * z_span))

        # Spread window centers apart with a bounded step + min separation.
        yz = prev_center.copy()
        found = False
        for _ in range(24):
            step = rng.normal(scale=max_step)
            step = np.clip(step, -max_step, max_step)
            cand = prev_center + step
            cand[0] = float(np.clip(cand[0], y_min + 0.5 * hole_y, y_max - 0.5 * hole_y))
            cand[1] = float(np.clip(cand[1], z_min + 0.5 * hole_z, z_max - 0.5 * hole_z))
            if float(np.linalg.norm(cand - prev_center)) >= min_center_sep:
                yz = cand
                found = True
                break
        if not found:
            # Fall back: take the farthest of a few bounded candidates.
            best = prev_center.copy()
            best_d = -1.0
            for _ in range(12):
                step = rng.normal(scale=max_step)
                step = np.clip(step, -max_step, max_step)
                cand = prev_center + step
                cand[0] = float(np.clip(cand[0], y_min + 0.5 * hole_y, y_max - 0.5 * hole_y))
                cand[1] = float(np.clip(cand[1], z_min + 0.5 * hole_z, z_max - 0.5 * hole_z))
                d = float(np.linalg.norm(cand - prev_center))
                if d > best_d:
                    best_d = d
                    best = cand.copy()
            yz = best

        y_c, z_c = float(yz[0]), float(yz[1])
        prev_center = np.array([y_c, z_c], dtype=float)

        y0, y1 = float(y_c - 0.5 * hole_y), float(y_c + 0.5 * hole_y)
        z0, z1 = float(z_c - 0.5 * hole_z), float(z_c + 0.5 * hole_z)
        hole_centers.append((float(y_c), float(z_c)))

        # Wall slab x-interval (clipped to bounds)
        x0 = float(np.clip(float(x_c) - 0.5 * thickness, x_min, x_max))
        x1 = float(np.clip(float(x_c) + 0.5 * thickness, x_min, x_max))
        wall_x0.append(x0)
        wall_x1.append(x1)
        holes_y.append((y0, y1))
        holes_z.append((z0, z1))

        pieces = _gate_wall_x(
            x_center=float(x_c),
            thickness=float(thickness),
            hole_y=(y0, y1),
            hole_z=(z0, z1),
            bounds=bounds,
        )
        wall_pieces_per_wall.append(list(pieces))
        obstacles.extend(pieces)

    # Connect consecutive windows with slope-plane tunnels (optional).
    if os.environ.get("NO_TUNNEL", "0") != "1":
        tunnel_margin = float(os.environ.get("TUNNEL_MARGIN", "0.15"))
        tunnel_pieces = _connect_windows_with_slope_planes(
            wall_x0=wall_x0,
            wall_x1=wall_x1,
            holes_y=holes_y,
            holes_z=holes_z,
            bounds=bounds,
            margin=tunnel_margin,
        )
        obstacles.extend(tunnel_pieces)

    # Build guiding waypoints:
    # For tight windows and an L-shape body, crossing at the *window center* can collide due to
    # body offsets. We therefore choose an entry/exit (y,z) inside each hole that yields a
    # collision-free straight segment across the slab when the robot follows the segment direction.
    guiding_waypoints: List[np.ndarray] = [start.copy()]
    # Robot component geometry for collision checks during waypoint construction.
    if shape == "lshape":
        comp_offsets, comp_halves = _robot_components_lshape_local(L=float(L), t=float(t), h=float(robot_h))
    else:
        # Rectangle prism fallback: single cuboid centered at origin.
        hx = 0.5 * float(max(1e-6, robot_len))
        hy = 0.5 * float(max(1e-6, robot_w))
        hz = 0.5 * float(max(1e-6, robot_h))
        comp_offsets = np.zeros((1, 3), dtype=float)
        comp_halves = np.array([[hx, hy, hz]], dtype=float)

    for i in range(n_walls):
        pre_x = float(np.clip(wall_x0[i] - robot_front_extent - x_clear, float(center_min[0]), float(center_max[0])))
        post_x = float(np.clip(wall_x1[i] + robot_back_extent + x_clear, float(center_min[0]), float(center_max[0])))
        (y0, y1) = holes_y[i]
        (z0, z1) = holes_z[i]
        margin = 0.01
        # Waypoints must respect center bounds (main_3d clips state to keep full body inside bounds).
        y_lo = float(max(y0 + margin, float(center_min[1])))
        y_hi = float(min(y1 - margin, float(center_max[1])))
        z_lo = float(max(z0 + margin, float(center_min[2])))
        z_hi = float(min(z1 - margin, float(center_max[2])))
        ys = (
            np.linspace(y_lo, y_hi, 7)
            if y_hi > y_lo
            else np.asarray([float(np.clip(0.5 * (y0 + y1), center_min[1], center_max[1]))])
        )
        zs = (
            np.linspace(z_lo, z_hi, 7)
            if z_hi > z_lo
            else np.asarray([float(np.clip(0.5 * (z0 + z1), center_min[2], center_max[2]))])
        )

        best_pair = None
        # Prefer a near-straight (yaw≈0, pitch≈0) segment through the window, because it is
        # easier for the MPC to track robustly.
        best_margin = -1.0
        for y in ys:
            for z in zs:
                a = np.array([pre_x, float(y), float(z)], dtype=float)
                b = np.array([post_x, float(y), float(z)], dtype=float)
                R = _rpy_to_R(0.0, 0.0, 0.0)
                ok = True
                for tt in np.linspace(0.0, 1.0, 80):
                    p = a * (1.0 - tt) + b * tt
                    if _collides_robot_aabbs(
                        p,
                        R,
                        comp_offsets,
                        comp_halves,
                        obstacles,
                        penetration_tol=1e-3,
                    ):
                        ok = False
                        break
                if ok:
                    m = float(min(y - y0, y1 - y, z - z0, z1 - z))
                    if m > best_margin:
                        best_margin = m
                        best_pair = (a, b)

        # If no straight segment works, search for the mildest diagonal (min yaw/pitch).
        if best_pair is None:
            best_cost = float("inf")
            for y_a in ys:
                for z_a in zs:
                    for y_b in ys:
                        for z_b in zs:
                            a = np.array([pre_x, float(y_a), float(z_a)], dtype=float)
                            b = np.array([post_x, float(y_b), float(z_b)], dtype=float)
                            d = b - a
                            yaw, pitch = _velocity_to_yaw_pitch(d)
                            R = _rpy_to_R(0.0, pitch, yaw)
                            ok = True
                            for tt in np.linspace(0.0, 1.0, 80):
                                p = a * (1.0 - tt) + b * tt
                                if _collides_robot_aabbs(
                                    p,
                                    R,
                                    comp_offsets,
                                    comp_halves,
                                    obstacles,
                                    penetration_tol=1e-3,
                                ):
                                    ok = False
                                    break
                            if ok:
                                cost = float(yaw * yaw + pitch * pitch)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pair = (a, b)

        if best_pair is None:
            # Fallback: use the window center if we couldn't find a diagonal segment quickly.
            yc, zc = hole_centers[i]
            yc = float(np.clip(float(yc), float(center_min[1]), float(center_max[1])))
            zc = float(np.clip(float(zc), float(center_min[2]), float(center_max[2])))
            a = np.array([pre_x, yc, zc], dtype=float)
            b = np.array([post_x, yc, zc], dtype=float)
            best_pair = (a, b)

        a, b = best_pair
        # Add a short x-only "approach" segment so yaw/pitch can settle before entering the slab.
        # This helps avoid local minima where the robot reaches the wall slightly misaligned and stops.
        if float(np.linalg.norm(a[1:] - b[1:])) < 1e-9:
            last_x = float(guiding_waypoints[-1][0])
            # Leave a small gap from both ends so we don't create nearly-duplicate points.
            avail = float(a[0] - last_x)
            buf_len = float(min(0.08, max(0.03, 0.35 * avail)))
            x_buf = float(np.clip(a[0] - buf_len, last_x + 0.02, a[0] - 0.01))
            if x_buf > last_x + 1e-6:
                guiding_waypoints.append(np.array([x_buf, float(a[1]), float(a[2])], dtype=float))

        guiding_waypoints.append(a)
        guiding_waypoints.append(b)

        # Add a short x-only "exit" segment (except after the last wall).
        if i < n_walls - 1 and float(np.linalg.norm(a[1:] - b[1:])) < 1e-9:
            next_pre_x = float(
                np.clip(
                    wall_x0[i + 1] - robot_front_extent - x_clear,
                    float(center_min[0]),
                    float(center_max[0]),
                )
            )
            gap = float(next_pre_x - b[0])
            buf_len2 = float(min(0.08, max(0.03, 0.25 * gap)))
            x_buf2 = float(np.clip(b[0] + buf_len2, b[0] + 0.01, next_pre_x - 0.02))
            if x_buf2 > b[0] + 1e-6:
                guiding_waypoints.append(np.array([x_buf2, float(b[1]), float(b[2])], dtype=float))

    # After the last wall, move straight in +x a bit before turning toward the goal.
    if hole_centers:
        post_last = float(guiding_waypoints[-1][0])
        yc_last, zc_last = float(guiding_waypoints[-1][1]), float(guiding_waypoints[-1][2])
        dx_to_goal = float(goal[0] - post_last)
        if dx_to_goal > 1e-6:
            exit_x = float(np.clip(post_last + 0.6 * dx_to_goal, float(center_min[0]), float(center_max[0])))
            if exit_x > post_last + 1e-6:
                guiding_waypoints.append(np.array([exit_x, yc_last, zc_last], dtype=float))

    if float(np.linalg.norm(guiding_waypoints[-1] - goal)) > 1e-9:
        guiding_waypoints.append(goal.copy())

    return EnvironmentSpec(
        name="random_3d_gate_maze",
        seed=int(seed),
        start=start,
        goal=goal,
        obstacles=obstacles,
        bounds=bounds,
        grid=(bounds, float(cell_size)),
        guiding_waypoints=np.asarray(guiding_waypoints, dtype=float),
    )


def load_environment(name: str = "random_3d_gate_maze", seed: int = 48) -> EnvironmentSpec:
    name_norm = str(name).strip().lower()
    if name_norm in {"random_3d_gate_maze", "random_3d_maze", "maze3d"}:
        return _create_random_3d_gate_maze(seed=int(seed))
    raise ValueError(f"Unknown environment {name!r}")


ENVIRONMENT = load_environment()
OBSTACLE_REGIONS = ENVIRONMENT.obstacles
OBSTACLE_HALFSPACES = [(A, b.reshape(-1)) for A, b in (region.get_convex_rep() for region in OBSTACLE_REGIONS)]
OBSTACLE_VERTICES = [region.vertices() for region in OBSTACLE_REGIONS]

