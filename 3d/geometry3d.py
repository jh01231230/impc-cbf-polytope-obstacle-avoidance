from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BoxRegion3D:
    """Axis-aligned 3D box (a convex polytope) with x/y/z bounds.

    The *interior* of the box is considered obstacle space.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        if not (self.x_min < self.x_max and self.y_min < self.y_max and self.z_min < self.z_max):
            raise ValueError("Invalid box bounds (min must be < max for all axes).")

    def get_convex_rep(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return half-space representation A x <= b."""
        A = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        b = np.array(
            [
                -float(self.x_min),
                float(self.x_max),
                -float(self.y_min),
                float(self.y_max),
                -float(self.z_min),
                float(self.z_max),
            ],
            dtype=float,
        ).reshape(-1, 1)
        return A, b

    def vertices(self) -> np.ndarray:
        """Return 8 corner vertices with shape (8,3)."""
        x0, x1 = float(self.x_min), float(self.x_max)
        y0, y1 = float(self.y_min), float(self.y_max)
        z0, z1 = float(self.z_min), float(self.z_max)
        return np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=float,
        )

    def inflate(self, margin: float) -> "BoxRegion3D":
        m = float(max(0.0, margin))
        return BoxRegion3D(
            x_min=self.x_min - m,
            x_max=self.x_max + m,
            y_min=self.y_min - m,
            y_max=self.y_max + m,
            z_min=self.z_min - m,
            z_max=self.z_max + m,
        )

    def contains(self, point: np.ndarray, tol: float = 0.0) -> bool:
        p = np.asarray(point, dtype=float).reshape(3)
        # Signed tolerance:
        # - tol > 0 expands the box (treat "nearby" as inside)
        # - tol < 0 shrinks the box (treat small penetrations as non-colliding)
        t = float(tol)
        x0 = float(self.x_min - t)
        x1 = float(self.x_max + t)
        y0 = float(self.y_min - t)
        y1 = float(self.y_max + t)
        z0 = float(self.z_min - t)
        z1 = float(self.z_max + t)
        if x0 > x1 or y0 > y1 or z0 > z1:
            # Shrunk past disappearance.
            return False
        return bool((x0 <= p[0] <= x1) and (y0 <= p[1] <= y1) and (z0 <= p[2] <= z1))

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        """Closest point on (or inside) the box to the query point."""
        p = np.asarray(point, dtype=float).reshape(3)
        q = np.empty(3, dtype=float)
        q[0] = float(np.clip(p[0], self.x_min, self.x_max))
        q[1] = float(np.clip(p[1], self.y_min, self.y_max))
        q[2] = float(np.clip(p[2], self.z_min, self.z_max))
        return q

    def closest_point_and_normal(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Return (closest_point, outward_unit_normal, distance, inside_flag).

        - If the point is outside, the normal points from the closest point on the box to the point.
        - If the point is inside, the normal points outward through the *nearest* face.
        """
        p = np.asarray(point, dtype=float).reshape(3)
        inside = self.contains(p, tol=0.0)
        if not inside:
            q = self.closest_point(p)
            diff = p - q
            dist = float(np.linalg.norm(diff))
            if dist <= 1e-12:
                # Degenerate numeric case: fall back to a stable axis normal.
                n = np.array([1.0, 0.0, 0.0], dtype=float)
                return q, n, 0.0, False
            n = diff / dist
            return q, n, dist, False

        # Inside: push out through the nearest face.
        dx0 = float(p[0] - self.x_min)
        dx1 = float(self.x_max - p[0])
        dy0 = float(p[1] - self.y_min)
        dy1 = float(self.y_max - p[1])
        dz0 = float(p[2] - self.z_min)
        dz1 = float(self.z_max - p[2])
        dists = np.array([dx0, dx1, dy0, dy1, dz0, dz1], dtype=float)
        face = int(np.argmin(dists))

        if face == 0:
            q = np.array([self.x_min, p[1], p[2]], dtype=float)
            n = np.array([-1.0, 0.0, 0.0], dtype=float)
        elif face == 1:
            q = np.array([self.x_max, p[1], p[2]], dtype=float)
            n = np.array([1.0, 0.0, 0.0], dtype=float)
        elif face == 2:
            q = np.array([p[0], self.y_min, p[2]], dtype=float)
            n = np.array([0.0, -1.0, 0.0], dtype=float)
        elif face == 3:
            q = np.array([p[0], self.y_max, p[2]], dtype=float)
            n = np.array([0.0, 1.0, 0.0], dtype=float)
        elif face == 4:
            q = np.array([p[0], p[1], self.z_min], dtype=float)
            n = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            q = np.array([p[0], p[1], self.z_max], dtype=float)
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        return q, n, 0.0, True


def aabb_intersects(a: BoxRegion3D, b: BoxRegion3D, penetration_tol: float = 0.0) -> bool:
    """Return True if AABBs intersect with overlap greater than penetration_tol on all axes."""
    tol = float(max(0.0, penetration_tol))
    ox = min(float(a.x_max), float(b.x_max)) - max(float(a.x_min), float(b.x_min))
    oy = min(float(a.y_max), float(b.y_max)) - max(float(a.y_min), float(b.y_min))
    oz = min(float(a.z_max), float(b.z_max)) - max(float(a.z_min), float(b.z_min))
    return bool((ox > tol) and (oy > tol) and (oz > tol))


def closest_points_between_aabbs(
    obstacle: BoxRegion3D, robot: BoxRegion3D
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Closest points between 2 axis-aligned boxes.

    Args:
      obstacle: obstacle AABB
      robot: robot AABB

    Returns:
      p_obs  : point on obstacle (3,)
      p_robot: point on robot (3,)
      n      : unit normal pointing from obstacle to robot (3,)
      dist   : Euclidean distance between p_obs and p_robot (>= 0)
      pen    : min penetration depth (only meaningful when dist==0), else 0
    """

    def _closest_1d(o0: float, o1: float, r0: float, r1: float) -> Tuple[float, float]:
        # robot strictly "below" obstacle interval
        if r1 < o0:
            return o0, r1
        # robot strictly "above" obstacle interval
        if o1 < r0:
            return o1, r0
        # overlap: choose a representative common coordinate
        c0 = max(o0, r0)
        c1 = min(o1, r1)
        c = 0.5 * (c0 + c1)
        return c, c

    o0x, o1x = float(obstacle.x_min), float(obstacle.x_max)
    o0y, o1y = float(obstacle.y_min), float(obstacle.y_max)
    o0z, o1z = float(obstacle.z_min), float(obstacle.z_max)
    r0x, r1x = float(robot.x_min), float(robot.x_max)
    r0y, r1y = float(robot.y_min), float(robot.y_max)
    r0z, r1z = float(robot.z_min), float(robot.z_max)

    px_o, px_r = _closest_1d(o0x, o1x, r0x, r1x)
    py_o, py_r = _closest_1d(o0y, o1y, r0y, r1y)
    pz_o, pz_r = _closest_1d(o0z, o1z, r0z, r1z)

    p_obs = np.array([px_o, py_o, pz_o], dtype=float)
    p_robot = np.array([px_r, py_r, pz_r], dtype=float)
    diff = p_robot - p_obs
    dist = float(np.linalg.norm(diff))
    if dist > 1e-12:
        n = diff / dist
        return p_obs, p_robot, n, dist, 0.0

    # Overlapping AABBs: choose a separating axis with minimum penetration.
    ox = min(o1x, r1x) - max(o0x, r0x)
    oy = min(o1y, r1y) - max(o0y, r0y)
    oz = min(o1z, r1z) - max(o0z, r0z)
    pen = float(min(ox, oy, oz))

    c_obs = np.array([(o0x + o1x) * 0.5, (o0y + o1y) * 0.5, (o0z + o1z) * 0.5], dtype=float)
    c_robot = np.array([(r0x + r1x) * 0.5, (r0y + r1y) * 0.5, (r0z + r1z) * 0.5], dtype=float)

    axis = int(np.argmin([ox, oy, oz]))
    n = np.zeros(3, dtype=float)
    sign = 1.0
    if float(c_robot[axis] - c_obs[axis]) < 0.0:
        sign = -1.0
    n[axis] = sign

    # Contact points on faces along chosen axis; other axes use overlap midpoints.
    midx = 0.5 * (max(o0x, r0x) + min(o1x, r1x))
    midy = 0.5 * (max(o0y, r0y) + min(o1y, r1y))
    midz = 0.5 * (max(o0z, r0z) + min(o1z, r1z))
    p_obs = np.array([midx, midy, midz], dtype=float)
    p_robot = np.array([midx, midy, midz], dtype=float)

    if axis == 0:
        if sign > 0:
            p_obs[0] = o1x
            p_robot[0] = r0x
        else:
            p_obs[0] = o0x
            p_robot[0] = r1x
    elif axis == 1:
        if sign > 0:
            p_obs[1] = o1y
            p_robot[1] = r0y
        else:
            p_obs[1] = o0y
            p_robot[1] = r1y
    else:
        if sign > 0:
            p_obs[2] = o1z
            p_robot[2] = r0z
        else:
            p_obs[2] = o0z
            p_robot[2] = r1z

    # dist is 0 by definition for overlap; n is axis-aligned.
    return p_obs, p_robot, n, 0.0, pen


@dataclass(frozen=True)
class HalfSpaceRegion3D:
    """Half-space obstacle {p : n·p <= d} with optional active x-range."""

    n: np.ndarray  # (3,) unit normal
    d: float  # plane equation n·p = d
    viz_quad: np.ndarray  # (4,3) corners for visualization
    x_lo: float = -np.inf
    x_hi: float = np.inf

    def contains(self, point: np.ndarray, tol: float = 0.0) -> bool:
        p = np.asarray(point, dtype=float).reshape(3)
        return bool(float(np.dot(self.n, p)) <= float(self.d) + float(tol))

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(3)
        n = np.asarray(self.n, dtype=float).reshape(3)
        n = n / float(np.linalg.norm(n))
        dist = float(np.dot(n, p) - self.d)
        return p - dist * n

    def closest_point_and_normal(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        n = np.asarray(self.n, dtype=float).reshape(3)
        n = n / float(np.linalg.norm(n))
        p = np.asarray(point, dtype=float).reshape(3)
        q = self.closest_point(p)
        inside = bool(float(np.dot(n, p)) <= float(self.d))
        dist = abs(float(np.dot(n, p) - self.d))
        return q, n.copy(), dist, inside

    def get_convex_rep(self) -> Tuple[np.ndarray, np.ndarray]:
        n = np.asarray(self.n, dtype=float).reshape(3)
        return n.reshape(1, 3), np.array([[float(self.d)]], dtype=float)

    def vertices(self) -> np.ndarray:
        """Return a thin slab around viz_quad (8 vertices) for generic plotting code."""
        v = np.asarray(self.viz_quad, dtype=float).reshape(4, 3)
        n = np.asarray(self.n, dtype=float).reshape(3)
        n = n / float(np.linalg.norm(n))
        eps = 1e-5
        v_back = v - eps * n
        return np.vstack([v_back, v]).astype(float)


def obb_intersects_halfspace(
    *,
    obb_center: np.ndarray,
    obb_R: np.ndarray,
    obb_half: np.ndarray,
    n: np.ndarray,
    d: float,
    penetration_tol: float = 0.0,
    x_lo: float = -np.inf,
    x_hi: float = np.inf,
) -> bool:
    """Return True if OBB intersects half-space {p : n·p <= d} within x-range [x_lo, x_hi]."""
    c = np.asarray(obb_center, dtype=float).reshape(3)
    R = np.asarray(obb_R, dtype=float).reshape(3, 3)
    h = np.asarray(obb_half, dtype=float).reshape(3)
    signs = np.array(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
        ],
        dtype=float,
    )
    verts = c + (R @ (signs * h).T).T
    verts_x = verts[:, 0]
    if float(np.max(verts_x)) < float(x_lo) - 1e-9 or float(np.min(verts_x)) > float(x_hi) + 1e-9:
        return False

    n_vec = np.asarray(n, dtype=float).reshape(3)
    n_vec = n_vec / float(np.linalg.norm(n_vec))
    vals = verts @ n_vec
    min_val = float(np.min(vals))
    if min_val > float(d) + float(penetration_tol):
        return False
    return True


def obb_intersects_aabb(
    *,
    obb_center: np.ndarray,
    obb_R: np.ndarray,
    obb_half: np.ndarray,
    aabb: BoxRegion3D,
    penetration_tol: float = 0.0,
) -> bool:
    """Return True if an oriented box (OBB) intersects an axis-aligned box (AABB).

    Uses the standard Separating Axis Theorem (SAT) test (15 candidate axes).

    `penetration_tol` is interpreted like the rest of this repo: overlaps smaller than this
    are treated as non-colliding.
    """
    tol = float(max(0.0, penetration_tol))

    cB = np.asarray(obb_center, dtype=float).reshape(3)
    R = np.asarray(obb_R, dtype=float).reshape(3, 3)
    eB = np.asarray(obb_half, dtype=float).reshape(3)

    # AABB as OBB with identity rotation.
    cA = np.array(
        [
            0.5 * (float(aabb.x_min) + float(aabb.x_max)),
            0.5 * (float(aabb.y_min) + float(aabb.y_max)),
            0.5 * (float(aabb.z_min) + float(aabb.z_max)),
        ],
        dtype=float,
    )
    eA = np.array(
        [
            0.5 * (float(aabb.x_max) - float(aabb.x_min)),
            0.5 * (float(aabb.y_max) - float(aabb.y_min)),
            0.5 * (float(aabb.z_max) - float(aabb.z_min)),
        ],
        dtype=float,
    )

    # Translation from A to B in A coordinates (world frame).
    t = cB - cA

    # Rotation matrix from A to B: R_ij = dot(A_i, B_j). With A=I, that's just B_j components.
    Rm = R
    eps = 1e-9
    absR = np.abs(Rm) + eps

    # Helper: if abs(x) > ra+rb - tol => separated (overlap <= tol is not collision)
    def _sep(val: float, ra: float, rb: float) -> bool:
        thr = ra + rb - tol
        if thr < 0.0:
            thr = 0.0
        return bool(abs(val) > thr)

    # Test axes L = A0, A1, A2 (world axes)
    for i in range(3):
        ra = float(eA[i])
        rb = float(eB[0] * absR[i, 0] + eB[1] * absR[i, 1] + eB[2] * absR[i, 2])
        if _sep(float(t[i]), ra, rb):
            return False

    # Test axes L = B0, B1, B2
    tB = Rm.T @ t
    for j in range(3):
        ra = float(eA[0] * absR[0, j] + eA[1] * absR[1, j] + eA[2] * absR[2, j])
        rb = float(eB[j])
        if _sep(float(tB[j]), ra, rb):
            return False

    # Test axis L = A_i x B_j
    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        for j in range(3):
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            val = float(t[i2] * Rm[i1, j] - t[i1] * Rm[i2, j])
            ra = float(eA[i1] * absR[i2, j] + eA[i2] * absR[i1, j])
            rb = float(eB[j1] * absR[i, j2] + eB[j2] * absR[i, j1])
            if _sep(val, ra, rb):
                return False

    return True

