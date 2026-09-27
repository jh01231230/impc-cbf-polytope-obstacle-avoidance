from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from matplotlib import patches
from scipy.spatial import ConvexHull


class ConvexRegion2D:
    """Abstract base for convex regions used by the reference CBF example."""

    def get_convex_rep(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_plot_patch(self):
        raise NotImplementedError

    def vertices(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class RectangleRegion(ConvexRegion2D):
    left: float
    right: float
    down: float
    up: float

    def get_convex_rep(self) -> Tuple[np.ndarray, np.ndarray]:
        mat_A = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype=float)
        vec_b = np.array([[-self.left], [-self.down], [self.right], [self.up]], dtype=float)
        return mat_A, vec_b

    def get_plot_patch(self):
        return patches.Rectangle(
            (self.left, self.down),
            self.right - self.left,
            self.up - self.down,
            linewidth=1,
            edgecolor="k",
            facecolor="r",
            alpha=0.4,
        )

    def vertices(self) -> np.ndarray:
        return np.array(
            [
                [self.left, self.down],
                [self.right, self.down],
                [self.right, self.up],
                [self.left, self.up],
            ],
            dtype=float,
        )


@dataclass
class PolytopeRegion(ConvexRegion2D):
    mat_A: np.ndarray
    vec_b: np.ndarray
    points: np.ndarray

    @classmethod
    def convex_hull(cls, points: np.ndarray) -> "PolytopeRegion":
        hull = ConvexHull(points)
        mat_A = hull.equations[:, :2]
        vec_b = -hull.equations[:, 2].reshape(-1, 1)
        ordered_points = points[hull.vertices]
        return cls(mat_A.astype(float), vec_b.astype(float), ordered_points.astype(float))

    def get_convex_rep(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mat_A.astype(float), self.vec_b.reshape(self.vec_b.shape[0], -1).astype(float)

    def get_plot_patch(self):
        return patches.Polygon(self.points, closed=True, linewidth=1, edgecolor="k", facecolor="r", alpha=0.4)

    def vertices(self) -> np.ndarray:
        return np.asarray(self.points, dtype=float)

