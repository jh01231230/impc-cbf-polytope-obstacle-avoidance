from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from cbf_geometry import PolytopeRegion, RectangleRegion


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    start: np.ndarray
    goal: np.ndarray
    obstacles: List[RectangleRegion | PolytopeRegion]
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    grid: Tuple[Tuple[float, float], float]


def _create_s_path() -> EnvironmentSpec:
    s = 1.0
    start = np.array([0.0 * s, 0.2 * s, 0.0])
    goal = np.array([1.0 * s, 0.8 * s])
    bounds = ((-0.2 * s, 0.0 * s), (1.2 * s, 1.2 * s))
    cell_size = 0.05 * s
    obstacles: List[RectangleRegion | PolytopeRegion] = [
        RectangleRegion(0.0 * s, 1.0 * s, 0.9 * s, 1.0 * s),
        RectangleRegion(0.0 * s, 0.4 * s, 0.4 * s, 1.0 * s),
        RectangleRegion(0.6 * s, 1.0 * s, 0.0 * s, 0.7 * s),
    ]
    return EnvironmentSpec("s_path", start, goal, obstacles, bounds, (bounds, cell_size))


def _create_maze() -> EnvironmentSpec:
    s = 0.15
    start = np.array([0.5 * s, 5.5 * s, -np.pi / 2.0])
    goal = np.array([12.5 * s, 0.5 * s])
    bounds = ((0.0 * s, 0.0 * s), (13.0 * s, 6.0 * s))
    cell_size = 0.25 * s
    obstacles = [
        RectangleRegion(0.0 * s, 3.0 * s, 0.0 * s, 3.0 * s),
        RectangleRegion(1.0 * s, 2.0 * s, 4.0 * s, 6.0 * s),
        RectangleRegion(2.0 * s, 6.0 * s, 5.0 * s, 6.0 * s),
        RectangleRegion(6.0 * s, 7.0 * s, 4.0 * s, 6.0 * s),
        RectangleRegion(4.0 * s, 5.0 * s, 0.0 * s, 4.0 * s),
        RectangleRegion(5.0 * s, 7.0 * s, 2.0 * s, 3.0 * s),
        RectangleRegion(6.0 * s, 9.0 * s, 1.0 * s, 2.0 * s),
        RectangleRegion(8.0 * s, 9.0 * s, 2.0 * s, 4.0 * s),
        RectangleRegion(9.0 * s, 12.0 * s, 3.0 * s, 4.0 * s),
        RectangleRegion(11.0 * s, 12.0 * s, 4.0 * s, 5.0 * s),
        RectangleRegion(8.0 * s, 10.0 * s, 5.0 * s, 6.0 * s),
        RectangleRegion(10.0 * s, 11.0 * s, 0.0 * s, 2.0 * s),
        RectangleRegion(12.0 * s, 13.0 * s, 1.0 * s, 2.0 * s),
        RectangleRegion(0.0 * s, 13.0 * s, 6.0 * s, 7.0 * s),
        RectangleRegion(-1.0 * s, 0.0 * s, -1.0 * s, 7.0 * s),
        RectangleRegion(0.0 * s, 13.0 * s, -1.0 * s, 0.0 * s),
        RectangleRegion(13.0 * s, 14.0 * s, -1.0 * s, 7.0 * s),
    ]
    return EnvironmentSpec("maze", start, goal, obstacles, bounds, (bounds, cell_size))


def _create_oblique_maze() -> EnvironmentSpec:
    s = 0.15
    start = np.array([1.0 * s, 1.5 * s, 0.0])
    goal = np.array([8.5 * s, 6.5 * s])
    bounds = ((0.0 * s, 1.0 * s), (10.0 * s, 7.0 * s))
    cell_size = 0.2 * s
    obstacles: List[RectangleRegion | PolytopeRegion] = [
        RectangleRegion(-1.0 * s, 0.0 * s, 0.0 * s, 8.0 * s),
        RectangleRegion(0.0 * s, 10.0 * s, 0.0 * s, 1.0 * s),
        RectangleRegion(0.0 * s, 8.0 * s, 7.0 * s, 8.0 * s),
        RectangleRegion(10.0 * s, 11.0 * s, 0.0 * s, 8.0 * s),
        PolytopeRegion.convex_hull(s * np.array([[0.0, 2.0], [1.25, 3.875], [2.875, 3.125], [2.5, 2.25]])),
        PolytopeRegion.convex_hull(s * np.array([[1, 4.75], [0.0, 5.0], [0.875, 7], [1.875, 6.375]])),
        PolytopeRegion.convex_hull(
            s * np.array([[2.75, 1], [4.2, 3.25], [5.125, 3.75], [6.625, 2.5], [6.5, 1.0]])
        ),
        PolytopeRegion.convex_hull(s * np.array([[6.0, 7.0], [6, 6], [6.5, 7.0]])),
        PolytopeRegion.convex_hull(
            s * np.array([[2.375, 4.875], [2.875, 5.875], [4.5, 5.875], [4.75, 4], [3.375, 4]])
        ),
        PolytopeRegion.convex_hull(s * np.array([[6.75, 1.0], [7.25, 2.375], [8.5, 2.0], [8.5, 1.0]])),
        PolytopeRegion.convex_hull(s * np.array([[8.625, 1.0], [10.0, 2.5], [10.0, 1.0]])),
        PolytopeRegion.convex_hull(s * np.array([[10.0, 2.875], [9.5, 5.75], [10.0, 5.875]])),
        PolytopeRegion.convex_hull(
            s
            * np.array(
                [[8.875, 3.125], [8.0, 5.5], [6.875, 6.375], [5.875, 5.875], [6.25, 4.375], [7.125, 3.5]]
            )
        ),
    ]
    return EnvironmentSpec("oblique_maze", start, goal, obstacles, bounds, (bounds, cell_size))


def load_environment(name: str = "oblique_maze") -> EnvironmentSpec:
    name = name.lower()
    if name == "s_path":
        return _create_s_path()
    if name == "maze":
        return _create_maze()
    if name == "oblique_maze":
        return _create_oblique_maze()
    raise ValueError(f"Unknown environment {name}")


ENVIRONMENT = load_environment()
OBSTACLE_REGIONS = ENVIRONMENT.obstacles
OBSTACLE_HALFSPACES = [(A, b.reshape(-1)) for A, b in (region.get_convex_rep() for region in OBSTACLE_REGIONS)]
OBSTACLE_VERTICES = [region.vertices() for region in OBSTACLE_REGIONS]

