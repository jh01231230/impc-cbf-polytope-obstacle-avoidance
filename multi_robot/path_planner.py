from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from cbf_geometry import ConvexRegion2D

try:
    import casadi as ca  # type: ignore
except ImportError:  # pragma: no cover - CasADi is optional
    ca = None


# ---------------------------------------------------------------------------
# Optional CasADi-based optimization planner (not used by default).
# ---------------------------------------------------------------------------

@dataclass
class OptimizationPlanner:
    obstacles: Sequence[ConvexRegion2D]
    margin: float
    horizon: int

    def __post_init__(self) -> None:
        if ca is None:
            raise RuntimeError("CasADi is required for the optimization planner.")
        self._opti = ca.Opti()
        self._p = self._opti.variable(2, self.horizon + 2)
        self._gamma = self._opti.variable(self.horizon + 1, 1)
        self._start = self._opti.parameter(2, 1)
        self._goal = self._opti.parameter(2, 1)
        self._build_problem()

    def _build_problem(self) -> None:
        self._opti.subject_to(self._p[:, 0] == self._start)
        self._opti.subject_to(self._p[:, -1] == self._goal)

        cost = 0.0
        for t in range(1, self.horizon + 2):
            diff = self._p[:, t] - self._p[:, t - 1]
            cost += ca.mtimes(diff.T, diff)

        for obstacle in self.obstacles:
            A_obs, b_obs = obstacle.get_convex_rep()
            A_obs = np.asarray(A_obs, dtype=float)
            b_obs = np.asarray(b_obs, dtype=float).reshape(-1, 1)
            mu = self._opti.variable(self.horizon + 1, A_obs.shape[0])
            for t in range(self.horizon + 1):
                normal = ca.mtimes(mu[t, :], A_obs)
                self._opti.subject_to(
                    -ca.mtimes(normal, normal.T) / 4
                    + ca.mtimes(normal, self._p[:, t + 1])
                    - ca.mtimes(mu[t, :], b_obs)
                    - self._gamma[t]
                    >= self.margin**2
                )
                self._opti.subject_to(self._gamma[t] + ca.mtimes(normal, (self._p[:, t] - self._p[:, t + 1])) >= 0)
                self._opti.subject_to(self._gamma[t] >= 0)
                self._opti.subject_to(mu[t, :] >= 0)
            self._opti.set_initial(mu, 0.05 * np.ones((self.horizon + 1, A_obs.shape[0])))

        for t in range(self.horizon + 1):
            self._opti.set_initial(self._p[:, t], 0.0)
            self._opti.set_initial(self._gamma[t], 0.0)

        self._opti.minimize(cost)
        self._opti.solver(
            "ipopt",
            {"verbose": False, "ipopt.print_level": 0, "print_time": 0},
            {"max_iter": 10000},
        )

    def plan(self, start: Iterable[float], goal: Iterable[float]) -> np.ndarray:
        start = np.asarray(start, dtype=float).reshape(2)
        goal = np.asarray(goal, dtype=float).reshape(2)
        self._opti.set_value(self._start, start.reshape(2, 1))
        self._opti.set_value(self._goal, goal.reshape(2, 1))

        for t in range(self.horizon + 1):
            guess = start + (goal - start) * (t / (self.horizon + 1))
            self._opti.set_initial(self._p[:, t], guess)
            self._opti.set_initial(self._gamma[t], 0.0)
        self._opti.set_initial(self._p[:, -1], goal)

        sol = self._opti.solve()
        return sol.value(self._p).T


# ---------------------------------------------------------------------------
# Graph-search planners (reference: planning/path_generator/astar.py)
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    pos: np.ndarray
    parent: "_Node | None" = None
    g_cost: float = math.inf
    f_cost: float = math.inf

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Node):
            return NotImplemented
        # Tie-breaking: use exact position equality on grid nodes.
        return bool(np.all(self.pos == other.pos))

    def __le__(self, other: "_Node") -> bool:
        if self.pos[0] == other.pos[0]:
            return bool(self.pos[1] <= other.pos[1])
        return bool(self.pos[0] <= other.pos[0])

    def __lt__(self, other: "_Node") -> bool:
        if self.pos[0] == other.pos[0]:
            return bool(self.pos[1] < other.pos[1])
        return bool(self.pos[0] < other.pos[0])


class _GridMap:
    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]], cell_size: float, quad: bool):
        self.bounds = bounds
        self.cell_size = cell_size
        self.quad = quad
        self.Nx = max(1, math.ceil((bounds[1][0] - bounds[0][0]) / cell_size))
        self.Ny = max(1, math.ceil((bounds[1][1] - bounds[0][1]) / cell_size))
        self.grid: List[List[_Node]] = [
            [
                _Node(
                    np.array(
                        [
                            bounds[0][0] + (i + 0.5) * cell_size,
                            bounds[0][1] + (j + 0.5) * cell_size,
                        ],
                        dtype=float,
                    )
                )
                for j in range(self.Ny)
            ]
            for i in range(self.Nx)
        ]

    def _index(self, pos: np.ndarray) -> Tuple[int, int]:
        # Assume query is within bounds; no clipping.
        ix = int(math.floor((pos[0] - self.bounds[0][0]) / self.cell_size))
        iy = int(math.floor((pos[1] - self.bounds[0][1]) / self.cell_size))
        return ix, iy

    def node(self, pos: np.ndarray) -> _Node:
        ix, iy = self._index(pos)
        return self.grid[ix][iy]

    def set_node(self, pos: np.ndarray, parent: _Node | None, g_cost: float, f_cost: float) -> _Node:
        node = self.node(pos)
        node.parent = parent
        node.g_cost = g_cost
        node.f_cost = f_cost
        return node

    def neighbours(self, node: _Node) -> Sequence[_Node]:
        ix, iy = self._index(node.pos)
        neigh: List[_Node] = []
        for i in range(ix - 1, ix + 2):
            for j in range(iy - 1, iy + 2):
                if i == ix and j == iy:
                    continue
                if self.quad and abs(i - ix) + abs(j - iy) > 1:
                    continue
                if 0 <= i < self.Nx and 0 <= j < self.Ny:
                    neigh.append(self.grid[i][j])
        return neigh


class _GraphSearch:
    def __init__(self, graph: _GridMap, obstacles: Sequence[ConvexRegion2D], margin: float):
        self.graph = graph
        self.obstacles = obstacles
        self.margin = margin

    def _collision(self, pos: np.ndarray) -> bool:
        for region in self.obstacles:
            A, b = region.get_convex_rep()
            A = np.asarray(A, dtype=float)
            b = np.asarray(b, dtype=float).reshape(-1)
            if np.all(A @ pos - b - self.margin * np.linalg.norm(A, axis=1) <= 0):
                return True
        return False

    def _heuristic(self, pos: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(goal - pos))

    def _edge_cost(self, n1: _Node, n2: _Node) -> float:
        return float(np.linalg.norm(n1.pos - n2.pos))

    def _reconstruct(self, node: _Node) -> np.ndarray:
        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        return np.asarray([n.pos for n in reversed(path)], dtype=float)

    def _reconstruct_nodes(self, node: _Node) -> List[_Node]:
        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        return [path[len(path) - i - 1] for i in range(len(path))]

    def line_of_sight(self, a: _Node, b: _Node) -> bool:
        # Standard A* on the occupancy grid.
        e = float(self.graph.cell_size)
        div = float(np.linalg.norm(b.pos - a.pos)) / e
        if div < 1e-9:
            return True
        for i in range(1, int(math.floor(div)) + 1):
            pos = (b.pos * i + a.pos * (div - i)) / div
            if self._collision(pos):
                return False
        return True

    def a_star(self, start: np.ndarray, goal: np.ndarray) -> List[_Node]:
        open_set: List[Tuple[float, _Node]] = []
        start_node = self.graph.set_node(start, None, 0.0, self._heuristic(start, goal))
        goal_node = self.graph.node(goal)
        heapq.heappush(open_set, (start_node.f_cost, start_node))

        while len(open_set) > 0:
            current = open_set[0][1]
            if current == goal_node:
                return self._reconstruct_nodes(current)

            heapq.heappop(open_set)

            for neigh in self.graph.neighbours(current):
                if self._collision(neigh.pos):
                    continue
                g_score = current.g_cost + self._edge_cost(current, neigh)
                if g_score < neigh.g_cost:
                    updated = self.graph.set_node(
                        neigh.pos, current, g_score, g_score + self._heuristic(neigh.pos, goal)
                    )
                    if not (neigh in (x[1] for x in open_set)):
                        heapq.heappush(open_set, (updated.f_cost, updated))
        return []

    def theta_star(self, start: np.ndarray, goal: np.ndarray) -> List[_Node]:
        open_set: List[Tuple[float, _Node]] = []
        start_node = self.graph.set_node(start, None, 0.0, self._heuristic(start, goal))
        goal_node = self.graph.node(goal)
        heapq.heappush(open_set, (start_node.f_cost, start_node))

        while len(open_set) > 0:
            current = open_set[0][1]
            if current == goal_node:
                return self._reconstruct_nodes(current)

            heapq.heappop(open_set)

            for neigh in self.graph.neighbours(current):
                if self._collision(neigh.pos):
                    continue
                if current.parent is not None and self.line_of_sight(current.parent, neigh):
                    parent = current.parent
                    g_score = parent.g_cost + self._edge_cost(parent, neigh)
                    if g_score < neigh.g_cost:
                        updated = self.graph.set_node(
                            neigh.pos, parent, g_score, g_score + self._heuristic(neigh.pos, goal)
                        )
                        _remove_node(open_set, neigh)
                        heapq.heappush(open_set, (updated.f_cost, updated))
                else:
                    g_score = current.g_cost + self._edge_cost(current, neigh)
                    if g_score < neigh.g_cost:
                        updated = self.graph.set_node(
                            neigh.pos, current, g_score, g_score + self._heuristic(neigh.pos, goal)
                        )
                        _remove_node(open_set, neigh)
                        heapq.heappush(open_set, (updated.f_cost, updated))
        return []

    def reduce_path(self, path: List[_Node]) -> List[_Node]:
        """Line-of-sight path reduction."""
        red_path: List[_Node] = []
        if len(path) > 1:
            for i in range(1, len(path)):
                if (
                    path[i].parent is not None
                    and path[i].parent.parent is not None
                    and self.line_of_sight(path[i], path[i].parent.parent)
                ):
                    path[i].parent = path[i].parent.parent
                else:
                    if path[i].parent is not None:
                        red_path.append(path[i].parent)
        if path:
            red_path.append(path[-1])
        return red_path


def _remove_node(open_set: List[Tuple[float, int, _Node]], node: _Node) -> None:
    for i, (_, _, cand) in enumerate(open_set):
        if cand == node:
            open_set[i] = open_set[-1]
            open_set.pop()
            if i < len(open_set):
                heapq._siftup(open_set, i)
                heapq._siftdown(open_set, 0, i)
            break


# ---------------------------------------------------------------------------
# Unified guiding path interface
# ---------------------------------------------------------------------------

def plan_path(
    start: Iterable[float],
    goal: Iterable[float],
    obstacles: Sequence[ConvexRegion2D],
    horizon: int = 40,
    margin: float = 0.05,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
    cell_size: float | None = None,
    use_grid: bool = False,
    method: str = "astar_los",
    quad: bool = False,
) -> np.ndarray:
    start_arr = np.asarray(start, dtype=float).reshape(2)
    goal_arr = np.asarray(goal, dtype=float).reshape(2)

    if ca is not None and not use_grid:
        planner = OptimizationPlanner(obstacles=obstacles, margin=margin, horizon=horizon)
        try:
            return planner.plan(start_arr, goal_arr)
        except Exception as e:
            print(f"Optimization planner failed: {e}. Falling back to grid search.")

    if bounds is None or cell_size is None:
        span = np.vstack([start_arr, goal_arr])
        bounds = (
            (float(span[:, 0].min()) - 1.0, float(span[:, 1].min()) - 1.0),
            (float(span[:, 0].max()) + 1.0, float(span[:, 1].max()) + 1.0),
        )
        cell_size = 0.05

    grid = _GridMap(bounds=bounds, cell_size=cell_size, quad=quad)
    search = _GraphSearch(grid, obstacles, float(margin))

    method_norm = str(method).strip().lower()
    if method_norm not in {"theta_star", "astar", "astar_los"}:
        raise ValueError(f"Unknown method={method!r}. Expected 'theta_star', 'astar', or 'astar_los'.")

    if method_norm == "theta_star":
        path_nodes = search.theta_star(start_arr, goal_arr)
    else:
        path_nodes = search.a_star(start_arr, goal_arr)
        if method_norm == "astar_los":
            path_nodes = search.reduce_path(path_nodes)

    if not path_nodes:
        raise RuntimeError("Global Path not found.")
    return np.asarray([p.pos for p in path_nodes], dtype=float)

