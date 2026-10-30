"""Baseline goal-selection policies (no learning).

All policies pick a high-level GOAL CELL each time their current goal is reached
(or is stale). The environment handles BFS path following to the goal, mirroring
how a ROS2 policy would send a Nav2 goal / drone setpoint.
"""
from __future__ import annotations

import math
import numpy as np
from collections import deque

from .map_gen import ROVER_OK, DRONE_OK
from .env import RescueEnv


def _frontiers(env: RescueEnv, ag):
    """Cells the agent has explored that border unexplored space, filtered by
    the agent's traversability."""
    allowed = DRONE_OK if ag.type == "drone" else ROVER_OK
    grid = env.bmap.grid
    H, W = grid.shape
    explored = ag.explored
    out = []
    ys, xs = np.where(explored)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if grid[y, x] not in allowed:
            continue
        # border unexplored?
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not explored[ny, nx]:
                out.append((x, y))
                break
    return out


def _bfs_dist(env: RescueEnv, ag):
    """Distances from agent to all reachable cells (BFS over traversable grid)."""
    allowed = DRONE_OK if ag.type == "drone" else ROVER_OK
    grid = env.bmap.grid
    H, W = grid.shape
    start = ag.pos
    dist = {start: 0}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] in allowed and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                q.append((nx, ny))
    return dist


def _random_traversable(env: RescueEnv, ag):
    allowed = DRONE_OK if ag.type == "drone" else ROVER_OK
    ys, xs = np.where(np.isin(env.bmap.grid, list(allowed)))
    if len(xs) == 0:
        return ag.pos
    i = env.rng.integers(0, len(xs))
    return (int(xs[i]), int(ys[i]))


class RandomPolicy:
    def __init__(self):
        self.goal = None

    def choose(self, env: RescueEnv, i: int):
        ag = env.agents[i]
        if self.goal is None or self.goal == ag.pos or not ag.path:
            fr = _frontiers(env, ag)
            self.goal = fr[env.rng.integers(0, len(fr))] if fr else _random_traversable(env, ag)
        return self.goal


class NearestFrontierPolicy:
    def __init__(self):
        self.goal = None

    def choose(self, env: RescueEnv, i: int):
        ag = env.agents[i]
        if self.goal is None or self.goal == ag.pos or not ag.path:
            fr = _frontiers(env, ag)
            if not fr:
                self.goal = _random_traversable(env, ag)
            else:
                dist = _bfs_dist(env, ag)
                self.goal = min(fr, key=lambda c: dist.get(c, 10 ** 9))
        return self.goal


class ExploreLitePolicy:
    """Frontier selection with information gain vs cost utility (explore_lite style)."""

    def __init__(self):
        self.goal = None

    def choose(self, env: RescueEnv, i: int):
        ag = env.agents[i]
        if self.goal is None or self.goal == ag.pos or not ag.path:
            fr = _frontiers(env, ag)
            if not fr:
                self.goal = _random_traversable(env, ag)
            else:
                dist = _bfs_dist(env, ag)
                explored = ag.explored
                H, W = env.bmap.H, env.bmap.W

                def utility(c):
                    x, y = c
                    gain = 0
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H and not explored[ny, nx]:
                            gain += 1
                    cost = dist.get(c, 10 ** 9)
                    return gain / (cost + 1.0)

                self.goal = max(fr, key=utility)
        return self.goal


BASELINES = {
    "random": RandomPolicy,
    "nearest_frontier": NearestFrontierPolicy,
    "explore_lite": ExploreLitePolicy,
}
