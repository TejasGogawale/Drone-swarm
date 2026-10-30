"""Rescue swarm environment (grid world, no ROS).

Heterogeneous agents (drones/rovers) explore a collapsed-building map, sense
survivors via thermal/acoustic, and share information peer-to-peer within comms
range. Movement is high-level: each step a policy chooses a goal cell, and the
agent follows a BFS path to it (this mirrors "pick a goal, Nav2/setpoint drives").

Designed so the SAME goal-selection interface transfers to the ROS2 policy nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .map_gen import BuildingMap, FREE, WALL, NARROW, RUBBLE, ROVER_OK, DRONE_OK, SPEED_FACTOR
from .sensors import line_of_sight, thermal_detect, acoustic_observe

ACOUSTIC_THRESHOLD = 1.0  # observed loudness above this => detected

# Agent specs (cells per step, sensor range in cells)
SPECS = {
    "drone": {
        "speed": 2, "sensor_range": 8, "fov_deg": 360, "coarse": True,
        "battery": 250, "thermal": True, "acoustic": False,
    },
    "rover": {
        "speed": 1, "sensor_range": 6, "fov_deg": 90, "coarse": False,
        "battery": 800, "thermal": True, "acoustic": True,
    },
}


@dataclass
class Agent:
    id: int
    type: str  # 'drone' | 'rover'
    pos: tuple
    heading: float = 0.0
    battery: float = 1.0
    alive: bool = True
    explored: np.ndarray = field(default=None)  # local explored map (HxW bool)
    known: set = field(default_factory=set)     # survivor ids it knows about
    goal: tuple | None = None
    path: list = field(default_factory=list)


class RescueEnv:
    def __init__(self, bmap: BuildingMap, n_drones: int, n_rovers: int,
                 comm_range: float = 10.0, max_steps: int = 300, seed: int = 0):
        self.bmap = bmap
        self.comm_range = comm_range
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.n_drones = n_drones
        self.n_rovers = n_rovers
        self.agents: list[Agent] = []
        self.step_count = 0
        self.collisions = 0
        self.first_found_step = None
        self.all_found_step = None
        self._new_found = 0
        self._new_confirmed = 0
        self._step_collisions = 0
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.step_count = 0
        self.collisions = 0
        self.first_found_step = None
        self.all_found_step = None
        self._new_found = 0
        self._new_confirmed = 0
        for s in self.bmap.survivors:
            s.found = False
            s.confirmed = False
            s.found_by = set()
            s.detected_via = set()
        H, W = self.bmap.H, self.bmap.W
        self.agents = []
        entry = self._entry_point()
        # spawn around the shared entry point with known initial poses
        spawns = self._spawn_positions(entry, len_ok=True)
        ids = []
        for t, n in (("drone", self.n_drones), ("rover", self.n_rovers)):
            for _ in range(n):
                ids.append(t)
        self.agents = []
        for i, t in enumerate(ids):
            pos = spawns[i] if i < len(spawns) else entry
            ag = Agent(id=i, type=t, pos=pos, heading=0.0)
            ag.explored = np.zeros((H, W), dtype=bool)
            self.agents.append(ag)
        self._mark_initial_explored()
        return self._observe()

    # ------------------------------------------------------------------ #
    def _entry_point(self):
        """Bottom-left-most rover-traversable cell."""
        ys, xs = np.where(np.isin(self.bmap.grid, list(ROVER_OK)))
        if len(xs) == 0:
            return (0, 0)
        i = int(np.argmin(ys + xs))  # top-left-ish; fine as a stable entry
        return (int(xs[i]), int(ys[i]))

    def _spawn_positions(self, entry, len_ok=True):
        """A few cells around the entry point, all rover-traversable."""
        x, y = entry
        cands = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                xx, yy = x + dx, y + dy
                if self.bmap.in_bounds(xx, yy) and self.bmap.rover_ok(xx, yy):
                    cands.append((xx, yy))
        if cands:
            self.rng.shuffle(cands)
            return cands
        return [entry]

    def _mark_initial_explored(self):
        for ag in self.agents:
            self._sense(ag, initial=True)

    # ------------------------------------------------------------------ #
    def _traversable(self, ag: Agent) -> set:
        return DRONE_OK if ag.type == "drone" else ROVER_OK

    def _bfs(self, ag: Agent, goal) -> list:
        """BFS path (list of cells) from agent pos to goal, excluding start."""
        allowed = self._traversable(ag)
        grid = self.bmap.grid
        H, W = grid.shape
        start = ag.pos
        if start == goal:
            return []
        from collections import deque
        q = deque([start])
        came = {start: None}
        while q:
            cur = q.popleft()
            cx, cy = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if grid[ny, nx] not in allowed:
                    continue
                nxt = (nx, ny)
                if nxt in came:
                    continue
                came[nxt] = cur
                if nxt == goal:
                    # reconstruct
                    path = []
                    p = nxt
                    while p is not None and p != start:
                        path.append(p)
                        p = came[p]
                    return path[::-1]
                q.append(nxt)
        return []  # unreachable

    # ------------------------------------------------------------------ #
    def _sense(self, ag: Agent, initial: bool = False):
        """Update local explored map and detect survivors (passive sensing)."""
        spec = SPECS[ag.type]
        ax, ay = ag.pos
        H, W = self.bmap.H, self.bmap.W
        r = spec["sensor_range"]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                xx, yy = ax + dx, ay + dy
                if 0 <= xx < W and 0 <= yy < H:
                    if math.hypot(dx, dy) <= r and line_of_sight(self.bmap, ax, ay, xx, yy):
                        ag.explored[yy, xx] = True
        # survivor detection
        for s in self.bmap.survivors:
            if s.found and s.confirmed:
                continue
            sx, sy = s.pos
            detected = False
            modality = None
            if spec["thermal"] and s.heat:
                if thermal_detect(self.bmap, ax, ay, ag.heading, sx, sy,
                                  sensor_range=spec["sensor_range"], fov_deg=spec["fov_deg"],
                                  rng=self.rng, coarse=spec["coarse"]):
                    detected = True
                    modality = "thermal"
            if spec["acoustic"] and s.sound and not detected:
                obs = acoustic_observe(self.bmap, ax, ay, sx, sy, s.loudness, self.rng)
                if obs is not None and obs >= ACOUSTIC_THRESHOLD:
                    detected = True
                    modality = "acoustic"
            if detected:
                if not s.found:
                    s.found = True
                    self._new_found += 1
                    if self.first_found_step is None:
                        self.first_found_step = self.step_count
                if ag.id not in s.found_by:
                    s.found_by.add(ag.id)
                s.detected_via.add(modality)
                # "confirmed" = seen by a 2nd robot OR a 2nd sensor type
                if not s.confirmed and (len(s.found_by) >= 2 or len(s.detected_via) >= 2):
                    s.confirmed = True
                    self._new_confirmed += 1
        # update known survivor ids
        for s in self.bmap.survivors:
            if s.found:
                ag.known.add(s.id)

    def _comms(self):
        """Peer-to-peer sharing within comms range: union explored maps + known ids."""
        n = len(self.agents)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ai, aj = self.agents[i], self.agents[j]
                d = math.hypot(ai.pos[0] - aj.pos[0], ai.pos[1] - aj.pos[1])
                if d <= self.comm_range:
                    ai.explored = np.logical_or(ai.explored, aj.explored)
                    ai.known |= aj.known

    def _move(self, ag: Agent):
        """Advance agent up to its speed along its path (mutates ag in place)."""
        spec = SPECS[ag.type]
        ag.battery -= 0.004 if ag.type == "drone" else 0.001
        if ag.battery <= 0:
            ag.alive = False
            ag.path = []
            return
        moved = 0
        while moved < spec["speed"] and ag.path:
            nxt = ag.path[0]
            cell = self.bmap.grid[nxt[1], nxt[0]]
            # rubble slows rovers: they may fail to advance this substep
            if ag.type == "rover" and cell == RUBBLE and self.rng.random() < 0.5:
                break
            prev = ag.pos
            ag.pos = nxt
            ag.path.pop(0)
            dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
            if dx != 0 or dy != 0:
                ag.heading = math.atan2(dy, dx)
            moved += 1

    def _resolve_collisions(self, new_positions, orig_positions):
        seen = {}
        for i, p in enumerate(new_positions):
            if p in seen:
                # keep first, the other waits (restored to its original position)
                self.collisions += 1
                self._step_collisions += 1
                new_positions[i] = orig_positions[i]
            else:
                seen[p] = i
        return new_positions

    # ------------------------------------------------------------------ #
    def step(self, goals):
        """goals: list of (x,y) goal cells (or None to hold) per agent.
        Returns (obs, reward, done, info). Reward is a shaped TEAM reward."""
        if self.step_count >= self.max_steps:
            return self._observe(), 0.0, True, self._info()

        # reset per-step event counters
        self._new_found = 0
        self._new_confirmed = 0
        self._step_collisions = 0

        for ag, g in zip(self.agents, goals):
            if not ag.alive:
                continue
            if g is not None and g != ag.goal:
                ag.goal = g
                ag.path = self._bfs(ag, g)

        # record explored count before, for reward
        before = self.union_explored_count()

        # move each alive agent
        orig_positions = [ag.pos for ag in self.agents]
        new_positions = [ag.pos for ag in self.agents]
        for i, ag in enumerate(self.agents):
            if ag.alive and ag.path:
                self._move(ag)
                new_positions[i] = ag.pos
        new_positions = self._resolve_collisions(new_positions, orig_positions)
        for i, ag in enumerate(self.agents):
            if ag.alive:
                ag.pos = new_positions[i]

        # sense + comms + metrics
        for ag in self.agents:
            if ag.alive:
                self._sense(ag)
        self._comms()

        after = self.union_explored_count()
        reward = self._shaped_reward(after - before)

        self.step_count += 1

        all_found = all(s.found for s in self.bmap.survivors)
        if all_found and self.all_found_step is None:
            self.all_found_step = self.step_count

        done = all_found or all(not ag.alive for ag in self.agents) or self.step_count >= self.max_steps
        return self._observe(), reward, done, self._info()

    def _shaped_reward(self, explored_delta):
        """Team reward combining exploration, discovery, confirmation, and costs."""
        w_explore = 0.1
        w_found = 5.0
        w_confirm = 3.0
        w_collision = -0.5
        w_time = -0.02
        w_overlap = -0.05

        r = w_explore * explored_delta
        r += w_found * self._new_found
        r += w_confirm * self._new_confirmed
        r += w_collision * self._step_collisions
        r += w_time
        # overlap penalty: pairs of same-type agents within 2 cells
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                ai, aj = self.agents[i], self.agents[j]
                if ai.alive and aj.alive:
                    d = math.hypot(ai.pos[0] - aj.pos[0], ai.pos[1] - aj.pos[1])
                    if d < 2.0:
                        r += w_overlap
        return float(r)

    # ------------------------------------------------------------------ #
    def union_explored(self):
        u = np.zeros((self.bmap.H, self.bmap.W), dtype=bool)
        for ag in self.agents:
            u = np.logical_or(u, ag.explored)
        return u

    def union_explored_count(self):
        u = self.union_explored()
        return int(np.count_nonzero(u & self.bmap.traversable_cells()))

    def total_traversable(self):
        return int(np.count_nonzero(self.bmap.traversable_cells()))

    def coverage(self):
        return self.union_explored_count() / max(1, self.total_traversable())

    def survivors_found(self):
        return [s for s in self.bmap.survivors if s.found]

    def _info(self):
        return {
            "coverage": self.coverage(),
            "survivors_found": len(self.survivors_found()),
            "survivors_confirmed": len([s for s in self.bmap.survivors if s.confirmed]),
            "survivors_total": len(self.bmap.survivors),
            "first_found_step": self.first_found_step,
            "all_found_step": self.all_found_step,
            "collisions": self.collisions,
            "alive": [ag.alive for ag in self.agents],
        }

    def _observe(self):
        """Per-agent observation dict (used by learned policy in P2)."""
        obs = []
        for ag in self.agents:
            obs.append({
                "pos": ag.pos,
                "type": ag.type,
                "heading": ag.heading,
                "battery": ag.battery,
                "alive": ag.alive,
                "explored": ag.explored,
                "known": frozenset(ag.known),
            })
        return obs
