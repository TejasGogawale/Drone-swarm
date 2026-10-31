"""Ground navigation for the rescue rovers, built on the shared voxel map (ROS-free).

Pipeline: VoxelMap (3-D occupancy) -> Traversability (2-D ground grid) -> A* -> string-pulled waypoints
-> pure-pursuit Rover kinematics, with blocked-route detection and replanning.

The 2-D grid is a projection of the SAME occupancy the viewer shows, not a separate hard-coded map:
  * obstacle  = any occupied voxel whose height range overlaps the rover body band [z_lo, z_hi];
  * blocked   = obstacle inflated by the rover radius + margin (the rover's footprint/clearance);
  * known free = ground cells that sensors have actually observed free;
  * unknown   = everything else; routed through only if the policy allows it.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NavParams:
    radius: float = 0.45            # rover half-width (m)
    margin: float = 0.2             # extra clearance (m)
    z_lo: float = 0.15              # obstacle band: below this is ground texture
    z_hi: float = 1.2               # above this the rover passes underneath (clearance)
    p_occ: float = 0.5
    free_l: float = 0.3             # ground log-odds needed to count as observed-free
    unknown_policy: str = "forbid"  # 'forbid' | 'penalize'
    unknown_cost: float = 3.0
    max_speed: float = 1.6
    max_yaw_rate: float = 1.8
    accel: float = 1.2
    lookahead: float = 1.4
    goal_tol: float = 0.35
    standoff: float = 1.5           # stop within this distance of the survivor
    target_min: float = 0.95        # ...but never closer than this (the survivor is a physical obstacle)
    target_exempt: float = 2.2      # inflation around the survivor is lifted out to here so the rover can reach the ring


def _disk(r_cells: int):
    return [(dr, dc) for dr in range(-r_cells, r_cells + 1) for dc in range(-r_cells, r_cells + 1)
            if dr * dr + dc * dc <= r_cells * r_cells]


def dilate(mask: np.ndarray, r_cells: int) -> np.ndarray:
    if r_cells <= 0:
        return mask.copy()
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dr, dc in _disk(r_cells):
        r0, r1 = max(0, dr), min(h, h + dr)
        c0, c1 = max(0, dc), min(w, w + dc)
        out[r0:r1, c0:c1] |= mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


class Traversability:
    """Ground-level navigation grid derived from a VoxelMap."""

    def __init__(self, vm, params: NavParams | None = None):
        self.vm = vm
        self.p = params or NavParams()
        self.version = -1
        self.rebuild()

    def rebuild(self):
        vm, p = self.vm, self.p
        res = vm.res
        n = vm.n
        obstacle = np.zeros((n, n), bool)
        ijk, prob, *_ = vm.occupied_arrays(p.p_occ)
        if len(ijk):
            sel = ((ijk[:, 2] + 1) * res > p.z_lo) & (ijk[:, 2] * res < p.z_hi)
            off = int(round(vm.half / res))
            rows, cols = ijk[sel, 0] + off, ijk[sel, 1] + off
            ok = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
            obstacle[rows[ok], cols[ok]] = True
        gl, _, _ = vm.ground_snapshot()
        r_cells = int(math.ceil((p.radius + p.margin) / res))
        self.obstacle = obstacle
        self.blocked = dilate(obstacle, r_cells)
        near = dilate(obstacle, r_cells + 2) & ~self.blocked
        self.known_free = (gl > p.free_l) & ~self.blocked
        self.cost = 1.0 + 2.0 * near.astype(np.float32)
        self.version = vm.version

    def allowed(self) -> np.ndarray:
        if self.p.unknown_policy == "penalize":
            return ~self.blocked
        return self.known_free

    def cell_blocked(self, row: int, col: int) -> bool:
        return (not self.vm.in_grid(row, col)) or bool(self.blocked[row, col])

    def segment_blocked(self, a, b, step: float | None = None, target=None) -> bool:
        """World-space segment a->b (n, e) crosses a blocked cell? Cells within the target-approach ring
        (inflation lifted around the survivor, see plan()) are not counted."""
        step = step or self.vm.res * 0.5
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        k = max(1, int(dist / step))
        for s in range(k + 1):
            f = s / k
            pn, pe = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            r, c = self.vm.cell(pn, pe)
            if self.cell_blocked(r, c):
                if target is not None and self.vm.in_grid(r, c) and not self.obstacle[r, c]:
                    cn, ce = self.vm.cell_center(r, c)            # same test as plan(): by CELL centre
                    if self.p.target_min <= math.hypot(cn - target[0], ce - target[1]) <= self.p.target_exempt:
                        continue
                return True
        return False

    def route_blocked(self, route, start_idx: int, horizon: float = 12.0, from_pos=None, target=None) -> bool:
        """Is any part of the next `horizon` metres of the route now blocked?"""
        pts = ([from_pos] if from_pos is not None else []) + list(route[start_idx:])
        travelled = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            if self.segment_blocked(a, b, target=target):
                return True
            travelled += math.hypot(b[0] - a[0], b[1] - a[1])
            if travelled > horizon:
                break
        return False

    # ------------------------------------------------------------------------ planning
    def plan(self, start, target, standoff: float | None = None, max_expand: int = 120_000):
        """A* from world point `start` to any allowed cell within `standoff` of `target`.
        Returns (waypoints [(n, e), ...] after line-of-sight simplification, length_m) or None."""
        vm, p = self.vm, self.p
        standoff = p.standoff if standoff is None else standoff
        allowed = self.allowed().copy()
        sr, sc = vm.cell(*start)
        if not vm.in_grid(sr, sc):
            return None
        tr, tc = vm.cell(*target)
        res = vm.res
        h, w = allowed.shape
        # The survivor is mapped as an obstacle and inflated by the rover's footprint, which would hide every
        # cell next to them. Lift the inflation (not the survivor itself) in a ring around the target.
        rr = int(math.ceil(p.target_exempt / res)) + 1
        for r0 in range(max(0, tr - rr), min(h, tr + rr + 1)):
            for c0 in range(max(0, tc - rr), min(w, tc + rr + 1)):
                n0, e0 = vm.cell_center(r0, c0)
                d0 = math.hypot(n0 - target[0], e0 - target[1])
                if p.target_min <= d0 <= p.target_exempt and not self.obstacle[r0, c0]:
                    if p.unknown_policy == "penalize" or self.vm.ground_l[r0, c0] > p.free_l:
                        allowed[r0, c0] = True
        # the rover may always leave its own cell and the ring around it (inflation can swallow it)
        escape = np.zeros_like(allowed)
        escape[max(0, sr - 2):sr + 3, max(0, sc - 2):sc + 3] = True
        free_cell = allowed | (escape & ~self.obstacle)

        def hfun(r, c):
            dr, dc = abs(r - tr), abs(c - tc)
            d = (max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)) * res
            return max(0.0, d - standoff)

        def at_goal(r, c):
            n, e = vm.cell_center(r, c)
            return p.target_min <= math.hypot(n - target[0], e - target[1]) <= standoff

        g = {(sr, sc): 0.0}
        came = {}
        heap = [(hfun(sr, sc), 0.0, sr, sc)]
        closed = set()
        expanded = 0
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        unknown_pen = p.unknown_cost if p.unknown_policy == "penalize" else 0.0
        goal = None
        while heap and expanded < max_expand:
            f, gc, r, c = heapq.heappop(heap)
            if (r, c) in closed:
                continue
            closed.add((r, c))
            expanded += 1
            if at_goal(r, c):
                goal = (r, c)
                break
            for dr, dc, mv in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not free_cell[nr, nc] or (nr, nc) in closed:
                    continue
                if dr and dc and not (free_cell[r + dr, c] and free_cell[r, c + dc]):
                    continue                      # no corner cutting through blocked cells
                step = mv * res * float(self.cost[nr, nc])
                if unknown_pen and not self.known_free[nr, nc]:
                    step += unknown_pen * res
                ng = gc + step
                if ng < g.get((nr, nc), 1e18):
                    g[(nr, nc)] = ng
                    came[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (ng + hfun(nr, nc), ng, nr, nc))
        if goal is None:
            return None
        cells = [goal]
        while cells[-1] in came:
            cells.append(came[cells[-1]])
        cells.reverse()
        pts = [vm.cell_center(r, c) for r, c in cells]
        pts[0] = (float(start[0]), float(start[1]))
        wp = self._simplify(pts, free_cell)
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(wp[:-1], wp[1:]))
        return wp, length

    def _simplify(self, pts, free_cell):
        """String pulling: keep only the corner points needed so straight runs stay straight."""
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._clear(pts[i], pts[j], free_cell):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def _clear(self, a, b, free_cell):
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        k = max(1, int(dist / (self.vm.res * 0.5)))
        for s in range(k + 1):
            f = s / k
            r, c = self.vm.cell(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            if not self.vm.in_grid(r, c) or not free_cell[r, c]:
                return False
        return True


# ------------------------------------------------------------------------------ rover
@dataclass
class Rover:
    id: int
    name: str
    color: tuple                     # BGR
    n: float
    e: float
    yaw: float = 0.0
    v: float = 0.0
    w: float = 0.0
    state: str = "idle"              # idle | navigating | arrived | blocked
    mission_id: int | None = None
    target: tuple | None = None
    route: list = field(default_factory=list)
    route_idx: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=400))
    wheel_phase: float = 0.0
    odometer: float = 0.0
    replans: int = 0
    last_replan_t: float = -1e9
    bumps: int = 0

    def pos(self):
        return (self.n, self.e)

    def set_route(self, waypoints):
        self.route = [tuple(map(float, w)) for w in waypoints]
        self.route_idx = 1 if len(self.route) > 1 else 0
        self.state = "navigating"

    def dist_to_end(self) -> float:
        if not self.route:
            return 0.0
        return math.hypot(self.route[-1][0] - self.n, self.route[-1][1] - self.e)

    def step(self, dt: float, p: NavParams, world=None):
        """Pure-pursuit tracking with yaw-rate and acceleration limits. Never teleports."""
        if self.state != "navigating" or not self.route:
            self.v = max(0.0, self.v - p.accel * dt)
            self.w = 0.0
            return
        # advance past waypoints we have reached
        while self.route_idx < len(self.route) - 1 and \
                math.hypot(self.route[self.route_idx][0] - self.n, self.route[self.route_idx][1] - self.e) < 0.5:
            self.route_idx += 1
        # lookahead point on the remaining polyline
        tgt = self._lookahead(p.lookahead)
        desired = math.atan2(tgt[1] - self.e, tgt[0] - self.n)
        err = (desired - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.w = float(np.clip(2.5 * err, -p.max_yaw_rate, p.max_yaw_rate))
        remaining = self.dist_to_end()
        v_target = p.max_speed * max(0.0, math.cos(err)) ** 2
        if abs(err) > math.radians(70):
            v_target = 0.0                                  # turn in place first
        v_target = min(v_target, max(0.25, remaining * 0.9))
        dv = float(np.clip(v_target - self.v, -p.accel * dt, p.accel * dt))
        self.v = max(0.0, self.v + dv)
        self.yaw = (self.yaw + self.w * dt + math.pi) % (2 * math.pi) - math.pi
        nn = self.n + self.v * math.cos(self.yaw) * dt
        ne = self.e + self.v * math.sin(self.yaw) * dt
        if world is not None and world.blocked_at(nn, ne, margin=p.radius * 0.8):
            self.v = 0.0                                    # touched something the map does not know yet
            self.bumps += 1
            return
        self.odometer += math.hypot(nn - self.n, ne - self.e)
        self.wheel_phase += self.v * dt / 0.18
        self.n, self.e = nn, ne
        self.trail.append((self.n, self.e))
        if remaining < p.goal_tol and self.route_idx >= len(self.route) - 1:
            self.state = "arrived"

    def _lookahead(self, dist: float):
        pts = [(self.n, self.e)] + self.route[self.route_idx:]
        acc = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if acc + seg >= dist and seg > 1e-9:
                f = (dist - acc) / seg
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            acc += seg
        return pts[-1]
