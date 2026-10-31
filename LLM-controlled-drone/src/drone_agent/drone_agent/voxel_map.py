"""Probabilistic 3-D voxel occupancy map (ROS-free).

This is the data layer behind the OctoMap viewer and the rover navigation. It keeps the same
conventions as the GUI's existing map: world frame = local NED horizontal axes (north, east) with
height z measured above the ground (AGL), so voxel (i, j, k) covers
    north in [i*res, (i+1)*res),  east in [j*res, (j+1)*res),  z in [k*res, (k+1)*res).

What it stores (the same kind of evidence an OctoMap keeps, but sparse and in plain Python):
  * per-voxel log-odds of occupancy (hit +l_hit, miss -l_miss, clamped), first-detection and
    last-update times -> occupancy probability, "recently updated", "newly detected obstacle";
  * coarse free-space voxels observed along sensor rays (for the optional free-space layer);
  * a ground-level grid: log-odds that a ground cell was observed free, which agents saw it and when
    -> unknown / free / explored-by layers and the rover traversability map.

It is NOT the octomap_server library: there is no octree and no pruning. See PROGRESS.md.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np


class VoxelMap:
    def __init__(self, res: float = 0.5, half: float = 60.0, l_hit: float = 0.85, l_miss: float = 0.4,
                 l_free: float = 0.5, l_min: float = -2.0, l_max: float = 3.5, max_voxels: int = 150_000,
                 max_free3d: int = 40_000):
        self.res = float(res)
        self.half = float(half)
        self.l_hit, self.l_miss, self.l_free = l_hit, l_miss, l_free
        self.l_min, self.l_max = l_min, l_max
        self.max_voxels = max_voxels
        self.max_free3d = max_free3d
        self.occ: dict[tuple, list] = {}          # (i, j, k) -> [logodds, t_first_occ|None, t_last, n_hits]
        self.free3d: dict[tuple, float] = {}      # coarse (2*res) free voxels -> t_last
        n = int(round(2 * self.half / self.res))
        self.n = n
        self.ground_l = np.zeros((n, n), np.float32)      # log-odds that the ground cell is free
        self.ground_seen = np.zeros((n, n), np.uint16)    # bitmask of agents that observed the cell
        self.ground_t = np.full((n, n), -1e9, np.float32)  # last time the cell was observed
        self.version = 0
        self.lock = threading.RLock()
        self.t0 = time.monotonic()

    # ----------------------------------------------------------------------- indexing
    def now(self) -> float:
        return time.monotonic() - self.t0

    def key(self, n: float, e: float, z: float) -> tuple:
        r = self.res
        return (int(math.floor(n / r)), int(math.floor(e / r)), int(math.floor(max(0.0, z) / r)))

    def center(self, key) -> tuple:
        r = self.res
        return ((key[0] + 0.5) * r, (key[1] + 0.5) * r, (key[2] + 0.5) * r)

    def cell(self, n: float, e: float) -> tuple:
        """Ground-grid (row, col) of a world point; may be outside the grid."""
        return (int(math.floor((n + self.half) / self.res)), int(math.floor((e + self.half) / self.res)))

    def cell_center(self, row: int, col: int) -> tuple:
        return (row + 0.5) * self.res - self.half, (col + 0.5) * self.res - self.half

    def in_grid(self, row: int, col: int) -> bool:
        return 0 <= row < self.n and 0 <= col < self.n

    @staticmethod
    def prob(logodds):
        return 1.0 / (1.0 + np.exp(-np.asarray(logodds, dtype=float)))

    # ------------------------------------------------------------------ integration
    def integrate_rays(self, origin, rays, agent_bit: int = 1, t: float | None = None):
        """origin = (n, e, z_agl) of the sensor.
        rays: (R, 5) array [n_end, e_end, z_lo, z_hi, hit]. For hit rays the column
        z in [z_lo, z_hi] at the end point is occupied; the horizontal path before it is free."""
        rays = np.asarray(rays, dtype=float).reshape(-1, 5)
        if rays.size == 0:
            return
        t = self.now() if t is None else t
        on, oe, oz = origin
        with self.lock:
            self._free_along(on, oe, oz, rays, agent_bit, t)
            hit = rays[rays[:, 4] > 0.5]
            for n_end, e_end, z_lo, z_hi, _ in hit:
                self._hit_column(n_end, e_end, z_lo, z_hi, t)
            self.version += 1

    def integrate_surface(self, points, t: float | None = None):
        """Occupied voxels from surface points (n, e, z) seen from above (roofs). No free-space update:
        the ground under a roof is NOT free just because the roof was seen."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(pts):
            return
        t = self.now() if t is None else t
        with self.lock:
            for n, e, z in pts:
                self._hit_column(n, e, z, z, t)
            self.version += 1

    def integrate_points(self, origin, hits, free_endpoints, agent_bit: int = 1, t: float | None = None):
        """Legacy contract of MapGuiNode._integrate_drone_scan: hits = [(n, e, z)], free_endpoints = [(n, e)]."""
        rows = [[hn, he, hz, hz, 1.0] for hn, he, hz in hits]
        rows += [[fn, fe, 0.0, 0.0, 0.0] for fn, fe in free_endpoints]
        if rows:
            self.integrate_rays(origin, np.array(rows), agent_bit, t)

    def _free_along(self, on, oe, oz, rays, bit, t, samples: int = 48):
        # sample every ray at `samples` points between the origin and its end point
        ends = rays[:, :2]
        hit = rays[:, 4] > 0.5
        frac = np.linspace(0.0, 1.0, samples, endpoint=False)[None, :, None]       # (1,K,1)
        pts = np.array([on, oe])[None, None, :] + (ends[:, None, :] - np.array([on, oe])[None, None, :]) * frac
        # stop one sample short of an occupied end point
        rows = np.floor((pts[..., 0] + self.half) / self.res).astype(np.int64)
        cols = np.floor((pts[..., 1] + self.half) / self.res).astype(np.int64)
        ok = (rows >= 0) & (rows < self.n) & (cols >= 0) & (cols < self.n)
        rows, cols = rows[ok], cols[ok]
        if rows.size:
            code = np.unique(rows * self.n + cols)
            r_idx, c_idx = code // self.n, code % self.n
            self.ground_l[r_idx, c_idx] = np.minimum(self.ground_l[r_idx, c_idx] + self.l_free, 3.0)
            self.ground_seen[r_idx, c_idx] |= np.uint16(bit)
            self.ground_t[r_idx, c_idx] = t
        # coarse free voxels at the sensor height + decay of occupied voxels the ray passes through
        k0 = max(0, int(math.floor(oz / self.res)))
        fr = 2 * self.res
        for pn, pe in {(int(math.floor(p[0] / fr)), int(math.floor(p[1] / fr))) for p in pts[:, ::4].reshape(-1, 2)}:
            if len(self.free3d) < self.max_free3d:
                self.free3d[(pn, pe, int(math.floor(oz / fr)))] = t
        if self.occ:
            uniq = {(int(math.floor(p[0] / self.res)), int(math.floor(p[1] / self.res)))
                    for p in pts[:, 1:-1:2].reshape(-1, 2)}
            for i, j in uniq:
                v = self.occ.get((i, j, k0))
                if v is not None:
                    was = v[0] > 0.0
                    v[0] = max(self.l_min, v[0] - self.l_miss)
                    if was and v[0] <= 0.0:
                        v[2] = t               # state change: no longer an obstacle
        _ = hit  # (kept for clarity: end points of hit rays are handled in _hit_column)

    def _hit_column(self, n_end, e_end, z_lo, z_hi, t):
        i = int(math.floor(n_end / self.res))
        j = int(math.floor(e_end / self.res))
        k_lo = max(0, int(math.floor(z_lo / self.res)))
        k_hi = max(k_lo, int(math.floor(z_hi / self.res)))
        for k in range(k_lo, k_hi + 1):
            v = self.occ.get((i, j, k))
            if v is None:
                if len(self.occ) >= self.max_voxels:
                    return
                v = self.occ[(i, j, k)] = [0.0, None, t, 0]     # t_last_change starts at creation
            was = v[0] > 0.0
            v[0] = min(self.l_max, v[0] + self.l_hit)
            v[3] += 1
            if v[0] > 0.0 and not was:
                v[2] = t                       # state change: now an obstacle (re-observing does NOT refresh this)
                if v[1] is None:
                    v[1] = t                   # first time it counts as an obstacle

    def clear(self):
        with self.lock:
            self.occ.clear()
            self.free3d.clear()
            self.ground_l.fill(0.0)
            self.ground_seen.fill(0)
            self.ground_t.fill(-1e9)
            self.version += 1

    # ----------------------------------------------------------------------- queries
    def occupied_arrays(self, p_thresh: float = 0.5):
        """Snapshot of occupied voxels as arrays: ijk (N,3) int32, p (N,), t_occ (N,), t_last (N,), hits (N,).
        t_last is the time of the voxel's last STATE CHANGE (created / became or stopped being an obstacle), not of
        its last observation, so the viewer's "recently updated" highlight marks real map changes only."""
        l_thresh = math.log(p_thresh / (1.0 - p_thresh)) if 0.0 < p_thresh < 1.0 else 0.0
        with self.lock:
            items = [(k, v) for k, v in self.occ.items() if v[0] > l_thresh]
            if not items:
                z = np.zeros(0)
                return np.zeros((0, 3), np.int32), z, z, z, z
            ijk = np.array([k for k, _ in items], dtype=np.int32)
            lo = np.array([v[0] for _, v in items])
            t_occ = np.array([(v[1] if v[1] is not None else -1e9) for _, v in items])
            t_last = np.array([v[2] for _, v in items])
            hits = np.array([v[3] for _, v in items], dtype=float)
        return ijk, self.prob(lo), t_occ, t_last, hits

    def free3d_array(self):
        with self.lock:
            if not self.free3d:
                return np.zeros((0, 3), np.int32), np.zeros(0)
            keys = np.array(list(self.free3d.keys()), dtype=np.int32)
            ts = np.array(list(self.free3d.values()))
        return keys, ts

    def ground_snapshot(self):
        with self.lock:
            return self.ground_l.copy(), self.ground_seen.copy(), self.ground_t.copy()

    def is_occupied(self, n: float, e: float, z: float, p_thresh: float = 0.5) -> bool:
        v = self.occ.get(self.key(n, e, z))
        return v is not None and self.prob(v[0]) > p_thresh

    def column_height(self, n: float, e: float, p_thresh: float = 0.5) -> float:
        """Top of the tallest occupied voxel in the column (0 if none)."""
        i, j, _ = self.key(n, e, 0.0)
        best = 0.0
        with self.lock:
            for (a, b, k), v in self.occ.items():
                if a == i and b == j and self.prob(v[0]) > p_thresh:
                    best = max(best, (k + 1) * self.res)
        return best

    def stats(self) -> dict:
        with self.lock:
            occ = sum(1 for v in self.occ.values() if v[0] > 0.0)
            return {"occupied": occ, "tracked": len(self.occ), "free3d": len(self.free3d),
                    "ground_known": int((self.ground_l > 0.0).sum()), "version": self.version}
