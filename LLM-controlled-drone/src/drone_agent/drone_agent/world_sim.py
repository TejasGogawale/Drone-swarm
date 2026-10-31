"""Simulated disaster world used by the GUI's mapping: obstacles, LiDAR scans, survivor detector.

IMPORTANT (honest labelling): the Gazebo drones in this project carry a camera but no LiDAR, so the
GUI's 3D map has always been built from a SIMULATED LiDAR that ray-casts against the hand-written list
of world obstacles in `MapGuiNode._get_world_obstacles()`. This module keeps that exact data and the
same visibility rule, but computes the ray hits analytically and densely (the old marching loop took
48 rays and one hit every 2 m of height, which is why walls looked like scattered dots).

Frame: x = north, y = east (as in the GUI), z = height above ground, yaw 0 = north, positive toward east.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def default_obstacles():
    """Same numbers as MapGuiNode._get_world_obstacles()."""
    boxes = [
        (26.0, 32.0, 14.0, 18.0, 11.5),     # Collapsed Fire Station NE
        (26.0, -32.0, 15.0, 15.0, 8.6),     # Collapsed Police Station NW
        (-24.0, 30.0, 12.0, 12.0, 6.0),     # Collapsed House SE
        (-24.0, -30.0, 12.0, 12.0, 6.0),    # Collapsed House SW
        (36.0, 0.0, 12.0, 14.0, 6.0),       # Collapsed House North
        (-36.0, 0.0, 14.0, 15.0, 8.6),      # Collapsed Police Station South
        (-8.0, 11.0, 5.0, 2.5, 2.4),        # Rescue Ambulance
        (3.0, 12.0, 4.4, 2.0, 1.5),         # Stranded Prius Car
        (-3.0, -11.0, 4.6, 2.2, 1.8),       # Damaged SUV
    ]
    cyls = [
        (8.0, 8.0, 0.6, 1.85),              # Walking Survivor #1
        (-8.0, -8.0, 0.6, 1.85),            # Walking Survivor #2
        (10.0, 4.0, 0.6, 1.90),             # Standing Survivor Male #1
        (6.0, -8.0, 0.6, 1.77),             # Standing Survivor Female #1
        (-5.0, 9.0, 0.6, 1.90),             # Standing Survivor Male #2
        (8.15, 2.36, 0.9, 0.6),             # Lying Victim Male #1
        (4.60, -6.78, 0.9, 0.6),            # Lying Victim Female #1
        (-8.54, -3.21, 0.9, 0.6),           # Lying Victim Male #2
        (-7.77, 6.58, 0.9, 0.6),            # Lying Victim Female #2
    ]
    return boxes, cyls


SURVIVOR_NAMES = ["Walking #1", "Walking #2", "Standing M1", "Standing F1", "Standing M2",
                  "Lying M1", "Lying F1", "Lying M2", "Lying F2"]


@dataclass
class Survivor:
    id: int
    name: str
    n: float
    e: float
    r: float
    h: float


class World:
    def __init__(self, boxes=None, cyls=None):
        b, c = default_obstacles()
        self.boxes = [tuple(x) for x in (b if boxes is None else boxes)]
        self.cyls = [tuple(x) for x in (c if cyls is None else cyls)]
        # every cylinder in the default list is a survivor (the remaining obstacles are buildings/cars)
        self.survivors = [Survivor(i, SURVIVOR_NAMES[i] if i < len(SURVIVOR_NAMES) else f"S{i}", *c)
                          for i, c in enumerate(self.cyls)] if cyls is None else []
        self.dynamic: list[tuple] = []     # (n, e, size_n, size_e, h): rubble dropped at run time
        self.version = 0

    # ------------------------------------------------------------------ obstacles
    def add_dynamic_box(self, n, e, size_n=2.0, size_e=2.0, h=1.2):
        self.dynamic.append((float(n), float(e), float(size_n), float(size_e), float(h)))
        self.version += 1

    def clear_dynamic(self):
        self.dynamic.clear()
        self.version += 1

    def _box_array(self):
        allb = self.boxes + self.dynamic
        return np.array(allb, dtype=float).reshape(-1, 5)

    def _cyl_array(self):
        return np.array(self.cyls, dtype=float).reshape(-1, 4)

    def blocked_at(self, n, e, margin: float = 0.0) -> bool:
        """True ground-truth collision test (used for rover safety, never for planning)."""
        for bn, be, sn, se, bh in self.boxes + self.dynamic:
            if abs(n - bn) <= sn * 0.5 + margin and abs(e - be) <= se * 0.5 + margin:
                return True
        for cn, ce, cr, ch in self.cyls:
            if math.hypot(n - cn, e - ce) <= cr + margin:
                return True
        return False

    # ------------------------------------------------------------------ LiDAR sim
    def scan(self, n0, e0, alt_agl, yaw, n_rays=192, max_range=28.0, min_range=1.5, rise=0.45,
             z_step=None, sensor="drone"):
        """Cast `n_rays` horizontal rays around the sensor. Returns array (R, 5):
        [n_end, e_end, z_lo, z_hi, hit]. Visibility rule kept from the original code: the part of an
        obstacle face that is visible grows with range, up to alt_agl + r*rise (a drone at altitude
        sees tall walls, a ground rover sees only the low part). Rays that hit nothing end at max_range."""
        ang = yaw + np.arange(n_rays) * (2.0 * math.pi / n_rays)
        dn, de = np.cos(ang), np.sin(ang)
        best = np.full(n_rays, np.inf)
        top = np.zeros(n_rays)
        boxes = self._box_array()
        for bn, be, sn, se, bh in boxes:
            t = _ray_aabb(n0, e0, dn, de, bn - sn / 2, bn + sn / 2, be - se / 2, be + se / 2)
            better = t < best
            best = np.where(better, t, best)
            top = np.where(better, bh, top)
        for cn, ce, cr, ch in self._cyl_array():
            t = _ray_circle(n0, e0, dn, de, cn, ce, cr)
            better = t < best
            best = np.where(better, t, best)
            top = np.where(better, ch, top)
        hit = np.isfinite(best) & (best >= min_range) & (best <= max_range)
        # an obstacle that starts closer than min_range (sensor inside it) is ignored, as in the original
        r = np.where(hit, best, max_range)
        if sensor == "rover":
            vis = np.minimum(top, 1.6)          # a ground sensor only sees up to its own mast height
            hit &= vis >= 0.2
        else:
            vis = np.minimum(top, alt_agl + r * rise)
            hit &= vis >= 0.5
        out = np.zeros((n_rays, 5))
        r_end = np.where(hit, r + 0.05, r)      # end point 5 cm inside the surface => first voxel INSIDE the obstacle
        out[:, 0] = n0 + r_end * dn
        out[:, 1] = e0 + r_end * de
        out[:, 2] = 0.0
        out[:, 3] = np.where(hit, vis, 0.0)
        out[:, 4] = hit.astype(float)
        return out


def _roof_points(self, n0, e0, alt_agl, max_range=28.0, spacing=0.5):
    """Top surfaces of obstacles that are lower than the sensor, as (N, 3) points [n, e, z_top]. A real
    aerial LiDAR sees roofs; without them buildings render as open walls. Only drone sensors use this."""
    pts = []
    for bn, be, sn, se, bh in self._box_array():
        if bh >= alt_agl - 0.4:
            continue
        ns = np.arange(bn - sn / 2 + spacing / 2, bn + sn / 2, spacing)
        es = np.arange(be - se / 2 + spacing / 2, be + se / 2, spacing)
        gn, ge = np.meshgrid(ns, es, indexing="ij")
        d = np.hypot(gn - n0, ge - e0)
        m = d <= max_range
        if m.any():
            pts.append(np.stack([gn[m], ge[m], np.full(m.sum(), bh - spacing * 0.5)], axis=1))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


World.roof_points = _roof_points


def _ray_aabb(n0, e0, dn, de, n_min, n_max, e_min, e_max):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (n_min - n0) / dn
        t2 = (n_max - n0) / dn
        t3 = (e_min - e0) / de
        t4 = (e_max - e0) / de
    tn = np.minimum(t1, t2)
    tf = np.maximum(t1, t2)
    en = np.minimum(t3, t4)
    ef = np.maximum(t3, t4)
    # parallel rays outside the slab never hit
    par_n = (np.abs(dn) < 1e-12) & ((n0 < n_min) | (n0 > n_max))
    par_e = (np.abs(de) < 1e-12) & ((e0 < e_min) | (e0 > e_max))
    tn = np.where(np.abs(dn) < 1e-12, -np.inf, tn)
    tf = np.where(np.abs(dn) < 1e-12, np.inf, tf)
    en = np.where(np.abs(de) < 1e-12, -np.inf, en)
    ef = np.where(np.abs(de) < 1e-12, np.inf, ef)
    t_enter = np.maximum(tn, en)
    t_exit = np.minimum(tf, ef)
    ok = (t_exit >= np.maximum(t_enter, 0.0)) & ~par_n & ~par_e
    # sensor inside the box: report the exit as "no useful hit" (handled by min_range)
    t = np.where(t_enter > 0, t_enter, 0.0)
    return np.where(ok, t, np.inf)


def _ray_circle(n0, e0, dn, de, cn, ce, cr):
    fn, fe = n0 - cn, e0 - ce
    b = fn * dn + fe * de
    c = fn * fn + fe * fe - cr * cr
    disc = b * b - c
    with np.errstate(invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
    t = -b - sq
    return np.where((disc >= 0) & (t > 0), t, np.inf)


# --------------------------------------------------------------------- simulated detector
@dataclass
class Detection:
    id: int                 # survivor id (world id for the simulated detector, registry id for the real one)
    n: float
    e: float
    conf: float
    drone_id: int
    t: float
    source: str = "sim"     # 'sim' = SIMULATED detector, 'swarm' = real YOLO registry
    name: str = ""


class SimDetector:
    """SIMULATED survivor detector: a drone 'sees' a survivor if it is inside a footprint that
    grows with altitude and the line of sight is not blocked by a building. Position error ~0.5 m.
    This stands in for YOLO until the real pipeline is accurate; the GUI labels it as simulated."""

    def __init__(self, world: World, rng_seed: int = 0, p_detect: float = 0.45, pos_sigma: float = 0.5):
        self.world = world
        self.rng = np.random.default_rng(rng_seed)
        self.p_detect = p_detect
        self.pos_sigma = pos_sigma
        self.seen: dict[int, int] = {}

    def step(self, drones, t: float):
        """drones: iterable of dicts with id, x(north), y(east), z(down, negative = up). Returns Detections."""
        out = []
        for d in drones:
            alt = max(0.0, -float(d["z"]))
            reach = min(24.0, 8.0 + 0.8 * alt)
            for s in self.world.survivors:
                dist = math.hypot(s.n - d["x"], s.e - d["y"])
                if dist > reach or self._los_blocked(d["x"], d["y"], s.n, s.e):
                    continue
                if self.rng.random() > self.p_detect:
                    continue
                conf = float(np.clip(0.97 - 0.02 * dist + self.rng.normal(0, 0.02), 0.5, 0.99))
                en, ee = self.rng.normal(0, self.pos_sigma, 2)
                self.seen[s.id] = self.seen.get(s.id, 0) + 1
                out.append(Detection(s.id, s.n + float(en), s.e + float(ee), conf, int(d["id"]), t, "sim", s.name))
        return out

    def _los_blocked(self, n0, e0, n1, e1) -> bool:
        dist = math.hypot(n1 - n0, e1 - e0)
        if dist < 1e-6:
            return False
        dn, de = (n1 - n0) / dist, (e1 - e0) / dist
        for bn, be, sn, se, bh in self.world._box_array():
            t = float(_ray_aabb(n0, e0, np.array([dn]), np.array([de]), bn - sn / 2, bn + sn / 2,
                                be - se / 2, be + se / 2)[0])
            if 0.0 < t < dist - 0.7:
                return True
        return False
