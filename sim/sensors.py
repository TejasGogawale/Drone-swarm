"""Sensor models: line-of-sight (thermal) and acoustic (through-wall)."""
from __future__ import annotations

import math
import numpy as np
from .map_gen import BuildingMap, LOS_BLOCK


def line_of_sight(bmap: BuildingMap, x0, y0, x1, y1) -> bool:
    """True if a straight ray from (x0,y0) to (x1,y1) crosses no LOS-blocking cell
    (walls). Uses Bresenham; start/end cells are not considered blockers."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if (x, y) != (x0, y0) and (x, y) != (x1, y1):
            if bmap.grid[y, x] in LOS_BLOCK:
                return False
        if (x, y) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _walls_between(bmap: BuildingMap, x0, y0, x1, y1) -> int:
    """Count wall cells crossed by a ray (for acoustic attenuation)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    count = 0
    while True:
        if (x, y) != (x0, y0) and (x, y) != (x1, y1) and bmap.grid[y, x] == 1:
            count += 1
        if (x, y) == (x1, y1):
            return count
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def thermal_detect(bmap: BuildingMap, ax, ay, aheading, sx, sy, *,
                   sensor_range: float, fov_deg: float, rng: np.random.Generator,
                   coarse: bool = False) -> bool:
    """Detect a heat source at (sx,sy) from agent (ax,ay) heading `aheading`.
    Conditions: in range, within FOV cone, line of sight, then a noisy detection
    probability that falls off with distance."""
    dx, dy = sx - ax, sy - ay
    dist = math.hypot(dx, dy)
    if dist > sensor_range:
        return False
    # angular difference between heading and bearing to survivor
    bearing = math.atan2(dy, dx)
    ang = abs((bearing - aheading + math.pi) % (2 * math.pi) - math.pi)
    if math.degrees(ang) > fov_deg / 2.0:
        return False
    if not line_of_sight(bmap, ax, ay, sx, sy):
        return False
    # detection probability: high when close, decays with distance; coarse (drone)
    # sensors are noisier -> lower effective probability.
    p = max(0.05, 1.0 - dist / (sensor_range * 1.2))
    if coarse:
        p *= 0.7
    return bool(rng.random() < p)


def acoustic_observe(bmap: BuildingMap, ax, ay, sx, sy, loudness: float,
                     rng: np.random.Generator, max_range: float = 10.0) -> float | None:
    """Return observed loudness of a sound source, or None if below threshold or
    beyond max_range. Sound attenuates with distance and number of walls crossed."""
    dist = max(0.5, math.hypot(sx - ax, sy - ay))
    if dist > max_range:
        return None
    walls = _walls_between(bmap, ax, ay, sx, sy)
    # inverse-square-ish distance attenuation in dB + per-wall attenuation
    atten_db = 20.0 * math.log10(dist) + walls * 12.0
    received = loudness * 100.0 / (10 ** (atten_db / 20.0))
    # add noise (sensor + environmental)
    received *= float(rng.uniform(0.6, 1.4))
    return received
