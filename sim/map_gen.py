"""Procedural collapsed-building map generator.

Grid cell types:
  FREE    (0) - open space, traversable by drones AND rovers
  WALL    (1) - solid, blocks movement and line-of-sight for both types
  NARROW  (2) - low/narrow gap: traversable by ROVERS only (drone cannot fit)
  RUBBLE  (3) - debris field: traversable by ROVERS only, slow (drone cannot operate)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
import math

FREE = 0
WALL = 1
NARROW = 2
RUBBLE = 3

# Traversability per robot type
ROVER_OK = {FREE, NARROW, RUBBLE}
DRONE_OK = {FREE}
# Cell types that block line-of-sight for sensors (thermal). Sound is attenuated
# by walls separately (see sensors.py); narrow/rubble do not block thermal LOS.
LOS_BLOCK = {WALL}

# Movement cost multipliers (rover). Drones ignore these (they fly over open cells).
SPEED_FACTOR = {FREE: 1.0, NARROW: 1.0, RUBBLE: 0.5}


@dataclass
class Survivor:
    """A trapped victim with sensor attributes."""
    id: int
    pos: tuple            # (x, y) grid cell
    heat: bool = True     # detectable by thermal camera (line of sight)
    sound: bool = True    # detectable by acoustic (sound travels through walls)
    loudness: float = 1.0
    found: bool = False   # set when first detected by any robot
    confirmed: bool = False  # set when detected by a 2nd robot or 2nd sensor type
    found_by: set = field(default_factory=set)     # robot ids that detected it
    detected_via: set = field(default_factory=set)  # {'thermal','acoustic'} seen


@dataclass
class BuildingMap:
    grid: np.ndarray
    survivors: list

    @property
    def H(self) -> int:
        return self.grid.shape[0]

    @property
    def W(self) -> int:
        return self.grid.shape[1]

    def rover_ok(self, x, y) -> bool:
        return self.grid[y, x] in ROVER_OK

    def drone_ok(self, x, y) -> bool:
        return self.grid[y, x] in DRONE_OK

    def in_bounds(self, x, y) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def traversable_cells(self) -> np.ndarray:
        """All cells a rover can reach (the 'ground truth' free space for coverage)."""
        return np.isin(self.grid, list(ROVER_OK))


def generate_building(
    seed: int = 0,
    width: int = 40,
    height: int = 40,
    n_rooms: int = 6,
    room_min: int = 5,
    room_max: int = 10,
    n_narrow: int = 3,
    n_rubble: int = 8,
    n_survivors: int = 6,
    min_entry_dist: int = 12,
    min_pair_dist: int = 4,
) -> BuildingMap:
    """Generate a labyrinthine collapsed building with rooms, corridors, narrow
    gaps, rubble patches and survivors with heat/sound attributes."""
    rng = np.random.default_rng(seed)
    grid = np.full((height, width), WALL, dtype=np.int8)

    # 1. place rooms
    rooms = []
    for _ in range(n_rooms):
        rw = int(rng.integers(room_min, room_max + 1))
        rh = int(rng.integers(room_min, room_max + 1))
        x0 = int(rng.integers(1, width - rw - 1))
        y0 = int(rng.integers(1, height - rh - 1))
        # reject rooms that overlap existing free space too much
        region = grid[y0:y0 + rh, x0:x0 + rw]
        if np.count_nonzero(region != WALL) > 0.3 * region.size:
            continue
        grid[y0:y0 + rh, x0:x0 + rw] = FREE
        rooms.append((x0, y0, rw, rh))

    if len(rooms) < 2:
        # fall back: carve one big room
        grid[5:-5, 5:-5] = FREE
        rooms = [(5, 5, width - 10, height - 10)]

    # 2. connect rooms in sequence with L-shaped corridors (FREE)
    centers = [(x + rw // 2, y + rh // 2) for (x, y, rw, rh) in rooms]
    for i in range(len(centers) - 1):
        (x1, y1), (x2, y2) = centers[i], centers[i + 1]
        _carve_corridor(grid, x1, y1, x2, y2, FREE)

    # 3. narrow gaps (rover-only) between some rooms
    for _ in range(n_narrow):
        if len(rooms) < 2:
            break
        a, b = rng.choice(len(rooms), size=2, replace=False)
        (x1, y1), (x2, y2) = centers[a], centers[b]
        _carve_corridor(grid, x1, y1, x2, y2, NARROW)

    # 4. rubble patches (rover-only, slow) sprinkled on free cells
    free_cells = np.argwhere(grid == FREE)
    if len(free_cells) > 0:
        for _ in range(n_rubble):
            idx = int(rng.integers(0, len(free_cells)))
            y, x = free_cells[idx]
            # convert a small cluster to rubble
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width and grid[yy, xx] == FREE:
                        grid[yy, xx] = RUBBLE

    # 5. survivors — with explicit, physically-motivated sensor profiles:
    #    - "open" (in free/narrow space): warm (heat=True); some are quiet (sound=False)
    #      -> detectable by thermal (drones wide / rovers narrow).
    #    - "buried" (under rubble): heat hidden (heat=False), sound=True
    #      -> detectable only by rover ACOUSTIC (sound travels through debris).
    #    Survivors are placed away from the entry point and spread apart so that
    #    exploration actually matters (a policy must search, not just spawn-detect).
    entry = _entry_cell(grid)
    rubble_cells = np.argwhere(grid == RUBBLE)
    open_cells = np.argwhere(np.isin(grid, [FREE, NARROW]))
    rubble_cells = _far_from_entry(rubble_cells, entry, min_entry_dist)
    open_cells = _far_from_entry(open_cells, entry, min_entry_dist)
    rng.shuffle(rubble_cells)
    rng.shuffle(open_cells)

    n_buried = min(len(rubble_cells), max(1, n_survivors // 3))
    n_open = n_survivors - n_buried
    if n_open > len(open_cells):
        n_open = len(open_cells)
        n_buried = min(len(rubble_cells), n_survivors - n_open)

    # build a candidate pool, then greedily pick with min pairwise separation
    cands = []
    for y, x in rubble_cells[:n_buried]:
        cands.append((int(x), int(y), "buried"))
    for y, x in open_cells[:n_open]:
        cands.append((int(x), int(y), "open"))
    rng.shuffle(cands)

    placements = []
    for c in cands:
        if len(placements) >= n_survivors:
            break
        cx, cy, _ = c
        if all(math.hypot(cx - px, cy - py) >= min_pair_dist for (px, py, _) in placements):
            placements.append(c)
    # if we couldn't satisfy separation, just fill from cands
    for c in cands:
        if len(placements) >= n_survivors:
            break
        if c not in placements:
            placements.append(c)

    survivors = []
    for i, (x, y, kind) in enumerate(placements[:n_survivors]):
        if kind == "buried":
            survivors.append(Survivor(id=i, pos=(x, y), heat=False, sound=True,
                                      loudness=float(rng.uniform(0.6, 1.0))))
        else:
            quiet = bool(rng.random() < 0.35)
            survivors.append(Survivor(id=i, pos=(x, y), heat=True, sound=not quiet,
                                      loudness=float(rng.uniform(0.6, 1.0))))

    return BuildingMap(grid=grid, survivors=survivors)


def _entry_cell(grid):
    """Top-left-most rover-traversable cell (matches env._entry_point)."""
    ys, xs = np.where(np.isin(grid, [FREE, NARROW, RUBBLE]))
    if len(xs) == 0:
        return (0, 0)
    i = int(np.argmin(ys + xs))
    return (int(xs[i]), int(ys[i]))


def _far_from_entry(cells, entry, min_dist):
    if len(cells) == 0:
        return cells
    ex, ey = entry
    keep = [c for c in cells if math.hypot(c[1] - ex, c[0] - ey) >= min_dist]
    return np.array(keep, dtype=cells.dtype) if keep else cells


def _carve_corridor(grid, x1, y1, x2, y2, cell_type) -> None:
    """L-shaped corridor between two points."""
    x, y = x1, y1
    while x != x2:
        grid[y, x] = cell_type
        x += 1 if x2 > x else -1
    while y != y2:
        grid[y, x] = cell_type
        y += 1 if y2 > y else -1
    grid[y2, x2] = cell_type
