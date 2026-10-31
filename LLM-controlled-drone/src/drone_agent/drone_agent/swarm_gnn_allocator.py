#!/usr/bin/env python3
"""Decentralized 4-Drone Swarm Task Allocator using GraphSAGE GNN + Heuristic Fallback.

Architecture:
1. Attempts to load the trained PyTorch GraphSAGE GNN (`ActorCritic` from `train/model.py`
   and weights from `results/checkpoints/gnn.pt`).
2. At each allocation step, extracts the 109-dim agent feature vector (`FA=109`) for each of
   the 4 drones, the 5-dim goal feature vector (`FG=5`) for `N_GOALS=20` candidate frontiers/tasks
   across the 120m x 120m disaster map, and the 4x4 peer-to-peer communication adjacency matrix
   `comm_adj` (edges active within `comm_range`).
3. Runs GraphSAGE message passing across the swarm graph to score (drone, goal) pairs.
4. Verifies GNN task assignments for conflict-free separation (>= 16m) and information gain.
   If GNN weights are unavailable, degenerate, or assign colliding goals, automatically falls back
   to a Decentralized Frontier & Sector Auction Heuristic so all 4 drones continuously scan the
   entire environment together without overlap.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# Ensure the swarm root is importable; only append .venv as fallback if torch is not in ~/.local
SWARM_ROOT = '/home/tushar/Desktop/swarm'
VENV_SITE = '/home/tushar/Desktop/swarm/.venv/lib/python3.12/site-packages'
if os.path.isdir(SWARM_ROOT) and SWARM_ROOT not in sys.path:
    sys.path.insert(0, SWARM_ROOT)

TORCH_AVAILABLE = False
try:
    try:
        import torch
    except ImportError:
        if os.path.isdir(VENV_SITE) and VENV_SITE not in sys.path:
            sys.path.append(VENV_SITE)
        import torch
    from train.model import ActorCritic
    from train.features import FA, FG, N_GOALS, PATCH
    TORCH_AVAILABLE = True
except Exception:
    FA, FG, N_GOALS, PATCH = 109, 5, 20, 7


@dataclass
class DroneAgentState:
    drone_id: int
    name: str
    x: float          # North (m, NED)
    y: float          # East (m, NED)
    z: float          # Down (m, NED, negative = up)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    battery: float = 1.0
    armed: bool = True
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = -10.0
    task_label: str = 'Initializing'
    alloc_mode: str = 'GNN'
    gnn_score: float = 0.0


# Strategic coverage waypoints across the 120m x 120m post-earthquake map (North, East, label)
# Ensures both inner survivor plaza and outer collapsed buildings/perimeters get 100% scanned.
STRATEGIC_WAYPOINTS_NED: list[tuple[float, float, str]] = [
    # Quadrant Core Sectors
    (18.0, 18.0, 'NE Sector (Fire Stn Approach)'),
    (18.0, -18.0, 'NW Sector (Police Stn Approach)'),
    (-18.0, -18.0, 'SW Sector (Damaged SUV & House)'),
    (-18.0, 18.0, 'SE Sector (Ambulance & House)'),
    # Inner Survivor & Victim Clusters
    (10.0, 7.0, 'NE Survivors (Walking & Lying)'),
    (7.0, -8.0, 'NW Survivors (Standing & Lying)'),
    (-9.0, -6.0, 'SW Survivors (Walking & Lying)'),
    (-7.0, 9.0, 'SE Survivors (Ambulance Zone)'),
    # Mid-Ring Perimeter Sweeps
    (32.0, 16.0, 'North-East Perimeter Sweep'),
    (32.0, -16.0, 'North-West Perimeter Sweep'),
    (-32.0, -16.0, 'South-West Perimeter Sweep'),
    (-32.0, 16.0, 'South-East Perimeter Sweep'),
    # Cardinal Outer Corridors
    (0.0, 34.0, 'East Road Corridor'),
    (0.0, -34.0, 'West Road Corridor'),
    (24.0, 0.0, 'North Plaza Corridor'),
    (-24.0, 0.0, 'South Plaza Corridor'),
    # Far Outer Corners (for 100% 120x120m map coverage)
    (38.0, 38.0, 'NE Far Corner Overlook'),
    (38.0, -38.0, 'NW Far Corner Overlook'),
    (-38.0, -38.0, 'SW Far Corner Overlook'),
    (-38.0, 38.0, 'SE Far Corner Overlook'),
]

# Known survivor positions in NED for proximity/reward hints
SURVIVOR_POSITIONS_NED: list[tuple[float, float]] = [
    (8.0, 8.0),
    (-8.0, -8.0),
    (10.0, 4.0),
    (6.0, -8.0),
    (-5.0, 9.0),
    (8.15, 2.36),
    (4.60, -6.78),
    (-8.54, -3.21),
    (-7.77, 6.58),
]

# Preferred sector angles (radians in NED: atan2(East, North)) for the 4 drones
# UAV-1: NE (+45 deg), UAV-2: NW (-45 deg), UAV-3: SW (-135 deg), UAV-4: SE (+135 deg)
DRONE_SECTOR_ANGLES = [
    math.pi * 0.25,
    -math.pi * 0.25,
    -math.pi * 0.75,
    math.pi * 0.75,
]


class SwarmGNNAllocator:
    """Decentralized GraphSAGE GNN + Heuristic Fallback task allocator for 4 drones."""

    def __init__(
        self,
        checkpoint_path: str = '/home/tushar/Desktop/swarm/results/checkpoints/gnn.pt',
        comm_range_m: float = 85.0,
        grid_half_m: float = 60.0,
        grid_res: float = 0.5,
    ):
        self.comm_range_m = comm_range_m
        self.grid_half_m = grid_half_m
        self.grid_res = grid_res
        self.grid_size = int((grid_half_m * 2.0) / grid_res)  # 240
        self.model = None
        self.gnn_loaded = False
        self.status_reason = 'Heuristic mode'
        self.visit_counts: dict[tuple[int, int], int] = {}
        self.last_alloc_time = 0.0

        if TORCH_AVAILABLE and os.path.exists(checkpoint_path):
            try:
                self.model = ActorCritic(FA, FG, n_goals=N_GOALS, hidden=64, use_gnn=True)
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.gnn_loaded = True
                self.status_reason = f'GraphSAGE GNN loaded ({os.path.basename(checkpoint_path)})'
            except Exception as exc:
                self.gnn_loaded = False
                self.status_reason = f'GNN load error ({exc}) -> Heuristic Fallback'
        else:
            self.status_reason = 'PyTorch/checkpoint unavailable -> Heuristic Fallback'

    def _ned_to_grid(self, north: float, east: float) -> tuple[int, int]:
        gx = int((east + self.grid_half_m) / self.grid_res)
        gy = int((self.grid_half_m - north) / self.grid_res)
        return max(0, min(self.grid_size - 1, gx)), max(0, min(self.grid_size - 1, gy))

    def _grid_to_ned(self, gx: int, gy: int) -> tuple[float, float]:
        east = gx * self.grid_res - self.grid_half_m
        north = self.grid_half_m - gy * self.grid_res
        return north, east

    def compute_comm_adj(self, drones: list[DroneAgentState]) -> np.ndarray:
        """Compute [nA, nA] communication adjacency matrix based on peer distance."""
        n = len(drones)
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(drones[i].x - drones[j].x, drones[i].y - drones[j].y)
                if d <= self.comm_range_m:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        return adj

    def _unexplored_gain(self, occ_grid: np.ndarray, north: float, east: float, radius_m: float = 14.0) -> float:
        """Return fraction [0..1] of cells within `radius_m` of (north, east) that are still unexplored (0)."""
        gx, gy = self._ned_to_grid(north, east)
        r_cells = max(2, int(radius_m / self.grid_res))
        x0 = max(0, gx - r_cells)
        x1 = min(self.grid_size, gx + r_cells + 1)
        y0 = max(0, gy - r_cells)
        y1 = min(self.grid_size, gy + r_cells + 1)
        patch = occ_grid[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        unexplored = float(np.count_nonzero(patch == 0))
        return unexplored / float(patch.size)

    def _extract_candidate_goals(
        self, occ_grid: np.ndarray
    ) -> list[tuple[float, float, str, float]]:
        """Build exactly N_GOALS=20 candidate goals combining dynamic frontiers and strategic waypoints.

        Returns list of (north, east, label, info_gain).
        """
        candidates: list[tuple[float, float, str, float]] = []

        # 1. Score all strategic waypoints by current unexplored information gain & visit history
        for wn, we, wlbl in STRATEGIC_WAYPOINTS_NED:
            gain = self._unexplored_gain(occ_grid, wn, we, radius_m=15.0)
            key = (int(round(wn / 4.0)), int(round(we / 4.0)))
            visits = self.visit_counts.get(key, 0)
            eff_gain = gain / (1.0 + 0.35 * visits)
            candidates.append((wn, we, wlbl, eff_gain))

        # 2. Sample dynamic frontier cells from the live occupancy grid (free cell adjacent to unknown)
        # Downsample grid by 4x (60x60) for fast frontier detection
        small = occ_grid[::4, ::4]
        free_mask = (small == 1)
        unk_mask = (small == 0)
        if np.any(free_mask) and np.any(unk_mask):
            # Find free cells next to unknown cells
            frontier_yx = []
            sh, sw = small.shape
            for sy in range(2, sh - 2, 2):
                for sx in range(2, sw - 2, 2):
                    if small[sy, sx] == 1:
                        nb = small[sy - 1:sy + 2, sx - 1:sx + 2]
                        if np.any(nb == 0) and not np.any(nb == 2):
                            gx, gy = sx * 4, sy * 4
                            fn, fe = self._grid_to_ned(gx, gy)
                            # Keep inside safe flyable bounds (-44m..+44m)
                            if abs(fn) <= 44.0 and abs(fe) <= 44.0:
                                frontier_yx.append((fn, fe))

            # Pick top well-spaced dynamic frontiers
            for fn, fe in frontier_yx[:12]:
                if all(math.hypot(fn - cn, fe - ce) > 10.0 for cn, ce, _, _ in candidates):
                    gain = self._unexplored_gain(occ_grid, fn, fe, radius_m=14.0)
                    sec = (
                        'NE' if (fn >= 0 and fe >= 0)
                        else ('NW' if (fn >= 0 and fe < 0) else ('SW' if fe < 0 else 'SE'))
                    )
                    candidates.append((fn, fe, f'Dynamic Frontier {sec} ({fn:+.0f},{fe:+.0f})', gain))

        # Sort by effective information gain descending and keep N_GOALS=20
        candidates.sort(key=lambda item: item[3], reverse=True)
        selected = candidates[:N_GOALS]
        while len(selected) < N_GOALS:
            selected.append((0.0, 0.0, 'Central Plaza Patrol', 0.2))
        return selected

    def _build_gnn_tensors(
        self,
        drones: list[DroneAgentState],
        occ_grid: np.ndarray,
        candidates: list[tuple[float, float, str, float]],
        comm_adj: np.ndarray,
    ):
        """Build observation tensors matching `train/features.py` (FA=109, FG=5, N_GOALS=20)."""
        nA = len(drones)
        span = self.grid_half_m * 2.0  # 120.0m
        diag = math.hypot(span, span)

        agent_feats = np.zeros((nA, FA), dtype=np.float32)
        goal_feats = np.zeros((nA, N_GOALS, FG), dtype=np.float32)
        goal_valid = np.ones((nA, N_GOALS), dtype=bool)

        for i, d in enumerate(drones):
            # Nearest survivor thermal hint
            best_surv_d = min(
                (math.hypot(d.x - sn, d.y - se) for sn, se in SURVIVOR_POSITIONS_NED),
                default=50.0,
            )
            thermal_hint = 1.0 / (1.0 + max(0.0, best_surv_d / 4.0))

            # Nearest high-gain frontier distance
            min_f_dist = min(
                (math.hypot(d.x - cn, d.y - ce) for cn, ce, _, g in candidates if g > 0.15),
                default=20.0,
            )
            ndist = min(1.0, min_f_dist / diag)

            # Local 7x7 occupancy patch + 7x7 survivor patch around drone
            gx, gy = self._ned_to_grid(d.x, d.y)
            half = PATCH // 2
            occ_patch = np.full((PATCH, PATCH), -1.0, dtype=np.float32)
            surv_patch = np.zeros((PATCH, PATCH), dtype=np.float32)
            step_c = 4  # 2.0m per patch cell
            for py in range(PATCH):
                for px in range(PATCH):
                    cx = gx + (px - half) * step_c
                    cy = gy + (py - half) * step_c
                    if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                        val = int(occ_grid[cy, cx])
                        if val == 1:
                            occ_patch[py, px] = 0.0   # FREE
                        elif val == 2:
                            occ_patch[py, px] = 1.0   # WALL/OBSTACLE
                        cn, ce = self._grid_to_ned(cx, cy)
                        if any(math.hypot(cn - sn, ce - se) <= 3.0 for sn, se in SURVIVOR_POSITIONS_NED):
                            surv_patch[py, px] = 1.0

            norm_x = (d.x + self.grid_half_m) / span
            norm_y = (d.y + self.grid_half_m) / span
            feat = np.concatenate([
                np.array([1.0, 0.0], dtype=np.float32),                   # drone one-hot
                np.array([d.battery], dtype=np.float32),                  # battery
                np.array([norm_x, norm_y], dtype=np.float32),             # normalized pos
                np.array([math.sin(d.yaw), math.cos(d.yaw)], dtype=np.float32),
                np.array([0.5], dtype=np.float32),                        # known survivors ratio
                np.array([thermal_hint], dtype=np.float32),               # thermal hint
                np.array([0.0], dtype=np.float32),                        # acoustic hint
                np.array([ndist], dtype=np.float32),                      # frontier dist
                occ_patch.reshape(-1),                                    # 49
                surv_patch.reshape(-1),                                   # 49
            ]).astype(np.float32)
            agent_feats[i] = feat

            for k, (cn, ce, _lbl, gain) in enumerate(candidates):
                dx = cn - d.x
                dy = ce - d.y
                dist = math.hypot(dx, dy)
                near_surv = 1.0 if any(
                    math.hypot(cn - sn, ce - se) <= 8.0 for sn, se in SURVIVOR_POSITIONS_NED
                ) else 0.0
                goal_feats[i, k] = [
                    dx / span,
                    dy / span,
                    dist / diag,
                    min(1.0, gain * 1.5),
                    near_surv,
                ]

        cov = float(np.count_nonzero(occ_grid > 0)) / float(occ_grid.size)
        return agent_feats, goal_feats, goal_valid, cov

    def allocate_tasks(
        self,
        drones: list[DroneAgentState],
        occ_grid: np.ndarray,
        mission_mode: str = 'scan',
    ) -> dict[int, tuple[float, float, str, str, float]]:
        """Allocate conflict-free (target_n, target_e, task_label, alloc_mode, score) for all 4 drones.

        Uses GraphSAGE GNN message passing first; deconflicts or falls back to Heuristic
        Frontier Auction if GNN assignments collide or stall.
        """
        self.last_alloc_time = time.time()
        nA = len(drones)
        comm_adj = self.compute_comm_adj(drones)
        candidates = self._extract_candidate_goals(occ_grid)

        # Record visit counts near current drone positions so drones keep advancing to new areas
        for d in drones:
            vkey = (int(round(d.x / 4.0)), int(round(d.y / 4.0)))
            self.visit_counts[vkey] = self.visit_counts.get(vkey, 0) + 1

        # ── Step 1: Evaluate GraphSAGE GNN if available ──────────────────
        gnn_probs = np.zeros((nA, N_GOALS), dtype=np.float32)
        gnn_ok = False
        if self.gnn_loaded and self.model is not None and TORCH_AVAILABLE:
            try:
                af, gf, gv, cov = self._build_gnn_tensors(drones, occ_grid, candidates, comm_adj)
                with torch.no_grad():
                    t_af = torch.as_tensor(af, dtype=torch.float32)
                    t_gf = torch.as_tensor(gf, dtype=torch.float32)
                    t_gv = torch.as_tensor(gv, dtype=torch.bool)
                    t_types = torch.zeros(nA, dtype=torch.long)  # all drones = 0
                    t_adj = torch.as_tensor(comm_adj, dtype=torch.float32)
                    t_cov = torch.tensor([cov], dtype=torch.float32)
                    logits, _val = self.model(t_af, t_gf, t_gv, t_types, t_adj, t_cov)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    if np.all(np.isfinite(probs)):
                        gnn_probs = probs
                        gnn_ok = True
            except Exception:
                gnn_ok = False

        # ── Step 2: Compute Heuristic Frontier & Quadrant Utility Matrix ─
        heur_scores = np.zeros((nA, N_GOALS), dtype=np.float32)
        for i, d in enumerate(drones):
            pref_angle = DRONE_SECTOR_ANGLES[i % len(DRONE_SECTOR_ANGLES)]
            for k, (cn, ce, _lbl, gain) in enumerate(candidates):
                dist = math.hypot(cn - d.x, ce - d.y)
                goal_angle = math.atan2(ce, cn)
                ang_diff = abs((goal_angle - pref_angle + math.pi) % (2.0 * math.pi) - math.pi)

                # Sector bonus keeps the 4 drones naturally spread across NE, NW, SW, SE
                # while allowing any drone to help in other sectors once its own sector is scanned
                sector_bonus = 0.45 * (1.0 - ang_diff / math.pi)

                # Prefer moderate travel distances (8m..35m) over staying still (<3m) or crossing entire map
                if dist < 3.5:
                    dist_term = -0.5  # finished current waypoint -> move on!
                else:
                    dist_term = -0.25 * (dist / 60.0)

                # Survivor search bonus when in survivor search mode
                surv_bonus = 0.0
                if mission_mode == 'survivor':
                    if any(math.hypot(cn - sn, ce - se) <= 9.0 for sn, se in SURVIVOR_POSITIONS_NED):
                        surv_bonus = 0.65

                heur_scores[i, k] = (1.35 * gain) + sector_bonus + dist_term + surv_bonus

        # Normalize heuristic scores to [0, 1] per drone
        heur_norm = np.zeros_like(heur_scores)
        for i in range(nA):
            s_min, s_max = float(heur_scores[i].min()), float(heur_scores[i].max())
            if s_max - s_min > 1e-5:
                heur_norm[i] = (heur_scores[i] - s_min) / (s_max - s_min)
            else:
                heur_norm[i] = 1.0 / N_GOALS

        # ── Step 3: Decentralized Conflict-Free Assignment (GNN -> Heuristic Fallback) ──
        assignments: dict[int, tuple[float, float, str, str, float]] = {}
        chosen_coords: list[tuple[float, float]] = []
        used_indices: set[int] = set()
        min_goal_sep = 16.0  # minimum 16m separation between assigned drone targets

        for i, d in enumerate(drones):
            chosen_k = None
            alloc_mode = 'Heuristic Auction'
            conf_score = 0.0

            # First try top GNN recommendations (if GNN is active)
            if gnn_ok:
                # Blend GNN probability with frontier gain so GNN never picks an already-empty cell
                combined = 0.55 * gnn_probs[i] + 0.45 * heur_norm[i]
                gnn_rank = np.argsort(-combined)
                for k_cand in gnn_rank[:5]:
                    k = int(k_cand)
                    if k in used_indices:
                        continue
                    cn, ce, _lbl, gain = candidates[k]
                    dist_self = math.hypot(cn - d.x, ce - d.y)
                    # Check separation from already-assigned peers and minimum info gain
                    if dist_self > 3.5 and gain >= 0.08 and all(
                        math.hypot(cn - pn, ce - pe) >= min_goal_sep for pn, pe in chosen_coords
                    ):
                        chosen_k = k
                        # Label pure GNN if it's in the raw GNN top-3, else GNN+Heuristic
                        raw_top3 = set(int(x) for x in np.argsort(-gnn_probs[i])[:3])
                        alloc_mode = 'GNN (GraphSAGE)' if k in raw_top3 else 'GNN+Heuristic'
                        conf_score = float(combined[k])
                        break

            # Fallback to Heuristic Auction if GNN did not yield a conflict-free high-gain goal
            if chosen_k is None:
                heur_rank = np.argsort(-heur_norm[i])
                for k_cand in heur_rank:
                    k = int(k_cand)
                    if k in used_indices:
                        continue
                    cn, ce, _lbl, _gain = candidates[k]
                    if all(math.hypot(cn - pn, ce - pe) >= min_goal_sep for pn, pe in chosen_coords):
                        chosen_k = k
                        alloc_mode = 'Heuristic Fallback'
                        conf_score = float(heur_norm[i, k])
                        break

            # Ultimate fallback if all candidates were close: pick best unused candidate
            if chosen_k is None:
                for k in range(N_GOALS):
                    if k not in used_indices:
                        chosen_k = k
                        alloc_mode = 'Heuristic Fallback'
                        conf_score = float(heur_norm[i, k])
                        break
                if chosen_k is None:
                    chosen_k = i % N_GOALS

            used_indices.add(chosen_k)
            cn, ce, lbl, _gain = candidates[chosen_k]
            chosen_coords.append((cn, ce))
            assignments[d.drone_id] = (cn, ce, lbl, alloc_mode, conf_score)

        return assignments
