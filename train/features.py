"""Observation / feature extraction for the MARL policy.

Builds a fixed-size observation per agent and the goal-candidate list + graph
edges, from the RescueEnv state. Decentralized: uses each agent's local (shared
via comms) explored map, not global ground truth.
"""
from __future__ import annotations

import math

import numpy as np

from sim.map_gen import ROVER_OK, DRONE_OK, FREE, WALL, NARROW, RUBBLE
from sim.sensors import line_of_sight, _walls_between

N_GOALS = 20        # max number of goal (frontier) candidates per agent
PATCH = 7           # local occupancy patch size (PATCH x PATCH)
OCC_CLASSES = 5     # -1 unknown, 0 free, 1 wall, 2 narrow, 3 rubble

# feature dimensions
FA = 2 + 1 + 2 + 2 + 1 + 1 + 1 + 1 + (PATCH * PATCH) * 2  # = 109
FG = 5              # goal feature dim


def _norm(x, denom):
    return float(x) / max(1.0, float(denom))


def _thermal_hint(env, ag) -> float:
    """Deterministic hint: closeness of the nearest un-found warm survivor in LOS."""
    ax, ay = ag.pos
    best = None
    for s in env.bmap.survivors:
        if s.found or not s.heat:
            continue
        sx, sy = s.pos
        d = math.hypot(sx - ax, sy - ay)
        if d > 8:
            continue
        if not line_of_sight(env.bmap, ax, ay, sx, sy):
            continue
        if best is None or d < best:
            best = d
    return 0.0 if best is None else 1.0 / (1.0 + best)


def _acoustic_hint(env, ag) -> float:
    """Deterministic hint: strongest un-found sound source (rovers only)."""
    if ag.type != "rover":
        return 0.0
    ax, ay = ag.pos
    best = 0.0
    for s in env.bmap.survivors:
        if s.found or not s.sound:
            continue
        sx, sy = s.pos
        d = max(0.5, math.hypot(sx - ax, sy - ay))
        walls = _walls_between(env.bmap, ax, ay, sx, sy)
        atten_db = 20.0 * math.log10(d) + walls * 12.0
        recv = s.loudness * 100.0 / (10 ** (atten_db / 20.0))
        best = max(best, min(1.0, recv / 50.0))
    return best


def _occupancy_patch(env, ag):
    """Two-channel PATCHxPATCH patch centered on agent: occupancy + known survivor."""
    H, W = env.bmap.H, env.bmap.W
    ax, ay = ag.pos
    half = PATCH // 2
    occ = np.full((PATCH, PATCH), -1.0, dtype=np.float32)
    surv = np.zeros((PATCH, PATCH), dtype=np.float32)
    # survivor positions the agent knows about
    known_pos = {}
    for s in env.bmap.survivors:
        if s.id in ag.known:
            known_pos[s.pos] = True
    for py in range(PATCH):
        for px in range(PATCH):
            gx = ax + (px - half)
            gy = ay + (py - half)
            if 0 <= gx < W and 0 <= gy < H:
                if ag.explored[gy, gx]:
                    occ[py, px] = float(env.bmap.grid[gy, gx])
                # survivor channel uses known survivors regardless of explored
                if (gx, gy) in known_pos:
                    surv[py, px] = 1.0
    return occ, surv


def agent_feature_dim() -> int:
    return FA


def agent_features(env, ag) -> np.ndarray:
    H, W = env.bmap.H, env.bmap.W
    occ, surv = _occupancy_patch(env, ag)
    # nearest frontier distance (from the agent's own explored map)
    fr = _frontiers(env, ag)
    if fr:
        ndist = min(math.hypot(ag.pos[0] - c[0], ag.pos[1] - c[1]) for c in fr)
        ndist = _norm(ndist, math.hypot(W, H))
    else:
        ndist = 1.0

    feat = np.concatenate([
        np.array([1.0, 0.0] if ag.type == "drone" else [0.0, 1.0], dtype=np.float32),  # type
        np.array([ag.battery], dtype=np.float32),                                        # battery
        np.array([ag.pos[0] / W, ag.pos[1] / H], dtype=np.float32),                      # norm pos
        np.array([math.sin(ag.heading), math.cos(ag.heading)], dtype=np.float32),        # heading
        np.array([len(ag.known) / max(1, len(env.bmap.survivors))], dtype=np.float32),   # known surv
        np.array([_thermal_hint(env, ag)], dtype=np.float32),                            # thermal
        np.array([_acoustic_hint(env, ag)], dtype=np.float32),                           # acoustic
        np.array([ndist], dtype=np.float32),                                             # frontier dist
        occ.reshape(-1),
        surv.reshape(-1),
    ]).astype(np.float32)
    assert feat.shape[0] == FA, f"feature dim {feat.shape[0]} != {FA}"
    return feat


def _frontiers(env, ag):
    from sim.baselines import _frontiers as _f
    return _f(env, ag)


def goal_features(env, ag):
    """Return (goals_feat [N_GOALS, FG], goal_cells [N_GOALS, 2], valid [N_GOALS])."""
    H, W = env.bmap.H, env.bmap.W
    fr = _frontiers(env, ag)
    # sort frontiers by distance to agent (nearest first) for a stable slot ordering
    fr = sorted(fr, key=lambda c: math.hypot(ag.pos[0] - c[0], ag.pos[1] - c[1]))
    ax, ay = ag.pos
    known_pos = {s.pos for s in env.bmap.survivors if s.id in ag.known}
    feats = np.zeros((N_GOALS, FG), dtype=np.float32)
    cells = np.zeros((N_GOALS, 2), dtype=np.float32)
    valid = np.zeros(N_GOALS, dtype=bool)
    for k in range(N_GOALS):
        if k >= len(fr):
            cells[k] = [ax, ay]  # pad with current pos -> "stay"
            continue
        x, y = fr[k]
        valid[k] = True
        cells[k] = [x, y]
        dx, dy = x - ax, y - ay
        dist = math.hypot(dx, dy)
        # info gain = number of unexplored neighbors
        gain = 0
        for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + ddx, y + ddy
            if 0 <= nx < W and 0 <= ny < H and not ag.explored[ny, nx]:
                gain += 1
        near_surv = 1.0 if any(math.hypot(x - sx, y - sy) <= 3 for (sx, sy) in known_pos) else 0.0
        feats[k] = [dx / W, dy / H, dist / math.hypot(W, H), gain / 4.0, near_surv]
    return feats, cells, valid


def comm_adj(env):
    """Dense agent-agent adjacency [nA, nA]: 1.0 if within comms range."""
    n = len(env.agents)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = env.agents[i], env.agents[j]
            if ai.alive and aj.alive:
                d = math.hypot(ai.pos[0] - aj.pos[0], ai.pos[1] - aj.pos[1])
                if d <= env.comm_range:
                    adj[i, j] = adj[j, i] = 1.0
    return adj


def build_obs(env):
    """One env's observation: everything the policy + critic need."""
    types = [ag.type for ag in env.agents]
    alive = [ag.alive for ag in env.agents]
    agent_f = np.stack([agent_features(env, ag) for ag in env.agents]).astype(np.float32)
    g_feats, g_cells, g_valid = [], [], []
    for ag in env.agents:
        f, c, v = goal_features(env, ag)
        g_feats.append(f)
        g_cells.append(c)
        g_valid.append(v)
    return {
        "agent_feats": agent_f,                                   # [nA, FA]
        "goal_feats": np.stack(g_feats).astype(np.float32),       # [nA, NG, FG]
        "goal_cells": np.stack(g_cells).astype(np.float32),       # [nA, NG, 2]
        "goal_valid": np.stack(g_valid),                          # [nA, NG] bool
        "types": types,
        "alive": np.array(alive),
        "comm_adj": comm_adj(env),                                # [nA, nA]
        "coverage": np.array([env.coverage()], dtype=np.float32), # [1]
    }
