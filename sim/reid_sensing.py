"""Noisy per-agent observation model feeding the Re-ID registry (lightweight sim).

Honest labels: positions carry a per-agent random-walk drift ("simulated VIO"), so
agents disagree about where a victim is, exactly the situation Re-ID must survive.
Ground-truth ids are attached for evaluation only.

Sensors per observation:
  thermal  - existing sensors.thermal_detect (range/FOV/LOS/p falls with distance)
  rgb      - night-vision, shorter range (0.6 x sensor range), needs LOS, gives appearance
  acoustic - existing sensors.acoustic_observe, poor localisation, no appearance
False positives: persistent hot rocks (thermal only) and transient thermal blips.
"""
from __future__ import annotations

import math

import numpy as np

from .map_gen import FREE
from .reid import Observation, EMB_DIM
from .sensors import line_of_sight, thermal_detect, acoustic_observe
from .env import SPECS, ACOUSTIC_THRESHOLD


def _unit(v):
    return v / np.linalg.norm(v)


class SensingModel:
    def __init__(self, env, seed: int = 0, n_rocks: int = 3, fp_rate: float = 0.02,
                 similar_frac: float = 0.34, drift: float = 0.03, dim: int = EMB_DIM,
                 acoustic: str = "bearing", align: bool = True, rel_sigma: float = 0.15,
                 bearing_sigma_deg: float = 8.0, range_frac: float = 0.25):
        self.env = env
        self.rng = np.random.default_rng(10_000 + seed)   # separate stream: env rng untouched
        self.dim = dim
        self.fp_rate = fp_rate
        self.drift = drift
        survs = env.bmap.survivors
        self.ident = {}
        for s in survs:
            self.ident[s.id] = _unit(self.rng.normal(size=dim))
        # some victims look alike (same clothing): correlated identity vectors
        ids = [s.id for s in survs]
        for k in range(0, len(ids) - 1, 2):
            if self.rng.random() < similar_frac:
                a, b = ids[k], ids[k + 1]
                self.ident[b] = _unit(self.ident[a] + 0.6 * _unit(self.rng.normal(size=dim)))
        free = np.argwhere(env.bmap.grid == FREE)
        self.rocks = []
        for _ in range(n_rocks):
            y, x = free[self.rng.integers(len(free))]
            self.rocks.append(((float(x), float(y)), _unit(self.rng.normal(size=dim))))
        self.bias = {ag.id: np.zeros(2) for ag in env.agents}
        self.pvar = {ag.id: 0.0 for ag in env.agents}   # pose variance per axis relative to the swarm
        self.acoustic_mode, self.align_on = acoustic, align
        self.rel_sigma, self.bearing_sigma, self.range_frac = rel_sigma, math.radians(bearing_sigma_deg), range_frac
        # oracle bookkeeping for evaluation (what was *observable*)
        self.gt_obs_count = {s.id: 0 for s in survs}
        self.gt_obs_sensors = {s.id: set() for s in survs}

    # ------------------------------------------------------------------ helpers
    def _emb(self, ident, s):
        n = self.rng.normal(size=self.dim) * (s / math.sqrt(self.dim))
        return _unit(ident + n)

    def _step_bias(self, ag):
        self.bias[ag.id] = self.bias[ag.id] + self.rng.normal(scale=self.drift, size=2)
        self.pvar[ag.id] += self.drift ** 2

    def align(self, i: int, j: int):
        """Inter-robot relative localisation on contact (ranging / mutual detection / loop closure).
        Pulls the two robots' drift toward their mean, leaving the swarm's COMMON drift untouched
        (it is unobservable and shifts every victim equally, so it cannot cause double counts).
        A simulation assumption, labelled as such in the report."""
        if not self.align_on:
            return
        m = 0.5 * (self.bias[i] + self.bias[j])
        e = self.rng.normal(scale=self.rel_sigma / math.sqrt(2), size=(2, 2))
        self.bias[i], self.bias[j] = m + e[0], m + e[1]
        v = 0.25 * (self.pvar[i] + self.pvar[j]) + self.rel_sigma ** 2
        self.pvar[i] = self.pvar[j] = v

    def observe(self, ag, t: int) -> list[Observation]:
        spec = SPECS[ag.type]
        self._step_bias(ag)
        ax, ay = ag.pos
        b = self.bias[ag.id]
        pose_var = self.pvar[ag.id]                        # honest, shrinks after alignment
        out: list[Observation] = []
        env = self.env
        rng = self.rng

        def pos_noise(sig):
            return rng.normal(scale=sig, size=2)

        for s in env.bmap.survivors:
            sx, sy = s.pos
            dist = math.hypot(sx - ax, sy - ay)
            sensors = set()
            if s.heat and thermal_detect(env.bmap, ax, ay, ag.heading, sx, sy,
                                         sensor_range=spec["sensor_range"],
                                         fov_deg=spec["fov_deg"], rng=rng, coarse=spec["coarse"]):
                sensors.add("thermal")
            if (s.heat and dist <= 0.6 * spec["sensor_range"] and rng.random() < 0.6
                    and line_of_sight(env.bmap, ax, ay, sx, sy)):
                sensors.add("rgb")
            if spec["acoustic"] and s.sound:
                a = acoustic_observe(env.bmap, ax, ay, sx, sy, s.loudness, rng)
                if a is not None and a >= ACOUSTIC_THRESHOLD:
                    sensors.add("acoustic")
            if not sensors:
                continue
            if "rgb" in sensors:
                sig, emb = 0.3, self._emb(self.ident[s.id], 0.45 + 0.4 * dist / spec["sensor_range"])
            elif "thermal" in sensors:
                sig = 0.8 if spec["coarse"] else 0.4
                emb = self._emb(self.ident[s.id], 0.9)         # thermal has little texture
            else:
                sig, emb = 1.5, None                            # acoustic-only
            cov = None
            if emb is None and self.acoustic_mode == "bearing":
                # microphone array: direction of arrival is good, range-from-loudness is poor
                rng_m = max(0.5, dist * (1.0 + rng.normal(scale=self.range_frac)))
                th = math.atan2(sy - ay, sx - ax) + rng.normal(scale=self.bearing_sigma)
                u = np.array([math.cos(th), math.sin(th)])
                w = np.array([-u[1], u[0]])
                s_r, s_t = max(0.3, self.range_frac * rng_m), max(0.3, rng_m * self.bearing_sigma)
                cov = s_r ** 2 * np.outer(u, u) + s_t ** 2 * np.outer(w, w)
                x = np.array([ax, ay], float) + b + rng_m * u
                sig = math.sqrt(0.5 * (s_r ** 2 + s_t ** 2))
            else:
                x = np.array([sx, sy], float) + b + pos_noise(sig)
            out.append(Observation(t, ag.id, x, sig ** 2, emb,
                                   frozenset(sensors), "person", pose_var=pose_var, cov=cov, gt=s.id))
            self.gt_obs_count[s.id] += 1
            self.gt_obs_sensors[s.id] |= sensors

        # persistent hot rocks: thermal only, never RGB-confirmable
        for (rx, ry), ident in self.rocks:
            if thermal_detect(env.bmap, ax, ay, ag.heading, rx, ry,
                              sensor_range=spec["sensor_range"], fov_deg=spec["fov_deg"],
                              rng=rng, coarse=spec["coarse"]):
                sig = 0.8 if spec["coarse"] else 0.4
                out.append(Observation(t, ag.id, np.array([rx, ry]) + b + pos_noise(sig),
                                       sig ** 2, self._emb(ident, 0.9),
                                       frozenset({"thermal"}), "person", pose_var=pose_var, gt=-1))
        # transient thermal blips (sun-heated debris, exhaust)
        if rng.random() < self.fp_rate:
            r = spec["sensor_range"]
            p = np.array([ax, ay]) + rng.uniform(-r, r, size=2)
            out.append(Observation(t, ag.id, p + b, 1.0, self._emb(_unit(rng.normal(size=self.dim)), 0.9),
                                   frozenset({"thermal"}), "person", pose_var=pose_var, gt=-1))
        return out

    def ideal_count_one(self, sid, n_confirm: int = 3) -> bool:
        """True if victim `sid` was observed enough to be confirmable (see ideal_count)."""
        n, sens = self.gt_obs_count[sid], self.gt_obs_sensors[sid]
        if sens == {"acoustic"}:
            return n >= 10
        return n >= n_confirm and len(sens) >= 2

    def ideal_count(self, n_confirm: int = 3) -> int:
        """Victims the fleet actually observed enough to be confirmable: isolates Re-ID
        error from exploration coverage."""
        return sum(1 for sid in self.gt_obs_count if self.ideal_count_one(sid, n_confirm))

    def min_separation(self) -> float:
        """Smallest distance between two confirmable victims (cells). Below ~4 cells the
        sensors cannot physically separate people, whatever the algorithm does."""
        pts = [s.pos for s in self.env.bmap.survivors if self.ideal_count_one(s.id)]
        d = [math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:]]
        return min(d) if d else float("inf")
