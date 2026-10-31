"""Distributed victim registry + multi-view Re-ID (no ROS).

Goal: a human is counted exactly once, however many robots see them, from how many
angles, or how often they re-enter view (see SAR_SWARM_SPEC.md section 4).

Design (CRDT-style, so it works with partitions and lossy gossip):
  * Every robot owns *tracks* it created. Only the owner updates a track, so every
    observation is fused into exactly one track (no double-counted information).
  * Tracks are replicated to neighbours; a replica is replaced only by a version with
    more observations (higher `n`). Replication is idempotent and order-independent.
  * "Same person" decisions are *links* between tracks, a grow-only set. Victims are
    the connected components (union-find), canonical id = min track uid.
  * The victim COUNT is derived from components. Observations can create or extend
    components, but a repeat observation of an already-known person can never add one.
  * Cannot-link constraints (two different detections in one camera frame are two
    different people) stop a transitive merge of distinct people.

Acoustic-only detections have no appearance and poor localisation, so they are NOT tracked
individually: they are splatted into a per-robot evidence map (a grow-only CRDT per owner) and
victims are the density peaks (source separation limit `ac_sep`). A peak that a confirmed
visual victim already explains is not counted again.

The grid sim is 2D, so the spatial gate is chi-square with 2 DoF (the spec's 3D gate
is chi2(3, 0.99) = 11.34). Ground-truth labels (`gt`) ride along ONLY for evaluation;
no decision in this file reads them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CHI2_2D_99 = 9.21
EMB_DIM = 32


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
def hungarian(cost) -> list[tuple[int, int]]:
    """Minimum-cost assignment on a rectangular matrix (O(n^2 m)); returns (row, col)
    pairs. Entries >= `INFEASIBLE` are legal but callers should drop such matches."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    n, m = cost.shape
    transpose = n > m
    if transpose:
        cost, n, m = cost.T, m, n
    inf = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j]]
    return [(c, r) for r, c in pairs] if transpose else pairs


INFEASIBLE = 1e6


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class ReIDConfig:
    chi2: float = CHI2_2D_99
    use_appearance: bool = True
    tau_app: float = 0.40        # obs vs track-mean cosine needed to associate
    tau_link: float = 0.50       # track vs track cosine needed to link
    tau_conflict: float = 0.15   # below this, a spatial match is a *different person*
    lam_app: float = 1.0
    q: float = 0.01              # victim process noise (cells^2 / step): near-static
    min_var: float = 0.02        # measurement-fusion floor; pose uncertainty is added separately
    link_persist: int = 3        # consecutive consolidation rounds a pair must pass (hysteresis)
    min_emb_for_link: int = 4    # embeddings per track before appearance may justify a LINK
    min_n_noapp_link: int = 3    # observations per track before a position-only link
    noapp_gate: float = 0.8      # fraction of chi2 gate when appearance is unavailable
    n_confirm: int = 3
    min_agents: int = 1          # distinct robots that must have seen the person (multi-view consensus)
    min_sensors: int = 2         # dual-spectrum rule; set 1 when only one camera type exists (Gazebo mono_cam)
    tight_link: float = 0.15     # fraction of chi2: position this tight links despite inconclusive appearance
    ac_min_mass: float = 10.0    # acoustic observations (mass) a source needs before it is counted
    ac_sep: float = 2.0          # cells: closer acoustic sources are physically unresolvable
    ac_reach: float = 4.5        # cells: evidence within this of a peak counts toward that source's mass
    ac_explain: float = 2.0      # cells: acoustic peak within this of a confirmed visual victim is the same person
    min_emb_for_app: int = 1     # embeddings needed before appearance is trusted


@dataclass
class Observation:
    t: int
    agent_id: int
    x: np.ndarray                 # world (x, y), the agent's own estimate (drifting VIO)
    var: float                    # measurement variance (cells^2)
    emb: np.ndarray | None        # unit appearance vector, None for acoustic-only
    sensors: frozenset            # subset of {'thermal', 'rgb', 'acoustic'}
    cls: str = "person"           # 'person' | 'responder' | 'robot'
    pose_var: float = 0.0         # observer pose (drift) variance: correlated, never averaged away
    cov: np.ndarray | None = None  # full 2x2 measurement covariance (bearing+range gives an elongated one)
    gt: int | None = None         # EVALUATION ONLY (-1 = false positive)


@dataclass
class Track:
    uid: tuple                    # (owner agent id, seq)
    x: np.ndarray
    var: float
    emb_sum: np.ndarray
    n_emb: int
    n: int
    sensors: set
    t_first: int
    t_last: int
    pose_var: float = 0.0
    gt_votes: dict = field(default_factory=dict)   # evaluation only

    @property
    def emb(self):
        norm = float(np.linalg.norm(self.emb_sum))
        return self.emb_sum / norm if norm > 0 else self.emb_sum

    def copy(self) -> "Track":
        return Track(self.uid, self.x.copy(), self.var, self.emb_sum.copy(), self.n_emb,
                     self.n, set(self.sensors), self.t_first, self.t_last, self.pose_var, dict(self.gt_votes))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class VictimRegistry:
    def __init__(self, agent_id: int, cfg: ReIDConfig | None = None):
        self.agent_id = agent_id
        self.cfg = cfg or ReIDConfig()
        self.tracks: dict[tuple, Track] = {}
        self.links: set[frozenset] = set()
        self.cannot: set[frozenset] = set()
        self.ambiguous: set[tuple] = set()
        self._seq = 0
        self.log: list[tuple] = []        # (t, 'link'|'ambiguous', details): auditable merges
        self._pending: dict[frozenset, int] = {}
        self.ac: dict[int, list] = {}      # owner -> [n_obs, {(ix, iy): mass}]  acoustic evidence

    # ------------------------------------------------------------- local tracking
    def _var_eff(self, tr: Track, t: int) -> float:
        return tr.var + tr.pose_var + self.cfg.q * max(0, t - tr.t_last)

    def _assoc_cost(self, o: Observation, tr: Track, t: int) -> float:
        c = self.cfg
        d2 = float(np.sum((o.x - tr.x) ** 2)) / (o.var + o.pose_var + self._var_eff(tr, t))
        use_app = (c.use_appearance and o.emb is not None and tr.n_emb >= c.min_emb_for_app)
        if use_app:
            cs = _cos(o.emb, tr.emb)
            if d2 > c.chi2 or cs < c.tau_app:
                return INFEASIBLE
            return d2 / c.chi2 + c.lam_app * (1.0 - cs)
        if d2 > c.chi2 * c.noapp_gate:
            return INFEASIBLE
        return d2 / c.chi2 + 0.3

    def _update(self, tr: Track, o: Observation):
        k = tr.var / (tr.var + o.var)
        tr.x = tr.x + k * (o.x - tr.x)
        tr.var = max(self.cfg.min_var, (1.0 - k) * tr.var)
        if o.emb is not None:
            # a track born without appearance has a placeholder vector of the wrong length
            tr.emb_sum = o.emb.copy() if tr.n_emb == 0 else tr.emb_sum + o.emb
            tr.n_emb += 1
        tr.n += 1
        tr.sensors |= set(o.sensors)
        tr.t_last = o.t
        tr.pose_var = max(tr.pose_var, o.pose_var)
        if o.gt is not None:
            tr.gt_votes[o.gt] = tr.gt_votes.get(o.gt, 0) + 1

    def _new_track(self, o: Observation) -> Track:
        self._seq += 1
        uid = (self.agent_id, self._seq)
        emb = o.emb if o.emb is not None else np.zeros(EMB_DIM)
        tr = Track(uid, o.x.copy(), max(self.cfg.min_var, o.var), emb.copy(),
                   1 if o.emb is not None else 0, 1, set(o.sensors), o.t, o.t, o.pose_var,
                   {o.gt: 1} if o.gt is not None else {})
        self.tracks[uid] = tr
        return tr

    def observe(self, observations: list[Observation], t: int):
        """Associate this agent's detections from ONE frame with its own tracks
        (Hungarian), update matches, spawn tentative tracks for the rest."""
        obs = [o for o in observations if o.cls == "person"]   # responders/robots never counted
        acoustic_only = frozenset({"acoustic"})
        for o in obs:
            if o.sensors == acoustic_only:        # sound has no appearance: evidence map, not a track
                self._splat(o)
        obs = [o for o in obs if o.sensors != acoustic_only]
        own = [tr for uid, tr in self.tracks.items() if uid[0] == self.agent_id]
        assigned: list[Track] = []
        used_obs = set()
        if obs and own:
            cost = np.full((len(obs), len(own)), INFEASIBLE)
            for i, o in enumerate(obs):
                for j, tr in enumerate(own):
                    cost[i, j] = self._assoc_cost(o, tr, t)
            for i, j in hungarian(cost):
                if cost[i, j] < INFEASIBLE:
                    self._update(own[j], obs[i])
                    assigned.append(own[j])
                    used_obs.add(i)
        for i, o in enumerate(obs):
            if i not in used_obs:
                assigned.append(self._new_track(o))
        # two detections in one frame from one camera are two different people
        for a in range(len(assigned)):
            for b in range(a + 1, len(assigned)):
                if assigned[a].uid != assigned[b].uid:
                    self.cannot.add(frozenset((assigned[a].uid, assigned[b].uid)))

    # ------------------------------------------------------------ acoustic evidence
    def _splat(self, o: Observation):
        """Add one acoustic detection as a unit-mass Gaussian (sigma from measurement +
        pose uncertainty) to this robot's own evidence map."""
        ent = self.ac.setdefault(self.agent_id, [0, {}])
        ent[0] += 1
        cov = (o.cov if o.cov is not None else np.eye(2) * o.var) + np.eye(2) * o.pose_var
        inv = np.linalg.inv(cov)
        r = int(math.ceil(3 * math.sqrt(float(np.linalg.eigvalsh(cov)[-1]))))
        cx, cy = int(round(o.x[0])), int(round(o.x[1]))
        cells, ws = [], []
        for ix in range(cx - r, cx + r + 1):
            for iy in range(cy - r, cy + r + 1):
                d = np.array([ix - o.x[0], iy - o.x[1]])
                cells.append((ix, iy))
                ws.append(math.exp(-0.5 * float(d @ inv @ d)))
        tot = sum(ws)
        grid = ent[1]
        for cell, w in zip(cells, ws):
            grid[cell] = grid.get(cell, 0.0) + w / tot

    def acoustic_sources(self) -> list[dict]:
        """Density peaks of the fused acoustic evidence (greedy non-max suppression with
        radius `ac_sep`). Deterministic given the replicated maps, so identical everywhere."""
        c = self.cfg
        dens: dict = {}
        for owner in sorted(self.ac):
            for cell, w in self.ac[owner][1].items():
                dens[cell] = dens.get(cell, 0.0) + w
        if not dens:
            return []
        # a peak is a cell no other cell within `ac_sep` beats (ties broken by cell index), so a
        # cloud's tail can never masquerade as a second source
        rad = int(c.ac_sep)
        offs = [(dx, dy) for dx in range(-rad, rad + 1) for dy in range(-rad, rad + 1)
                if 0 < dx * dx + dy * dy <= c.ac_sep ** 2]
        peaks = []
        for cell in sorted(dens):
            v = dens[cell]
            if v < 0.05:
                continue
            if all((dens.get((cell[0] + dx, cell[1] + dy), 0.0), (cell[0] + dx, cell[1] + dy))
                   < (v, cell) for dx, dy in offs):
                peaks.append(cell)
        mass = [0.0] * len(peaks)
        sx = [0.0] * len(peaks)
        sy = [0.0] * len(peaks)
        reach2 = c.ac_reach ** 2
        for cell, w in dens.items():
            best, bd = -1, reach2
            for i, q in enumerate(peaks):
                d = (cell[0] - q[0]) ** 2 + (cell[1] - q[1]) ** 2
                if d < bd:
                    best, bd = i, d
            if best >= 0:
                mass[best] += w
                sx[best] += w * cell[0]
                sy[best] += w * cell[1]
        return [{"x": np.array([sx[i] / mass[i], sy[i] / mass[i]]), "mass": mass[i]}
                for i in range(len(peaks)) if mass[i] >= c.ac_min_mass]

    # ------------------------------------------------------------------ gossip
    def merge(self, other: "VictimRegistry"):
        """Idempotent, commutative, associative join of two replicas."""
        for uid, tr in other.tracks.items():
            mine = self.tracks.get(uid)
            if mine is None or tr.n > mine.n:
                self.tracks[uid] = tr.copy()
        for owner, (n_obs, grid) in other.ac.items():
            mine = self.ac.get(owner)
            if mine is None or n_obs > mine[0]:
                self.ac[owner] = [n_obs, dict(grid)]
        self.links |= other.links
        self.cannot |= other.cannot
        self.ambiguous |= other.ambiguous
        if other.agent_id != self.agent_id:
            for uid in other.tracks:
                if uid[0] == self.agent_id:
                    self._seq = max(self._seq, uid[1])

    # ------------------------------------------------------------ union-find
    def _find_map(self) -> dict:
        parent = {uid: uid for uid in self.tracks}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for link in self.links:
            a, b = tuple(link)
            if a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    # canonical = smaller uid => identical on every replica
                    parent[max(ra, rb)] = min(ra, rb)
        return {uid: find(uid) for uid in parent}

    def consolidate(self, t: int):
        """Link tracks that are the same person. Single-linkage over a spatial gate
        (equivalent to DBSCAN with min_samples=1) AND an appearance check."""
        c = self.cfg
        roots = self._find_map()
        uids = sorted(self.tracks)
        pairs = []
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                a, b = self.tracks[uids[i]], self.tracks[uids[j]]
                if roots[a.uid] == roots[b.uid]:
                    continue
                d2 = float(np.sum((a.x - b.x) ** 2)) / (self._var_eff(a, t) + self._var_eff(b, t))
                if d2 < c.chi2:
                    pairs.append((d2, a.uid, b.uid))
        pairs.sort()
        passed = set()
        for d2, ua, ub in pairs:
            roots = self._find_map()
            ra, rb = roots[ua], roots[ub]
            if ra == rb:
                continue
            if any((roots.get(x) == ra and roots.get(y) == rb) or (roots.get(x) == rb and roots.get(y) == ra)
                   for x, y in map(tuple, self.cannot) if x in roots and y in roots):
                continue
            a, b = self.tracks[ua], self.tracks[ub]
            key = frozenset((ua, ub))
            if c.use_appearance and a.n_emb >= c.min_emb_for_link and b.n_emb >= c.min_emb_for_link:
                cs = _cos(a.emb, b.emb)
                if cs >= c.tau_link:
                    ok = True
                elif cs >= c.tau_conflict and d2 < c.chi2 * c.tight_link:
                    ok = True        # appearance inconclusive (e.g. weak thermal texture) but positions coincide
                elif cs < c.tau_conflict:
                    # same spot, clearly different appearance => two people; never auto-merge
                    if not (a.uid in self.ambiguous and b.uid in self.ambiguous):
                        self.log.append((t, "ambiguous", ua, ub, round(d2, 2), round(cs, 2)))
                    self.ambiguous.update((ua, ub))
                    ok = False
                else:
                    ok = False
                info = round(cs, 2)
            elif (not c.use_appearance and d2 < c.chi2 * c.noapp_gate
                  and a.n >= c.min_n_noapp_link and b.n >= c.min_n_noapp_link):
                ok = True            # position-only variant (ablation)
                info = None
            else:
                ok = False
            if ok:
                passed.add(key)
                self._pending[key] = self._pending.get(key, 0) + 1
                if self._pending[key] >= c.link_persist:
                    self.links.add(key)
                    self.log.append((t, "link", ua, ub, round(d2, 2), info))
        for key in list(self._pending):
            if key not in passed:
                del self._pending[key]

    # -------------------------------------------------------------- serialisation
    def snapshot(self) -> dict:
        """JSON-safe replica state for a real comms link (ground-truth labels are NOT sent)."""
        return {
            "agent_id": self.agent_id,
            "tracks": [{"uid": list(t.uid), "x": t.x.tolist(), "var": t.var,
                        "emb_sum": [round(float(v), 4) for v in t.emb_sum], "n_emb": t.n_emb, "n": t.n,
                        "sensors": sorted(t.sensors), "t_first": t.t_first, "t_last": t.t_last,
                        "pose_var": t.pose_var} for t in self.tracks.values()],
            "links": [sorted(map(list, l)) for l in self.links],
            "cannot": [sorted(map(list, l)) for l in self.cannot],
            "ambiguous": [list(u) for u in self.ambiguous],
            "ac": {str(o): [n, [[ix, iy, round(w, 4)] for (ix, iy), w in g.items() if w >= 0.005]]
                   for o, (n, g) in self.ac.items()},
        }

    @classmethod
    def from_snapshot(cls, d: dict, cfg: ReIDConfig | None = None) -> "VictimRegistry":
        r = cls(int(d["agent_id"]), cfg)
        for t in d["tracks"]:
            uid = tuple(t["uid"])
            r.tracks[uid] = Track(uid, np.array(t["x"], float), float(t["var"]),
                                  np.array(t["emb_sum"], float), int(t["n_emb"]), int(t["n"]),
                                  set(t["sensors"]), int(t["t_first"]), int(t["t_last"]),
                                  float(t["pose_var"]))
        r.links = {frozenset(tuple(u) for u in l) for l in d["links"]}
        r.cannot = {frozenset(tuple(u) for u in l) for l in d["cannot"]}
        r.ambiguous = {tuple(u) for u in d["ambiguous"]}
        r.ac = {int(o): [int(n), {(ix, iy): w for ix, iy, w in cells}] for o, (n, cells) in d["ac"].items()}
        return r

    def merge_snapshot(self, d: dict):
        self.merge(VictimRegistry.from_snapshot(d, self.cfg))

    # ------------------------------------------------------------------ output
    def victims(self) -> list[dict]:
        roots = self._find_map()
        groups: dict[tuple, list[Track]] = {}
        for uid, tr in self.tracks.items():
            groups.setdefault(roots[uid], []).append(tr)
        out = []
        for cid, members in sorted(groups.items()):
            w = np.array([1.0 / m.var for m in members])
            x = sum(wi * m.x for wi, m in zip(w, members)) / w.sum()
            sensors = set().union(*(m.sensors for m in members))
            n_total = sum(m.n for m in members)
            n_agents = len({m.uid[0] for m in members})
            confirmed = (n_total >= self.cfg.n_confirm and len(sensors) >= self.cfg.min_sensors
                         and n_agents >= self.cfg.min_agents)
            votes: dict = {}
            for m in members:
                for k, v in m.gt_votes.items():
                    votes[k] = votes.get(k, 0) + v
            out.append({"id": cid, "x": x, "n": n_total, "sensors": sensors,
                        "confirmed": confirmed, "members": [m.uid for m in members],
                        "ambiguous": any(m.uid in self.ambiguous for m in members),
                        "gt_votes": votes, "kind": "visual"})
        vis = [v["x"] for v in out if v["confirmed"]]
        for k, src in enumerate(self.acoustic_sources()):
            if any(np.linalg.norm(src["x"] - xv) <= self.cfg.ac_explain for xv in vis):
                continue                      # a confirmed visual victim already explains this sound
            out.append({"id": ("ac", int(round(src["x"][0])), int(round(src["x"][1]))),
                        "x": src["x"], "n": src["mass"], "sensors": {"acoustic"},
                        "confirmed": True, "members": [], "ambiguous": False,
                        "gt_votes": {}, "kind": "acoustic"})
        return out

    def count(self) -> int:
        return sum(1 for v in self.victims() if v["confirmed"])
