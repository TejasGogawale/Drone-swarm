"""Unit + integration tests for the victim Re-ID registry (sim/reid.py).

Run:  python -m pytest tests/test_reid.py -q
"""
import itertools
import math

import numpy as np
import pytest

from sim.reid import (EMB_DIM, Observation, ReIDConfig, VictimRegistry, hungarian)

RNG = np.random.default_rng(0)


def unit(v):
    return v / np.linalg.norm(v)


def identity(seed):
    return unit(np.random.default_rng(seed).normal(size=EMB_DIM))


def emb_obs(ident, rng, s=0.45):
    return unit(ident + rng.normal(size=EMB_DIM) * (s / math.sqrt(EMB_DIM)))


def obs(t, agent, xy, emb, var=0.15, sensors=("thermal", "rgb"), pose_var=0.05, cls="person"):
    return Observation(t, agent, np.array(xy, float), var, emb, frozenset(sensors), cls,
                       pose_var=pose_var)


def converge(regs, t=1000, rounds=4):
    for _ in range(rounds):
        for a in regs:
            for b in regs:
                a.merge(b)
        for r in regs:
            r.consolidate(t)
    for a in regs:
        for b in regs:
            a.merge(b)


def watch(reg, ident, xy, agent, t0, n, rng, bias=(0.0, 0.0), sensors=("thermal", "rgb")):
    for k in range(n):
        x = np.array(xy) + np.array(bias) + rng.normal(scale=0.2, size=2)
        reg.observe([obs(t0 + k, agent, x, emb_obs(ident, rng), sensors=sensors)], t0 + k)


# ------------------------------------------------------------------ assignment
def test_hungarian_matches_bruteforce_square():
    rng = np.random.default_rng(1)
    for _ in range(20):
        c = rng.uniform(0, 10, size=(5, 5))
        got = sum(c[i, j] for i, j in hungarian(c))
        best = min(sum(c[i, p[i]] for i in range(5)) for p in itertools.permutations(range(5)))
        assert got == pytest.approx(best)


@pytest.mark.parametrize("shape", [(3, 5), (5, 3)])
def test_hungarian_rectangular_is_optimal(shape):
    rng = np.random.default_rng(2)
    n, m = shape
    for _ in range(20):
        c = rng.uniform(0, 10, size=shape)
        pairs = hungarian(c)
        assert len(pairs) == min(n, m)
        got = sum(c[i, j] for i, j in pairs)
        k = min(n, m)
        if n <= m:
            best = min(sum(c[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        else:
            best = min(sum(c[p[j], j] for j in range(m)) for p in itertools.permutations(range(n), m))
        assert got == pytest.approx(best)


# --------------------------------------------------------------- counting rules
def test_same_victim_two_agents_is_one_victim():
    rng = np.random.default_rng(3)
    ident = identity(10)
    a, b = VictimRegistry(0), VictimRegistry(1)
    watch(a, ident, (10, 10), 0, 0, 8, rng, bias=(0.3, -0.2))
    watch(b, ident, (10, 10), 1, 0, 8, rng, bias=(-0.2, 0.3))
    assert a.count() == 1 and b.count() == 1          # each alone sees one
    converge([a, b])
    assert a.count() == 1 and b.count() == 1
    assert len(a.victims()) == 1                      # no leftover duplicate component


def test_repeat_observations_never_raise_count():
    rng = np.random.default_rng(4)
    ident = identity(11)
    r = VictimRegistry(0)
    counts = []
    for k in range(200):
        r.observe([obs(k, 0, np.array([5, 5]) + rng.normal(scale=0.2, size=2), emb_obs(ident, rng))], k)
        if k % 10 == 0:
            r.consolidate(k)
        counts.append(r.count())
    assert max(counts) == 1
    assert len(r.tracks) == 1


def test_count_is_monotone_under_merges():
    rng = np.random.default_rng(5)
    regs = [VictimRegistry(i) for i in range(3)]
    for i, r in enumerate(regs):
        for v in range(4):
            watch(r, identity(100 + v), (10 + 8 * v, 10), i, 0, 6, rng, bias=rng.normal(scale=0.3, size=2))
    before = sum(r.count() for r in regs)             # naive sum double counts
    converge(regs)
    assert all(r.count() == 4 for r in regs)
    assert before > 4


def test_two_victims_close_together_need_appearance():
    rng = np.random.default_rng(6)
    ia, ib = identity(20), identity(21)

    def build(use_app):
        cfg = ReIDConfig(use_appearance=use_app)
        a, b = VictimRegistry(0, cfg), VictimRegistry(1, cfg)
        watch(a, ia, (10, 10), 0, 0, 10, rng)          # robot A only ever sees person A
        watch(b, ib, (11.2, 10), 1, 0, 10, rng)        # robot B only ever sees person B
        converge([a, b])
        return a.count()

    assert build(True) == 2           # appearance keeps two people apart
    assert build(False) == 1          # position-only would wrongly merge them


def test_same_frame_detections_are_never_merged():
    rng = np.random.default_rng(7)
    ident = identity(30)
    a, helper = VictimRegistry(0), VictimRegistry(1)
    for k in range(10):
        # two detections 0.3 cells apart in ONE camera frame => two people
        a.observe([obs(k, 0, (4.0, 4.0), emb_obs(ident, rng)),
                   obs(k, 0, (4.3, 4.0), emb_obs(ident, rng))], k)
    watch(helper, ident, (4.15, 4.0), 1, 0, 10, rng)   # a third view straddles both
    converge([a, helper])
    assert a.count() == 2


def test_responders_and_robots_are_not_counted():
    rng = np.random.default_rng(8)
    ident = identity(40)
    r = VictimRegistry(0)
    for k in range(20):
        r.observe([obs(k, 0, (3, 3), emb_obs(ident, rng), cls="responder"),
                   obs(k, 0, (6, 6), emb_obs(ident, rng), cls="robot")], k)
    assert r.count() == 0 and len(r.tracks) == 0


def test_thermal_only_hot_rock_is_never_confirmed():
    rng = np.random.default_rng(9)
    r = VictimRegistry(0)
    ident = identity(50)
    for k in range(300):
        r.observe([obs(k, 0, (7, 7), emb_obs(ident, rng, 0.9), sensors=("thermal",))], k)
    assert r.count() == 0


def test_buried_acoustic_victim_needs_enough_evidence():
    rng = np.random.default_rng(10)
    r = VictimRegistry(0)
    for k in range(5):
        r.observe([obs(k, 0, np.array([9, 9]) + rng.normal(scale=1.0, size=2), None, var=2.25,
                       sensors=("acoustic",))], k)
    assert r.count() == 0                     # 5 coarse detections: not enough
    for k in range(5, 40):
        r.observe([obs(k, 0, np.array([9, 9]) + rng.normal(scale=1.0, size=2), None, var=2.25,
                       sensors=("acoustic",))], k)
    assert r.count() == 1


def test_reentry_after_long_gap_keeps_the_same_victim():
    rng = np.random.default_rng(11)
    ident = identity(60)
    r = VictimRegistry(0)
    watch(r, ident, (12, 12), 0, 0, 6, rng)
    cid = r.victims()[0]["id"]
    watch(r, ident, (12.4, 11.8), 0, 500, 6, rng)       # back after 500 steps, slight pose shift
    assert r.count() == 1
    assert len(r.tracks) == 1 and r.victims()[0]["id"] == cid


# ------------------------------------------------------------------- CRDT laws
def _three_replicas():
    rng = np.random.default_rng(12)
    regs = [VictimRegistry(i) for i in range(3)]
    for i, r in enumerate(regs):
        for v in range(3):
            watch(r, identity(200 + v), (6 + 9 * v, 8), i, 0, 6, rng, bias=rng.normal(scale=0.25, size=2))
    return regs


def _fingerprint(r):
    return (sorted(r.tracks), sorted(map(tuple, map(sorted, r.links))),
            [(v["id"], v["confirmed"]) for v in r.victims()])


def test_merge_is_idempotent_and_order_independent():
    regs = _three_replicas()
    a1 = VictimRegistry(9)
    for r in regs:
        a1.merge(r)
    a2 = VictimRegistry(9)
    for r in reversed(regs):
        a2.merge(r)
    a2.merge(regs[0])                                   # duplicate delivery
    assert sorted(a1.tracks) == sorted(a2.tracks)
    fp = _fingerprint(a1)
    a1.merge(a2)
    a1.merge(a1)
    assert _fingerprint(a1) == fp


def test_canonical_ids_identical_on_every_replica():
    regs = _three_replicas()
    converge(regs)
    ids = [[v["id"] for v in r.victims()] for r in regs]
    assert ids[0] == ids[1] == ids[2]


def test_partition_then_heal_gives_same_answer_as_always_connected():
    rng = np.random.default_rng(13)
    ident = identity(70)
    a, b = VictimRegistry(0), VictimRegistry(1)
    watch(a, ident, (10, 10), 0, 0, 10, rng, bias=(0.2, 0.1))
    watch(b, ident, (10, 10), 1, 0, 10, rng, bias=(-0.1, 0.2))
    assert a.count() + b.count() == 2                  # partitioned: each reports its own
    converge([a, b])
    assert a.count() == b.count() == 1                 # healed: one person


# ------------------------------------------------------------------ integration
def test_swarm_episode_beats_naive_counting():
    from sim.run_reid import run
    dup_reid = dup_local = 0
    for seed in range(5):
        a = run(seed, "reid", 4, 2, 300, 40)
        b = run(seed, "local_only", 4, 2, 300, 40)
        dup_reid += a["duplicates"]
        dup_local += b["duplicates"]
        assert a["false_pos"] == 0
    assert dup_reid * 3 <= dup_local          # sharing + Re-ID removes most double counts


def test_swarm_episode_replicas_agree():
    from sim.run_reid import run
    r = run(1, "reid", 4, 2, 300, 40)
    assert r["consistent"] and r["false_pos"] == 0


# -------------------------------------------------------------- acoustic evidence
def ac(reg, rng, xy, n, agent=None, bias=(0.0, 0.0), t0=0, sigma=1.5):
    for k in range(n):
        x = np.array(xy, float) + np.array(bias) + rng.normal(scale=sigma, size=2)
        reg.observe([obs(t0 + k, reg.agent_id if agent is None else agent, x, None, var=sigma ** 2,
                         sensors=("acoustic",), pose_var=0.1)], t0 + k)


def test_two_buried_victims_far_apart_are_two_sources_on_every_replica():
    rng = np.random.default_rng(20)
    a, b = VictimRegistry(0), VictimRegistry(1)
    ac(a, rng, (10, 10), 20, bias=(0.4, -0.3))
    ac(b, rng, (10, 10), 20, bias=(-0.3, 0.4))
    ac(a, rng, (22, 14), 20, bias=(0.4, -0.3), t0=20)
    ac(b, rng, (22, 14), 20, bias=(-0.3, 0.4), t0=20)
    converge([a, b])
    assert a.count() == b.count() == 2
    assert [v["id"] for v in a.victims()] == [v["id"] for v in b.victims()]


def test_one_buried_victim_heard_by_two_robots_is_counted_once():
    rng = np.random.default_rng(21)
    a, b = VictimRegistry(0), VictimRegistry(1)
    ac(a, rng, (15, 15), 25, bias=(0.6, 0.0))
    ac(b, rng, (15, 15), 25, bias=(-0.5, 0.2))
    assert a.count() + b.count() == 2                 # each alone
    converge([a, b])
    assert a.count() == b.count() == 1


def test_too_few_acoustic_detections_are_not_a_victim():
    rng = np.random.default_rng(22)
    r = VictimRegistry(0)
    ac(r, rng, (5, 5), 6)
    assert r.count() == 0


def test_acoustic_peak_is_explained_by_confirmed_visual_victim_only():
    rng = np.random.default_rng(23)
    ident = identity(80)
    r = VictimRegistry(0)
    watch(r, ident, (12, 12), 0, 0, 8, rng)           # confirmed (rgb+thermal) at (12, 12)
    ac(r, rng, (12.5, 12), 30, t0=10)                 # sound from the same spot: same person
    assert r.count() == 1
    ac(r, rng, (22, 12), 30, t0=50)                   # sound from elsewhere: a buried second victim
    assert r.count() == 2


def test_unconfirmed_hot_rock_does_not_hide_a_buried_victim():
    rng = np.random.default_rng(24)
    r = VictimRegistry(0)
    for k in range(60):
        r.observe([obs(k, 0, (9, 9), emb_obs(identity(90), rng, 0.9), sensors=("thermal",))], k)
    ac(r, rng, (9.5, 9), 30, t0=70)
    assert r.count() == 1                             # the rock is not a victim, the sound is


def test_victims_closer_than_acoustic_resolution_are_one_source():
    """Documented limit: buried people ~1 cell apart can't be separated by sound alone."""
    rng = np.random.default_rng(25)
    r = VictimRegistry(0)
    ac(r, rng, (10, 10), 20)
    ac(r, rng, (11, 10), 20, t0=20)
    assert r.count() == 1


def test_acoustic_merge_is_idempotent_and_order_independent():
    rng = np.random.default_rng(26)
    regs = [VictimRegistry(i) for i in range(3)]
    for i, r in enumerate(regs):
        ac(r, rng, (8 + 9 * i, 10), 18)
    x, y = VictimRegistry(9), VictimRegistry(9)
    for r in regs:
        x.merge(r)
    for r in reversed(regs):
        y.merge(r)
    y.merge(regs[1])
    assert x.ac.keys() == y.ac.keys()
    assert [v["id"] for v in x.victims()] == [v["id"] for v in y.victims()]
    before = [v["id"] for v in x.victims()]
    x.merge(x)
    assert [v["id"] for v in x.victims()] == before


# ----------------------------------------------- bearing acoustics + pose alignment
def _sensing(**kw):
    from sim.env import RescueEnv
    from sim.map_gen import generate_building
    from sim.reid_sensing import SensingModel
    env = RescueEnv(generate_building(seed=0, width=40, height=40, n_survivors=6), 2, 1,
                    comm_range=10, max_steps=50, seed=0)
    return env, SensingModel(env, seed=0, **kw)


def test_bearing_observation_is_unbiased_with_elongated_covariance():
    env, sm = _sensing(acoustic="bearing")
    rover = env.agents[-1]
    truth = [sv for sv in env.bmap.survivors if sv.sound and not sv.heat][0]   # audible, not visible
    pts, covs = [], []
    for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3), (4, 0), (-4, 0), (0, 4), (0, -4)):
        pos = (truth.pos[0] + dx, truth.pos[1] + dy)
        if not env.bmap.in_bounds(*pos):
            continue
        rover.pos = pos
        pts, covs = [], []
        for t in range(400):
            sm.bias[rover.id][:] = 0.0                  # isolate the sensor model from drift
            for o in sm.observe(rover, t):
                if o.gt == truth.id and o.emb is None and o.cov is not None:
                    pts.append(o.x)
                    covs.append(o.cov)
        if len(pts) > 30:
            break
    assert len(pts) > 30, "no clear listening position found"
    err = np.mean(pts, axis=0) - np.array(truth.pos, float)
    assert np.linalg.norm(err) < 0.8                    # roughly unbiased
    ev = np.linalg.eigvalsh(covs[0])
    assert ev[-1] > 1.3 * ev[0]                         # elongated, not isotropic
    assert np.allclose(covs[0], covs[0].T)


def test_two_bearing_views_triangulate_better_than_isotropic_range_blobs():
    """Two robots hear one source from perpendicular directions. Anisotropic splats cross at
    the source; isotropic blobs of equal total variance cannot do better than their own width."""
    truth = np.array([20.0, 20.0])

    def peak_error(aniso, seed):
        rng = np.random.default_rng(seed)
        reg = VictimRegistry(0)
        for k in range(30):
            for pos in (np.array([20.0, 8.0]), np.array([8.0, 20.0])):   # south and west of source
                d = truth - pos
                r = np.linalg.norm(d)
                u = d / r
                w = np.array([-u[1], u[0]])
                s_r, s_t = 0.25 * r, r * math.radians(8)
                if aniso:
                    cov = s_r ** 2 * np.outer(u, u) + s_t ** 2 * np.outer(w, w)
                    x = pos + (r + rng.normal(scale=s_r)) * u + rng.normal(scale=s_t) * w
                    reg.observe([Observation(k, 0, x, 0.5 * (s_r ** 2 + s_t ** 2), None, frozenset({"acoustic"}),
                                             cov=cov)], k)
                else:
                    sig = math.sqrt(0.5 * (s_r ** 2 + s_t ** 2))
                    x = truth + rng.normal(scale=sig, size=2)
                    reg.observe([Observation(k, 0, x, sig ** 2, None, frozenset({"acoustic"}))], k)
        src = reg.acoustic_sources()
        assert src, "no source found"
        return min(np.linalg.norm(a["x"] - truth) for a in src)

    # the isotropic variant is given the *unbiased* position (best case for it); bearing must still win or tie
    err_a = np.mean([peak_error(True, s) for s in range(15)])
    err_i = np.mean([peak_error(False, s) for s in range(15)])
    assert err_a < 1.0
    assert err_a < err_i + 0.35


def test_pose_alignment_pulls_robots_together_but_keeps_common_drift():
    env, sm = _sensing(align=True)
    sm.bias[0], sm.bias[1] = np.array([1.0, 0.5]), np.array([-0.6, 0.1])
    sm.pvar[0] = sm.pvar[1] = 0.3
    mean_before = 0.5 * (sm.bias[0] + sm.bias[1])
    sm.align(0, 1)
    assert np.linalg.norm(sm.bias[0] - sm.bias[1]) < 0.6          # was 1.7
    assert np.linalg.norm(0.5 * (sm.bias[0] + sm.bias[1]) - mean_before) < 0.3   # common drift kept
    assert sm.pvar[0] == sm.pvar[1] < 0.3


def test_alignment_disabled_changes_nothing():
    env, sm = _sensing(align=False)
    sm.bias[0], sm.bias[1] = np.array([1.0, 0.5]), np.array([-0.6, 0.1])
    sm.pvar[0] = sm.pvar[1] = 0.3
    sm.align(0, 1)
    assert np.allclose(sm.bias[0], [1.0, 0.5]) and sm.pvar[0] == 0.3


def test_bearing_and_alignment_help_in_swarm_episodes():
    from sim.run_reid import run
    base = better = 0
    for seed in range(20, 30):
        a = run(seed, "reid", 4, 2, 300, 40, acoustic="range", align=False)
        b = run(seed, "reid", 4, 2, 300, 40, acoustic="bearing", align=True)
        base += a["duplicates"] + abs(a["error"])
        better += b["duplicates"] + abs(b["error"])
        assert b["false_pos"] == 0
    assert better < base
