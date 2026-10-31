"""Evaluate victim Re-ID / de-duplication across a swarm episode.

Variants (same maps, same sensing noise per seed):
  reid        full system: gossip + Hungarian association + appearance + consolidation
  pos_only    same, appearance disabled (position gate only)
  local_only  no gossip: each robot counts its own confirmed tracks; fleet count = SUM
              (what naive per-robot counting gives: the double-count problem)

Usage:  python -m sim.run_reid --seeds 0 1 2 3 4 --drones 4 --rovers 2
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from .map_gen import generate_building
from .env import RescueEnv
from .baselines import BASELINES
from .reid import VictimRegistry, ReIDConfig
from .reid_sensing import SensingModel


def _label(v, survivors=()):
    """Majority ground-truth label of a component (evaluation only). Acoustic sources carry
    no per-observation labels, so they are matched to the nearest true survivor (<= 4 cells)."""
    if v.get("kind") == "acoustic":
        best = min(survivors, key=lambda s: math.hypot(s.pos[0] - v["x"][0], s.pos[1] - v["x"][1]))
        near = math.hypot(best.pos[0] - v["x"][0], best.pos[1] - v["x"][1]) <= 4.0
        return (best.id if near else -1), 0.0
    if not v["gt_votes"]:
        return None, 0.0
    lab = max(v["gt_votes"], key=v["gt_votes"].get)
    tot = sum(v["gt_votes"].values())
    return lab, 1.0 - v["gt_votes"][lab] / tot


def run(seed: int, variant: str, drones: int, rovers: int, steps: int, size: int,
        policy: str = "explore_lite", comm_range: float = 10.0, drift: float = 0.03,
        similar_frac: float = 0.34, consolidate_every: int = 5, cfg_kw: dict | None = None,
        acoustic: str = "bearing", align: bool = True):
    bmap = generate_building(seed=seed, width=size, height=size, n_survivors=6)
    env = RescueEnv(bmap, drones, rovers, comm_range=comm_range, max_steps=steps, seed=seed)
    sens = SensingModel(env, seed=seed, drift=drift, similar_frac=similar_frac,
                        acoustic=acoustic, align=align)
    cfg = ReIDConfig(use_appearance=(variant != "pos_only"), **(cfg_kw or {}))
    regs = {ag.id: VictimRegistry(ag.id, cfg) for ag in env.agents}
    pol = BASELINES[policy]()
    pols = [BASELINES[policy]() for _ in env.agents]
    gossip = variant != "local_only"
    bytes_sent = 0
    for t in range(steps):
        goals = [p.choose(env, i) if env.agents[i].alive else None for i, p in enumerate(pols)]
        _, _, done, _ = env.step(goals)
        for ag in env.agents:
            if ag.alive:
                regs[ag.id].observe(sens.observe(ag, t), t)
        if gossip:
            for a in env.agents:
                for b in env.agents:
                    if a.id < b.id and a.alive and b.alive and \
                            math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]) <= comm_range:
                        regs[a.id].merge(regs[b.id])
                        regs[b.id].merge(regs[a.id])
                        sens.align(a.id, b.id)
                        bytes_sent += 2 * (len(regs[a.id].tracks) * (4 * 8 + 32 * 2 + 16)
                                           + sum(len(g[1]) for g in regs[a.id].ac.values()) * 6)
            if t % consolidate_every == 0:
                for r in regs.values():
                    r.consolidate(t)
        if done:
            break
    if gossip:                        # final convergence (assume a base reconciles at the end)
        for k in range(4):               # >= link_persist rounds so hysteresis can clear
            for a in regs.values():
                for b in regs.values():
                    a.merge(b)
            for r in regs.values():
                r.consolidate(steps)
        for a in regs.values():
            for b in regs.values():
                a.merge(b)

    ideal = sens.ideal_count()
    if gossip:
        per_agent = {i: r.count() for i, r in regs.items()}
        fleet = per_agent[min(per_agent)]
        views = [r.victims() for r in regs.values()][:1]
        victims = [v for v in views[0] if v["confirmed"]]
        consistent = len(set(per_agent.values())) == 1
    else:
        per_agent = {i: r.count() for i, r in regs.items()}
        fleet = sum(per_agent.values())
        victims = [v for r in regs.values() for v in r.victims() if v["confirmed"]]
        consistent = True

    frag = {}
    fp = merged = 0
    for v in victims:
        lab, impurity = _label(v, bmap.survivors)
        if lab == -1:
            fp += 1
            continue
        if lab is not None:
            frag[lab] = frag.get(lab, 0) + 1
        if impurity > 0.25:
            merged += 1
    duplicates = sum(max(0, n - 1) for n in frag.values())
    missed = sum(1 for sid in sens.gt_obs_count if sid not in frag and
                 sens.ideal_count_one(sid))
    return {"seed": seed, "variant": variant, "ideal": ideal, "count": fleet,
            "error": fleet - ideal, "duplicates": duplicates, "false_pos": fp,
            "merged_distinct": merged, "missed_observable": missed,
            "consistent": consistent, "kB": bytes_sent / 1024,
            "min_sep": sens.min_separation(),
            "truth_total": len(bmap.survivors)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--drones", type=int, default=4)
    ap.add_argument("--rovers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--drift", type=float, default=0.03)
    ap.add_argument("--similar-frac", type=float, default=0.34)
    ap.add_argument("--acoustic", choices=["bearing", "range"], default="bearing")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--variants", nargs="+", default=["reid", "pos_only", "local_only"])
    a = ap.parse_args(argv)

    rows = []
    for v in a.variants:
        for s in a.seeds:
            rows.append(run(s, v, a.drones, a.rovers, a.steps, a.size,
                            drift=a.drift, similar_frac=a.similar_frac,
                            acoustic=a.acoustic, align=not a.no_align))
    print(f"{'variant':<11}{'seed':>5}{'ideal':>7}{'count':>7}{'err':>5}{'dup':>5}"
          f"{'FP':>4}{'merge':>6}{'miss':>5}{'agree':>6}{'kB':>7}")
    for r in rows:
        print(f"{r['variant']:<11}{r['seed']:>5}{r['ideal']:>7}{r['count']:>7}{r['error']:>5}"
              f"{r['duplicates']:>5}{r['false_pos']:>4}{r['merged_distinct']:>6}"
              f"{r['missed_observable']:>5}{str(r['consistent']):>6}{r['kB']:>7.0f}")
    print("\nsummary (ideal = victims the fleet observed enough to confirm; truth_total = 6).")
    print("'resolvable' = every pair of confirmable victims >= 4 cells apart (people closer than that")
    print("cannot be separated by these sensors); 'crowded' = at least one pair closer.")
    for v in a.variants:
        rs = [r for r in rows if r["variant"] == v]
        for name, sel in (("all", rs), ("resolvable", [r for r in rs if r["min_sep"] >= 4.0]),
                          ("crowded", [r for r in rs if r["min_sep"] < 4.0])):
            if not sel:
                continue
            exact = sum(r["error"] == 0 and r["duplicates"] == 0 and r["false_pos"] == 0 for r in sel)
            print(f"  {v:<11}{name:<11} exact {exact}/{len(sel)}  mean|err| {np.mean([abs(r['error']) for r in sel]):.2f}"
                  f"  over {sum(max(r['error'], 0) for r in sel)} under {sum(max(-r['error'], 0) for r in sel)}"
                  f"  dup {sum(r['duplicates'] for r in sel)}  FP {sum(r['false_pos'] for r in sel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
