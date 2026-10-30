"""Compare trained policies (GNN vs no-GNN) against baselines on held-out seeds."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.map_gen import generate_building
from sim.env import RescueEnv
from sim.baselines import BASELINES
from sim.metrics import run_episode
from train.features import build_obs, FA, FG, N_GOALS
from train.model import ActorCritic
from train.ppo import act


def load_model(path, use_gnn, device):
    model = ActorCritic(FA, FG, n_goals=N_GOALS, hidden=64, use_gnn=use_gnn).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def eval_model(model, device, seed, size, n_survivors, max_steps):
    bmap = generate_building(seed=seed, width=size, height=size, n_survivors=n_survivors)
    env = RescueEnv(bmap, n_drones=2, n_rovers=2, comm_range=10.0, max_steps=max_steps, seed=seed)
    cur = build_obs(env)
    done = False
    while not done:
        nA = len(env.agents)
        action, _, _ = act(model, cur, device, deterministic=True)
        gc = cur["goal_cells"][np.arange(nA), action.cpu().numpy()]
        _, _, done, info = env.step([tuple(int(v) for v in g) for g in gc])
        cur = build_obs(env)
    return {
        "coverage_final": info["coverage"],
        "survivors_found": info["survivors_found"],
        "survivors_confirmed": info["survivors_confirmed"],
        "survivors_total": info["survivors_total"],
        "first_found_step": info["first_found_step"],
        "all_found_step": info["all_found_step"],
        "collisions_total": info["collisions"],
        "steps": env.step_count,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default="results/checkpoints/gnn.pt")
    ap.add_argument("--no-gnn", default="results/checkpoints/no_gnn.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3000, 3001, 3002])
    ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--n-survivors", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    gnn = load_model(args.gnn, True, device) if os.path.exists(args.gnn) else None
    no_gnn = load_model(args.no_gnn, False, device) if os.path.exists(args.no_gnn) else None

    names = ["random", "nearest_frontier", "explore_lite"]
    if gnn is not None:
        names.append("gnn")
    if no_gnn is not None:
        names.append("no_gnn")

    summary = {n: {} for n in names}
    for seed in args.seeds:
        bmap = generate_building(seed=seed, width=args.size, height=args.size,
                                 n_survivors=args.n_survivors)
        for n in ["random", "nearest_frontier", "explore_lite"]:
            env = RescueEnv(bmap, n_drones=2, n_rovers=2, comm_range=10.0,
                            max_steps=args.max_steps, seed=seed)
            summary[n][seed] = run_episode(env, BASELINES[n], seed)["summary"]
        if gnn is not None:
            summary["gnn"][seed] = eval_model(gnn, device, seed, args.size, args.n_survivors, args.max_steps)
        if no_gnn is not None:
            summary["no_gnn"][seed] = eval_model(no_gnn, device, seed, args.size, args.n_survivors, args.max_steps)

    # print table
    print("\n" + "=" * 96)
    print(f"{'method':<18}{'cov%':<8}{'surv':<8}{'conf':<8}{'1st@':<8}{'all@':<8}{'coll':<6}")
    print("-" * 96)
    for n in names:
        for seed in args.seeds:
            s = summary[n][seed]
            first = "-" if s["first_found_step"] is None else s["first_found_step"]
            allf = "-" if s["all_found_step"] is None else s["all_found_step"]
            conf = s.get("survivors_confirmed", "-")
            print(f"{n:<18}{s['coverage_final']*100:>6.1f}%{s['survivors_found']:>4}/{s['survivors_total']:<3}"
                  f"{str(conf):>8}{str(first):>8}{str(allf):>8}{s['collisions_total']:>6}")

    print("\nAverages:")
    print(f"{'method':<18}{'cov%':<8}{'surv%':<8}{'conf%':<8}{'1st@':<8}{'all@':<8}{'coll':<6}")
    print("-" * 72)
    for n in names:
        ss = summary[n]
        cov = np.mean([s["coverage_final"] for s in ss.values()])
        surv = np.mean([s["survivors_found"] / max(1, s["survivors_total"]) for s in ss.values()])
        confs = [s.get("survivors_confirmed", 0) / max(1, s["survivors_total"]) for s in ss.values()]
        conf = np.mean(confs)
        firsts = [s["first_found_step"] for s in ss.values() if s["first_found_step"] is not None]
        alls = [s["all_found_step"] for s in ss.values() if s["all_found_step"] is not None]
        coll = np.mean([s["collisions_total"] for s in ss.values()])
        f = f"{np.mean(firsts):.0f}" if firsts else "-"
        a = f"{np.mean(alls):.0f}" if alls else "-"
        print(f"{n:<18}{cov*100:>6.1f}%{surv*100:>6.0f}%{conf*100:>6.0f}%{f:>8}{a:>8}{coll:>6.1f}")
    print("-" * 72)

    # bar plot: coverage and survivors found per method
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = names
    covs = [np.mean([summary[n][s]["coverage_final"] for s in args.seeds]) * 100 for n in names]
    survs = [np.mean([summary[n][s]["survivors_found"] / max(1, summary[n][s]["survivors_total"])
                      for s in args.seeds]) * 100 for n in names]
    axes[0].bar(labels, covs, color="steelblue"); axes[0].set_title("Coverage %"); axes[0].set_ylim(0, 105)
    axes[1].bar(labels, survs, color="darkorange"); axes[1].set_title("Survivors found %"); axes[1].set_ylim(0, 105)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    cmp_path = os.path.join(args.out, "p2_comparison.png")
    fig.savefig(cmp_path, dpi=110); plt.close()
    print(f"\nComparison plot -> {cmp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
