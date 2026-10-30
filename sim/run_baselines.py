"""Run baselines on N map seeds; print a metrics table and save plots.

Usage:
  python -m sim.run_baselines --seeds 0 1 2 --drones 2 --rovers 2
"""
from __future__ import annotations

import argparse
import os
import sys

from .map_gen import generate_building
from .env import RescueEnv
from .baselines import BASELINES
from .metrics import run_episode, print_summary_table, plot_results

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--drones", type=int, default=2)
    ap.add_argument("--rovers", type=int, default=2)
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--comm-range", type=float, default=10.0)
    ap.add_argument("--baselines", nargs="+", default=list(BASELINES.keys()))
    args = ap.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # build one map per seed (same map reused across baselines for fair comparison)
    results = {b: {} for b in args.baselines}
    traces = {b: {} for b in args.baselines}
    for seed in args.seeds:
        bmap = generate_building(seed=seed, width=args.size, height=args.size,
                                 n_survivors=6)
        print(f"--- seed {seed}: map {bmap.W}x{bmap.H}, "
              f"{len(bmap.survivors)} survivors, "
              f"traversable={bmap.traversable_cells().sum()} ---")
        for name in args.baselines:
            env = RescueEnv(bmap, n_drones=args.drones, n_rovers=args.rovers,
                            comm_range=args.comm_range, max_steps=args.max_steps, seed=seed)
            res = run_episode(env, BASELINES[name], seed)
            results[name][seed] = res["summary"]
            traces[name][seed] = res["trace"]
            print(f"  {name:<18} cov={res['summary']['coverage_final']*100:5.1f}%  "
                  f"survivors={res['summary']['survivors_found']}/{res['summary']['survivors_total']}  "
                  f"steps={res['summary']['steps']}")

    print_summary_table(results)

    plot_path = os.path.join(RESULTS_DIR, "p1_baselines.png")
    plot_results(results, traces, plot_path)
    print(f"\nPlots saved to {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
