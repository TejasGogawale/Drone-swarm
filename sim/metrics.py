"""Metrics collection and plotting for baseline runs."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def run_episode(env: RescueEnv, policy_cls, seed: int) -> dict:
    """Run one episode with a fresh policy instance and return a metrics trace."""
    env.reset()
    policy = {i: policy_cls() for i in range(len(env.agents))}
    trace = {
        "coverage": [],
        "survivors_found": [],
        "collisions": [],
    }
    done = False
    while not done:
        goals = []
        for i, ag in enumerate(env.agents):
            if ag.alive:
                goals.append(policy[i].choose(env, i))
            else:
                goals.append(None)
        _, _, done, info = env.step(goals)
        trace["coverage"].append(info["coverage"])
        trace["survivors_found"].append(info["survivors_found"])
        trace["collisions"].append(info["collisions"])
    # final summary
    summary = {
        "coverage_final": info["coverage"],
        "survivors_found": info["survivors_found"],
        "survivors_total": info["survivors_total"],
        "first_found_step": info["first_found_step"],
        "all_found_step": info["all_found_step"],
        "collisions_total": info["collisions"],
        "steps": env.step_count,
    }
    return {"summary": summary, "trace": trace}


def print_summary_table(results):
    """results: dict[name][seed] = summary dict."""
    print("\n" + "=" * 100)
    hdr = f"{'baseline':<18}{'seed':<6}{'cov%':<8}{'surv':<8}{'1st@':<8}{'all@':<8}{'steps':<8}{'coll':<6}"
    print(hdr)
    print("-" * 100)
    for name in results:
        for seed in results[name]:
            s = results[name][seed]
            first = "-" if s["first_found_step"] is None else s["first_found_step"]
            allf = "-" if s["all_found_step"] is None else s["all_found_step"]
            print(f"{name:<18}{seed:<6}{s['coverage_final']*100:>6.1f}%{s['survivors_found']:>4}/{s['survivors_total']:<3}"
                  f"{str(first):>8}{str(allf):>8}{s['steps']:>8}{s['collisions_total']:>6}")
    print("=" * 100)

    # averages per baseline
    print("\nAverages across seeds:")
    print(f"{'baseline':<18}{'cov%':<8}{'surv%':<8}{'1st@':<8}{'all@':<8}{'coll':<6}")
    print("-" * 60)
    for name in results:
        seeds = results[name]
        cov = np.mean([s["coverage_final"] for s in seeds.values()])
        surv = np.mean([s["survivors_found"] / max(1, s["survivors_total"]) for s in seeds.values()])
        firsts = [s["first_found_step"] for s in seeds.values() if s["first_found_step"] is not None]
        alls = [s["all_found_step"] for s in seeds.values() if s["all_found_step"] is not None]
        coll = np.mean([s["collisions_total"] for s in seeds.values()])
        f = f"{np.mean(firsts):.0f}" if firsts else "-"
        a = f"{np.mean(alls):.0f}" if alls else "-"
        print(f"{name:<18}{cov*100:>6.1f}%{surv*100:>6.0f}%{f:>8}{a:>8}{coll:>6.1f}")
    print("-" * 60)


def plot_results(results, traces, out_path):
    """Plot coverage curves per baseline/seed."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1, ax2 = axes
    for name in results:
        for seed in results[name]:
            cov = traces[name][seed]["coverage"]
            ax1.plot(range(len(cov)), [c * 100 for c in cov], alpha=0.6, label=f"{name} s{seed}")
            surv = traces[name][seed]["survivors_found"]
            ax2.plot(range(len(surv)), surv, alpha=0.6, label=f"{name} s{seed}")
    ax1.set_xlabel("step"); ax1.set_ylabel("coverage %"); ax1.set_title("Coverage over time")
    ax2.set_xlabel("step"); ax2.set_ylabel("survivors found"); ax2.set_title("Survivors found over time")
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
