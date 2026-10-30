"""Train the MARL+GNN policy (and a no-GNN ablation) and export it.

Usage:
  .venv/bin/python -m train.run_train --iters 300 --rollout 80 --envs 8 \
      --out results
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.map_gen import generate_building
from sim.env import RescueEnv
from train.features import build_obs, FA, FG, N_GOALS
from train.model import ActorCritic
from train.ppo import PPO, act, forward_one


def make_env(seed, size=24, n_survivors=5, max_steps=100):
    bmap = generate_building(seed=seed, width=size, height=size, n_survivors=n_survivors)
    return RescueEnv(bmap, n_drones=2, n_rovers=2, comm_range=10.0,
                     max_steps=max_steps, seed=seed)


def collect(envs, model, device, T):
    B = len(envs)
    nA = len(envs[0].agents)
    cur_obs = [build_obs(e) for e in envs]
    obs_list, actions, logprobs, values, rewards, dones = [], [], [], [], [], []
    for t in range(T):
        acts = np.zeros((B, nA), dtype=np.int64)
        lps = np.zeros((B, nA), dtype=np.float32)
        vals = np.zeros(B, dtype=np.float32)
        rws = np.zeros(B, dtype=np.float32)
        dns = np.zeros(B, dtype=bool)
        for b in range(B):
            action, lp, value = act(model, cur_obs[b], device, deterministic=False)
            acts[b] = action.cpu().numpy()
            lps[b] = lp.cpu().numpy()
            vals[b] = value.item()
        obs_list.append(list(cur_obs))
        actions.append(acts)
        logprobs.append(lps)
        values.append(vals)

        for b in range(B):
            gc = cur_obs[b]["goal_cells"][np.arange(nA), acts[b]]
            obs, r, done, info = envs[b].step([tuple(int(v) for v in g) for g in gc])
            rws[b] = r
            dns[b] = done
            cur_obs[b] = build_obs(envs[b])
            if done:
                envs[b].reset()
                cur_obs[b] = build_obs(envs[b])
        rewards.append(rws)
        dones.append(dns)
    return obs_list, actions, logprobs, values, rewards, dones


@torch.no_grad()
def evaluate(model, device, seeds, size=24, n_survivors=5, max_steps=100, n_envs=8):
    """Return mean metrics over rollouts (deterministic policy)."""
    envs = [make_env(s, size, n_survivors, max_steps) for s in seeds]
    envs = envs[:n_envs]
    totals = {"coverage": 0.0, "survivors_found": 0.0, "survivors_confirmed": 0.0,
              "collisions": 0.0, "steps": 0.0}
    n = 0
    for env in envs:
        cur = build_obs(env)
        done = False
        while not done:
            nA = len(env.agents)
            action, _, _ = act(model, cur, device, deterministic=True)
            gc = cur["goal_cells"][np.arange(nA), action.cpu().numpy()]
            _, _, done, info = env.step([tuple(int(v) for v in g) for g in gc])
            cur = build_obs(env)
        totals["coverage"] += info["coverage"]
        totals["survivors_found"] += info["survivors_found"]
        totals["survivors_confirmed"] += info["survivors_confirmed"]
        totals["collisions"] += info["collisions"]
        totals["steps"] += env.step_count
        n += 1
    return {k: v / n for k, v in totals.items()}


def export_actor(model, path):
    """Export the actor (policy) to TorchScript via tracing."""
    model.eval()
    device = next(model.parameters()).device
    nA = 4
    example = (
        torch.randn(nA, FA, device=device),
        torch.randn(nA, N_GOALS, FG, device=device),
        torch.ones(nA, N_GOALS, dtype=torch.bool, device=device),
        torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device),
        torch.ones(nA, nA, device=device),
        torch.tensor([0.5], device=device),
    )
    traced = torch.jit.trace(model, example, check_trace=False)
    traced.save(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--rollout", type=int, default=80)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--n-survivors", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    print(f"device={device}, FA={FA}, FG={FG}, N_GOALS={N_GOALS}")

    results = {}
    for tag, use_gnn in (("gnn", True), ("no_gnn", False)):
        print(f"\n===== training {tag} (use_gnn={use_gnn}) =====")
        model = ActorCritic(FA, FG, n_goals=N_GOALS, hidden=args.hidden, use_gnn=use_gnn).to(device)
        ppo = PPO(model, lr=args.lr, device=device, epochs=args.epochs)
        envs = [make_env(1000 + i, args.size, args.n_survivors, args.max_steps)
                for i in range(args.envs)]
        curve = []
        t0 = time.time()
        for it in range(args.iters):
            obs_list, actions, logprobs, values, rewards, dones = collect(
                envs, model, device, args.rollout)
            mean_r = float(np.mean(rewards))
            ppo.update(obs_list, actions, logprobs, values, rewards, dones)
            curve.append(mean_r)
            if it % 20 == 0 or it == args.iters - 1:
                ev = evaluate(model, device, [2000 + i for i in range(args.envs)],
                              args.size, args.n_survivors, args.max_steps, args.envs)
                print(f"  [{tag}] it {it:4d} mean_r={mean_r:7.2f} "
                      f"cov={ev['coverage']*100:5.1f}% surv={ev['survivors_found']:.1f} "
                      f"conf={ev['survivors_confirmed']:.1f} coll={ev['collisions']:.1f} "
                      f"({time.time()-t0:.0f}s)")
        results[tag] = curve
        # save checkpoint + export
        ckpt = os.path.join(args.out, "checkpoints", f"{tag}.pt")
        torch.save(model.state_dict(), ckpt)
        export_actor(model, os.path.join(args.out, "checkpoints", f"{tag}_actor.pt"))
        print(f"  saved {ckpt}")

    # learning curves
    plt.figure(figsize=(8, 5))
    for tag, curve in results.items():
        plt.plot(range(len(curve)), curve, label=tag)
    plt.xlabel("iteration"); plt.ylabel("mean team reward"); plt.title("Learning curves")
    plt.legend(); plt.grid(True, alpha=0.3)
    curve_path = os.path.join(args.out, "p2_learning_curves.png")
    plt.savefig(curve_path, dpi=110); plt.close()
    print(f"\nLearning curves -> {curve_path}")

    with open(os.path.join(args.out, "p2_summary.json"), "w") as f:
        json.dump({k: v[-10:] for k, v in results.items()}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
