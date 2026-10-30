"""Visualize the lightweight swarm simulator (map, robots, survivors, comms).

Render the current state to a matplotlib figure, and/or produce an animated GIF/MP4
of an episode so the swarm's exploration can be watched.

Usage:
  python3 -m sim.visualize --seed 0 --drones 2 --rovers 2 --policy nearest_frontier \
      --out results/p1_episode.gif
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

from .map_gen import generate_building, FREE, WALL, NARROW, RUBBLE
from .env import RescueEnv
from .baselines import BASELINES

# cell colors (rgb 0-1) and text
CELL_COLORS = {
    WALL: (0.18, 0.18, 0.18),
    FREE: (0.96, 0.96, 0.96),
    NARROW: (0.45, 0.65, 0.95),
    RUBBLE: (0.70, 0.55, 0.38),
}
UNKNOWN = (0.85, 0.85, 0.85)
AGENT_COLOR = {"drone": "#1f77b4", "rover": "#2ca02c"}


def _cell_color(env, x, y):
    if env.bmap.grid[y, x] in CELL_COLORS:
        return CELL_COLORS[env.bmap.grid[y, x]]
    return UNKNOWN


def render(env, ax, show_unknown=True):
    """Draw the current env state onto axes `ax`."""
    H, W = env.bmap.H, env.bmap.W
    union = env.union_explored()
    # background: draw wall cells always; free cells colored by explored state
    img = np.ones((H, W, 3))
    for y in range(H):
        for x in range(W):
            cell = env.bmap.grid[y, x]
            if cell == WALL:
                img[y, x] = CELL_COLORS[WALL]
            elif not union[y, x] and show_unknown:
                img[y, x] = UNKNOWN
            else:
                img[y, x] = CELL_COLORS.get(cell, UNKNOWN)
    ax.imshow(img, origin="upper", interpolation="nearest")

    # survivors
    for s in env.bmap.survivors:
        x, y = s.pos
        if s.found:
            color, ms = "#d62728", 11
        elif not s.heat:  # buried -> acoustic only
            color, ms = "#9467bd", 9
        elif not s.sound:  # quiet -> thermal only
            color, ms = "#ff7f0e", 9
        else:
            color, ms = "#d62728", 8
        marker = "*" if (not s.heat) else ("D" if (not s.sound) else "o")
        ax.plot(x, y, marker=marker, color=color, markersize=ms,
                markeredgecolor="k", markeredgewidth=0.5, zorder=4)

    # comms links
    for i in range(len(env.agents)):
        for j in range(i + 1, len(env.agents)):
            ai, aj = env.agents[i], env.agents[j]
            if ai.alive and aj.alive:
                d = np.hypot(ai.pos[0] - aj.pos[0], ai.pos[1] - aj.pos[1])
                if d <= env.comm_range:
                    ax.plot([ai.pos[0], aj.pos[0]], [ai.pos[1], aj.pos[1]],
                            ls="--", color="gray", lw=0.8, alpha=0.6, zorder=3)

    # agents
    for ag in env.agents:
        if not ag.alive:
            continue
        x, y = ag.pos
        color = AGENT_COLOR[ag.type]
        marker = "^" if ag.type == "drone" else "s"
        ax.plot(x, y, marker=marker, color=color, markersize=12,
                markeredgecolor="k", markeredgewidth=0.7, zorder=5)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def animate(env, policy_cls, out_path, max_frames=200, fps=8):
    """Run an episode and save an animated GIF."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))
    policy = {i: policy_cls() for i in range(len(env.agents))}
    state = {"done": False}

    def update(frame):
        if not state["done"]:
            goals = [policy[i].choose(env, i) if ag.alive else None
                     for i, ag in enumerate(env.agents)]
            _, _, done, info = env.step(goals)
            state["done"] = done
        else:
            info = env._info()
        ax.clear()
        render(env, ax)
        ax.set_title(f"step {env.step_count} | survivors {info['survivors_found']}/"
                     f"{info['survivors_total']} | cov {info['coverage']*100:.0f}%")
        return []

    anim = FuncAnimation(fig, update, frames=max_frames, interval=1000 // fps,
                         repeat=False, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drones", type=int, default=2)
    ap.add_argument("--rovers", type=int, default=2)
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--n-survivors", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--policy", default="nearest_frontier", choices=list(BASELINES.keys()))
    ap.add_argument("--out", default="results/p1_episode.gif")
    ap.add_argument("--snapshot", action="store_true", help="also save a static PNG")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bmap = generate_building(seed=args.seed, width=args.size, height=args.size,
                             n_survivors=args.n_survivors)
    env = RescueEnv(bmap, n_drones=args.drones, n_rovers=args.rovers,
                    comm_range=7.0, max_steps=args.max_steps, seed=args.seed)
    out = animate(env, BASELINES[args.policy], args.out, max_frames=args.max_steps)
    print(f"saved animation -> {out}")

    if args.snapshot:
        snap = args.out.rsplit(".", 1)[0] + ".png"
        env.reset()
        fig, ax = plt.subplots(figsize=(6, 6))
        # run a few steps then snapshot
        pol = {i: BASELINES[args.policy]() for i in range(len(env.agents))}
        for _ in range(20):
            goals = [pol[i].choose(env, i) if a.alive else None
                     for i, a in enumerate(env.agents)]
            _, _, d, _ = env.step(goals)
            if d:
                break
        render(env, ax)
        ax.set_title("swarm snapshot (step 20)")
        fig.savefig(snap, dpi=110); plt.close(fig)
        print(f"saved snapshot -> {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
