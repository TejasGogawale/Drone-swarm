# PROGRESS.md — running log

## Phase 0 — Recon (in progress)

### Environment detected
- OS: Ubuntu 24.04.5 LTS (noble)
- ROS 2: Jazzy (`/opt/ros/jazzy`), `ros2` CLI works
- Gazebo: Gazebo Sim (gz sim) **8.15.0** (Harmonic) — no classic `gazebo`
- Python: 3.12.3 (system), pip 24.0, `venv` available; **torch NOT installed** (ML venv to create)
- CPU: 12 cores (`nproc` = 12)
- RAM: **7.5 GiB total**, ~1.9 GiB available at detect time, 4 GiB swap (LOW — this is the main constraint)
- GPU: NVIDIA GeForce RTX 2050, 4 GiB VRAM, driver 580.178.04, CUDA 13.0
- Disk: 27 GiB free on `/`
- Ollama: NOT installed
- Display: `DISPLAY=:0` (GUI possible; prefer headless to save RAM)

### ROS packages already present (system)
`ros_gz_bridge`, `ros_gz_image`, `ros_gz_interfaces`, `ros_gz_sim`, `ros_gz_sim_demos`
**Missing (install later, P5):** slam_toolbox, nav2, rtabmap_ros, explore_lite, multirobot_map_merge

### Existing user workspace (DO NOT TOUCH)
`/home/tushar/ros2_ws` has packages: `ur_description`, `eyantra_kepler_colony`, `ebot_description`, `algorithms`.
Our project lives in `/home/tushar/Desktop/swarm` (separate). We will not modify the user's ros2_ws.

### Decisions so far
- Project root = `/home/tushar/Desktop/swarm` (cwd already named "swarm"). Git initialized.
- Robot counts for live Gazebo (RAM-limited): **1 drone + 1 rover**, headless. Scalability (10+) in lightweight sim.
- LLM commander: default **rule-based fallback allocator** (Ollama not installed, RAM too tight for 7B).
  Optional cloud/local via env var.

### Acceptance test P0 — PASSED
- [x] Gazebo Harmonic world runs headless (`gz sim -r -s` on a camera world).
- [x] Camera topic `/camera` (`sensor_msgs/msg/Image`) visible in ROS2 via `ros_gz_bridge`, ~30 Hz.
- World kept at `ros_ws/src/worlds/p0_camera_test.sdf`.

### References cloned (~/refs, shallow) + licenses
VMAS (GPLv3), PettingZoo (MIT), hemac (Thales Non-Production), IR2 (Apache-2.0),
m-explore-ros2 (BSD). Recorded in THIRD_PARTY.md. We build our OWN simulator (avoid VMAS GPL).

### Final stack decision
- ROS 2 Jazzy + Gazebo Harmonic (existing, untouched).
- Lightweight sim: **custom pure-Python** (numpy; torch for GNN in P2), NOT VMAS/hemac (faster + no license risk).
- MARL: custom PPO (torch) + PyTorch Geometric GNN (small models; GPU 4GB or CPU).
- LLM commander: **Groq cloud API** — `qwen/qwen3.8-27b`, key in `.env` (verified working). Fallback = rule-based.
- Robot counts (live Gazebo): **2 drones + 2 rovers, headless** (user's choice). Lightweight sim: 3/6/10+.
- Deadline: 48 h (assumed).

### Next
P1 lightweight simulator (no ROS).

## Phase 1 — Lightweight simulator (DONE, acceptance PASSED)

### Built
- `sim/map_gen.py` — procedural collapsed-building generator: rooms + corridors + narrow gaps
  (rover-only) + rubble (rover-only, slow) + survivors with explicit sensor profiles
  (open/quiet = thermal; buried-in-rubble = acoustic-only). Every survivor detectable by >=1 modality.
- `sim/env.py` — heterogeneous grid env: drones (fast, wide thermal, short battery) vs rovers
  (slow, narrow thermal + acoustic, longer battery); passive sensing; peer-to-peer comms sharing
  within range; BFS goal-following; collision handling; battery drain.
- `sim/sensors.py` — thermal (FOV + LOS + distance + noise; coarse for drones) and acoustic
  (distance + per-wall attenuation + noise) models.
- `sim/baselines.py` — random, nearest-frontier, explore-lite frontier policies.
- `sim/metrics.py` + `sim/run_baselines.py` — metrics (coverage %, survivors found, first/all-found
  step, collisions), table + plot output to `results/p1_baselines.png`.

### Acceptance test P1 — PASSED
Command: `python3 -m sim.run_baselines --seeds 0 1 2 --drones 2 --rovers 2 --max-steps 300`
Ran on 3 map seeds, metrics printed and plotted. Runtime ~2.4 s total.

Results (2 drones + 2 rovers, 40x40, 6 survivors):
| baseline | avg cov% | avg surv% | avg first@ | avg all@ | avg collisions |
|---|---|---|---|---|---|
| random | 78.9 | 100 | 5 | 104 | 4.3 |
| nearest_frontier | 75.3 | 100 | 4 | 56 | 2.7 |
| explore_lite | 81.8 | 100 | 4 | 76 | 2.0 |

### Known issues / notes
- Maps are "easy" (all baselines find all survivors). For P2/P3 I'll make it harder
  (larger maps, more/buried survivors, shorter comms range) so MARL has room to win on
  time-to-all, coverage efficiency, and collisions.
- Movement is grid-step based; drones 2 cells/step, rovers 1 (0.5 through rubble).

### Next
P2: MARL + GNN training (custom PPO + PyTorch Geometric; separate venv with torch).
