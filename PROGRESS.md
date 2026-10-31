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

## Phase 2 — MARL + GNN training (in progress)

### Environment setup
- `python3.12-venv` missing + no sudo + PyPI throttled large files. Resolved with:
  - venv created via `python3 -m venv --without-pip`; torch installed from a local wheel
    (`torch-2.15.0+cpu`, ~200 MB downloaded earlier) with `pip --target --no-deps`.
  - `system_site.pth` adds `/usr/lib/python3/dist-packages` for numpy 1.26.4 + matplotlib.
  - `typing_extensions` 4.16.0 (system had 4.10, too old for torch 2.15). **CPU torch** (GPU
    CUDA wheel is 539 MB @ ~13 KB/s = infeasible; tiny models train fine on 12 CPU cores).
  - torch_geometric NOT installed — GNN (GraphSAGE) implemented in pure PyTorch for reliable
    TorchScript export + one light dependency. Honest note: brief asked for PyTorch Geometric;
    we use GraphSAGE-style message passing implemented in PyTorch (same algorithm, no dep).

### Built
- `train/features.py` — observation builder (FA=109, FG=5, N_GOALS=20): type one-hot, battery,
  pos, heading, known survivors, thermal/acoustic hints, nearest-frontier dist, 2-channel 7x7
  occupancy patch (occupancy + known-survivor); goal (frontier) features; comm adjacency.
- `train/model.py` — ActorCritic with comm-weighted GraphSAGE GNN (per-type encoders, shared
  GNN, per-type scorer heads, centralized critic). Constant graph topology + dynamic comm weights
  => clean TorchScript export.
- `train/ppo.py` — MAPPO-style PPO (centralized critic, decentralized actors), GAE, clip.
- `train/run_train.py` — trains GNN + no-GNN ablation, saves learning curves + checkpoints +
  TorchScript actor.
- `train/evaluate.py` — compares gnn / no-gnn / baselines on held-out seeds.

### Verification (smoke)
- collect+PPO update run without error; TorchScript export loads and matches eager model
  (comm on/off changes logits as expected). All smoke tests PASS.

### Acceptance test P2 (pending full run)
- Training run in progress (40-iter validation first).

## Phase 3 — 4-Drone Decentralized Swarm (GNN + Heuristic) & Multi-Camera Vision (COMPLETE)

### Built & Verified
1. **Heterogeneous 4-Drone Swarm in Gazebo Harmonic**:
   - `start_all.sh` spawns 4 PX4 SITL instances (`-i 0..3`) with models `gz_x500_mono_cam` at staggered NED origins `(0,3,0)`, `(3,0,0)`, `(0,-3,0)`, `(-3,0,0)`.
   - `drone_agent.launch.py` bridges all 4 cameras simultaneously via `ros_gz_bridge`:
     - UAV-1 (`x500_mono_cam_0`) -> `/camera`
     - UAV-2 (`x500_mono_cam_1`) -> `/uav_1/camera`
     - UAV-3 (`x500_mono_cam_2`) -> `/uav_2/camera`
     - UAV-4 (`x500_mono_cam_3`) -> `/uav_3/camera`
2. **Decentralized Swarm Task Allocation (GNN + Heuristic Fallback)**:
   - `swarm_gnn_allocator.py`: GraphSAGE GNN model loaded from `results/checkpoints/gnn.pt` (`FA=109`, `FG=5`, `N_GOALS=20`).
   - Dynamically predicts goal affinity scores over mutual peer communication graph.
   - Robust Heuristic Frontier & Sector Fallback when frontiers require geometric partitioning or network partitions occur.
3. **Multi-Camera Visual GUI (`map_gui_node.py`)**:
   - 2x2 Quad-Cam Grid displaying real-time feeds from all 4 drones with HUD telemetry (altitude, heading, speed, GNN/Heuristic task).
   - Instant single-drone camera enlargement via top-bar buttons (`[4-CAM GRID]`, `[UAV-1 CAM]`, `[UAV-2 CAM]`, `[UAV-3 CAM]`, `[UAV-4 CAM]`) or direct clicking on camera feeds.
   - Page 2 3D Collaborative OctoMap with 4-Cam PiP.
4. **Collision Avoidance & Universal Offboard Control**:
   - 6.5m inter-drone separation buffer and 3D obstacle avoidance around collapsed buildings in `command_translator.py`.
   - Universal target system ID 0 for offboard trajectory setpoints.

## Phase R — Victim Re-ID / de-duplication in the lightweight sim (spec §4)
- Added `sim/reid.py` (CRDT victim registry, Hungarian association, appearance + pose-aware gating,
  cannot-link, hysteresis linking), `sim/reid_sensing.py` (noisy multi-view obs, drifting "simulated VIO",
  hot rocks, transient false positives), `sim/run_reid.py` (eval), `tests/test_reid.py` (17 tests, pass).
- Run: `python -m sim.run_reid --seeds 0 1 2 3 4 5 6 7 8 9 --drones 4 --rovers 2`
- 10 seeds, 4 UAV + 2 UGV, vs "ideal" (victims the fleet observed enough to confirm), drift 0.03/step:
  exact 5/10 (reid), 4/10 (position-only), 0/10 (per-robot counting, sum of 6 counts; 39 duplicates).
  Spec target (>=95% exact) NOT met. Remaining errors: buried acoustic-only victims (no appearance,
  sigma 1.5 cells) fragment or merge near hot rocks. No false-positive confirmations in any run.
- Known issues / next: acoustic localisation (multi-robot triangulation), cross-agent pose alignment,
  then Gazebo cameras (real detections), then ROS node wrapper.

### Phase R update — acoustic-only victims (evidence map)
- Acoustic-only detections no longer form individual tracks; they are splatted into a per-robot evidence
  map (CRDT, owner-versioned) and victims are density peaks (`ac_sep` 2.0, `ac_reach` 4.5, mass >= 10).
  A peak within `ac_explain` (2.0 cells) of a CONFIRMED visual victim is the same person; an unconfirmed
  hot rock never explains one away. Linking now needs mature appearance (>=4 embeddings) except in the
  position-only ablation; tight position (<15% of the gate) links when appearance is merely inconclusive.
- 24 tests pass (`python -m pytest tests/test_reid.py -q`).
- 20 seeds (0-19), 4 UAV + 2 UGV, drift 0.03/step. Exact = count == ideal, 0 duplicates, 0 false positives.
  reid: resolvable maps 5/8 exact (all victims >= 4 cells apart), crowded maps 1/12; position-only 5/8, 2/12;
  per-robot counting 1/8, 0/12 (49 duplicates over 20 maps). False positives: 0 for reid in all 20 maps.
- HONEST: on seeds 0-9 the overall exact rate (3/10) is NOT better than the track-based version (5/10).
  The evidence map removed the acoustic fragments but undercounts clusters of victims < ~3 cells apart,
  which these sensors cannot separate. Appearance matching adds little on these maps.
- Next levers: bearing-capable acoustic array (not single-range noise), inter-robot pose alignment,
  then real Gazebo detections. Do not tune thresholds further against this 20-seed set.

### Phase R update 2 — bearing acoustics + inter-robot pose alignment
- `reid_sensing.py`: rovers' microphone array now reports direction of arrival (8 deg) plus a poor
  loudness-based range (25%), giving an elongated 2x2 covariance (`Observation.cov`); the acoustic evidence
  map splats it anisotropically so views from different directions cross at the source.
  `SensingModel.align()`: robots in contact pull their drift toward each other (ranging / mutual detection /
  loop-closure stand-in) and shrink their reported pose variance; common swarm drift is untouched.
  SIMULATION ASSUMPTIONS, not measured hardware. Flags: `--acoustic bearing|range`, `--no-align`.
- 29 tests pass. Ablation on FRESH seeds 20-39 (not used for tuning), reid, 4 UAV + 2 UGV, drift 0.03:
  range/no-align exact 6/20, dup 15 | range/align 7/20, dup 8 | bearing/no-align 9/20, dup 15 |
  bearing/align 11/20, dup 8, FP 0 in every configuration.
  Resolvable maps (victims >= 4 cells apart): 4/8 -> 6/9 exact. Subset sizes differ per configuration
  because "resolvable" depends on which victims were observed. Still short of the >=95% target.
- Next: Gazebo cameras (real detections) to replace the synthetic observation model.

## Phase R3 — Re-ID bridge to Gazebo/ROS 2 (code written, NOT run against Gazebo)
Built and unit-tested without ROS (`python -m pytest tests -q`: 47 pass):
- `drone_agent/victim_geometry.py`: pixel -> ground point (exact inverse of the detector's projection,
  round-trip tested), covariance propagation, colour-histogram appearance placeholder.
- `drone_agent/victim_fusion.py`: poses + raw YOLO boxes -> per-drone `VictimRegistry`, snapshot gossip
  between drones within range. `min_sensors=1`, `min_agents=2` (a person must be seen by 2 drones).
- `drone_agent/victim_registry_node.py` (+ entry point `victim_registry`, added to the launch file):
  publishes `/swarm/victims` JSON. `yolo_detector.py` now also publishes `/yolo/raw_detections`.
- `sim/reid.py`: `snapshot()/from_snapshot()` for a real link; `min_agents`, `min_sensors` options;
  fixed a crash when a track born without appearance later receives an embedding.
FINDING (violates principle 5, not changed): `yolo_detector._get_disaster_targets_ned()` is a hard-coded
list of the world's people and cars (animated by wall-clock time); every one inside the camera frustum is
published on `/yolo/detections` at confidence ~0.91 whether or not YOLO saw it. `/yolo/detections`,
the annotated video and anything built on them reflect simulator ground truth. The Re-ID node uses only
`/yolo/raw_detections` (real YOLO boxes) to avoid inheriting this.
First-run checklist on the Ubuntu laptop: (1) `colcon build` + source; (2) verify `origins_ned` matches
the PX4 spawn offsets in start_all.sh; (3) check `ros2 topic echo /yolo/raw_detections` shows boxes on
people in the Gazebo world (COCO yolov8n may not detect the SDF human models at all); (4) check
`/swarm/victims` count vs the world's real number of people; (5) recalibrate tau_app/tau_link on real
crops; (6) add a thermal-like second sensor if a mannequin/poster false positive appears.

### Update — ground-truth overlay is now opt-in
`yolo_detector` parameter `gt_overlay` (default false; launch arg `gt_overlay:=true`) gates
`_get_disaster_targets_ned()`. Off by default: `/yolo/detections`, the annotated video and the GUI now
show only real YOLO boxes (fewer/no "lock-ons" if yolov8n misses the SDF humans). On: old demo look,
with a startup warning. Not run on ROS/Gazebo (syntax-checked only).

## WSL run — root cause of "drones don't fly / cameras dead" and live result (2026-10-31)
- ROOT CAUSE: Gazebo ran at ~0.1% real time (real_time_factor 0.0009). Mesa's d3d12 driver chose adapter 0,
  the Intel UHD iGPU, so the RTX 3050 sat at 0%. Same with the original, unmodified start_all.sh.
  FIX: `export MESA_D3D12_DEFAULT_ADAPTER_NAME="${...:-NVIDIA}"` in start_all.sh and launch_drone.sh.
  After the fix: real_time_factor ~0.7, drones reach their 9-13.5 m altitudes, camera topics publish.
- `ros2 topic hz` on the camera topics reports "no messages" even when they flow (use `echo --once`).
- Workspace `LLM-controlled-drone/install` was stale (no gt_overlay / raw_detections / victim_registry);
  rebuilt with `colcon build --packages-select drone_agent`.
- LIVE RESULT, current code, overlay off, 4 drones, ~3 min: /swarm/victims reports 10 confirmed vs 9 real
  people, but only 3 match a real person within 3 m; 7 are false/mislocalised tracks (two with 100+
  observations near the origin) and 6 people are not found. The matching count is a coincidence.
  Not fixed: suspects are camera/odometry frame conventions (PX4 FRD quaternion, yaw), the
  flat-ground foot projection, YOLOv8n false positives on static objects, 640x480 camera.
- Observed (not diagnosed): UAV-2..4 hold within ~2 m of one spot while UAV-1 patrols; the brain keeps
  re-issuing far-corner goals (+42N,+39E) that they do not reach.
- Temp files I left in the repo root: .stack.* .sample.* .ros_test.* .build_wsl.* ; WSL: ~/swarm_ws,
  /home/tushar/Desktop/swarm symlink.

## 3D voxel viewer rebuild + rescue rovers (2026-10-31)
FINDINGS before changing anything: the "OctoMap" was not an OctoMap. `voxels_3d` was a dict of 1 m integer cells
(cap 9000, no probability, no 3D free space) filled by a SIMULATED LiDAR that marched 48 rays through the hand-written
obstacle list in `_get_world_obstacles()` (Gazebo drones have no LiDAR), one hit per 2 m of height, drawn as a fixed
7 px square in a cv2 painter. That is why it looked like scattered dots. Old code kept as `_render_3d_octomap_viewer_legacy`.
NEW (ROS-free, in drone_agent/): voxel_map.py (log-odds voxels, free space, ground coverage), world_sim.py (analytic
dense LiDAR sim + roofs, dynamic obstacles, SIMULATED survivor detector), voxel_viewer.py (cube renderer, lighting, AO,
shadows, colour modes, layers, orbit camera, rover/drone/survivor models, routes, transmissions, legend),
rover_nav.py (traversability from the voxel map, A*, string-pulling, pure-pursuit rover, replanning),
rescue_coordinator.py (dedupe, nearest-by-route dispatch, FIFO queue, mission state). map_gui_node.py page 2 uses them
(Tk controls: layers, thresholds, height filter, opacity, voxel size, colour mode, views, focus, rover ops).
TESTS: 77 pass (tests/test_rover_stack.py, test_voxel_viewer.py new). LIVE in WSL: 9/9 simulated survivors dispatched and
rescued by 2 rovers; viewer ~5 fps inside the GUI while the machine is at load ~15 (84-124 ms/frame benchmark under load).
NOT DONE / CAVEATS: detections are SIMULATED by default (real YOLO registry selectable, still inaccurate); rovers are
simulated in the GUI, not Gazebo models; Tk controls and mouse not exercised by hand; rubble-replan verified in unit
tests, not live; ambient occlusion/shadows are approximations (per-face AO, ground shadows only).

## Drone stability + rover camera feeds (2026-10-31)
MEASURED (scripts/flight_metrics.py, PX4 odometry, 60 s of flight, mean of 4 drones; SITL, so simulated physics):
  original: tilt RMS 27.6 deg, body-rate p95 164 deg/s, yaw-rate p95 157 deg/s, accel RMS 5.6 m/s^2
  final (run 8, all four healthy): tilt RMS 5.6 deg, body-rate p95 13.8 deg/s, yaw-rate p95 4.8 deg/s, accel RMS 0.9 m/s^2
CAUSES: (1) raw far-away target sent as position setpoint at 10 Hz; (2) yaw setpoint snapped to the bearing of every
new target (position_ned sets target_yaw) and flipped at a 1.2 m threshold; (3) swarm allocator re-targets each UAV every
~8 s, ping-ponging between two nearby points (NOT fixed: coverage grid marks a 32 m radius explored at once, visit counts
are reset at >94% coverage); (4) limits were in wall-clock time while Gazebo ran at ~0.58x real time => 3x harsher in
sim time; (5) CPU oversubscription (load ~17 on 12 cores) starved a PX4 instance: EKF lost, "blind land", flyaway.
FIXES: flight_smoothing.py (jerk-limited carrot + yaw smoother, 6 deg/s), setpoints with velocity/accel feed-forward at
25 Hz, smoother steps in SIM time (rtf estimated from odometry timestamps), heading from low-passed consistent travel
direction, gentler PX4 limits in start_all.sh, GUI at nice 12, YOLO ~3 inf/s/camera, rover cams at 2.5 Hz.
ROVER CAMERAS: VoxelViewer.first_person + render_pov, worker thread; page 1 buttons ROVER CAMS / ROVER 1 CAM / ROVER 2 CAM
and page-2 panels. SIMULATED views rendered from the voxel map (rovers are not Gazebo models).
KNOWN: UAV-2 lost its estimator in two heavy-load runs (runs 6, 7); no repeat in run 8 after load cuts, but root cause
(CPU starvation) is unproven; add a guard against following a diverging estimate.
