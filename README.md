# Drone-Swarm (SwarmRescue)
## Autonomous Heterogeneous Multi-Agent UAV + UGV Swarm for GPS-Denied Disaster Search & Rescue

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros)](https://docs.ros.org/en/jazzy/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-FF6F00?logo=gazebo)](https://gazebosim.org/docs/harmonic/)
[![PX4 Autopilot](https://img.shields.io/badge/PX4-Autopilot%20v1.14+-blue?logo=px4)](https://px4.io/)
[![PyTorch GNN](https://img.shields.io/badge/PyTorch-GraphSAGE%20GNN-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![MARL MAPPO](https://img.shields.io/badge/MARL-MAPPO-green)](#algorithmic-framework)
[![Unit Tests](https://img.shields.io/badge/Tests-87%20Passed-brightgreen)](#testing--verification)

> **SwarmRescue** is an end-to-end, decentralized, heterogeneous swarm robotics platform (**4 Autonomous UAVs + 2 Ground Rovers**) engineered to search GPS-denied disaster zones, detect surface and buried survivors via thermal, visual, and acoustic modalities, de-duplicate sightings across the fleet using a conflict-free replicated data type (CRDT) registry, construct a live 3D probabilistic voxel map, and automatically dispatch ground rovers to reach every survivor—commanded in natural language by a single operator.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Capabilities & Innovations](#key-capabilities--innovations)
3. [System Architecture](#system-architecture)
4. [Algorithmic Framework](#algorithmic-framework)
   - [1. Three-Layer Autonomy Hierarchy](#1-three-layer-autonomy-hierarchy)
   - [2. Decentralized MARL + GraphSAGE GNN](#2-decentralized-marl--graphsage-gnn)
   - [3. Multi-Modal Sensing (Thermal, RGB YOLOv8, Acoustic DoA)](#3-multi-modal-sensing-thermal-rgb-yolov8-acoustic-doa)
   - [4. Victim Re-ID & CRDT Registry (Counting Every Human Once)](#4-victim-re-id--crdt-registry-counting-every-human-once)
   - [5. 3D Probabilistic Voxel Mapping & Shaded Renderer](#5-3d-probabilistic-voxel-mapping--shaded-renderer)
   - [6. Air-to-Ground Autonomous Rover Dispatch Loop](#6-air-to-ground-autonomous-rover-dispatch-loop)
   - [7. Jerk-Limited Flight Smoothing & Collision Avoidance](#7-jerk-limited-flight-smoothing--collision-avoidance)
5. [Real-Time Ground Control GUI (Quad-Cam + Rover POV + 3D Voxel Map)](#real-time-ground-control-gui-quad-cam--rover-pov--3d-voxel-map)
6. [Repository Structure](#repository-structure)
7. [Prerequisites & Installation](#prerequisites--installation)
8. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Quickstart 1: Pure-Python Multi-Agent Simulator](#quickstart-1-pure-python-multi-agent-simulator-no-ros-required)
   - [Quickstart 2: MARL + GraphSAGE GNN Training & Evaluation](#quickstart-2-marl--graphsage-gnn-training--evaluation)
   - [Quickstart 3: Automated Verification Suite (87 Tests)](#quickstart-3-automated-verification-suite-87-tests)
   - [Quickstart 4: Full 4-UAV + 2-UGV Gazebo Harmonic Swarm & GUI](#quickstart-4-full-4-uav--2-ugv-gazebo-harmonic-swarm--gui)
9. [Experimental Results & Benchmarks](#experimental-results--benchmarks)
10. [Design Principles & Engineering Commitments](#design-principles--engineering-commitments)

---

## Project Overview

In the first 72 hours following a major structural collapse or earthquake, survivor mortality rises sharply while road networks, cellular towers, and GPS availability inside collapsed structures fail. Conventional single-pilot drones require dedicated operators per aircraft, lack shared situational awareness, double-count victims observed from overlapping viewpoints, and cannot traverse interior rubble.

**Drone-Swarm (SwarmRescue)** addresses these challenges with ~11,400 lines of custom Python across simulation, reinforcement learning, computer vision, 3D mapping, and ROS 2 / PX4 flight control:
- **4 Unmanned Aerial Vehicles (UAVs):** Perform rapid 3D exploration, wide-FOV thermal and YOLOv8 visual detection, 3D raycasting ground projection, and peer-to-peer mesh relay.
- **2 Unmanned Ground Vehicles (UGVs / Rovers):** Perform acoustic Direction-of-Arrival (DoA) sensing for buried survivors, navigate around collapsed rubble using A* planning and pure-pursuit kinematics over the shared voxel map, dynamically replan when debris blocks routes, and stream first-person POV camera feeds.
- **Single-Operator Natural Language Command:** High-level mission directives (*"Map the whole area"*, *"Search northern sector for survivors"*, *"Hold position"*) are translated by a local/cloud LLM (`Ollama`, `Groq`, or `Gemini`) into structured task specifications without ever placing the LLM in the real-time flight control loop.

---

## Key Capabilities & Innovations

| Capability | Technical Implementation | Source Modules |
|---|---|---|
| **Decentralized Swarm Brain** | Identical policy bundle on every agent; peer-to-peer range-limited comms; GraphSAGE GNN goal allocator with automatic geometric frontier fallback | `train/model.py`, `drone_agent/swarm_gnn_allocator.py`, `drone_agent/brain_node.py` |
| **Heterogeneous Air + Ground Fleet** | 4 PX4 UAVs (fast aerial sweep, thermal/RGB) + 2 UGVs (rubble traversal, microphone array acoustic triangulation, A* + pure-pursuit navigation) | `sim/env.py`, `drone_agent/rover_nav.py`, `drone_agent/rescue_coordinator.py` |
| **GPS-Denied Operation** | Local-frame odometry/VIO only; explicit noisy drift modeling ($\sigma = 0.03$/step) and uncertainty covariance propagation | `sim/reid_sensing.py`, `drone_agent/victim_geometry.py` |
| **Multi-Modal Survivor Sensing** | LWIR thermal raycasting (open victims) + YOLOv8 camera detection + anisotropic acoustic DoA splatting (buried victims) | `sim/sensors.py`, `drone_agent/yolo_detector.py`, `sim/reid_sensing.py` |
| **Zero-Duplicate Victim Re-ID** | 3D raycast ground lift + 128-d appearance embeddings + Hungarian hysteresis matching + gossip-replicated Union-Find CRDT registry | `sim/reid.py`, `drone_agent/victim_fusion.py`, `drone_agent/victim_registry_node.py` |
| **3D Probabilistic Voxel Mapping** | Log-odds occupancy grid (`occupied`, `free`, `unknown`, `recently changed`) rendered as shaded 3D cubes with interactive orbit camera & rover POV | `drone_agent/voxel_map.py`, `drone_agent/voxel_viewer.py` |
| **Air-to-Ground Rescue Dispatch** | Confirmed aerial detections automatically dispatch the nearest available rover by A* path length, with dynamic replanning and mission queuing | `drone_agent/rescue_coordinator.py`, `drone_agent/rover_nav.py` |
| **Sim-Time Jerk-Limited Smoothing** | 25–50 Hz position/velocity/acceleration feed-forward setpoints with sim-time RTF awareness and $6^\circ/\text{s}$ yaw-rate smoothing, cutting drone tilt by 80% | `drone_agent/flight_smoothing.py`, `drone_agent/command_translator.py`, `scripts/flight_metrics.py` |

---

## System Architecture

```
 ┌────────────────────────────── OPERATOR / GROUND CONTROL STATION ─────────────────────────────────┐
 │  Natural Language Commander (Ollama / Groq / Gemini) ──► Structured Task Spec JSON               │
 │  PySide6 Multi-Page GUI: 4-UAV Live Quad-Cam + 2-Rover POV Feeds + 3D Shaded Voxel Map Viewer    │
 └───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │ Mesh Gossip (Task spec deltas + CRDT updates)
 ┌───────────────────────────────────────────────▼──────────────────────────────────────────────────┐
 │                         DISTRIBUTED SWARM BLACKBOARD (Replicated CRDT)                           │
 │  • Victim Registry (Union-Find CRDT)       • Frontier Claims & Dynamic Leases                    │
 │  • 3D Probabilistic Voxel Map Deltas       • Air-to-Ground Rover Rescue Dispatch Queue           │
 └───────▲───────────────────────────────────────────────────────────────────────────▲──────────────┘
         │ (Peer-to-Peer Ad-hoc Wireless Mesh)                                       │
 ┌───────┴─────────────────────── UAV Stack (×4) ───────────────────────┐   ┌────────┴────── UGV Stack (×2) ──────┐
 │ • Sensors: RGB Camera + LWIR Thermal + 3D LiDAR + Local VIO          │   │ • Sensors: Microphone Array (DoA),  │
 │ • Perception: YOLOv8 Detector + 3D Pinhole Raycasting Ground Lift    │   │   Simulated Voxel Depth + Odometry  │
 │ • Re-ID Engine: 128-d Appearance Embedding + Covariance Propagation  │   │ • Acoustic Mapper: Anisotropic      │
 │ • Tactical Planner: GraphSAGE GNN Policy over Dynamic Peer Graph     │   │   Gaussian Evidence Splatting       │
 │ • Flight Control: Sim-Time Jerk-Limited Carrot + 6.5m APF Buffer     │   │ • Navigation: 2D A* on Voxel Map +  │
 │ • Autopilot: PX4 SITL Offboard Mode (Pos/Vel/Accel Feed-Forward)     │   │   Pure-Pursuit + Dynamic Replanning │
 └──────────────────────────────────────────────────────────────────────┘   └─────────────────────────────────────┘
```

---

## Algorithmic Framework

### 1. Three-Layer Autonomy Hierarchy
Strict separation of concerns guarantees real-time flight safety:
- **Layer 1 — Advisory LLM Commander (10–30 s):** Parses plain-English operator prompts into structured JSON mission parameters (target sector, search priority, speed profile). **Strictly decoupled from the flight control loop.**
- **Layer 2 — Tactical MARL + GraphSAGE Allocator (1 Hz + Event-Triggered):** Performs message passing over the dynamic peer communication graph to assign exploration frontiers and avoid redundant coverage. Automatically falls back to a deterministic geometric frontier heuristic if GNN weights or peer links are unavailable.
- **Layer 3 — Real-Time Flight & Obstacle Control (25–50 Hz):** Executes jerk-limited trajectory tracking, Artificial Potential Field (APF) obstacle repulsion, 6.5 m inter-drone separation, and PX4 offboard feed-forward setpoint publishing.

---

### 2. Decentralized MARL + GraphSAGE GNN
Trained via **Multi-Agent Proximal Policy Optimization (MAPPO)** under Centralized Training with Decentralized Execution (CTDE):
- **Agent Node Features ($F_A = 109$):** Agent type one-hot (UAV vs. UGV), normalized battery, local coordinates, heading, detected survivor count, acoustic hint vector, nearest-frontier distance, and a flattened $7 \times 7 \times 2$ local occupancy/victim patch.
- **Candidate Frontier Features ($N = 20, F_G = 5$):** Relative distance, unit direction vector, frontier cluster size, information gain prior, and terrain compatibility.
- **Distance-Attenuated GraphSAGE Message Passing:**
  $$\mathbf{h}_i^{(l+1)} = \sigma \left( \mathbf{W}_1 \mathbf{h}_i^{(l)} + \mathbf{W}_2 \cdot \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} w_{ij} \mathbf{h}_j^{(l)} \right)$$
  where edge weight $w_{ij}$ decays with inter-robot distance and RF wall attenuation.

---

### 3. Multi-Modal Sensing (Thermal, RGB YOLOv8, Acoustic DoA)
- **LWIR Thermal & YOLOv8 Visual Detection (UAVs):** Conical field-of-view raycasting with Line-of-Sight (LOS) occlusion checks and real-time `yolov8n.pt` person detection on live Gazebo camera streams.
- **Acoustic DoA Triangulation (UGVs):** Models a 4-microphone array estimating Direction-of-Arrival ($\pm 8^\circ$ bearing noise) and inverse-square attenuation ($I = I_0 / (1 + \alpha d^2)$). Splatting anisotropic Gaussian ellipses onto a shared acoustic evidence grid enables cross-bearing triangulation of survivors buried under opaque rubble.

---

### 4. Victim Re-ID & CRDT Registry (Counting Every Human Once)
1. **3D Raycasting Ground Lift:** Projects 2D bounding-box centers $(u, v)$ to 3D ground coordinates using drone altitude and roll/pitch/yaw attitude:
   $$\mathbf{P}_{\text{ground}} = \mathbf{P}_{\text{drone}} + z_{\text{drone}} \cdot \mathbf{R}_{\text{body}}^{\text{world}} \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
2. **Analytical Covariance Propagation:** Propagates pixel and attitude uncertainty via the Jacobian $\Sigma_{\text{ground}} = \mathbf{J} \Sigma_{\text{pixel}} \mathbf{J}^T$.
3. **Appearance + Spatial Hungarian Matching:** Combines Mahalanobis/Euclidean spatial gating with 128-d appearance cosine similarity and hysteresis thresholds.
4. **Union-Find CRDT Gossip Registry:** Replicates sightings across partitioned sub-swarms and merges equivalence classes deterministically upon reconnection with zero duplicate counts.

---

### 5. 3D Probabilistic Voxel Mapping & Shaded Renderer
- **Log-Odds 3D Occupancy Grid (`voxel_map.py`):** Maintains multi-layer voxel states (`OCCUPIED`, `FREE`, `UNKNOWN`, `CHANGED`) updated continuously from drone depth/LiDAR rays and dynamic rubble collapse events.
- **High-Speed Shaded Voxel Engine (`voxel_viewer.py`):** Custom painter's-algorithm 3D cube renderer with directional lighting, interactive orbit/pan/zoom camera, height-slice filtering, layer visibility toggles, and real-time first-person POV rendering (`render_pov`) for ground rovers (~60 ms/frame).

---

### 6. Air-to-Ground Autonomous Rover Dispatch Loop
- **Rescue Coordinator (`rescue_coordinator.py`):** Subscribes to confirmed CRDT victim records and computes the true A* path length from each ground rover to the survivor over the 2D ground projection of the 3D voxel map.
- **Rover Navigation (`rover_nav.py`):** Dispatches the closest idle rover (or queues the target if both rovers are busy). Uses 8-connected A* with obstacle inflation and pure-pursuit steering controller, automatically triggering dynamic replanning when new rubble blocks the active route.

---

### 7. Jerk-Limited Flight Smoothing & Collision Avoidance
- **Sim-Time Aware Trajectory Smoother (`flight_smoothing.py`):** Generates jerk-limited position, velocity, and acceleration feed-forward setpoints stepped in **simulation time** (estimating Gazebo Real-Time Factor from PX4 odometry timestamps) with a $6^\circ/\text{s}$ yaw-rate limiter aligned to low-pass filtered travel velocity.
- **Safety Filter:** Combines a 6.5 m inter-drone spatial separation buffer with Artificial Potential Field (APF) obstacle repulsion.

---

## Real-Time Ground Control GUI (Quad-Cam + Rover POV + 3D Voxel Map)

The PySide6 Ground Control Station (`drone_agent/map_gui_node.py`) provides full fleet observability:
- **Page 1 — Multi-Camera Tactical Feeds:**
  - `[4-CAM GRID]`: Live $2 \times 2$ video matrix from all 4 UAVs with HUD overlays (altitude, ground speed, battery, compass heading, YOLOv8 bounding boxes, and GNN target state).
  - `[UAV-1 CAM]` – `[UAV-4 CAM]`: One-click full-screen enlargement for any aerial camera.
  - `[ROVER CAMS]` / `[ROVER 1 CAM]` / `[ROVER 2 CAM]`: Real-time first-person 3D voxel-rendered POV streams showing what each ground rover sees as it navigates rubble toward survivors.
- **Page 2 — Interactive 3D Collaborative Voxel Map:**
  - Orbitable 3D shaded voxel map with layer toggles (`Occupied`, `Free`, `Changed`, `Unknown`), height cut-plane slider, live UAV/UGV trajectories, survivor markers, and Picture-in-Picture (PiP) camera feeds.

---

## Repository Structure

```text
Drone-swarm/
├── README.md                          # Comprehensive project documentation & execution guide
├── SAR_SWARM_SPEC.md                  # Full mathematical & systems engineering specification
├── AGENTS.md                          # Architectural principles & engineering invariants
├── PROGRESS.md                        # Phase-by-phase engineering & benchmark verification log
├── THIRD_PARTY.md                     # Open-source attributions & license documentation
├── build_px4.sh                       # Automated PX4 SITL build helper for Gazebo Harmonic
├── download_models.sh                 # Gazebo Fuel 3D disaster world asset downloader
├── yolov8n.pt                         # Ultralytics YOLOv8n weights for edge person detection
│
├── sim/                               # Pure-Python Lightweight Multi-Agent Simulator
│   ├── env.py                         # Heterogeneous grid world (UAVs + UGVs, battery, RF comms)
│   ├── map_gen.py                     # Procedural collapsed building & rubble generator
│   ├── sensors.py                     # Thermal LOS raycasting & acoustic attenuation models
│   ├── baselines.py                   # Exploration baselines (Random, Nearest Frontier, Explore Lite)
│   ├── reid.py                        # Union-Find CRDT victim registry & Hungarian Re-ID matcher
│   ├── reid_sensing.py                # Simulated VIO drift, hot-rock false positives, mic-array DoA
│   ├── run_baselines.py               # Multi-agent exploration benchmark runner
│   ├── run_reid.py                    # Multi-seed Re-ID de-duplication evaluation suite
│   ├── visualize.py                   # Episode GIF/PNG trajectory & coverage renderer
│   └── metrics.py                     # Coverage %, time-to-first-victim, collision statistics
│
├── train/                             # Decentralized MARL + GraphSAGE GNN Framework
│   ├── model.py                       # GraphSAGE Actor-Critic policy (TorchScript exportable)
│   ├── features.py                    # Node/edge observation encoder (FA=109, FG=5)
│   ├── ppo.py                         # Multi-Agent PPO (MAPPO) with Centralized Critic & GAE
│   ├── run_train.py                   # Training pipeline with GNN vs. No-GNN ablation support
│   └── evaluate.py                    # Held-out disaster seed policy evaluator
│
├── LLM-controlled-drone/              # ROS 2 Jazzy + PX4 SITL + Gazebo Harmonic Full Stack
│   ├── start_all.sh                   # One-command launcher (4 PX4 UAVs, bridges, YOLO, GUI)
│   ├── setup_workspace.sh             # ROS 2 dependency installer & colcon build script
│   ├── launch_drone.sh                # Single-drone PX4 SITL launcher helper
│   ├── worlds/
│   │   ├── drone_demo.sdf             # 3D disaster environment (collapsed structures, victims)
│   │   └── baylands_demo.sdf          # Outdoor test environment
│   └── src/
│       ├── drone_agent/               # Core ROS 2 Swarm Autonomy Package
│       │   ├── drone_agent/
│       │   │   ├── brain_node.py          # Mission state machine & LLM task orchestrator
│       │   │   ├── swarm_gnn_allocator.py # Decentralized GraphSAGE frontier goal allocator
│       │   │   ├── command_translator.py  # 25-50 Hz PX4 Offboard publisher & 6.5m APF safety filter
│       │   │   ├── flight_smoothing.py    # Sim-time jerk-limited carrot & yaw-rate smoother
│       │   │   ├── map_gui_node.py        # PySide6 Quad-Cam HUD, Rover POV & 3D Voxel GUI
│       │   │   ├── voxel_map.py           # Probabilistic 3D log-odds occupancy voxel grid
│       │   │   ├── voxel_viewer.py        # Shaded 3D voxel renderer, orbit cam & Rover POV engine
│       │   │   ├── rover_nav.py           # UGV 2D A* planner, pure-pursuit controller & replanner
│       │   │   ├── rescue_coordinator.py  # Air-to-ground survivor dispatch & rover queue manager
│       │   │   ├── world_sim.py           # Dynamic disaster world simulator for voxel/rover loop
│       │   │   ├── yolo_detector.py       # Real-time YOLOv8 person detector node
│       │   │   ├── victim_geometry.py     # 3D pinhole raycasting ground projection & Jacobian
│       │   │   ├── victim_fusion.py       # Multi-agent victim fusion & CRDT gossip sync
│       │   │   ├── victim_registry_node.py# ROS 2 /swarm/victims registry publisher
│       │   │   └── llm_client.py          # Ollama / Groq / Gemini natural-language client
│       │   ├── launch/
│       │   │   └── drone_agent.launch.py  # Multi-UAV ROS-Gazebo bridge & node launch file
│       │   ├── package.xml
│       │   └── setup.py
│       └── px4_msgs/                  # PX4 ROS 2 message definitions
│
├── scripts/
│   └── flight_metrics.py              # PX4 odometry analyzer (tilt, angular rate, yaw, accel)
│
├── tests/                             # Automated Unit & Integration Test Suite (87 Passing)
│   ├── test_reid.py                   # 29 tests: CRDT registry, Hungarian Re-ID, acoustic DoA
│   ├── test_victim_fusion.py          # 18 tests: 3D raycasting ground lift & covariance math
│   ├── test_flight_smoothing.py       # 11 tests: jerk-limited trajectory & yaw wrap-around
│   ├── test_rover_stack.py            # 14 tests: A* obstacle routing, dynamic replanning, dispatch
│   ├── test_voxel_viewer.py           # 15 tests: probabilistic voxel updates, 3D projection, POV
│   └── ros_integration_victim_registry.py # End-to-end ROS 2 victim registry pipeline check
│
├── results/                           # Trained Checkpoints, Plots & Episode Recordings
│   ├── checkpoints/                   # Saved GraphSAGE MARL weights (gnn.pt, gnn_actor.pt)
│   ├── p1_baselines.png               # Exploration baseline comparison plots
│   ├── p1_episode.gif                 # Animated multi-agent SAR exploration rollout
│   ├── p2_learning_curves.png         # MAPPO training reward curves
│   └── p2_summary.json                # Quantitative evaluation metrics
│
├── pitch/                             # Presentation Deck & Sourced Evidence Brief
│   ├── SwarmRescue_Pitch.md           # Sourced disaster statistics, benchmarks & architecture brief
│   └── SwarmRescue_AI_Manthan.pptx    # Slide deck presentation
│
└── dashboard/
    └── app.py                         # Web telemetry monitoring dashboard
```

---

## Prerequisites & Installation

### System Requirements
- **OS:** Ubuntu 24.04 LTS (Noble) or Windows 11 with WSL2 (`DISPLAY=:0` GUI support)
- **ROS 2 (for full Gazebo stack):** ROS 2 Jazzy Jalisco (`ros-jazzy-desktop`)
- **Simulator:** Gazebo Harmonic (`gz sim 8+`) & PX4 Autopilot SITL (`v1.14+`)
- **Python:** Python 3.10 – 3.12 (Pure-Python simulator, MARL training, 3D voxel engine, and all 87 unit tests run standalone without ROS 2!)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/TejasGogawale/Drone-swarm.git
cd Drone-swarm

# 2. Create a virtual environment and install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy matplotlib opencv-python ultralytics PySide6 pytest

# 3. (Optional — for full Gazebo + PX4 simulation on Ubuntu 24.04 / WSL2)
sudo apt update && sudo apt install -y \
    python3-colcon-common-extensions \
    ros-jazzy-desktop ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image ros-jazzy-cv-bridge
```

---

## Step-by-Step Execution Guide

### Quickstart 1: Pure-Python Multi-Agent Simulator (No ROS Required)
Benchmark **Random Walk**, **Nearest Frontier**, and **Explore Lite** across heterogeneous teams (UAVs + UGVs) in procedurally generated collapsed buildings:

```bash
python3 -m sim.run_baselines --seeds 0 1 2 --drones 2 --rovers 2 --max-steps 300
```

Render an animated GIF of the swarm exploring rubble and triaging survivors:
```bash
python3 -m sim.visualize --seed 42 --drones 2 --rovers 2 --out results/p1_episode.gif
```

---

### Quickstart 2: MARL + GraphSAGE GNN Training & Evaluation
Train the decentralized GraphSAGE policy with Multi-Agent PPO (MAPPO) and evaluate against baselines:

```bash
# Train GraphSAGE Actor-Critic policy
python3 -m train.run_train --iterations 40 --drones 2 --rovers 2

# Evaluate trained checkpoint on held-out disaster seeds
python3 -m train.evaluate --checkpoint results/checkpoints/gnn.pt --seeds 10 11 12
```

---

### Quickstart 3: Automated Verification Suite (87 Tests)
Run the complete unit test suite covering 3D raycasting geometry, CRDT victim de-duplication, acoustic DoA triangulation, jerk-limited flight smoothing, 3D voxel rendering, and A* rover dispatch:

```bash
# Execute all 87 automated tests
PYTHONPATH=.:LLM-controlled-drone/src/drone_agent pytest tests -q

# Run the 20-seed multi-agent Victim Re-ID benchmark under simulated VIO drift
python3 -m sim.run_reid --seeds 0 1 2 3 4 5 6 7 8 9 --drones 4 --rovers 2
```

---

### Quickstart 4: Full 4-UAV + 2-UGV Gazebo Harmonic Swarm & GUI
Launch 4 PX4 SITL quadrotors in Gazebo Harmonic alongside the ROS 2 bridge, YOLOv8 victim detectors, decentralized GNN allocators, autonomous ground rovers, and the PySide6 Ground Control GUI:

```bash
# 1. Download 3D disaster world assets
bash download_models.sh

# 2. Build the ROS 2 workspace
cd LLM-controlled-drone
bash setup_workspace.sh

# 3. Source ROS 2 Jazzy and workspace overlay
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# 4. Launch the complete 4-UAV + 2-UGV autonomous stack
bash start_all.sh
```

---

## Experimental Results & Benchmarks

### 1. Sim-Time Flight Smoothing & Attitude Stability (PX4 SITL Odometry, 4 UAVs)
Measured directly from `/px4_*/fmu/out/vehicle_odometry` over 60-second autonomous exploration flights (`scripts/flight_metrics.py`):

| Metric (Mean Across 4 UAVs) | Before Smoothing | After Sim-Time Jerk + Yaw Smoothing | Improvement |
|---|:---:|:---:|:---:|
| **Mean Tilt / Lean Angle** | $27.6^\circ$ | **$5.6^\circ$** | **$-80\%$** |
| **95th-Percentile Body Angular Rate** | $164.0^\circ/\text{s}$ | **$13.8^\circ/\text{s}$** | **$-92\%$** |
| **95th-Percentile Yaw Rate** | $157.0^\circ/\text{s}$ | **$4.8^\circ/\text{s}$** | **$-97\%$** |
| **Mean Linear Acceleration** | $5.6\text{ m/s}^2$ | **$0.9\text{ m/s}^2$** | **$-84\%$** |

---

### 2. Multi-Agent Exploration Benchmark ($40 \times 40$ Collapsed Structure Maps)
Evaluated with 2 UAVs + 2 UGVs and 6 buried/surface survivors across multiple disaster seeds:

| Strategy | Map Coverage (%) | Survivors Found (%) | Time to First Victim (steps) | Time to All Victims (steps) | Collisions |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Walk** | 78.9% | 100% | 5 | 104 | 4.3 |
| **Nearest Frontier** | 75.3% | 100% | 4 | 56 | 2.7 |
| **Explore Lite (Info Gain)** | 81.8% | 100% | 4 | 76 | 2.0 |
| **SwarmRescue (MARL + GraphSAGE)** | **88.4%** | **100%** | **3** | **48** | **0.8** |

---

### 3. Victim Re-ID & De-duplication Under VIO Drift (4 UAVs + 2 UGVs, $\sigma_{\text{drift}} = 0.03$/step)

| Fusion Method | Total Duplicate Sightings (20 Seeds) | False Positives | Air-to-Ground Rover Dispatch Reach Rate |
|---|:---:|:---:|:---:|
| **Naive Per-Robot Sum** | 49 duplicates | 0 | N/A |
| **Position-Only Clustering** | 8 duplicates | 0 | N/A |
| **SwarmRescue (Appearance + Acoustic DoA + CRDT)** | **0 – 8 fragments (84% reduction)** | **0** | **9 / 9 survivors reached (24 dynamic replans)** |

---

## Design Principles & Engineering Commitments

1. **Truly Decentralized Execution:** Every agent runs its own local perception and GraphSAGE allocator; communication is strictly peer-to-peer within wireless range limits.
2. **Honest GPS-Denied Modeling:** Agents rely exclusively on local odometry frames with injected VIO drift rather than simulator ground-truth coordinates.
3. **LLM Outside the Safety Loop:** Natural-language parsing is strictly advisory; if the LLM is offline or slow, the swarm continues full autonomous exploration and rescue dispatch.
4. **Reproducible Verification:** All 87 unit and integration tests pass deterministically out of the box.

---

## License & Acknowledgments

Developed by **Team LOVE** (*Tejas Gogawale & Contributors*) for autonomous disaster response robotics. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for open-source attributions.
