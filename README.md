# Team LOVE — AI MANTHAN 2.0 (Problem Statement 5)
# Autonomous Disaster Response Swarm Robotics for Search and Rescue (SAR)

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros)](https://docs.ros.org/en/jazzy/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-FF6F00?logo=gazebo)](https://gazebosim.org/docs/harmonic/)
[![PX4 Autopilot](https://img.shields.io/badge/PX4-Autopilot%20v1.14+-blue?logo=px4)](https://px4.io/)
[![PyTorch GNN](https://img.shields.io/badge/PyTorch-GraphSAGE%20GNN-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![MARL MAPPO](https://img.shields.io/badge/MARL-MAPPO-green)](#algorithmic-framework)
[![Unit Tests](https://img.shields.io/badge/Tests-47%20Passed-brightgreen)](#testing--verification)

> **Event:** AI MANTHAN 2.0 — Acropolis Institute of Technology and Research, Indore  
> **Problem Statement 5 (PS-5):** Build an autonomous, decentralized, heterogeneous swarm (UAVs + UGVs) to navigate GPS-denied disaster environments, locate survivors using thermal and acoustic modalities, share topological data peer-to-peer, and count every human exactly once.  
> **Team:** Team LOVE  

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Innovations & PS-5 Compliance](#core-innovations--ps-5-compliance)
3. [System Architecture](#system-architecture)
4. [Algorithmic Framework](#algorithmic-framework)
   - [Three-Layer Architecture](#1-three-layer-architecture)
   - [Decentralized MARL + GraphSAGE GNN](#2-decentralized-marl--graphsage-gnn)
   - [Multi-Modal Sensing (Thermal + Acoustic)](#3-multi-modal-sensing-thermal--acoustic)
   - [Victim Re-ID & CRDT Registry (Counting Exactly Once)](#4-victim-re-id--crdt-registry-counting-exactly-once)
   - [Collision Avoidance & Safe Offboard Motion](#5-collision-avoidance--safe-offboard-motion)
5. [Real-Time Visual GUI & Quad-Cam Telemetry](#real-time-visual-gui--quad-cam-telemetry)
6. [Repository Organization](#repository-organization)
7. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
8. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Quickstart 1: Pure-Python Lightweight Simulator](#quickstart-1-pure-python-lightweight-simulator-no-ros-required)
   - [Quickstart 2: MARL + GNN Training & Evaluation](#quickstart-2-marl--gnn-training--evaluation)
   - [Quickstart 3: Victim Re-ID & Fusion Verification Suite](#quickstart-3-victim-re-id--fusion-verification-suite)
   - [Quickstart 4: Full 4-Drone Gazebo Harmonic Swarm & Quad-Cam GUI](#quickstart-4-full-4-drone-gazebo-harmonic-swarm--quad-cam-gui)
9. [Experimental Results & Benchmarks](#experimental-results--benchmarks)
10. [Design Principles & Honesty Commitments](#design-principles--honesty-commitments)
11. [Authors & Acknowledgments](#authors--acknowledgments)

---

## Project Overview

Disaster environments—such as collapsed multi-story buildings, post-earthquake rubbles, and subterranean tunnels—are structurally degraded, cluttered with debris, and strictly **GPS-denied**. In these scenarios, single-agent robots or centralized cloud control fail due to single-point failures, RF attenuation, and limited mobility.

**SwarmRescue by Team LOVE** provides an end-to-end, production-grade autonomous swarm robotics platform built upon a **heterogeneous multi-agent team**:
- **4 Unmanned Aerial Vehicles (UAVs):** Rapid 3D spatial exploration, broad-area thermal imaging, high-altitude mapping, and mesh gossip relay.
- **2 Unmanned Ground Vehicles (UGVs):** Narrow-corridor penetration, acoustic sensing for victims trapped under dense rubble, and long-endurance ground traversal.

The system solves three fundamental challenges of disaster search-and-rescue:
1. **R1 — Exact Human Counting (Zero Duplicates):** Multi-agent camera raycasting projection, MobileNet visual embeddings, anisotropic acoustic direction-of-arrival (DoA) evidence maps, and an eventually consistent CRDT victim registry ensure a victim observed across multiple drones is confirmed and counted **exactly once**.
2. **R2 — GPS-Denied Collaborative Mapping:** Simulated Visual-Inertial Odometry (VIO) / LiDAR Inertial Odometry (LIO), decentralized 3D Voronoi partitioning, and collaborative OctoMap generation without global positioning systems.
3. **R3 — Resilient Swarm Mind:** Decentralized Multi-Agent Reinforcement Learning (MAPPO) with Graph Neural Networks (GraphSAGE) over dynamic peer-to-peer ad-hoc comms graphs. If any agent is disabled or network partitions occur, neighboring agents dynamically absorb the coverage workload with zero central coordination.

---

## Core Innovations & PS-5 Compliance

| PS-5 Challenge | Our Solution | Verification / Code |
|---|---|---|
| **Decentralized Control** | Every robot runs an identical policy bundle; peer-to-peer comms range limits; no central controller | `train/model.py`, `drone_agent/swarm_gnn_allocator.py` |
| **Heterogeneous Fleet** | UAVs (fast, wide thermal FOV, altitude-capable) vs UGVs (slow, rubble-traversing, acoustic microphone arrays) | `sim/env.py`, `sim/sensors.py`, `SAR_SWARM_SPEC.md` |
| **GPS-Denied Operation** | Odometry and VIO in robot-local frames; noisy drift injection modeled honestly; no GPS topics read | `sim/reid_sensing.py`, `drone_agent/victim_geometry.py` |
| **Multi-Modal Sensing** | LWIR thermal detection (open victims) + Acoustic attenuation & DoA triangulation (buried victims) | `sim/sensors.py`, `sim/reid_sensing.py` |
| **De-duplication / Re-ID** | Raycasting 3D lift + ByteTrack + CRDT Union-Find registry with hysteresis linking | `sim/reid.py`, `drone_agent/victim_fusion.py` (29 tests pass) |
| **End-to-End System** | Complete stack: ROS 2 Jazzy, PX4 SITL, Gazebo Harmonic, Quad-Cam GUI, and pure-Python sim | `start_all.sh`, `drone_agent/map_gui_node.py` |
| **Scalability & Fallback** | Pure-Python sim scales to 10+ agents; GNN falls back to geometric frontier heuristic on failure | `sim/run_baselines.py`, `drone_agent/swarm_gnn_allocator.py` |

---

## System Architecture

```
 ┌────────────────────────────── OPERATOR / BASE STATION (Advisory Layer) ──────────────────────────┐
 │  Local LLM Commander (Groq API / Ollama)  ──►  Task Spec JSON (target area, priority, rules)    │
 │  Multi-Camera Real-Time Dashboard (PySide6 Quad-Cam Grid, HUD Telemetry, 3D OctoMap Viewer)     │
 └───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │ Mesh Gossip (Task spec deltas + CRDT updates)
 ┌───────────────────────────────────────────────▼──────────────────────────────────────────────────┐
 │                         DISTRIBUTED SWARM BLACKBOARD (Replicated CRDT)                            │
 │  • Victim Registry (Union-Find CRDT)       • Frontier Claims & Dynamic Leases                    │
 │  • Dynamic Comms Graph Adjacency           • Agent Health & Battery State                        │
 └───────▲───────────────────────────────────────────────────────────────────────────▲───────────────┘
         │ (Peer-to-Peer Ad-hoc Wi-Fi / Mesh Link)                                     │
 ┌───────┴─────────────────────── UAV Stack (×4) ───────────────────────┐   ┌─────────┴───── UGV Stack (×2) ──────┐
 │ • Sensors: Low-light RGB + LWIR Thermal + 3D LiDAR + IMU / Baro      │   │ • Sensors: Microphone Array (DoA),  │
 │ • Edge Perception: YOLOv8 Detector + 3D Raycasting Ground Lift       │   │   2D/3D LiDAR, Wheel Odometry, RGB  │
 │ • Re-ID Engine: 128-d Appearance Embedding + Covariance Propagation  │   │ • Acoustic Mapper: Anisotropic      │
 │ • Tactical Planner: GraphSAGE GNN Policy over Dynamic Peer Graph     │   │   Evidence Map Splatting (Trapped)  │
 │ • Safety Filter: 6.5 m Inter-Drone Buffer + ORCA / APF Avoidance     │   │ • Nav2 Stack: Theta* Global Planner │
 │ • Autopilot: PX4 Offboard Mode (50 Hz Position/Velocity Setpoints)   │   │   + DWB Local Controller            │
 └──────────────────────────────────────────────────────────────────────┘   └─────────────────────────────────────┘
```

---

## Algorithmic Framework

### 1. Three-Layer Architecture
Strict separation of concerns ensures rock-solid fail-safety:
- **Layer 1: High-Level LLM Commander (Advisory, 10–30s):** Interprets natural language mission directives (e.g., *"Sweep northern sector with priority on collapsed warehouse"*) and translates them into structured JSON task specifications. **The LLM is strictly decoupled from the real-time flight loop.**
- **Layer 2: Tactical MARL + GNN Allocator (1 Hz + Event-Triggered):** GraphSAGE message passing over ad-hoc peer communication links predicts utility scores across candidate exploration frontiers. Automatically falls back to geometric sector partitioning if communication is severed.
- **Layer 3: Real-Time Flight & Obstacle Control (50 Hz):** Low-level trajectory tracking, Artificial Potential Field (APF) obstacle repulsion, 6.5 m inter-drone separation, and PX4 offboard setpoint publishing.

```
Mission Prompt ──► [LLM Commander] ──► Task Spec JSON
                                             │
                        Dynamic Comms Adjacency Graph
                                             ▼
                 [GraphSAGE GNN Tactical Allocator] ◄── Fallback to Frontier Heuristic
                                             │
                                  Candidate Waypoint
                                             ▼
                       [APF & 6.5m Collision Filter]
                                             │
                               50 Hz Setpoints
                                             ▼
                          [PX4 Autopilot Offboard]
```

---

### 2. Decentralized MARL + GraphSAGE GNN
The tactical coordination engine is trained via **Multi-Agent PPO (MAPPO)**:
- **Centralized Training with Decentralized Execution (CTDE):** A centralized critic observes the joint state during training; decentralized actors execute independently using only local observations and 1-hop peer communication embeddings.
- **Observation Space:** 
  - Agent Features ($F_A = 109$): Agent type one-hot, normalized battery, local coordinates, heading, detected survivors count, acoustic hint vector, nearest-frontier distance, and a $7 \times 7$ 2-channel local occupancy/victim patch.
  - Candidate Goals ($N = 20$, $F_G = 5$): Distance, direction, frontier cluster size, information gain prior, and type compatibility.
- **Graph Message Passing:**
  $$\mathbf{h}_i^{(l+1)} = \sigma \left( \mathbf{W}_1 \mathbf{h}_i^{(l)} + \mathbf{W}_2 \cdot \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} w_{ij} \mathbf{h}_j^{(l)} \right)$$
  where $w_{ij}$ is dynamic communication channel quality based on physical distance and RF attenuation.

---

### 3. Multi-Modal Sensing (Thermal + Acoustic)
In disaster zones, victims are either visible on the surface or buried beneath concrete rubble:
- **LWIR Thermal Sensing (UAVs):** Conical field-of-view (FOV) raycasting with Line-of-Sight (LOS) occlusion checks. Detects survivors in open spaces or under light dust.
- **Acoustic Sensing & Triangulation (UGVs):** Microphone array measures Direction-of-Arrival (DoA, $\pm 8^\circ$) combined with loudness-based range decay ($I = I_0 / (1 + \alpha d^2)$). Anisotropic evidence grids splat sound observations into 2D Gaussian ellipses, enabling multi-robot cross-bearing triangulation to pinpoint buried victims without visual contact.

---

### 4. Victim Re-ID & CRDT Registry (Counting Exactly Once)
To prevent victim overcounting when multiple drones observe the same person from different perspectives:
1. **Raycasting 3D Ground Lift:** Drone pinhole camera model inverts pixel coordinates $(u, v)$ to local ground coordinates $(x, y, 0)$ via altitude and gimbal attitude:
   $$\mathbf{P}_{\text{ground}} = \mathbf{P}_{\text{drone}} + z_{\text{drone}} \cdot \mathbf{R}_{\text{body}}^{\text{world}} \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
2. **Covariance Propagation:** Detection uncertainty $\Sigma_{\text{ground}} = \mathbf{J} \Sigma_{\text{pixel}} \mathbf{J}^T$ grows quadratically with altitude and tilt angle.
3. **Re-ID Appearance Embedding:** Fast 128-dimensional embedding vector compares color histograms and visual features.
4. **Hysteresis Hungarian Matching:** Merges observations into existing tracklets if spatial distance $< \tau_{\text{pos}}$ and cosine similarity $> \tau_{\text{app}}$.
5. **CRDT Union-Find Registry:** Gossip-based state-based CRDT replicates victim registries across all robots. Partitioned sub-swarms reconcile monotonically upon reconnection with zero duplicate IDs.

---

### 5. Collision Avoidance & Safe Offboard Motion
- **6.5 m Dynamic Separation Buffer:** Real-time spatial hash prevents UAV trajectory crossings.
- **Artificial Potential Fields (APF):** Repulsive force fields push agents away from building façades and peer vehicles.
- **Universal Offboard Targeting:** Clean 50 Hz trajectory setpoints (`TrajectorySetpoint.msg`) published via Micro XRCE-DDS to PX4 offboard flight controllers.

---

## Real-Time Visual GUI & Quad-Cam Telemetry

The real-time ground control station (`drone_agent/map_gui_node.py`) is engineered for multi-agent situational awareness:
- **2×2 Quad-Cam Grid:** Synchronously renders live video streams from all 4 UAVs via `ros_gz_bridge`.
- **HUD Telemetry Overlay:** Real-time altitude, ground speed, battery percentage, heading compass, and current GNN allocation state for each drone.
- **One-Click Camera Enlargement:** Top-bar selector buttons (`[4-CAM GRID]`, `[UAV-1 CAM]`, `[UAV-2 CAM]`, `[UAV-3 CAM]`, `[UAV-4 CAM]`) or direct clicking on camera feeds to maximize view.
- **Page 2 3D Collaborative OctoMap:** Global 3D occupancy voxel grid showing real-time point clouds, merged frontiers, and survivor waypoints with Picture-in-Picture (PiP) feeds.

---

## Repository Organization

```
Team-LOVE-AI-MANTHAN-2.0-PS-5/
├── README.md                      # Master documentation (architecture, setup, guide)
├── SAR_SWARM_SPEC.md              # Complete engineering specification & math derivations
├── AGENTS.md                      # Design principles, non-negotiables, working rules
├── PROGRESS.md                    # Verification log across development phases (P0 to P3, R)
├── THIRD_PARTY.md                 # Attribution & open-source license audits
├── build_px4.sh                   # Automated PX4 SITL build configuration for Gazebo Harmonic
├── download_models.sh             # Gazebo Fuel 3D disaster assets downloader
├── yolov8n.pt                     # Edge YOLOv8 object detection model weights
│
├── sim/                           # Pure-Python Lightweight Multi-Agent Simulator
│   ├── env.py                     # Heterogeneous grid world (UAVs + UGVs, battery, comms)
│   ├── map_gen.py                 # Procedural collapsed building & rubble generator
│   ├── sensors.py                 # Multi-modal thermal & acoustic sensing models
│   ├── baselines.py               # Exploration baselines (Random, Nearest Frontier, Explore Lite)
│   ├── reid.py                    # CRDT victim registry, Hungarian Re-ID, hysteresis linking
│   ├── reid_sensing.py            # Noisy VIO drift, hot-rock false positives, bearing acoustics
│   ├── run_baselines.py           # Multi-agent benchmark runner & metric exporter
│   ├── run_reid.py                # Re-ID de-duplication evaluation benchmark
│   ├── visualize.py               # Animated episode GIF/PNG renderer
│   └── metrics.py                 # Coverage %, time-to-first-victim, collision statistics
│
├── train/                         # MARL + Graph Neural Network Framework
│   ├── model.py                   # GraphSAGE Actor-Critic architecture (TorchScript exportable)
│   ├── features.py                # Graph node/edge feature extractor (FA=109, FG=5)
│   ├── ppo.py                     # Multi-Agent PPO (MAPPO) with Centralized Critic & GAE
│   ├── run_train.py               # Training loop with GNN vs No-GNN ablation
│   └── evaluate.py                # Policy evaluation over held-out disaster seeds
│
├── results/                       # Benchmarks, Learning Curves & Checkpoints
│   ├── checkpoints/               # Trained GNN weights (gnn.pt, gnn_actor.pt)
│   ├── p1_baselines.png           # P1 baseline exploration curves
│   ├── p1_episode.gif             # Animated multi-agent SAR exploration episode
│   ├── p2_learning_curves.png     # P2 MAPPO GNN training reward curves
│   └── p2_summary.json            # Numerical comparison metrics
│
├── tests/                         # Unit & Integration Verification Suite (47/47 Passing)
│   ├── test_reid.py               # 29 tests: CRDT registry, Hungarian matching, DoA acoustics
│   ├── test_victim_fusion.py      # 18 tests: 3D raycasting ground lift & covariance propagation
│   └── ros_integration_victim_registry.py # ROS 2 node pipeline verification
│
├── LLM-controlled-drone/          # ROS 2 Jazzy + PX4 SITL + Gazebo Harmonic Platform
│   ├── start_all.sh               # Master orchestrator (4 PX4 instances, bridges, GUI)
│   ├── setup_workspace.sh         # Dependency installer & colcon workspace build
│   ├── launch_drone.sh            # Single-drone launcher helper
│   ├── worlds/
│   │   ├── drone_demo.sdf         # 3D disaster world (collapsed buildings, rubble, victims)
│   │   └── baylands_demo.sdf      # Outdoor test world
│   └── src/
│       ├── drone_agent/           # Custom ROS 2 Swarm Coordination Package
│       │   ├── drone_agent/
│       │   │   ├── brain_node.py              # Orchestrator linking task spec to state machine
│       │   │   ├── swarm_gnn_allocator.py     # Decentralized GraphSAGE goal allocator
│       │   │   ├── command_translator.py      # Offboard setpoint generator & 6.5m buffer
│       │   │   ├── map_gui_node.py            # Quad-cam real-time HUD GUI & OctoMap viewer
│       │   │   ├── yolo_detector.py           # Real-time YOLOv8 person detection node
│       │   │   ├── victim_geometry.py         # Raycasting pixel-to-ground 3D projection
│       │   │   ├── victim_fusion.py           # Multi-agent victim registry & gossip sync
│       │   │   ├── victim_registry_node.py    # /swarm/victims publisher node
│       │   │   └── llm_client.py              # Groq / Ollama natural language interface
│       │   ├── launch/
│       │   │   └── drone_agent.launch.py      # 4-UAV camera bridge & agent launcher
│       │   ├── package.xml
│       │   └── setup.py
│       └── px4_msgs/              # PX4 ROS 2 message definitions (v1.14/v1.15)
│
└── dashboard/                     # Web-based Fleet Telemetry Dashboard
    └── app.py                     # Streamlit / Web monitoring interface
```

---

## Prerequisites & Environment Setup

### System Requirements
- **OS:** Ubuntu 24.04.5 LTS (Noble) or WSL2 with GUI support (`DISPLAY=:0`)
- **ROS 2:** Jazzy Jalisco (`desktop` install)
- **Gazebo:** Gazebo Harmonic (`gz sim 8.15+`)
- **Python:** 3.10 – 3.12 with PyTorch, NumPy, OpenCV, and Ultralytics

### Quick Installation

```bash
# 1. Update system packages
sudo apt update && sudo apt install -y \
    python3-pip python3-colcon-common-extensions \
    ros-jazzy-desktop ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image ros-jazzy-cv-bridge

# 2. Clone this repository
git clone https://github.com/TejasGogawale/Team-LOVE-AI-MANTHAN-2.0-PS-5.git
cd Team-LOVE-AI-MANTHAN-2.0-PS-5

# 3. Install Python dependencies
pip install torch numpy matplotlib opencv-python ultralytics pytest
```

---

## Step-by-Step Execution Guide

### Quickstart 1: Pure-Python Lightweight Simulator (No ROS Required!)
Run the high-speed multi-agent simulation benchmark in seconds. Compares **Random**, **Nearest Frontier**, and **Explore Lite** exploration algorithms across heterogeneous fleets (drones + rovers) in procedurally generated disaster ruins:

```bash
python3 -m sim.run_baselines --seeds 0 1 2 --drones 2 --rovers 2 --max-steps 300
```

To generate an animated visual recording of the swarm navigating rubble:
```bash
python3 -m sim.visualize --seed 42 --drones 2 --rovers 2 --out results/p1_episode.gif
```

---

### Quickstart 2: MARL + GNN Training & Evaluation
Train the decentralized GraphSAGE policy using Multi-Agent PPO (MAPPO):

```bash
# Run training with GraphSAGE message passing (CPU or CUDA GPU)
python3 -m train.run_train --iterations 40 --drones 2 --rovers 2

# Evaluate trained GNN vs No-GNN ablation vs Baselines
python3 -m train.evaluate --checkpoint results/checkpoints/gnn.pt --seeds 10 11 12
```

---

### Quickstart 3: Victim Re-ID & Fusion Verification Suite
Execute the 47-test automated verification suite testing 3D raycasting projection, covariance propagation, Hungarian matching, DoA acoustic triangulation, and CRDT victim registry synchronization:

```bash
# Run all unit tests
python3 -m pytest tests -q

# Run the synthetic Re-ID multi-agent benchmark (10 disaster seeds)
python3 -m sim.run_reid --seeds 0 1 2 3 4 5 6 7 8 9 --drones 4 --rovers 2
```

---

### Quickstart 4: Full 4-Drone Gazebo Harmonic Swarm & Quad-Cam GUI
Launch the complete end-to-end multi-robot system with 4 PX4 SITL instances in Gazebo Harmonic, real-time ROS 2 camera bridges, YOLOv8 victim detectors, decentralized GNN allocators, and the real-time Quad-Cam GUI:

```bash
# 1. Download disaster models (collapsed buildings, victims, vehicles)
bash download_models.sh

# 2. Build the ROS 2 workspace
cd LLM-controlled-drone
bash setup_workspace.sh

# 3. Source ROS 2 and workspace
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# 4. Launch the entire 4-drone autonomous swarm with Quad-Cam GUI
bash start_all.sh
```

**GUI Controls:**
- `[4-CAM GRID]`: Returns to the 2×2 quad-camera live telemetry view.
- `[UAV-1 CAM]` .. `[UAV-4 CAM]`: Enlarges the corresponding drone's live feed.
- Click directly on any video feed to maximize that camera stream.
- Switch to **Page 2** for the 3D Collaborative OctoMap view.

---

## Experimental Results & Benchmarks

### 1. Multi-Agent Exploration Baseline Comparison
Evaluated on $40 \times 40$ collapsed structure maps with 6 buried/open victims (2 Drones + 2 Rovers, averaged over 3 seeds):

| Exploration Strategy | Coverage (%) | Survivors Found (%) | Time to First Victim (steps) | Time to All Victims (steps) | Collisions |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Walk** | 78.9% | 100% | 5 | 104 | 4.3 |
| **Nearest Frontier** | 75.3% | 100% | 4 | 56 | 2.7 |
| **Explore Lite (Information Gain)** | **81.8%** | **100%** | **4** | **76** | **2.0** |
| **SwarmRescue (MARL + GraphSAGE)** | **88.4%** | **100%** | **3** | **48** | **0.8** |

---

### 2. Victim Re-ID & De-duplication Ablation (Counting Exactly Once)
Ablation on held-out seeds (4 UAVs + 2 UGVs, simulated VIO drift $\sigma = 0.03$/step):

| Configuration | Resolvable Maps (Exact Count) | Total Duplicates Across 20 Seeds | False Positives |
|---|:---:|:---:|:---:|
| **Naive Per-Robot Sum** | 1 / 8 | 49 duplicates | 0 |
| **Position-Only Tracking** | 5 / 8 | 8 duplicates | 0 |
| **Full SwarmRescue Re-ID (Appearance + DoA + CRDT)** | **7 / 9** | **0 duplicates** | **0** |

---

## Design Principles & Honesty Commitments

1. **Decentralized Execution:** No global coordinator node in the loop. Every drone runs its own GNN inference, and agents only communicate within their local wireless range.
2. **GPS-Denied Truthfulness:** Positions originate from local odometry/VIO frames. Simulated VIO drift is explicitly added and benchmarked rather than assuming ground truth.
3. **Decoupled LLM Commander:** High-level natural language control is advisory. If the LLM connection is severed or delayed, the swarm continues full autonomous execution via the GNN/heuristic allocator.
4. **Reproducible & Verified:** All 47 test cases run without proprietary dependencies or physical hardware; every baseline and metric reported is reproducible from the commands in this repository.

---

## Authors & Acknowledgments

**Team LOVE** — AI MANTHAN 2.0 (Problem Statement 5)  
Acropolis Institute of Technology and Research, Indore  

*Developed with pride for autonomous disaster response robotics.*
