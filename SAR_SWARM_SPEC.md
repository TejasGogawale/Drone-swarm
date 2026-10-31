# SAR Swarm — Architecture & Algorithmic Specification
**Fleet:** 4 UAV (`uav1..4`) + 2 UGV (`ugv1..2`) · GPS-denied · decentralized · ROS 2 Jazzy · Gazebo Harmonic · PX4 SITL
**Status key:** ✅ exists in repo (per PROGRESS.md) · 🟡 partial · ⬜ proposed in this spec

Three requirements drive every decision below:

| # | Requirement | Where it is solved |
|---|---|---|
| R1 | **A human is counted exactly once**, no matter how many drones/rovers see them, from how many angles, or how often they re-enter view | §3, §7.2 |
| R2 | **A much better 3D map** (dense, loop-closed, merged across agents) so rovers can navigate reliably | §4, §5, §7.3 |
| R3 | **One shared "swarm mind"**: one command reaches everyone, all act at once, and neighbours absorb disturbances locally | §6 |

---

## 0. Honest baseline (what exists vs. what this adds)

| Capability | Today | This spec adds |
|---|---|---|
| Lightweight sim, 2D grid, thermal+acoustic, baselines | ✅ `sim/` | — |
| MARL+GNN (GraphSAGE, MAPPO, centralized critic), rule fallback | ✅ `train/`, `swarm_gnn_allocator.py` | goal-set generation from 3D Voronoi cells |
| 4× PX4 x500 mono-cam in Gazebo, per-drone camera bridge, quad-cam GUI | ✅ `start_all.sh`, `drone_agent.launch.py` | thermal+RGB fusion, LiDAR |
| Collaborative OctoMap view | 🟡 GUI page 2 | TSDF/ESDF fusion, loop closure, map-merge, traversability layers |
| Victim detection (YOLO) | 🟡 `yolo_detector.py` | tracker + Re-ID + global registry (R1) |
| Collision avoidance | 🟡 6.5 m separation buffer | ORCA + APF + min-snap (d_min ≥ 2 m, not 6.5) |
| LLM commander | 🟡 Groq cloud, advisory | **local** LLM, typed task spec, gossip broadcast (R3) |

Constraint carried from `AGENTS.md` and kept here: **the LLM is never in the motion loop; the swarm works with it off; no robot reads ground truth at runtime; simulated VIO is labelled as such.**

> **Note on "same local AI" (R3):** every robot runs an *identical* policy bundle and a *replicated* world model (CRDT, eventually consistent). That is what makes them behave as one mind. A single remote brain would be a single point of failure and useless in a collapsed building with no uplink. The current Groq cloud LLM should become local-first (Ollama on a base station) with Groq only as an optional fallback — a disaster zone will not have internet.

---

## 1. System architecture

```
 ┌────────────────────────────── OPERATOR / BASE (optional, loses nothing if offline) ─────────────┐
 │  Local LLM commander (Ollama)  ─►  Task Spec JSON (validated, signed, seq#)                      │
 │  Ops UI: disaster map, victim list, per-robot status                                             │
 └───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │  mesh gossip (task + map-tile + registry deltas)
 ┌───────────────────────────────────────────────▼──────────────────────────────────────────────────┐
 │                         SWARM BLACKBOARD  (replicated on every robot, CRDT)                       │
 │  task_set (LWW-reg) · victim_registry (union-find CRDT) · map_tiles (versioned) · agent_health     │
 │  frontier_claims (OR-set + lease) · hazard_layer · comms_graph                                     │
 └───────▲───────────────────────────────────────────────────────────────────────────▲───────────────┘
         │ same code, same models, on every robot                                      │
 ┌───────┴─────────────────────── per-UAV stack (×4) ─────────────────────┐   ┌──────┴──── per-UGV stack (×2) ────┐
 │ Sensors: LWIR + low-light RGB + 3D LiDAR + IMU (+ baro, rangefinder)   │   │ Sensors: 2D/3D LiDAR, RGB, IMU,   │
 │   │                                                                    │   │   wheel odom, (thermal, mic)      │
 │   ├─► Time-sync + calib ─► Edge inference (YOLOv11n TRT-FP16)          │   │   ├─► LIO (local odom/map)         │
 │   │       thermal detector + RGB detector + dual-spectrum verifier     │   │   ├─► Map consumer (traversability │
 │   │       + Re-ID embedder (MobileNet-ReID, 128-d)                      │   │   │    tiles from UAVs + own LiDAR)│
 │   ├─► LIO / VIO (LIO-SAM or FAST-LIO2; ORB-SLAM3 as backup)            │   │   ├─► Nav2: global (Theta*) +      │
 │   │       ─► pose graph ─► loop closure ─► map-merge (inter-robot)     │   │   │    local MPC/DWB on own LiDAR  │
 │   ├─► Mapper: nvblox/Voxblox TSDF ─► ESDF + mesh + OctoMap             │   │   ├─► GNN policy (UGV head)        │
 │   │       ─► traversability tiles (§5) ─► broadcast                    │   │   └─► Victim approach + payload    │
 │   ├─► Tracker: ByteTrack (per-camera) ─► 3D lift ─► local tracks       │   └────────────────────────────────────┘
 │   ├─► GNN policy (UAV head) over comms graph ─► goal                   │
 │   ├─► Planner: min-snap in safe-flight-corridor ─► ORCA+APF filter     │
 │   └─► PX4 offboard (position/velocity/acc setpoints @ 50 Hz)           │
 └────────────────────────────────────────────────────────────────────────┘
```

**Pipeline in one line:** Sensors → time-sync → edge inference → SLAM/pose-graph → TSDF/ESDF map → blackboard → GNN planner → min-snap + ORCA/APF → PX4 / Nav2 → UGV reaches victim.

### 1.1 Rates and latency budget

| Stage | Rate | Budget | Where |
|---|---|---|---|
| IMU / LIO odometry | 200 Hz / 10–20 Hz | <20 ms | UAV/UGV |
| Detector (thermal+RGB, 640×512) | 15–30 Hz | <35 ms (TRT FP16, YOLO11n) | UAV |
| Tracker + 3D lift + Re-ID embedding | 10–15 Hz | <30 ms | UAV |
| Mapper (TSDF integrate + ESDF) | 5 Hz | <150 ms | UAV |
| Traversability tile delta broadcast | 1–2 Hz | <500 ms | UAV→mesh |
| GNN goal selection | 1 Hz + event-triggered | <100 ms | all |
| Min-snap re-solve | 5–10 Hz | <20 ms | UAV |
| ORCA+APF filter, setpoints | 50 Hz | <5 ms | UAV |
| LLM commander | on demand | 5–30 s, advisory | base |

### 1.2 Compute and link assumptions (state these in the demo)
- Per UAV: Jetson Orin NX 16 GB (or equivalent). Per UGV: Orin NX + x86 optional for Nav2.
- Mesh: 5 GHz or 900 MHz mesh, ~1–2 Mbps per link usable, range-limited and lossy. **Everything on the blackboard must degrade gracefully at 10× less bandwidth.**
- Dev laptop (7.5 GB RAM, RTX 2050 4 GB): live Gazebo is limited; run 4 UAV + 2 UGV only with headless cameras at reduced rate, otherwise use the lightweight sim for the 6-robot scalability results. Report which is which.

---

## 2. Collaborative exploration & search

### 2.1 3D dynamic partitioning (Voronoi, coverage-weighted)
- Search volume discretised into a 3D **cell grid** (UAV: 1 m; rover plane: 0.5 m), each cell has state `unknown / free / occupied` and `p_unexplored`.
- Each UAV owns the cells closest to it under a **weighted** metric (power diagram): distance divided by a capability weight `w_i = battery_i · speed_i · sensor_i`. Weak or low-battery UAVs get smaller cells automatically.
- Every ~5 s (or on agent loss) run **one Lloyd step** with density `ρ(c) = p_unexplored(c) · priority(c)`; centroids drive the sweep waypoints. Altitude layers: mappers fly high for coverage, searchers fly low for small-target detection.
- UGVs partition the **traversable 2.5D surface** only, and are assigned what UAVs can't see (under rubble, indoors, narrow gaps).

### 2.2 Hybrid grid-sweep + frontier search
- **Global:** boustrophedon sweep (lawnmower) inside the agent's Voronoi cell, line spacing set by thermal footprint at current altitude (≥30 % overlap).
- **Local:** in clutter or when the sweep line hits unknown/occupied space, switch to **frontier exploration**: frontiers are free cells adjacent to unknown cells, clustered (DBSCAN, 1.5 m) into candidate goals.
- The GNN scores the candidate goals (already implemented: `N_GOALS=20`, `FG=5`); a geometric heuristic (§7.1) is the fallback and also the prior fed to the GNN.

### 2.3 Dynamic re-planning
Triggers: heartbeat lost >3 s (agent presumed gone), battery below return threshold (reserve = path-home energy × 1.3), new map info invalidates a claim, comms partition.
Action: the owner's cells and frontier claims are released (lease expiry), then re-auctioned among neighbours (§7.1). **Leases + hysteresis prevent thrash.**

---

## 3. Perception and victim detection

### 3.1 Sensing and calibration
- LWIR (e.g. 640×512) + low-light RGB, hardware-triggered, extrinsically calibrated (thermal–RGB homography at working range, refined with a heated checkerboard).
- **Honest physics:** a clothed person seen from 10–30 m will *not* read 36.5–37.5 °C on a radiometric camera; apparent temperature is far lower and depends on clothing, distance, and emissivity. So the detector uses **relative contrast** against local background plus shape, with the 36.5–37.5 °C figure as a calibration-range prior for *exposed skin close-up*, not an absolute threshold. State this in the report.

### 3.2 Detectors
- Thermal and RGB each run YOLOv11n (TensorRT FP16), trained for aerial small-target persons (slicing/tiling inference or a P2 head at 1280 input if needed). Classes: `person_upright`, `person_prone`, `person_partial` (posture). Include hard negatives: warm rocks, engines, exhaust, sun-heated debris.

### 3.3 Dual-spectrum verification (false-positive killer)
A thermal detection becomes a **candidate** only. It is *confirmed* when:
1. an RGB/night-vision detection projects within the thermal box (IoU ≥ 0.3 after homography), **or** the thermal blob passes shape/size priors at known range and persists, **and**
2. it survives M-of-N frames (e.g. 4 of 6) from ≥1 sensor, **and**
3. a "heat-source" classifier doesn't flag it (engine/exhaust patterns: very high contrast, rectangular, tied to a vehicle).
Buried victims with only acoustic evidence are marked `acoustic_only` with lower confidence and tasked to a UGV.

---

## 4. Distributed localization, tracking and Re-ID — humans are counted once (R1)

### 4.1 From pixel to world
For each tracked detection, lift to a world estimate `x ∈ ℝ³` with covariance `Σ`:
- ray-cast the bbox foot/centre against the **current ESDF/mesh** (preferred), or use LiDAR points in the frustum, or triangulate across two views;
- propagate pose uncertainty from SLAM into `Σ` (important: a bad pose should widen the gate, not create a ghost victim);
- publish the *observation*, not just a point: `(t, x, Σ, embedding e, thermal signature, class, confidence, agent_id)`.

### 4.2 Per-agent tracking
ByteTrack per camera stream (high+low-score association) gives short-term track continuity in image space; each tracklet is lifted and filtered with a 3-state constant-position EKF (victims are mostly static; small process noise).

### 4.3 Global de-duplication (the core of R1)
Association cost between a new observation `o` and a registry victim `v`:

```
d_M²   = (x_o − x_v)ᵀ (Σ_o + Σ_v)⁻¹ (x_o − x_v)         # spatial, Mahalanobis
d_app  = 1 − cos(e_o, e_v)                                 # appearance (CosFace-trained MobileNet-ReID)
d_th   = |thermal_sig_o − thermal_sig_v| / scale           # thermal signature
cost   = λ1·d_M² + λ2·d_app + λ3·d_th
gate   : d_M² < χ²₃(0.99) = 11.34   AND   (d_app < τ_app  OR  n_obs(v) < 3)
```
Solved with the **Hungarian algorithm** per batch; unmatched observations spawn *tentative* victims.

Rules that guarantee a human is never recounted:
1. **Count = number of confirmed union-find components.** It is a *derived* quantity from the registry, never an incrementing counter. Merging two records can only lower the count; seeing the same person again can't raise it.
2. **IDs are never reissued.** A victim keeps its canonical ID (`min` of member UUIDs) forever, even after leaving all views.
3. **Re-entry uses a time-inflated gate:** the gate grows with `Δt` since last seen (`Σ_v += q·Δt`) so a victim who was lost and re-detected after a re-position still matches the old record rather than creating a new one.
4. **Separation safeguard:** two distinct people can be <1 m apart (a rescued person next to a casualty). Don't merge on position alone — require `d_app` agreement once both records have ≥3 observations; if embeddings disagree strongly, keep them separate and flag for operator review.
5. **DBSCAN safety net** (eps ≈ 1.5 m, min_samples = 1) over confirmed victim positions every few seconds: any two records in one cluster with compatible embeddings are merged; clusters with incompatible embeddings are flagged `ambiguous`, never auto-counted twice.
6. **Rescuers and the robots themselves** are excluded: a person class "responder" (hi-vis/helmet or marked by operator) is tracked but not counted as a victim; robots are masked by their own shared pose.
7. **Merge/split are logged** (who, when, why) so the final count is auditable — consistent with the project's honesty rule.

### 4.4 Distributed consensus and neighbour convergence
- The registry is a **CRDT** (union-find with deterministic canonical ID + per-field last-writer-wins on position/embedding by observation count). Agents gossip *deltas*; duplicates/reorderings are harmless; partitions heal by merging.
- **Trigger:** when a victim reaches confidence ≥ θ_hi from one agent, a `VICTIM_LOCK_REQUEST(victim_id, x, Σ)` goes to neighbours. Each neighbour decides *locally*, via its GNN with this as a high-priority goal, whether to take a second view: viewpoint chosen to be ≥45° from existing bearings and reachable within battery. Max 2 helpers, so the rest keep searching. No ground station needed.
- A UGV is dispatched when the registry marks a victim `confirmed` and traversability to it exists (or after a UAV scouts a route).

---

## 5. 3D reconstruction and the disaster map (R2)

### 5.1 Why the current map isn't good enough
A single-resolution OctoMap from mono-camera drones gives sparse, noisy occupancy with drift and no ground-truth for slope/step height. Rovers need **geometry they can trust**.

### 5.2 Upgraded mapping stack
| Layer | Method | Purpose |
|---|---|---|
| Odometry | LIO (LIO-SAM / FAST-LIO2) on 3D LiDAR + IMU; ORB-SLAM3 / RTAB-Map as visual backup for LiDAR-poor scenes (smoke, glass) | GPS-denied pose, 200 Hz IMU-rate propagation |
| Drift control | Local submaps + **pose graph**; loop closure by scan-context / Cartographer-style matching; **inter-robot** closure when two agents overlap | Globally consistent map |
| Dense geometry | **TSDF fusion** (nvblox on GPU, or Voxblox) with 5 cm voxels near the agent, coarser far away | Dense surfaces instead of speckle points |
| Derived products | **ESDF** (distance to nearest obstacle) · surface **mesh** (marching cubes) · coloured/thermal point cloud for the operator | Planning clearance, visualisation |
| Dynamic objects | Mask detected humans/robots **out of the static map** (via the detector's masks), integrate them in the victim layer instead | Stops people becoming "walls" or ghost blobs |
| Cleanup | Statistical outlier removal, voxel downsample, ground segmentation (patchwork++), confidence per voxel (observation count) | Fewer spurious obstacles |

Drones fly **map-first passes** at two altitudes (high sweep for coverage, low pass over detected approach corridors for detail), with the LiDAR tilted to see the ground ahead.

### 5.3 The disaster map layout (layered 2.5D + 3D)
One shared, tile-based map (e.g. 8 m × 8 m tiles at 0.1 m resolution), each layer separately versioned:

| Layer | Content | Producer | Consumer |
|---|---|---|---|
| L0 Occupancy / TSDF | 3D free / occupied / unknown | UAVs, UGVs | planners |
| L1 Elevation | height map (max & mean z), mesh | UAVs | rovers |
| L2 Traversability cost | per-cell cost for **rover** and for **crawl-gap** class, with *unknown* kept distinct from *free* | derived (§7.3) | Nav2 |
| L3 Clearance (ESDF) | distance to obstacles in 3D | mapper | UAV planner (corridors), UGV |
| L4 Coverage / information | `p_unexplored`, last-seen time | all | frontier allocation |
| L5 Victims | confirmed/tentative victims, confidence, status (seen / approached / reached) | registry | all, operator |
| L6 Hazards | unstable structure, fire/heat, water, no-fly | operator + perception | planners |
| L7 Comms | measured link quality, relay positions | comms-bridge | GNN, relay placement |

Operator view: a top-down 2.5D "disaster map" with coverage heat, traversability (green/amber/red/black-for-unknown), victim markers with ID and confidence, drone/rover icons, and links drawn with quality — plus a 3D mesh view.

---

## 6. Shared intelligence: "swarm mind" (R3)

### 6.1 Command path — one ask, everyone acts at once
```
Operator: "Search the east wing, prioritise the stairwell, send a rover to anyone found."
   │
   ▼  Local LLM  (structured output, JSON-schema constrained; never raw text to motors)
TaskSpec {
  id, seq, issued_at,
  area:       polygon / named region (resolved against the map),
  objective:  SEARCH | MAP | RELAY | DELIVER | RETURN | HOLD,
  priority:   0..1,
  constraints:{ max_alt, no_go[], min_battery, speed_cap },
  rules:      [ "ugv_approach_confirmed_victims", ... ]
}
   │  validate (schema + safety envelope) ─► sign ─► gossip
   ▼
Every robot's blackboard receives TaskSpec (LWW by seq). Each robot's GNN consumes it as extra
goal-features and *decides its own role*. No per-robot micromanagement from the LLM.
```
Result: all 6 act simultaneously on the same intent, but each picks its share (Voronoi + GNN), so there is no single choke point. If the link to base drops, robots continue the last TaskSpec; a rule-based fallback (already in repo) runs if the LLM is unavailable.

### 6.2 Disturbance handling — neighbours absorb it locally
Disturbance = lost agent, low battery, new obstacle, comms partition, wind gust, hazard appears.
1. Detect locally (heartbeat timeout, battery model, map change, tracking error).
2. Raise an **event** only to the **2-hop neighbourhood** (not the whole swarm).
3. Neighbours run a **local re-auction** (§7.1) of the orphaned cells/frontiers/victim-lock; the GNN provides the scores; ties broken by `agent_id`.
4. If a partition splits the swarm, each fragment continues independently with its own copy of the blackboard; on reconnection the CRDTs merge. A relay drone may be tasked to restore connectivity when the GNN's comms-aware reward indicates it.
5. The LLM only receives a *summary* ("UAV3 lost at east stair, cells re-assigned to UAV2/UAV4") and may advise; it can't block recovery.

### 6.3 How the GNN helps
Already implemented: GraphSAGE over the comms graph, per-type encoders (drone/rover), shared GNN, per-type scorer heads, centralized critic (MAPPO) with decentralized execution.
Extensions (⬜): add features — Voronoi-cell residual, victim-lock requests, link quality, relay need, map-tile staleness, wind estimate; add reward terms for **de-duplicated victim discovery**, rover-reachability of found victims, connectivity, and low collision/jitter. Always report the no-GNN ablation and baselines honestly (as `evaluate.py` already does).

### 6.4 Roles for 4 UAV + 2 UGV (dynamic, not fixed)
| Agent | Default role | Can switch to |
|---|---|---|
| uav1, uav2 | **Mappers** (high altitude, LiDAR-forward, coverage) | searcher, relay |
| uav3, uav4 | **Searchers** (low altitude, thermal+RGB, small-target) | second-view helper, relay |
| ugv1 | **Responder**: goes to confirmed victims, carries payload | scout under rubble |
| ugv2 | **Scout / relay**: narrow-gap & acoustic search, mesh relay | responder |

---

## 7. Algorithms

### 7.1 Dynamic frontier-search allocation

```python
# Runs on every agent, event-triggered + every 2 s. Uses only local copy of the blackboard.
def allocate(self, bb, now):
    # 1. Who is alive / reachable?
    alive = {a for a in bb.agents if now - a.last_heartbeat < T_DEAD and a.battery > a.return_threshold()}
    # 2. Release stale claims (leases) -> orphaned work becomes available again
    for claim in bb.frontier_claims:
        if claim.owner not in alive or claim.lease_expired(now):
            bb.frontier_claims.release(claim)

    # 3. Candidate goals = frontier clusters  ∪  sweep waypoints in my Voronoi cell
    F = cluster_frontiers(bb.map.free_unknown_boundary(), eps=1.5)
    F += sweep_waypoints(my_cell=weighted_voronoi(bb, alive)[self.id])
    F = [f for f in F if not bb.frontier_claims.held_by_other(f)]

    # 4. Score: information gain / cost, with learned prior from the GNN
    for f in F:
        gain   = unknown_volume_in_sensor_footprint(f, self.sensor)
        cost   = path_length(self.pos, f, bb.map.esdf, self.type) / self.speed \
                 + energy_to_return(f, self.battery)                # reject if cannot get home
        reach  = feasible(f, self.type)                             # rovers: traversable; UAV: clearance
        prior  = self.gnn.score(f, neighbors=bb.neighbors(self.id, hops=2))
        f.u    = (gain / cost) * (1 + β * prior) if reach else -inf
        f.u   -= γ * overlap_with_neighbors(f, bb)                  # discourage duplicate coverage

    # 5. Local auction among 2-hop neighbours (CBBA-lite): bid, resolve ties by agent_id
    my_bid = argmax(F, key=lambda f: f.u)
    winners = auction(self.id, my_bid, neighbors=bb.neighbors(self.id, hops=2), tie_break="agent_id")
    if self.id in winners:
        bb.frontier_claims.claim(my_bid, owner=self.id, lease=T_LEASE)
        # 6. Hysteresis: only switch goal if the new utility beats current by margin δ
        if self.goal is None or my_bid.u > self.goal.u * (1 + δ):
            self.goal = my_bid
    # 7. Victim-lock events pre-empt exploration if utility(second_view) > exploration utility
    if (lock := bb.pending_victim_lock_for(self.id)):
        self.goal = lock.viewpoint
```

### 7.2 Multi-angle victim Re-ID and de-duplication

```python
# Observation  = (t, x, Σ, e, thermal_sig, cls, conf, agent_id)
def on_observation(registry, obs):
    if obs.cls == "responder" or is_robot(obs.x, registry.robot_poses):      # never counted
        return
    cands = [v for v in registry.victims if gate(v, obs)]                    # §4.3 gate, time-inflated Σ
    if cands:
        C = [[cost(v, o) for o in [obs]] for v in cands]                     # batch many obs in practice
        v = cands[hungarian(C).best_for(obs)]
        if appearance_conflict(v, obs):                                      # confident different person
            v = registry.new_tentative(obs)                                  # keep separate, flag
            v.flags.add("close_to_other_victim")
        else:
            v.update(obs)                                                    # EKF pos, EMA embedding, n_obs++
    else:
        v = registry.new_tentative(obs)

    if v.state == "tentative" and v.n_obs >= N_CONFIRM and dual_spectrum_ok(v):
        v.state = "confirmed"

def periodic_consolidate(registry):                                          # every ~3 s
    pts = [v.x for v in registry.victims if v.state in ("tentative", "confirmed")]
    for cluster in dbscan(pts, eps=1.5, min_samples=1):
        group = [registry.victims[i] for i in cluster]
        if len(group) > 1:
            if all(compatible_embeddings(a, b) for a, b in pairs(group)):
                registry.union(group)                                        # canonical ID = min UUID
            else:
                for v in group: v.flags.add("ambiguous_needs_review")

def victim_count(registry):
    return len({registry.find(v.id) for v in registry.victims if v.state == "confirmed"})
    # derived, monotone under merges; observations alone can never raise it
```
Gossip: send only changed records `(canonical_id, x, Σ, e, n_obs, state, last_seen)`. Merge = union-find join, per-field pick the record with larger `n_obs`.

### 7.3 Real-time air-to-ground traversability transfer

```python
# UAV side: derive tiles from the fused map and publish only what changed.
def publish_traversability(self, map, bb, link_budget_bps):
    for tile in map.dirty_tiles():                                           # tiles whose voxels changed
        H = tile.height_map(res=0.1)                                         # z of ground surface from TSDF/mesh
        slope  = grad_mag(H);  step = max_step(H, footprint_radius=0.3)
        rough  = local_std(H, win=0.5)
        clear  = tile.esdf_above_ground(min_height=rover.height + margin)    # overhead clearance
        known  = tile.observation_count >= K                                 # unknown stays unknown
        cost   = where(~known, UNKNOWN,
                  where((slope > S_MAX) | (step > STEP_MAX) | (clear < C_MIN) | tile.hazard, LETHAL,
                        w1*slope + w2*rough + w3*inflate(tile.obstacle_dist)))
        tile.update(version=map.version(tile), cost=cost, confidence=tile.confidence, stamp=now)
    # Prioritise: tiles near rover path / victim first; send deltas (changed cells only), quantised to 8 bit
    for tile in sort_by(map.dirty_tiles(), key=priority(rover.path, victims)):
        if bytes_used >= link_budget_bps * dt: break
        mesh.send(tile.delta_compressed())                                   # zstd on RLE of cost deltas

# UGV side: consume tiles, plan, and re-plan only when it matters.
def on_tile(self, tile):
    if tile.version > self.costmap.version(tile):
        self.costmap.apply(tile)                                             # Nav2 costmap plugin layer
        if self.path and intersects(self.path, tile.changed_cells) and cost_increase(self.path, tile) > ε:
            self.replan()

def plan_to_victim(self, victim):
    goal = nearest_reachable(victim.x, self.costmap, standoff=1.0)           # never drive onto the person
    path = theta_star(self.pos, goal, self.costmap, forbid=[UNKNOWN, LETHAL])
    if path is None:
        bb.request_scout(victim.x, reason="no_known_route")                  # UAV scouts the corridor
        return
    self.follow(path, local_planner=MPC_or_DWB(own_lidar, speed_cap=near_victim_cap))
```
Rules: unknown is never driven through; the rover's own LiDAR can override a stale tile (local costmap has priority for micro-obstacles); a rover reports traversal outcome (`slipped`, `blocked`) back into L2 so the map learns.

---

## 8. Smooth kinematics and collision avoidance

### 8.1 Min-snap trajectory (per UAV)
Piecewise polynomial `p(t) ∈ ℝ³` over `M` segments, each of degree 7 (so snap is continuous), through waypoints chosen inside a **safe flight corridor** (sequence of convex free-space polytopes extracted from the ESDF/TSDF).

```
minimise     J = Σ_k ∫₀^{T_k} || d⁴p_k(t)/dt⁴ ||² dt  +  ρ · Σ T_k
subject to   p_1(0) = p_start,  ṗ_1(0)=v_0, p̈_1(0)=a_0   (continuity with current state, not hover)
             p_M(T_M) = p_goal,  ṗ=0, p̈=0 at stop (or free if replanning on the move)
             d^j p_k(T_k) = d^j p_{k+1}(0),   j = 0..3                      (C³ continuity)
             p_k(t) ∈ Polytope_k   ∀ t       (via Bernstein/B-spline control points ⊂ polytope)
             ||ṗ|| ≤ v_max,  ||p̈|| ≤ a_max,  ||p⃛|| ≤ j_max                  (convex hull of derivative control points)
             tilt/thrust:   ||p̈ + g·e_z|| ∈ [T_min/m, T_max/m],  angle ≤ θ_max (differential flatness)
             (optionally) altitude floor above ESDF for terrain following
```
- Written in B-spline/Bernstein basis, this is a **convex QP**: the convex-hull property gives *guaranteed* bounds on v/a/j and corridor membership from control points alone. Solve with OSQP/qpOASES, warm-started, ~5–10 Hz.
- Time allocation `T_k`: trapezoidal initialisation, then a few iterations of gradient descent on `T` (the `ρΣT` term trades speed vs smoothness).
- **Differential flatness:** flat outputs `(p, ψ)` map to full state and inputs, so the controller tracks `p, ṗ, p̈` (and optional jerk feed-forward) via PX4 offboard.
- **Wind:** estimate `d̂` with the extended state in the ESKF (or a DOB); add `−d̂/m` to the acceleration feed-forward and tighten corridor margin when `||d̂||` is large (a gust shrinks the usable corridor instead of causing an overshoot).
- **Replanning smoothness:** each new plan starts from the *current measured* `(p, v, a)` and blends over a short overlap, so changing a goal never causes a step in acceleration.
- The yaw `ψ(t)` follows the velocity direction or the sensor's point-of-interest with a rate limiter.

### 8.2 Collision avoidance: ORCA + APF
- **ORCA (velocity layer, 3D):** for each neighbour within range, build the half-space of velocities that avoids collision for time horizon `τ` with *reciprocal* responsibility 0.5. Combined radius `r_i + r_j ≥ 2.0 m + margin` to guarantee `d_min ≥ 2.0 m`; margin = `k·σ_pos` (localisation uncertainty) + control-tracking error bound. Solve the small LP; pick the velocity closest to the min-snap reference velocity.
- **APF (static obstacles + short range):** repulsive term from ESDF gradient, `a_rep = k_rep (1/d − 1/d₀) / d² · ∇d` for `d < d₀`, **added to the reference** (not applied directly), with saturation to keep it jerk-bounded. APF alone gets stuck in local minima, so the global min-snap corridor remains the guide; APF only nudges.
- **UGVs** are included as agents in ORCA (ground-plane) so UAVs yield to rovers; rovers obey their own local planner + ORCA with UAVs below them ignored (altitude separation).
- **Priority and deadlock breaking:** victim-lock helpers > mappers; symmetric deadlocks broken by `agent_id` (small lateral bias).
- **Failsafe:** if the ORCA LP is infeasible, command the maximum-clearance direction with zero closing speed, or climb to the agent's altitude lane.
- The existing 6.5 m buffer is **replaced** by the above; report measured minimum separation, not just the design target.

---

## 9. Verification plan (report what really happens)

| Claim | Test | Pass criterion |
|---|---|---|
| Humans counted once (R1) | N=6 known victims, 4 UAV + 2 UGV, random flight patterns, victims revisited, 2 placed 1 m apart; 20 seeds | `|count − 6| = 0` in ≥95 % of runs; zero double-count after re-entry; ID switches/victim reported |
| Dual-spectrum filter | place warm rock, engine, sun-heated debris | false-positive rate vs thermal-only, reported |
| Map quality (R2) | compare fused map to ground-truth world | surface RMSE (cm), fraction of rover-traversable cells correct, rover success rate vs old OctoMap |
| Rover navigation | UGV to every confirmed victim | success %, path length, replans, time; unknown-cell violations = 0 |
| Swarm mind (R3) | issue TaskSpec, kill one UAV mid-mission, partition link | time-to-reassign, coverage lost, count unchanged, CRDT converges |
| Smoothness | wind 0–8 m/s | RMS jerk, tracking error, d_min ≥ 2.0 m always |
| Honesty | baselines, no-GNN ablation, separate **simulated** vs **real** | tables as in `evaluate.py`; any baseline win is reported |

## 10. Build order (fits the fallback ladder in AGENTS.md)
1. Victim registry + Re-ID in the lightweight sim first (fast iteration; R1 is the top priority).
2. Dual-spectrum + tracker on Gazebo cameras; add a simulated thermal channel if the model lacks LWIR.
3. LiDAR on x500 models, nvblox/Voxblox + traversability tiles; rover consumes them in Nav2.
4. Blackboard (CRDT) + TaskSpec gossip; local LLM; disturbance injection tests.
5. Min-snap + ORCA/APF; remove the 6.5 m buffer once measured separation is safe.
Fallbacks if blocked >45 min: real SLAM → noisy odom labelled "simulated VIO"; nvblox → OctoMap with mesh post-processing; 6 live robots → 2+2 in Gazebo + 4+2 in lightweight sim.
