# SwarmRescue — pitch pack
**Team LOVE · AI MANTHAN 2.0 (PS-5) · "The first 72 hours, automated."**

> How to read this file. Every number is tagged:
> **[S]** sourced from a public page (link given) · **[M]** measured by us on this codebase, in simulation ·
> **[A]** our assumption, to be replaced by a vendor quote before you say it on stage · **[D]** derived by arithmetic from the tagged numbers above it.
> Nothing here has been field-tested on real hardware. Section 9 says exactly what we can and cannot claim.

---

## 1. The problem, in one slide

When a big earthquake hits, people who are alive under rubble die of **time**, not of the quake.
Two things consume that time:

1. **Nobody can get there.** Roads and bridges fail, helicopters are few, teams are small.
2. **Nobody can talk.** Power fails, towers fail, the networks that survive jam. Rescuers don't know where to go, and survivors cannot say where they are.

> **One line for the stage:** *"The disaster doesn't kill people at second zero. It kills them on hour 30, while we are still looking for the road."*

---

## 2. What actually happened (evidence)

| Disaster | What happened | Where response failed | Source |
|---|---|---|---|
| **Nepal, April 2015** (M7.8) | ~9,000 dead; ~16,800 (Britannica) to 22,000+ (other sources) injured; 600,000–800,000 structures damaged or destroyed | Roads damaged and networks down: "many rural areas left to cope on their own". Search teams were in the Kathmandu valley while remote villages were still waiting. Rescue teams reached **all** remote villages only after **~2 weeks**. | [Britannica](https://www.britannica.com/print/article/2024843) · [Transcend](https://www.transcend.org/tms/?p=58694) · [Global Voices](https://globalvoices.org/2015/04/28/nepalquake-a-tale-of-personal-and-national-tragedy/) · [PDC](https://www.pdc.org/emergency_communications_systems_improve_as_response_operations_broaden/) |
| **Japan, Tōhoku, March 2011** (M9.0 + tsunami) | ~15,800 dead and ~3,400–3,700 missing; >90% of deaths and disappearances were drowning | Communications: ~**1.5 million** fixed-line circuits and ~**6,700** mobile base-station units affected; **>80%** of disconnections caused by power loss; voice traffic **restricted by 70–95%** from congestion; ~**65,000** telephone poles flooded or collapsed. Restoration: 91% of KDDI base stations by 7 April (**~4 weeks**). | [UNESCAP](https://www.unescap.org/sites/default/files/Item%206%20Resilient%20Networks%20in%20Japan.pdf) · [NTT](https://www.ntt.com/en/about-us/press-releases/news/article/2011/20110330.html) · [ITU](https://itu.int/en/ITU-T/wtsa12/Documents/side_events/disaster_relief/Araki-Presentation.pdf) · [KDDI](https://www.kddi.com/english/corporate/news_release/2011/0408/) · [Kenken](https://iisee.kenken.go.jp/symposium/pdf/Reference_4_201202_Tohoku.pdf) |
| **Türkiye–Syria, Feb 2023** (M7.8 + M7.5) | >59,000 dead (reported; possibly higher) | Destroyed roads, weather, lack of heavy equipment kept rescuers from the worst-hit areas; poor internet between cities hindered rescue teams; **the Turkish president admitted the response "didn't go as quickly as hoped"**. | [EBSCO](https://ebsco.com/research-starters/construction-and-building/2023-turkey-syria-earthquake) · [Al Jazeera](https://www.aljazeera.com/news/2023/2/8/turkey-syria-earthquake-search-intensifies-as-death-toll-climbs) · [CBS](https://cbsnews.com/amp/sanfrancisco/news/turkey-syria-earthquake-victims-survivors-rescue-teams-not-fast-enough) |
| **Japan, Noto, Jan 2024** | 128+ dead within a week | Roads cut by ~1,000 landslides; ≥2,000 people in cut-off communities; day 3: water, power and mobile service still down in places; the Prime Minister said rescuers found the northern tip "very difficult to reach"; snow then slowed everything. Japan, the best-prepared country on Earth, still sent troops **on foot** to isolated villages. | [Gulf News](https://gulfnews.com/world/asia/hundreds-cut-off-after-japan-quake-that-killed-78-1.1704361649843) · [Jamaica Observer/AP](https://jamaicaobserver.com/2024/01/07/snow-hinders-rescues-and-aid-deliveries-to-isolated-communities-after-japan-quakes-kill-128-people) |
| **India, Gujarat (Bhuj), Jan 2001** | roughly 13,800–20,000 dead (official counts 15,000–16,900), 55,000–167,000 injured depending on report date | ~25,000 rescue personnel deployed; survivors were still being pulled out at **36 h** and **60 h** — proof that people were alive and waiting. | [ADRC situation reports](https://www.adrc.asia/counterpart_report/India010126_28.pdf) · [EastMojo](https://eastmojo.com/news/2021/01/26/20-years-of-gujarat-earthquake-numbers-that-highlight-the-devastation/) · [Tribune](https://www.tribuneindia.com/2001/20010204/nation.htm) |

**The global picture.** 1998–2017 disasters killed **1.3 million** people; **earthquakes and tsunamis alone caused 747,234 deaths = 56%** ([UNDRR/CRED](https://www.undrr.org/news/un-20-year-review-earthquakes-and-tsunamis-kill-more-people-while-climate-change-driving)) [S].
That is ≈ **65,000 deaths a year** from all disasters and ≈ **37,400 a year (≈ 100 a day)** from earthquakes and tsunamis alone [D].

---

## 3. The statistics that decide the pitch: survival decays by the hour

The most-quoted survival estimate for people trapped in collapsed buildings (Prof. Steven Godby, Nottingham Trent University, quoted in coverage of Türkiye 2023) is **74% at 24 h, 22% at 72 h, 6% by day 5** [S, as reported in coverage of the 2023 quake, e.g. [Globe and Mail](https://www.theglobeandmail.com/world/article-death-toll-in-turkey-syria-earthquake-surpasses-15000-2/) — we saw this via search results and did not open the page, so check it before quoting a percentage on stage].
It is one expert's estimate and other regions report lower numbers (a 2005 Bangladesh fire-service instructor quoted ~25% at 48 h and ~5% at 72 h) — so we use it as a **shape**, not a law.

Fitting a decay curve through those three points gives [D]:

| Window | Decay rate | Half-life of survival odds |
|---|---|---|
| 24 h → 72 h | −2.5% per hour | **≈ 27 h** |
| 72 h → 120 h | −2.7% per hour | ≈ 26 h |

The two windows agree, so a simple exponential is a fair summary: **every ~27 hours of delay halves the number of people still alive to be saved.**

| Rescuers reach the trapped at… | Alive per 1,000 people trapped [D] | Lives lost vs. 24 h |
|---|---|---|
| 24 h | 740 | — |
| 36 h | 546 | −194 |
| 48 h | 403 | −337 |
| 60 h | 298 | −442 |
| 72 h | 220 | **−520** |

> **Say it plainly:** *for every 1,000 people trapped, the difference between finding them on day 1 and on day 3 is about 500 lives* (on this curve). Nepal's remote villages waited up to two weeks.

### The communications statistic
In Japan 2011 voice traffic was cut by **70–95%**, so a survivor trying to call had **5–30%** of normal capacity [D from S], and over **80%** of outages were simply the power going out — a problem that does not get fixed in hours. Nepal 2015 and Noto 2024 repeat the pattern. **Any rescue system that needs a cell tower or the internet is a system that fails exactly when it is needed.**

### What this means for product requirements
| Fact | Requirement it creates |
|---|---|
| Survival halves every ~27 h | Locate survivors in **hours**, not days → autonomous, wide-area, parallel search |
| Roads and bridges fail | Reach by **air first**, ground robots for the last metres |
| Networks fail and congest | **Zero infrastructure**: peer-to-peer mesh, no towers, no cloud, no GPS |
| Rubble is dark, smoky, unstable | **Thermal + acoustic** sensing, robots instead of people in unstable structures |
| Many teams, one survivor seen many times | **Count each person exactly once**, or resources get sent to ghosts |
| Few trained operators | **Natural-language command**, one operator for the whole swarm |

---

## 4. What exists today, and where it breaks

| Today's tool | What it does well | Where it breaks in the first 72 h |
|---|---|---|
| **Manual USAR teams** (e.g. NDRF sent two teams of ~50 to Türkiye; US task forces ~78–79 people plus dogs) [S: [NDRF](https://ndrf.gov.in/en/node/89944), [LA County](https://www.foxla.com/news/la-county-sending-rescue-teams-to-turkey-following-deadly-quake)] | Skilled, can extract people | Must be airlifted in; ~7-day deployment windows; reach tens of sites, not thousands; put rescuers inside unstable rubble |
| **Rescue helicopters** | Fast, can lift casualties | **$1,200–3,000 per flight hour** (typical SAR range, low-authority source: [Revector](https://www.revector.com/reducing-costs-of-search-rescue/)) [S]; few of them; grounded by weather and smoke |
| **Piloted enterprise drones** (DJI Matrice 350 RTK + H30T thermal; Parrot ANAFI USA; Skydio X10) | Excellent sensors | One drone, one pilot (+ observer). Needs a link to the pilot. No shared map, no coordination, counts the same person once per drone |
| **Robot dogs / ground robots** (Boston Dynamics Spot ≈ $74,500 base kit) | Rough terrain | One robot, one operator, very expensive; no aerial overview to tell it *where* to go |
| **Rescue dogs, listening devices, cameras on poles** | Proven in rubble | Slow, one pile at a time |
| **Phones / satellite phones** | Direct voice | Phones: tower- and power-dependent, congested; satellite phones: few survivors have one |

**The gap:** nobody sells a *coordinated, infrastructure-free, air-plus-ground team that finds people, counts them once, and sends a robot to them, run by one person.* That is the product.

---

## 5. Our product: SwarmRescue

**One sentence.** *A kit of 4 autonomous drones and 2 ground rovers that launch from a van, search a collapsed area by themselves without GPS or a network, find and count survivors with thermal and acoustic sensing, build a 3-D map, and dispatch a rover to each survivor — commanded in plain language by a single operator.*

### 5.1 How it works (every part exists in the repository)

| Capability | What it does | Where in the code | Why it matters in a disaster |
|---|---|---|---|
| **Decentralised swarm brain** | Every robot runs the same policy and talks only to peers in range. Layer 1: slow LLM commander (advisory only, never in the motion loop); Layer 2: fast MARL + GraphSAGE GNN that picks goals; Layer 3: flight / navigation control. Rule-based fallback if the GNN or LLM is off. | `train/` (MAPPO + GNN), `drone_agent/swarm_gnn_allocator.py`, `brain_node.py`, `llm_client.py` | No single point of failure; works when a drone is lost; works with no internet |
| **Natural-language control** | "Map the whole area", "Search for survivors", "Hold" → structured task → every robot acts at once | `llm_client.py` (local Ollama, Groq or Gemini), `command_translator.py` | One operator, no pilot training |
| **GPS-denied operation** | Local-frame odometry only; simulated VIO drift is injected and labelled | `AGENTS.md` principle 3, `reid_sensing.py` | Collapsed cities and tunnels have no GPS |
| **Thermal + acoustic + camera sensing** | Thermal for open victims, acoustic direction-of-arrival for victims buried under rubble (heard, not seen), YOLO on cameras | `sim/sensors.py`, `reid_sensing.py` (microphone-array model), `yolo_detector.py` | Buried people are exactly the ones who need us |
| **Count each human exactly once** | Multi-view Re-ID registry that survives lost links and merges later (CRDT); acoustic evidence map; dual-sensor and two-viewpoint confirmation | `sim/reid.py`, `victim_fusion.py`, `victim_geometry.py` | Stops phantom victims; tells commanders the *real* number to plan for |
| **3-D map for people and robots** | Probabilistic voxel map (occupancy, free, unknown, recently changed), rendered as real shaded cubes with orbit camera, layer toggles, survivor and rover overlays | `voxel_map.py`, `voxel_viewer.py`, `map_gui_node.py` | Commanders see the structure; rovers plan on it |
| **Air-to-ground rescue loop** | A drone detects → coordinator de-duplicates → nearest rover **by route length** is dispatched → A* on the same voxel map → pure-pursuit driving → **replans when rubble blocks the way** → queue if both busy | `rover_nav.py`, `rescue_coordinator.py` | Closes the gap between "we see someone" and "someone is on the way" |
| **Rover camera feeds** | First-person view from each rover | `VoxelViewer.first_person`, GUI | The operator sees what the rover sees |
| **Steady flight** | Jerk-limited setpoints in simulation time, slow yaw, gentle PX4 limits | `flight_smoothing.py`, `command_translator.py` | Stable camera = usable footage and reliable detections |
| **Multi-drone mapping workspace** | Multiple Iris drones with lidar and a merged global OctoMap in Gazebo (ArduPilot flavour) | `Ardupilot_Swarm/` | Second flight stack proves the approach is not tied to one autopilot |
| **LLM-to-drone origin** | Natural-language PX4 drone control demo (the repo also keeps the reference paper *"Taking Flight with Dialogue"*, which is not ours) | `LLM-controlled-drone/`, `paper.pdf` | Where the LLM-to-drone pipeline started |

Size of what is built: **≈ 11,400 lines of our own Python** (sim 1,966 · training 799 · ROS/drone agent 6,750 · tests 1,477 · scripts and ArduPilot helpers 383) and **87 automated tests, all passing** [M].

### 5.2 Evidence we can show (all simulation) [M]

| Result | Number | Notes |
|---|---|---|
| Drone steadiness, PX4's own odometry, 60 s of flight, mean of 4 drones | Lean angle **27.6° → 5.6°**; body rotation (95th pct) **164 → 13.8 °/s**; yaw rate **157 → 4.8 °/s**; acceleration **5.6 → 0.9 m/s²** | Same airframe, same sim; before/after in `PROGRESS.md`, script `scripts/flight_metrics.py` |
| Counting survivors exactly once, 20 unseen maps (4 UAV + 2 UGV, drifting positions) | Exact on **11 of 20** maps, **8 duplicate fragments**, **0 false positives**. Naive per-robot counting on a different set of 20 maps (seeds 0–19): **49 duplicates** | The failures are almost all victims < 3 cells apart that the sensors physically cannot separate |
| Dispatch loop, GUI run | **9 of 9** simulated survivors detected, queued, dispatched and reached by 2 rovers; 24 replans | Detections in this run are a *simulated* detector, labelled on screen |
| Rover routing | Routes around walls, never touches blocked cells (unit-tested), replans after rubble is dropped on the route | |
| Exploration baselines (2 UAV + 2 UGV, 3 maps, lightweight sim) | Random 78.9% / nearest-frontier 75.3% / explore-lite 81.8% coverage | Baselines only; see §9 on the learned policy |
| Real-time 3-D viewer | ~60 ms per frame on a quiet machine, ~85–125 ms while the full simulation runs | |

---

## 6. How we are different (feature matrix)

*(Competitor columns are "to the best of our knowledge" from public product pages — verify before presenting.)*

| | **SwarmRescue** | DJI Matrice 350 + H30T | Skydio X10 | Parrot ANAFI USA | Boston Dynamics Spot | SAR helicopter | Manual USAR team |
|---|---|---|---|---|---|---|---|
| Multiple agents coordinate **autonomously**, no pilot per vehicle | **Yes** | No (one pilot each) | Single-drone autonomy | No | Single-robot autonomy | n/a | n/a |
| Air **and** ground in one loop | **Yes** (4 UAV + 2 UGV) | Air only | Air only | Air only | Ground only | Air | Ground |
| Works with **no network and no GPS** | **Designed for it** (peer mesh, local odometry, local LLM) | Needs a link to the pilot; GPS-dependent flight | Strong visual autonomy; link to the operator | Link to the operator | Link to the operator | Radio | Radio |
| Finds **buried** victims (acoustic) | **Yes** (rover mic array model) | Thermal/visual only | Thermal/visual only | Thermal/visual only | Add-ons | Thermal on some | Listening gear, dogs |
| **Counts each person once** across all sensors and agents | **Yes** | No | No | No | No | No | Manual tally |
| One operator, **plain-language** command | **Yes** | No | Partly (app) | Partly (app) | Partly | No | No |
| Dispatches a robot to each survivor automatically | **Yes** | No | No | No | No | n/a | n/a |
| Open stack, runs on commodity PX4/ROS 2 | **Yes** | No (closed) | No (closed) | Partly | No (closed) | n/a | n/a |

---

## 7. Cost: what it costs to buy and to run

### Table A. What the existing options cost

| Item | Price | Notes / source |
|---|---|---|
| DJI Matrice 350 RTK + Zenmuse H30T thermal kit | **$31,399** per aircraft [S] | US reseller: [Maverick Drone](https://maverickdrone.com/collections/dji/products/dji-matrice-350-rtk-single-h30t-payload-kit); H30T camera alone $11,999 |
| Skydio X10 (base) | **≈ $11,000–12,000** [S] | [Robotomated](https://robotomated.com/compare/parrot-anafi-usa-vs-skydio-x10), [Jisaku](https://jisaku.com/glossary/skydio-x10-2023); thermal config is extra |
| Parrot ANAFI USA (thermal) | **≈ $7,000** [S] | [BlueSkies](https://blueskiesdroneshop.com/products/parrot-anafi-usa) |
| Boston Dynamics Spot, Explorer kit | **≈ $74,500** [S] | [Lowyat](https://lowyat.net/2020/215312/boston-dynamics-spot-us74500) (2020 launch price) |
| SAR helicopter operation | **$1,200–3,000 per flight hour** [S] | [Revector](https://www.revector.com/reducing-costs-of-search-rescue/) — low-authority; older fuel-and-maintenance-only figures are lower. Military heavy lift (Black Hawk) ≈ $2,200–3,100/h. |
| Urban search and rescue task force | **~50–80 people + dogs per team** [S] | [NDRF](https://ndrf.gov.in/en/node/89944), [LA County](https://www.foxla.com/news/la-county-sending-rescue-teams-to-turkey-following-deadly-quake). **We found no public per-team cost; do not quote one.** |

**An off-the-shelf fleet equivalent to our kit** (4 aircraft + 2 ground robots, but *not* coordinated) [D]:

| Option | Hardware cost | People to run it |
|---|---|---|
| 4× DJI M350 + H30T and 2× Spot | 4 × $31,399 + 2 × $74,500 = **$274,596** | ~8 pilots/observers + 2 robot operators [A] |
| 4× Skydio X10 base + 2× Spot | 4 × $11,990 + 2 × $74,500 = **≈ $197,000** | ~4 + 2 operators [A] |
| 4× Parrot ANAFI USA only (air only) | 4 × $7,000 = **$28,000** | ~4–8 [A] |

### Table B. What our kit costs to build and to run

All prices **[A]** except where marked **[V]** (a listing we found this session). **Replace the [A] rows with real vendor quotes before presenting.**

| Per-unit bill of materials | **Lean** kit (USD) | **Pro** kit (USD) |
|---|---|---|
| Airframe + PX4 flight-controller kit (Holybro X500 v2 dev kit **[V $508](https://shop.holybro.com/x500-kit_p1180.html)**) | 508 | 508 |
| Companion computer | Jetson Orin Nano class 300 | Jetson Orin NX 16 GB 700 |
| RGB camera | 60 | 60 |
| Thermal camera | FLIR Lepton 3.5 **[V $164](https://groupgets.com/collections/thermal-imaging)** + breakout 40 = 204 | FLIR Boson 640 **[V $4,204](https://groupgets.com/products/flir-boson-640)** |
| Lidar | 2-D, 100 | 3-D (Livox Mid-360 class), 750 |
| Batteries ×2, mesh radio, cables/props/spares | 180 + 40 + 100 | 180 + 40 + 100 |
| **One UAV** | **$1,492** | **$6,542** |
| **One UGV** (chassis, computer, lidar, cameras, thermal, 4-mic array, batteries, radio) | **$1,634** | **$7,284** |
| Ground station (GPU laptop, mesh antenna, cases) | 1,700 | 1,700 |
| **Kit: 4 UAV + 2 UGV + station** | $10,936 | $42,436 |
| + 15% contingency | **$12,576 (≈ ₹10.7 lakh)** | **$48,801 (≈ ₹41.5 lakh)** |

*(INR at an assumed ₹85/$.)*

**Running cost per deployment day** [A, D]: amortise the kit over 300 operating days, add $40/day batteries/props, add 2 operators at $100/day each:

| | Lean | Pro | 8 h of one helicopter | Off-the-shelf fleet in Table A |
|---|---|---|---|---|
| Per day | **≈ $282** | **≈ $403** | **$9,600–24,000** | amortised $274,596/300 ≈ $915 + 10 people |

### Table C. Side by side

| | SwarmRescue Lean | SwarmRescue Pro | Off-the-shelf fleet (M350 + Spot) | One SAR helicopter, 1 day |
|---|---|---|---|---|
| Up-front | **≈ $12.6k** | **≈ $48.8k** | **≈ $274.6k** | n/a |
| Cost vs. the off-the-shelf fleet | **≈ 4.6% (22× cheaper)** | **≈ 17.8% (5.6× cheaper)** | 100% | — |
| Running cost / day | **≈ $282** | **≈ $403** | ≈ $915 + 10 people | $9,600–24,000 |
| Operators | 1–2 | 1–2 | ≥ 8–10 | pilots + crew |
| Autonomous coordination | **Yes** | **Yes** | No | n/a |

**Honest framing for judges.** We are not replacing the helicopter. A helicopter **lifts** people; we **find** them. The pitch is *"point the expensive assets at the right place the first time."* The Lean kit also means a **district** can own one, not just a national agency.

### Business model (proposal [A])
| Line | Offer |
|---|---|
| **Hardware kit** | District Readiness Kit (Lean) at ~$25–30k → ≈ 50–58% gross margin on a $12.6k cost; Pro kit for national forces and defence |
| **Software subscription** | Updates, swarm policy models, local-LLM packs, map-matcher regions: per-kit per-year |
| **Training and drills** | Annual certification with the district disaster authority |
| **Customers** | State/district disaster management authorities, NDRF/SDRF, fire services, municipal corporations, NGOs and UN agencies (INSARAG teams), insurers, infrastructure operators. *(Market size: compute bottom-up as number of districts × kits — verify the count before putting a TAM on a slide.)* |

---

## 8. One product, one ecosystem

> **Important:** the VLM, map-matcher and BitChat-app parts of your deck were not attached to this request (and I found no matching file on disk), so the boxes below describe **what each layer must do in the story**, using only their names. Make the wording match your slides.

```
 SURVIVOR PHONE              DRONE + ROVER SWARM               COMMAND
 ─────────────               ───────────────────               ───────
 BitChat-style offline  ──►  mesh gateway on every robot  ──►  Command dashboard
 mesh app: "I'm alive,       (relay + pins to map)             • 3-D voxel map + map matcher
 here is my area"                                              • Survivor list (counted once)
                              Thermal + acoustic + RGB         • Rover missions & routes
                              ─► detections ─► Re-ID registry  • LLM commander (plain language)
                              ─► VLM describes scenes          • VLM briefing: hazards, access
                              ─► voxel map ─► map matcher      Local LLM, no internet
                                  aligns to known maps
                              ─► rover A* ─► reach survivor
```

**The story, minute by minute** (this is how every feature connects):

| Time | What happens | Feature that does it |
|---|---|---|
| **T + 0** | Earthquake. Towers and power fail. | — |
| **T + 5 min** | Survivors' phones form an **offline mesh** and broadcast "alive, here" with no infrastructure. | BitChat-style mesh app |
| **T + 30 min** | Kit leaves the van. The operator says: *"Map and scan the whole area."* | Local-LLM natural-language command |
| **T + 40 min** | 4 drones spread out and fly themselves, sharing only with neighbours; no GPS, no network. | Decentralised MARL + GNN swarm, steady flight |
| **T + 1 h** | Thermal and camera find people in the open; the **map matcher** snaps the swarm's 3-D map to existing maps; the **VLM** explains what the cameras see ("collapsed 3-storey block, open gap on the east side"). | VLM, map matcher, voxel map |
| **T + 1–2 h** | Rovers listen along the rubble; **acoustic direction-of-arrival** reveals a buried person. | Acoustic evidence map |
| **T + 2 h** | Each person is **counted once** across all views and robots; the dashboard shows the real number. | Re-ID registry (CRDT) |
| **T + 2 h** | Nearest rover is dispatched along a path planned on the same map; it replans around new rubble. | Rescue coordinator + A* |
| **T + 3 h** | Human rescue teams arrive with **coordinates, routes, a map and a count**, instead of searching blind. | Dashboard handoff |

*Result:* the search that took Nepal's remote villages **two weeks** is compressed to **hours** — inside the window where, on the survival curve above, ~1 in 2 more trapped people are still alive than at day 3.

---

## 9. What we can and cannot claim (read before you go on stage)

**We can say:**
- It is a **working software system in simulation**: 87 passing tests; a full 4-drone PX4/Gazebo swarm; a live GUI with a 3-D map and two rovers; a drone → coordinator → rover loop.
- The camera shake was **measured and fixed**: 12× lower body rotation rate, 5× lower lean.
- The counting method beats naive counting by a wide margin on unseen maps (8 vs 49 duplicate fragments, 0 false positives).
- The design needs **no GPS, no towers, no internet** by construction (peer mesh, local LLM, local odometry).

**We must not say:**
- That it has flown or driven on real hardware. It has not.
- That it detects people with YOLO in the Gazebo world reliably. In our live test the real YOLO detections were mostly wrong (3 of 9 real people matched, 7 false tracks). The rescue demo uses a **simulated detector**, labelled as such. Say: *"detection is the next milestone; the coordination, mapping and dispatch are done."*
- That the map comes from a real lidar. In Gazebo the drones carry a camera only; the 3-D map is built from a **simulated lidar on the world's obstacle list**.
- That the learned GNN policy clearly beats the alternatives. Our stored comparison of GNN vs no-GNN training reward is **0.626 vs 0.587 over 10 evaluations — not a significant difference**. The honest claim is "the learned policy runs and is competitive; the baselines are strong."
- The cost figures as final. They are a bill-of-materials plan; **[A]** rows need quotes.
- **Do not repeat the headline numbers in the current root `README.md`** (a "88.4% coverage / 48 steps / 0.8 collisions" row and a "7/9, 0 duplicates" Re-ID row). We found **no result file that produces them**, and our measured Re-ID numbers differ. Fix the README before judges read it (see the hand-off note).

**Known limits (our own Q&A answers):**
| Judge's question | Honest answer |
|---|---|
| "Drones only fly 20 minutes." | True. Ground rovers have longer endurance; the swarm hands over tasks; Pro kit adds swap batteries. Tethered/relay drones are a roadmap item. |
| "Why do the drones shuttle between two points?" | The current allocator marks a 32 m radius as explored instantly and resets visit counts; we know the cause and it is the first software item on the roadmap. |
| "Can it work indoors / in tunnels?" | The design is GPS-denied; real LiDAR-inertial odometry is the planned hardware step. Today it is simulated VIO. |
| "What if the LLM is wrong?" | It is advisory and never in the motion loop; a rule-based fallback runs the swarm; commands are validated before any robot acts. |
| "Regulation?" | Drone use by disaster authorities in India falls under DGCA drone rules and NDMA protocols; pilots with the authority, not private flights. (Verify the current rules before presenting.) |

---

## 10. Roadmap
| Phase | Deliverable | Evidence of done |
|---|---|---|
| **Now** | Simulation-validated swarm, mapping, Re-ID, rover dispatch | 87 tests; GUI demo |
| **Next 2 months** | Fix allocator exploration; real YOLO/thermal detection accuracy in Gazebo; hardware-in-the-loop on 1 UAV | Detection metric against the world file; HIL flight log |
| **3–6 months** | 2 real UAVs + 1 UGV with thermal and mesh radio; outdoor test field | Field trial video; real flight-metric table |
| **6–12 months** | Pilot with a district disaster authority / fire service; mock-disaster drill | Signed pilot, drill report |
| **12+ months** | Pro kit, certification, state procurement | Purchase order |

---

## 11. Three-minute pitch script

**(0:00)** In Nepal in 2015, rescue teams reached the remote villages two weeks after the quake. In Japan in 2011, a million and a half phone lines went dark and the phone network that survived was throttled by up to 95%. In Türkiye in 2023, over fifty-nine thousand people died, and the president admitted the response was slower than hoped. Different countries, same story: **we are late, and we are blind.**

**(0:30)** And lateness is not a vague word. The best survival estimate we have says three out of four trapped people are alive at hour 24, and one in five at hour 72. Survival halves roughly every 27 hours. For every thousand people trapped, finding them on day one instead of day three is the difference of about **five hundred lives**.

**(1:00)** So we built **SwarmRescue**: four drones and two rovers that launch from a van, with no GPS, no towers and no internet. One operator says *"map and scan the whole area"* — in plain language. The drones coordinate with each other, not with a ground station. Thermal finds people in the open. Microphones on the rovers hear people buried under the rubble. Every person is **counted exactly once**, however many robots saw them. And the moment someone is found, **the nearest rover is already driving to them**, on a path planned on the same 3-D map the commander is looking at — and if the rubble shifts, it replans.

**(1:45)** This is not a slide. It is eleven thousand lines of code, eighty-seven passing tests, and a live demo where nine of nine simulated survivors are found, counted once and reached. We measured our own mistakes, too: our drones' camera lean fell from twenty-eight degrees to under six, because a rescue camera that shakes finds nobody.

**(2:15)** It connects to the rest of our ecosystem: survivors' phones form an **offline mesh** with the BitChat-style app, the **VLM** explains what the cameras see, the **map matcher** puts the swarm's map on the real map — one product, from the first phone signal to the rover at the survivor's side.

**(2:35)** And it is affordable. A comparable off-the-shelf fleet — four DJI Matrice 350s and two Spot robots — is about **$275,000** and still doesn't coordinate. Our lean kit is about **$12,600**. A district can own one. We are not replacing the helicopter. We are making sure it flies once, to the right place.

**(2:55)** The disaster does not kill at second zero. It kills at hour thirty. **SwarmRescue gets us there before that.**

---

## 12. Sources (all pages opened or returned by search this session)
UNDRR 20-year review (<https://www.undrr.org/news/un-20-year-review-earthquakes-and-tsunamis-kill-more-people-while-climate-change-driving>) ·
Britannica, Nepal 2015 (<https://www.britannica.com/print/article/2024843>) ·
UNESCAP, resilient networks in Japan (<https://www.unescap.org/sites/default/files/Item%206%20Resilient%20Networks%20in%20Japan.pdf>) ·
NTT 2011 (<https://www.ntt.com/en/about-us/press-releases/news/article/2011/20110330.html>) ·
ITU (<https://itu.int/en/ITU-T/wtsa12/Documents/side_events/disaster_relief/Araki-Presentation.pdf>) ·
KDDI (<https://www.kddi.com/english/corporate/news_release/2011/0408/>) ·
Kenken, Tōhoku (<https://iisee.kenken.go.jp/symposium/pdf/Reference_4_201202_Tohoku.pdf>) ·
EBSCO Türkiye–Syria (<https://ebsco.com/research-starters/construction-and-building/2023-turkey-syria-earthquake>) ·
CBS (<https://cbsnews.com/amp/sanfrancisco/news/turkey-syria-earthquake-victims-survivors-rescue-teams-not-fast-enough>) ·
Gulf News, Noto (<https://gulfnews.com/world/asia/hundreds-cut-off-after-japan-quake-that-killed-78-1.1704361649843>) ·
Jamaica Observer/AP (<https://jamaicaobserver.com/2024/01/07/snow-hinders-rescues-and-aid-deliveries-to-isolated-communities-after-japan-quakes-kill-128-people>) ·
ADRC Gujarat reports (<https://www.adrc.asia/counterpart_report/India010126_28.pdf>) ·
Tribune India (<https://www.tribuneindia.com/2001/20010204/nation.htm>) ·
Globe and Mail, survival figures (<https://www.theglobeandmail.com/world/article-death-toll-in-turkey-syria-earthquake-surpasses-15000-2/>) ·
Maverick Drone M350 + H30T (<https://maverickdrone.com/collections/dji/products/dji-matrice-350-rtk-single-h30t-payload-kit>) ·
Skydio / ANAFI comparison (<https://robotomated.com/compare/parrot-anafi-usa-vs-skydio-x10>) ·
Spot price (<https://lowyat.net/2020/215312/boston-dynamics-spot-us74500>) ·
Revector, SAR cost (<https://www.revector.com/reducing-costs-of-search-rescue/>) ·
Holybro X500 v2 (<https://shop.holybro.com/x500-kit_p1180.html>) ·
GroupGets thermal (<https://groupgets.com/collections/thermal-imaging>, <https://groupgets.com/products/flir-boson-640>) ·
NDRF Türkiye (<https://ndrf.gov.in/en/node/89944>).
