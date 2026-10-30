# AGENTS.md — working rules (kept in context)

## Project (Section 1)
Autonomous Disaster-Response Swarm (drones + rovers) for Search & Rescue.
AI Manthan 2.0, Acropolis Institute of Technology and Research, Indore.
Build a **decentralized MARL** framework coordinating a **heterogeneous** swarm of drones + rovers:
autonomously explore unknown, **GPS-denied**, labyrinthine spaces; map survivors via **thermal + acoustic**
sensors; share **topological data peer-to-peer**; locate victims efficiently.
Must visibly satisfy: decentralized, MARL, heterogeneous (drones AND rovers, different roles),
GPS-denied, thermal+acoustic sensing, peer-to-peer topological sharing, efficient victim location,
end-to-end working system, scalability, real-world applicability.

## Design principles (Section 3) — NON-NEGOTIABLE
1. THREE LAYERS: (a) LLM commander — slow (10–30s + on events), advisory only, NEVER in motion loop;
   (b) MARL+GNN policy — fast, decentralized, one per robot, picks goals;
   (c) low-level control/SLAM/nav (Nav2 / drone controller). Works even with LLM off (rule-based fallback).
2. DECENTRALIZED — each robot runs its own policy node; robots talk only via comms-bridge node
   (enforces range, latency, packet drop). No robot reads global ground truth at runtime.
3. GPS-DENIED — no GPS topic read anywhere in our code; positions from odometry/SLAM in local frame;
   if sim only has ground-truth odom, inject noise+drift and label "simulated VIO" honestly.
4. HETEROGENEOUS — drones (fast, wide FOV, short battery, no narrow gaps) vs rovers
   (slow, narrow, longer endurance, can enter narrow gaps/rubble); reward must make the difference matter.
5. HONESTY — never fake results; show real output; if a baseline wins, report it; separate real vs simulated.
6. FALLBACK LADDER — if blocked >45 min, log in PROGRESS.md and drop: real SLAM -> noisy odom;
   Gazebo multi-robot -> fewer robots; live demo -> scripted recording. Working end-to-end > perfect parts.

## Working rules (Section 11)
- **Always keep a runnable artifact committed** (highest priority per user).
- Use git; commit after every working step with a clear message.
- Test everything you write; show real command output; never say "done" without running it.
- Keep the laptop usable: limit parallel Gazebo instances, prefer headless, watch RAM/swap.
- Secrets only in `.env` (never committed/hardcoded). Track licenses in THIRD_PARTY.md (VMAS = GPLv3).
- Always source the correct ROS setup files; keep ML venv and ROS Python separate.
- End of every phase report: what was built, how to run, test results, known issues, next step.
- If plan vs reality conflict, tell the user plainly and propose smallest change that keeps demo working.
- Only ask the user when blocked, or before system-wide/destructive actions.

## Stack (from Phase 0 detection)
- Ubuntu 24.04.5 LTS, ROS 2 **Jazzy**, Gazebo **Harmonic (gz sim 8.15)**.
- RAM 7.5 GiB total (LOW) → live Gazebo: 1 drone + 1 rover max (headless); scalability in lightweight sim.
- GPU: RTX 2050 (4 GiB VRAM) for training; CPU (12 cores) fallback.
- LLM: default rule-based fallback; Ollama/cloud optional via env var.
