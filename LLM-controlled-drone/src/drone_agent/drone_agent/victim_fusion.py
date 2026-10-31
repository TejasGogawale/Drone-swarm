"""ROS-free victim fusion: raw YOLO person boxes + drone poses -> de-duplicated victim count.

Hosts one `VictimRegistry` per drone (the sim/reid.py CRDT registry) and mimics the comms link by
exchanging serialised snapshots only between drones that are within `comm_range_m`. The ROS node
(victim_registry_node.py) is a thin wrapper around this class, so the whole pipeline can be tested
without ROS or Gazebo.

Honest limits of this setup (see PROGRESS.md):
  * mono RGB camera only -> no thermal cross-check, so `min_sensors=1`. To keep a single drone's
    repeated false positive out of the count, a person must be seen by >= 2 different drones
    (`min_agents=2`). Cost: someone only ever seen by one drone is not counted. A static
    person-shaped object seen by two drones (mannequin, poster) WILL still be counted; ruling it
    out needs an LWIR camera in the Gazebo model.
  * appearance = colour histogram (placeholder), thresholds were tuned on synthetic identity
    vectors and need recalibration on real crops.
  * ground-plane foot localisation (flat ground).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from drone_agent.victim_geometry import (Camera, Pose, RunningCentering, ground_covariance,
                                         pixel_to_ground)


def _import_reid(swarm_root: str | None = None):
    """sim/reid.py lives at the repo root, outside the ROS package: find it."""
    candidates = [swarm_root, os.environ.get("SWARM_ROOT")]
    here = Path(__file__).resolve()
    candidates += [str(p) for p in here.parents]
    for c in candidates:
        if c and (Path(c) / "sim" / "reid.py").is_file():
            if c not in sys.path:
                sys.path.insert(0, c)
            from sim import reid
            return reid
    raise ImportError("sim/reid.py not found: set the swarm_root parameter or SWARM_ROOT")


# NED origins of each PX4 instance in the shared world frame (from start_all.sh spawn offsets).
DEFAULT_ORIGINS = [(3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0)]


class VictimFusion:
    def __init__(self, n_drones: int = 4, origins_ned=None, camera: Camera | None = None,
                 comm_range_m: float = 25.0, pose_var: float = 0.25, px_sigma: float = 8.0,
                 min_conf: float = 0.35, max_var: float = 25.0, reid_cfg_kw: dict | None = None,
                 swarm_root: str | None = None):
        self.reid = _import_reid(swarm_root)
        self.n = n_drones
        self.origins = [tuple(o) for o in (origins_ned or DEFAULT_ORIGINS)][:n_drones]
        self.cam = camera or Camera()
        self.comm_range = comm_range_m
        self.pose_var = pose_var
        self.px_sigma = px_sigma
        self.min_conf = min_conf
        self.max_var = max_var
        # one camera type => no dual-spectrum check; demand a second viewpoint instead
        kw = {"min_sensors": 1, "min_agents": 2}
        kw.update(reid_cfg_kw or {})
        self.cfg = self.reid.ReIDConfig(**kw)
        self.regs = {i: self.reid.VictimRegistry(i, self.cfg) for i in range(n_drones)}
        self.poses: dict[int, Pose] = {}
        self.center = RunningCentering()
        self.stats = {"boxes": 0, "accepted": 0, "no_ground": 0, "low_conf": 0, "no_pose": 0}
        self.bytes_gossiped = 0

    # ----------------------------------------------------------------------- inputs
    def set_pose(self, drone: int, pose: Pose):
        self.poses[drone] = pose

    def world_position(self, drone: int):
        p, o = self.poses.get(drone), self.origins[drone]
        return None if p is None else np.array([p.n + o[0], p.e + o[1]])

    def on_detections(self, payload: dict, tick: int) -> int:
        """payload = JSON of /yolo/raw_detections. Returns how many boxes became observations."""
        drone = int(payload["drone_id"])
        pose = self.poses.get(drone)
        cam = Camera(int(payload.get("width", self.cam.width)), int(payload.get("height", self.cam.height)),
                     self.cam.hfov, self.cam.mount_pitch)
        obs = []
        for det in payload["dets"]:
            self.stats["boxes"] += 1
            if det["conf"] < self.min_conf:
                self.stats["low_conf"] += 1
                continue
            if pose is None:
                self.stats["no_pose"] += 1
                continue
            x1, y1, x2, y2 = det["box"]
            u, v = 0.5 * (x1 + x2), float(y2)                    # foot point: bottom-centre of the box
            g = pixel_to_ground(cam, pose, u, v)
            cov = ground_covariance(cam, pose, u, v, px_sigma=self.px_sigma + 0.03 * (y2 - y1))
            if g is None or cov is None or 0.5 * float(np.trace(cov)) > self.max_var:
                self.stats["no_ground"] += 1
                continue
            o = self.origins[drone]
            world = g + np.array(o)
            emb = self.center(np.array(det["emb"], float)) if det.get("emb") else None
            obs.append(self.reid.Observation(
                tick, drone, world, 0.5 * float(np.trace(cov)), emb, frozenset({"rgb"}), "person",
                pose_var=self.pose_var, cov=cov))
        if obs:
            self.regs[drone].observe(obs, tick)
            self.stats["accepted"] += len(obs)
        return len(obs)

    # ------------------------------------------------------------------- comms + merge
    def step(self, tick: int, consolidate: bool = True):
        """Gossip between drones in range (serialised snapshots), then re-link."""
        pos = {i: self.world_position(i) for i in range(self.n)}
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if pos[a] is None or pos[b] is None:
                    continue
                if math.hypot(*(pos[a] - pos[b])) <= self.comm_range:
                    sa, sb = self.regs[a].snapshot(), self.regs[b].snapshot()
                    self.regs[a].merge_snapshot(sb)
                    self.regs[b].merge_snapshot(sa)
                    self.bytes_gossiped += len(str(sa)) + len(str(sb))
        if consolidate:
            for r in self.regs.values():
                r.consolidate(tick)

    # ----------------------------------------------------------------------- outputs
    def report(self, drone: int = 0) -> dict:
        r = self.regs[drone]
        victims = [{"id": list(map(str, v["id"])) if isinstance(v["id"], tuple) else str(v["id"]),
                    "n_ned": [round(float(v["x"][0]), 2), round(float(v["x"][1]), 2)],
                    "obs": round(float(v["n"]), 1), "confirmed": bool(v["confirmed"]),
                    "kind": v["kind"], "ambiguous": bool(v["ambiguous"])} for v in r.victims()]
        return {"count": sum(1 for v in victims if v["confirmed"]),
                "per_drone_count": {str(i): self.regs[i].count() for i in range(self.n)},
                "victims": victims, "stats": dict(self.stats)}
