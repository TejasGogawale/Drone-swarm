"""Central coordinator: survivor detections -> rover missions (ROS-free).

Workflow (matches the dispatch spec):
  1. a drone raises a survivor detection event (id, world coordinates, confidence, drone id);
  2. the coordinator de-duplicates it (one mission per survivor; repeated sightings only refine the position);
  3. it checks rover availability and plans a ground route for each free rover on the shared voxel map;
  4. the free rover with the SHORTEST ROUTE gets the mission; extra missions wait in a FIFO queue;
  5. the rover drives the route, replanning when the map shows the route blocked;
  6. on arrival the mission is marked rescued and the rover becomes free again.

Everything the viewer shows (routes, rover poses, mission state, transmissions) is read from here, so the
drawn route is literally the one the rover is following.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from drone_agent.rover_nav import NavParams, Rover, Traversability
from drone_agent.world_sim import World


@dataclass
class Mission:
    id: int
    survivor_id: int
    name: str
    n: float
    e: float
    conf: float
    drone_id: int
    source: str
    t_created: float
    state: str = "queued"       # queued | waiting_map | assigned | en_route | rescued
    rover_id: int | None = None
    route_len: float = 0.0
    t_assigned: float | None = None
    t_done: float | None = None
    replans: int = 0
    note: str = ""


class RescueCoordinator:
    def __init__(self, vm, world: World, rovers: list[Rover], params: NavParams | None = None,
                 sense_period: float = 0.2, plan_period: float = 1.0, rebuild_period: float = 0.4):
        self.vm = vm
        self.world = world
        self.rovers = rovers
        self.p = params or NavParams()
        self.trav = Traversability(vm, self.p)
        self.missions: dict[int, Mission] = {}
        self.queue: deque[int] = deque()
        self.events: deque[dict] = deque(maxlen=80)
        self.transmissions: list[dict] = []          # animated drone -> coordinator -> rover links
        self._next_id = 1
        self._by_survivor: dict[int, int] = {}
        self.t = 0.0
        self._last_sense = -1e9
        self._last_plan = -1e9
        self._last_build = -1e9
        self.sense_period, self.plan_period, self.rebuild_period = sense_period, plan_period, rebuild_period

    # ------------------------------------------------------------------ events
    def log(self, kind: str, **kw):
        self.events.append({"t": self.t, "kind": kind, **kw})

    def on_detection(self, det, drone_pos=None) -> Mission | None:
        """det: world_sim.Detection (or anything with id, n, e, conf, drone_id, source, name)."""
        mid = self._by_survivor.get(det.id)
        if mid is not None:
            m = self.missions[mid]
            if m.state in ("queued", "waiting_map", "assigned", "en_route"):
                # refine the target with a running average, keep the best confidence
                m.n, m.e = 0.8 * m.n + 0.2 * det.n, 0.8 * m.e + 0.2 * det.e
                m.conf = max(m.conf, det.conf)
            return None
        m = Mission(self._next_id, det.id, det.name or f"Survivor {det.id}", det.n, det.e, det.conf,
                    det.drone_id, det.source, self.t)
        self._next_id += 1
        self.missions[m.id] = m
        self._by_survivor[det.id] = m.id
        self.queue.append(m.id)
        self.log("detected", mission=m.id, survivor=det.id, drone=det.drone_id, conf=det.conf)
        self.transmissions.append({"t0": self.t, "kind": "to_coordinator", "from": drone_pos,
                                   "drone": det.drone_id, "target": (m.n, m.e), "mission": m.id})
        return m

    # ------------------------------------------------------------------ main loop
    def tick(self, dt: float, now: float | None = None):
        self.t = self.t + dt if now is None else now
        if self.vm.version != self.trav.version and self.t - self._last_build >= self.rebuild_period:
            self.trav.rebuild()
            self._last_build = self.t
        if self.t - self._last_sense >= self.sense_period:
            self._rover_sense()
            self._last_sense = self.t
        if self.t - self._last_plan >= self.plan_period:
            self._dispatch()
            self._last_plan = self.t
        for r in self.rovers:
            if r.state == "navigating":
                self._check_route(r)
            r.step(dt, self.p, self.world)
            if r.state == "arrived":
                self._arrive(r)
        self.transmissions = [x for x in self.transmissions if self.t - x["t0"] < 4.0]

    def _rover_sense(self):
        """Rovers scan their surroundings with a ground sensor: this is what discovers new/dropped obstacles."""
        for r in self.rovers:
            rays = self.world.scan(r.n, r.e, 0.4, r.yaw, n_rays=96, max_range=9.0, min_range=0.3, sensor="rover")
            self.vm.integrate_rays((r.n, r.e, 0.4), rays, agent_bit=1 << (4 + r.id))

    def _free_rovers(self):
        return [r for r in self.rovers if r.state in ("idle", "arrived") and r.mission_id is None]

    def _dispatch(self):
        self.retry_blocked()
        waiting = [mid for mid in self.queue]
        for mid in waiting:
            free = self._free_rovers()
            if not free:
                break
            m = self.missions[mid]
            best = None
            for r in free:
                plan = self.trav.plan(r.pos(), (m.n, m.e))
                if plan is not None and (best is None or plan[1] < best[1][1]):
                    best = (r, plan)
            if best is None:
                if m.state != "waiting_map":
                    m.state = "waiting_map"
                    m.note = "no known-free route yet (map still growing)"
                    self.log("no_route", mission=m.id)
                continue
            r, (wp, length) = best
            self.queue.remove(mid)
            r.mission_id = m.id
            r.target = (m.n, m.e)
            r.set_route(wp)
            m.state, m.rover_id, m.route_len, m.t_assigned, m.note = "en_route", r.id, length, self.t, ""
            self.log("assigned", mission=m.id, rover=r.id, route_m=round(length, 1))
            self.transmissions.append({"t0": self.t, "kind": "to_rover", "rover": r.id, "target": (m.n, m.e),
                                       "mission": m.id})

    def _check_route(self, r: Rover):
        if r.mission_id is None or r.target is None:
            return
        if not self.trav.route_blocked(r.route, r.route_idx, horizon=14.0, from_pos=r.pos(), target=r.target):
            return
        if self.t - r.last_replan_t < 0.6:
            return
        r.last_replan_t = self.t
        plan = self.trav.plan(r.pos(), r.target)
        m = self.missions[r.mission_id]
        if plan is None:
            r.state = "blocked"
            r.v = 0.0
            m.note = "route blocked, no alternative yet"
            self.log("blocked", mission=m.id, rover=r.id)
            return
        r.set_route(plan[0])
        r.replans += 1
        m.replans += 1
        m.route_len = plan[1]
        self.log("replanned", mission=m.id, rover=r.id, route_m=round(plan[1], 1))

    def retry_blocked(self):
        for r in self.rovers:
            if r.state == "blocked" and r.target is not None:
                plan = self.trav.plan(r.pos(), r.target)
                if plan is not None:
                    r.set_route(plan[0])
                    r.replans += 1
                    self.log("replanned", mission=r.mission_id, rover=r.id, route_m=round(plan[1], 1))

    def _arrive(self, r: Rover):
        m = self.missions.get(r.mission_id)
        if m is not None:
            m.state, m.t_done = "rescued", self.t
            self.log("arrived", mission=m.id, rover=r.id)
        r.mission_id = None
        r.target = None
        r.state = "idle"

    # ------------------------------------------------------------------ views
    def active_routes(self):
        """[(rover, remaining polyline starting at the rover)] for drawing."""
        out = []
        for r in self.rovers:
            if r.route and r.state in ("navigating", "blocked"):
                out.append((r, [r.pos()] + r.route[r.route_idx:]))
        return out

    def summary(self) -> dict:
        ms = list(self.missions.values())
        return {"detected": len(ms), "queued": sum(m.state in ("queued", "waiting_map") for m in ms),
                "en_route": sum(m.state == "en_route" for m in ms),
                "rescued": sum(m.state == "rescued" for m in ms),
                "replans": sum(m.replans for m in ms)}
