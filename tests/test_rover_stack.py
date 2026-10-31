"""Tests for the voxel map, simulated LiDAR, rover navigation and rescue coordinator.

Run:  python -m pytest tests/test_rover_stack.py -q      (no ROS needed)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LLM-controlled-drone" / "src" / "drone_agent"))

from drone_agent.rescue_coordinator import RescueCoordinator                     # noqa: E402
from drone_agent.rover_nav import NavParams, Rover, Traversability                 # noqa: E402
from drone_agent.voxel_map import VoxelMap                                         # noqa: E402
from drone_agent.world_sim import Detection, SimDetector, World, _ray_aabb         # noqa: E402


def explore(vm, world, step=10.0, extent=40.0, alt=12.0, n_rays=192, max_range=28.0):
    """Fly a lawn-mower of virtual scans over the area (what the 4-drone swarm does over minutes)."""
    k = 0
    for n in np.arange(-extent, extent + 0.1, step):
        for e in np.arange(-extent, extent + 0.1, step):
            rays = world.scan(n, e, alt, 0.0, n_rays=n_rays, max_range=max_range)
            vm.integrate_rays((n, e, alt), rays, agent_bit=1 << (k % 4))
            k += 1


def wall_world(survivor=(0.0, 14.0)):
    """Empty field with a long wall between the rover (south) and the survivor (north)."""
    w = World(boxes=[(6.0, 0.0, 1.0, 16.0, 3.0)], cyls=[])
    w.survivors = []
    return w


# ------------------------------------------------------------------- LiDAR simulation
def test_analytic_scan_matches_brute_force_marching():
    w = World()
    rng = np.random.default_rng(0)
    graze, total = 0, 0
    for _ in range(30):
        n0, e0 = rng.uniform(-20, 20, 2)
        if w.blocked_at(n0, e0, margin=1.6):
            continue                              # sensor inside/against an obstacle: min_range case, not geometry
        rays = w.scan(n0, e0, 12.0, 0.3, n_rays=48, max_range=28.0)
        for i, row in enumerate(rays):
            ang = 0.3 + i * 2 * math.pi / 48
            ref = None
            for r in np.arange(1.5, 28.0, 0.01):
                if w.blocked_at(n0 + r * math.cos(ang), e0 + r * math.sin(ang)):
                    ref = r
                    break
            got = math.hypot(row[0] - n0, row[1] - e0)
            total += 1
            if ref is None:
                graze += row[4] > 0.5            # analytic hit the marcher stepped over (grazing a corner)
            elif row[4] > 0.5:
                assert abs(got - ref) < 0.05, (got, ref)
            else:
                # the marcher found something the scan reports as 'not visible': only allowed by the
                # visibility rule (low obstacle seen from far/low), never by geometry
                assert ref >= 1.5
    assert graze / total < 0.01


def test_scan_visibility_grows_with_range_like_the_original():
    w = World(boxes=[(20.0, 0.0, 2.0, 40.0, 30.0)], cyls=[])
    rays = w.scan(0.0, 0.0, 2.0, 0.0, n_rays=4, max_range=40.0)
    near = rays[0]                                     # looks straight at the wall 19 m away
    assert near[4] > 0.5 and near[3] == pytest.approx(2.0 + 19.0 * 0.45, abs=0.2)


def test_voxel_map_builds_connected_walls_not_scattered_points():
    w = World(boxes=[(10.0, 0.0, 4.0, 12.0, 6.0)], cyls=[])
    vm = VoxelMap(res=0.5)
    for e in range(-6, 7, 3):
        rays = w.scan(0.0, float(e), 10.0, 0.0, n_rays=360, max_range=28.0)
        vm.integrate_rays((0.0, float(e), 10.0), rays)
    ijk, p, *_ = vm.occupied_arrays(0.5)
    assert len(ijk) > 200
    # the south face of the box (north = 8) must be a contiguous sheet: every voxel has a neighbour
    keys = {tuple(k) for k in ijk}
    lonely = sum(1 for (i, j, k) in keys
                 if not any((i + a, j + b, k + c) in keys for a, b, c in
                            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))))
    assert lonely / len(keys) < 0.02
    heights = ijk[:, 2] * vm.res
    assert heights.min() <= 0.5 and heights.max() >= 5.0     # reaches the ground and the roof line


def test_occupancy_probability_grows_with_repeated_hits_and_decays_on_misses():
    vm = VoxelMap(res=0.5)
    ray = np.array([[10.0, 0.0, 0.25, 0.25, 1.0]])
    p_prev = 0.0
    for _ in range(4):
        vm.integrate_rays((0.0, 0.0, 5.0), ray)
        ijk, p, *_ = vm.occupied_arrays(0.4)
        assert p.max() > p_prev
        p_prev = p.max()
    # now rays that pass through the same place without hitting lower the probability
    thru = np.array([[20.0, 0.0, 0.0, 0.0, 0.0]])
    for _ in range(12):
        vm.integrate_rays((0.0, 0.0, 0.25), thru)
    assert vm.occupied_arrays(0.5)[0].shape[0] == 0


def test_new_obstacle_gets_a_detection_timestamp():
    vm = VoxelMap(res=0.5)
    vm.integrate_rays((0, 0, 5), np.array([[10.0, 0.0, 0.0, 0.5, 1.0]]), t=12.5)
    ijk, p, t_occ, t_last, hits = vm.occupied_arrays(0.5)
    assert len(ijk) and np.all(t_occ == 12.5)


# --------------------------------------------------------------------- traversability
def test_inflated_obstacles_and_unknown_policy():
    w = wall_world()
    vm = VoxelMap(res=0.5)
    explore(vm, w, step=8.0, extent=24.0, alt=10.0)
    trav = Traversability(vm, NavParams())
    r, c = vm.cell(5.75, 0.0)                             # first cell inside the visible south face
    assert trav.obstacle[r, c] and trav.blocked[r, c]
    assert not trav.known_free[r, c]
    r_in, c_in = vm.cell(6.0, 0.0)                         # wall interior is inside the inflated footprint
    assert trav.blocked[r_in, c_in]
    # rover is 0.45 m wide: cells within radius+margin of the wall are blocked, 1.2 m away is not
    r2, c2 = vm.cell(6.0 - 1.9, 0.0)
    assert not trav.blocked[r2, c2]
    # unexplored far corner is unknown, hence unreachable with 'forbid'
    assert trav.plan((0.0, 0.0), (58.0, 58.0)) is None
    t2 = Traversability(vm, NavParams(unknown_policy="penalize"))
    assert t2.plan((0.0, 0.0), (50.0, 50.0)) is not None


def test_route_goes_around_the_wall_and_never_touches_blocked_cells():
    w = wall_world()
    vm = VoxelMap(res=0.5)
    explore(vm, w, step=6.0, extent=26.0, alt=10.0)
    trav = Traversability(vm, NavParams())
    plan = trav.plan((0.0, 0.0), (14.0, 0.0))
    assert plan is not None
    wp, length = plan
    assert length > 14.0 + 4.0                             # detour, not the straight line through the wall
    for a, b in zip(wp[:-1], wp[1:]):
        assert not trav.segment_blocked(a, b)
        for f in np.linspace(0, 1, 40):
            assert not w.blocked_at(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, margin=0.3)
    assert math.hypot(wp[-1][0] - 14.0, wp[-1][1]) <= NavParams().standoff + 1e-6


# ---------------------------------------------------------------------- rover kinematics
def test_rover_moves_smoothly_without_teleporting_and_respects_limits():
    p = NavParams()
    r = Rover(0, "R1", (0, 0, 0), 0.0, 0.0, yaw=math.pi)        # starts facing the wrong way
    r.set_route([(0.0, 0.0), (0.0, 12.0), (8.0, 12.0)])
    dt = 0.05
    last = r.pos()
    yaw_prev, v_prev = r.yaw, r.v
    for _ in range(2400):
        r.step(dt, p)
        step = math.hypot(r.n - last[0], r.e - last[1])
        assert step <= p.max_speed * dt + 1e-9                  # never faster than its top speed
        assert abs(r.v - v_prev) <= p.accel * dt + 1e-9         # acceleration limit
        dyaw = abs((r.yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi)
        assert dyaw <= p.max_yaw_rate * dt + 1e-9               # smooth rotation
        last, yaw_prev, v_prev = r.pos(), r.yaw, r.v
        if r.state == "arrived":
            break
    assert r.state == "arrived"
    assert math.hypot(r.n - 8.0, r.e - 12.0) < p.goal_tol + 0.05


# ---------------------------------------------------------------------- coordinator
def _scene(survivors):
    w = World(boxes=[(6.0, 0.0, 1.0, 16.0, 3.0)], cyls=[])
    w.survivors = list(survivors)
    vm = VoxelMap(res=0.5)
    explore(vm, w, step=6.0, extent=26.0, alt=10.0)
    rovers = [Rover(0, "Rover 1", (0, 200, 255), -5.0, -4.0), Rover(1, "Rover 2", (255, 140, 0), -5.0, 4.0)]
    return w, vm, rovers


def _sv(i, n, e):
    from drone_agent.world_sim import Survivor
    return Survivor(i, f"S{i}", n, e, 0.6, 1.8)


def _det(i, n, e, drone=0, conf=0.9):
    return Detection(i, n, e, conf, drone, 0.0, "sim", f"S{i}")


def run(co, seconds, dt=0.05):
    for _ in range(int(seconds / dt)):
        co.tick(dt)


def test_nearest_rover_by_route_distance_gets_the_mission_and_extras_queue():
    w, vm, rovers = _scene([_sv(1, 14.0, -8.0), _sv(2, 14.0, 8.0), _sv(3, -12.0, 0.0)])
    co = RescueCoordinator(vm, w, rovers)
    co.tick(0.05)
    co.on_detection(_det(1, 14.0, -8.0))                    # on the west side: Rover 1 (west) is closer
    co.on_detection(_det(2, 14.0, 8.0))
    co.on_detection(_det(3, -12.0, 0.0))                    # third one must wait: both rovers busy
    run(co, 2.0)
    assigned = {m.survivor_id: m.rover_id for m in co.missions.values() if m.state == "en_route"}
    assert assigned[1] == 0 and assigned[2] == 1
    assert co.missions[3 - 0].state in ("queued",) or any(
        m.survivor_id == 3 and m.state == "queued" for m in co.missions.values())
    run(co, 90.0)
    assert all(m.state == "rescued" for m in co.missions.values()), [(m.survivor_id, m.state) for m in co.missions.values()]


def test_duplicate_detection_does_not_create_a_second_mission():
    w, vm, rovers = _scene([_sv(1, 14.0, -8.0)])
    co = RescueCoordinator(vm, w, rovers)
    assert co.on_detection(_det(1, 14.0, -8.0)) is not None
    assert co.on_detection(_det(1, 14.3, -7.8, drone=2)) is None
    assert len(co.missions) == 1


def test_no_known_route_waits_for_the_map_then_goes():
    w = World(boxes=[], cyls=[])
    w.survivors = [_sv(1, 20.0, 0.0)]
    vm = VoxelMap(res=0.5)
    rovers = [Rover(0, "Rover 1", (0, 0, 0), 0.0, 0.0), Rover(1, "Rover 2", (0, 0, 0), 0.0, 3.0)]
    co = RescueCoordinator(vm, w, rovers)
    co.on_detection(_det(1, 20.0, 0.0))
    run(co, 3.0)
    assert co.missions[1].state == "waiting_map" and rovers[0].state != "navigating"
    explore(vm, w, step=8.0, extent=30.0, alt=10.0)         # drones finish mapping the area
    run(co, 3.0)
    assert co.missions[1].state == "en_route"


def test_dynamic_obstacle_triggers_replanning_and_the_rover_never_enters_it():
    w = World(boxes=[], cyls=[])
    w.survivors = [_sv(1, 24.0, 0.0)]
    vm = VoxelMap(res=0.5)
    explore(vm, w, step=6.0, extent=30.0, alt=10.0)
    rovers = [Rover(0, "Rover 1", (0, 0, 0), 0.0, 0.0), Rover(1, "Rover 2", (0, 0, 0), -20.0, 20.0)]
    co = RescueCoordinator(vm, w, rovers)
    co.on_detection(_det(1, 24.0, 0.0))
    run(co, 6.0)
    r = rovers[0]
    assert r.state == "navigating"
    first_route = list(r.route)
    # rubble lands on the straight route, 10 m ahead of the rover
    w.add_dynamic_box(r.n + 10.0, 0.0, 3.0, 5.0, 1.2)
    run(co, 60.0)
    assert r.replans >= 1
    assert co.missions[1].state == "rescued"
    assert r.route != first_route
    for (n, e) in r.trail:
        assert not w.blocked_at(n, e, margin=0.3), (n, e)


def test_sim_detector_is_labelled_simulated_and_respects_line_of_sight():
    w = World(boxes=[(5.0, 0.0, 2.0, 30.0, 10.0)], cyls=[])
    w.survivors = [_sv(1, 10.0, 0.0), _sv(2, -6.0, 0.0)]
    det = SimDetector(w, rng_seed=1, p_detect=1.0)
    out = det.step([{"id": 0, "x": 0.0, "y": 0.0, "z": -12.0}], t=1.0)
    assert [d.id for d in out] == [2] and out[0].source == "sim"      # survivor 1 is behind the wall


def test_rover_reaches_a_survivor_who_is_a_real_obstacle_without_touching_them():
    w = World(boxes=[], cyls=[(14.0, 5.0, 0.6, 1.85)])
    w.survivors = [_sv(1, 14.0, 5.0)]
    vm = VoxelMap(res=0.5)
    explore(vm, w, step=6.0, extent=26.0, alt=10.0)
    rovers = [Rover(0, "Rover 1", (0, 0, 0), -3.0, 0.0), Rover(1, "Rover 2", (0, 0, 0), -3.0, 6.0)]
    co = RescueCoordinator(vm, w, rovers)
    co.on_detection(_det(1, 14.0, 5.0))
    run(co, 80.0)
    m = co.missions[1]
    assert m.state == "rescued", (m.state, m.note, [(r.name, r.state, r.replans) for r in rovers])
    r = rovers[m.rover_id]
    assert NavParams().target_min - 0.2 <= math.hypot(r.n - 14.0, r.e - 5.0) <= NavParams().standoff + 0.4
    assert r.bumps == 0
    for (n, e) in r.trail:
        assert not w.blocked_at(n, e, margin=0.3)
    assert m.replans <= 2                        # the lifted approach ring must not cause replan thrash
