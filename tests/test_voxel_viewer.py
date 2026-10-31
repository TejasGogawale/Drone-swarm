"""Tests for the 3-D voxel viewer (software renderer). Run: python -m pytest tests/test_voxel_viewer.py -q"""
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LLM-controlled-drone" / "src" / "drone_agent"))

from drone_agent.voxel_map import VoxelMap                                              # noqa: E402
from drone_agent.voxel_viewer import (OrbitCamera, Scene, ViewOptions, VoxelViewer,       # noqa: E402
                                      build_geometry, ROVER_COLORS, SURVIVOR_COLOR)

W, H = 800, 600


def block(vm, i0, j0, k0, ni, nj, nk, hits=4):
    for i in range(i0, i0 + ni):
        for j in range(j0, j0 + nj):
            for k in range(k0, k0 + nk):
                for _ in range(hits):
                    vm._hit_column((i + 0.5) * vm.res, (j + 0.5) * vm.res, (k + 0.5) * vm.res, (k + 0.5) * vm.res, 1.0)
    vm.version += 1


def city():
    vm = VoxelMap(res=0.5)
    block(vm, 20, -10, 0, 12, 10, 16)
    block(vm, -30, 14, 0, 8, 8, 6)
    return vm


# ------------------------------------------------------------------------------ geometry
def test_exposed_faces_only_no_internal_faces():
    vm = VoxelMap(res=0.5)
    block(vm, 0, 0, 0, 1, 1, 1)
    assert len(build_geometry(vm, ViewOptions()).verts) == 6
    vm2 = VoxelMap(res=0.5)
    block(vm2, 0, 0, 0, 2, 2, 2)
    assert len(build_geometry(vm2, ViewOptions()).verts) == 24            # 6 faces x 4 per side, no inner faces
    vm3 = VoxelMap(res=0.5)
    block(vm3, 0, 0, 0, 2, 1, 1)
    assert len(build_geometry(vm3, ViewOptions()).verts) == 10            # two touching cubes share one wall


def test_voxel_edge_equals_map_resolution_times_scale():
    vm = VoxelMap(res=0.5)
    block(vm, 4, 4, 0, 1, 1, 1)
    for scale in (1.0, 0.6):
        g = build_geometry(vm, ViewOptions(voxel_scale=scale))
        edge = np.linalg.norm(g.verts[0][1] - g.verts[0][0])
        assert edge == pytest.approx(0.5 * scale, abs=1e-5)
    g = build_geometry(vm, ViewOptions())
    cube = g.verts.reshape(-1, 3)
    assert cube.min(axis=0) == pytest.approx([2.0, 2.0, 0.0], abs=1e-5)       # voxel 4 spans [2.0, 2.5)


def test_filters_change_the_drawn_geometry_but_never_the_map():
    vm = city()
    before = (vm.version, copy.deepcopy(vm.occ), vm.ground_l.copy())
    base = build_geometry(vm, ViewOptions())
    low = build_geometry(vm, ViewOptions(z_max=3.0))
    high = build_geometry(vm, ViewOptions(z_min=4.0))
    nothing = build_geometry(vm, ViewOptions(show_occupied=False))
    strict = build_geometry(vm, ViewOptions(occ_threshold=0.99))
    ground_only = build_geometry(vm, ViewOptions(show_obstacle_voxels=False))
    assert low.n_vox < base.n_vox and high.n_vox < base.n_vox and nothing.n_vox == 0
    assert strict.n_vox <= base.n_vox and ground_only.n_vox < base.n_vox
    assert vm.version == before[0] and vm.occ == before[1] and np.array_equal(vm.ground_l, before[2])


def test_colour_modes_give_different_colours():
    vm = city()
    cols = {m: build_geometry(vm, ViewOptions(color_mode=m)).colors.mean(axis=0) for m in ("height", "occupancy", "age", "class")}
    assert len({tuple(np.round(c)) for c in cols.values()}) == 4


def test_occupancy_colour_is_more_saturated_for_higher_probability():
    from drone_agent.voxel_viewer import voxel_colors
    o = ViewOptions()
    lo = voxel_colors("occupancy", np.array([1.0]), np.array([0.55]), np.zeros(1), 0.5, o)[0]
    hi = voxel_colors("occupancy", np.array([1.0]), np.array([0.98]), np.zeros(1), 0.5, o)[0]
    sat = lambda c: (c.max() - c.min()) / max(1.0, c.max())
    assert sat(hi) > sat(lo)


# --------------------------------------------------------------------------------- camera
def test_camera_presets_orbit_zoom_pan_focus_reset():
    cam = OrbitCamera()
    cam.preset("top")
    for _ in range(80):
        cam.update(0.05)
    assert cam.el > 85
    cam.preset("side")
    for _ in range(80):
        cam.update(0.05)
    assert cam.el < 10
    cam.preset("iso")
    for _ in range(80):
        cam.update(0.05)
    assert 30 < cam.el < 40
    d0 = cam.goal["dist"]
    cam.zoom(2.0)
    assert cam.goal["dist"] < d0
    cam.focus((20.0, -7.0, 1.0), dist=15.0)
    for _ in range(120):
        cam.update(0.05)
    assert np.allclose(cam.target, [20.0, -7.0, 1.0], atol=0.1) and abs(cam.dist - 15.0) < 0.2
    u, v, z = cam.project(np.array([[20.0, -7.0, 1.0]]), W, H)
    assert abs(u[0] - W / 2) < 2 and abs(v[0] - H / 2) < 2                  # focus target is centred
    cam.reset()
    for _ in range(200):
        cam.update(0.05)
    assert np.allclose(cam.target, cam.home["target"], atol=0.1)


def test_camera_motion_is_smooth_not_a_jump():
    cam = OrbitCamera()
    cam.focus((30.0, 30.0, 0.0), dist=20.0)
    steps = []
    last = cam.target.copy()
    for _ in range(30):
        cam.update(1 / 30)
        steps.append(float(np.linalg.norm(cam.target - last)))
        last = cam.target.copy()
    assert max(steps) < 0.5 * float(np.linalg.norm(np.array([30.0, 30.0, 0.0]) - OrbitCamera().target))
    assert steps[0] > 0


def test_east_is_to_the_right_when_looking_north():
    cam = OrbitCamera(target=(0, 0, 0), az=180.0, el=30.0, dist=50)          # camera south of target, looks north
    cam.target = cam.goal["target"] = np.zeros(3)
    u_e, _, _ = cam.project(np.array([[0.0, 10.0, 0.0]]), W, H)
    u_w, _, _ = cam.project(np.array([[0.0, -10.0, 0.0]]), W, H)
    assert u_e[0] > u_w[0]


# --------------------------------------------------------------------------------- render
def _scene_with_everything():
    sc = Scene(now=10.0, hub=(0.0, 0.0))
    sc.drones = [dict(id=0, name="UAV-1", n=-10.0, e=5.0, alt=12.0, yaw=0.5, trail=[(-14.0, 0.0, 10.0), (-10.0, 5.0, 12.0)])]
    sc.rovers = [dict(id=0, name="Rover 1", n=-5.0, e=-5.0, yaw=0.3, color=ROVER_COLORS[0], state="navigating", wheel=0.0,
                      trail=[(-8.0, -8.0), (-5.0, -5.0)], route=[(-5.0, -5.0), (0.0, -3.0), (10.0, 4.0)])]
    sc.survivors = [dict(id=1, name="Standing M1", n=10.0, e=4.0, conf=0.91, drone_id=0, state="detected", radius=3.0)]
    sc.stats = {"detected": 1, "queued": 0, "en_route": 1, "rescued": 0, "replans": 0, "lines": ["M1 Standing M1 -> Rover 1"]}
    return sc


def test_render_shapes_and_voxels_appear_in_the_image():
    vm = city()
    v = VoxelViewer(vm)
    v.cam.focus((0, 0, 3), dist=60)
    for _ in range(100):
        v.cam.update(0.05)
    empty = VoxelViewer(VoxelMap(res=0.5))
    for _ in range(100):
        empty.cam.update(0.05)
    a = v.render(W, H, Scene(), async_geometry=False)
    b = empty.render(W, H, Scene(), async_geometry=False)
    assert a.shape == (H, W, 3) and a.dtype == np.uint8
    assert np.abs(a.astype(int) - b.astype(int)).sum() > 1e6                 # the map is visibly drawn
    assert v.draw_faces > 100


def test_viewer_options_never_modify_the_map_and_each_changes_the_image():
    vm = city()
    snap = (vm.version, copy.deepcopy(vm.occ), vm.ground_l.copy())
    base_opts = ViewOptions()
    v = VoxelViewer(vm, base_opts)
    for _ in range(60):
        v.cam.update(0.05)
    sc = _scene_with_everything()
    ref = v.render(W, H, sc, async_geometry=False).astype(int)
    changes = dict(show_occupied=False, show_unknown=False, show_shadows=False, show_routes=False, show_survivors=False,
                   show_coverage=True, show_free=True, opacity=0.4, voxel_scale=0.6, color_mode="occupancy",
                   z_max=4.0, show_trails=False, show_edges=False)
    for k, val in changes.items():
        v2 = VoxelViewer(vm, ViewOptions(**{k: val}))
        for _ in range(60):
            v2.cam.update(0.05)
        img = v2.render(W, H, sc, async_geometry=False).astype(int)
        assert np.abs(img - ref).sum() > 0, f"{k} had no visible effect"
    assert vm.version == snap[0] and vm.occ == snap[1] and np.array_equal(vm.ground_l, snap[2])


def test_survivor_marker_and_rover_route_are_drawn_in_map_coordinates():
    vm = city()
    v = VoxelViewer(vm)
    v.opts.show_recent = False
    v.cam.focus((3.0, 0.0, 0.0), dist=45.0)
    for _ in range(100):
        v.cam.update(0.05)
    sc = _scene_with_everything()
    img = v.render(W, H, sc, async_geometry=False)
    # the route's middle waypoint (0, -3) must be a rover-1 coloured pixel (line + waypoint dot)
    u, vv, z = v.cam.project(np.array([[0.0, -3.0, 0.25]]), W, H)
    patch = img[int(vv[0]) - 3:int(vv[0]) + 4, int(u[0]) - 3:int(u[0]) + 4].reshape(-1, 3)
    assert any(np.abs(p.astype(int) - np.array(ROVER_COLORS[0])).sum() < 40 for p in patch)
    # the survivor's vertical beam lives exactly above its (n, e)
    u2, v2, _ = v.cam.project(np.array([[10.0, 4.0, 5.0]]), W, H)
    patch2 = img[int(v2[0]) - 3:int(v2[0]) + 4, int(u2[0]) - 3:int(u2[0]) + 4].reshape(-1, 3)
    assert any(np.abs(p.astype(int) - np.array(SURVIVOR_COLOR)).sum() < 60 for p in patch2)


def test_recent_updates_and_new_obstacles_are_highlighted_then_fade():
    vm = VoxelMap(res=0.5)
    now = vm.now()
    block(vm, 10, 0, 0, 3, 3, 3)
    for key, val in vm.occ.items():
        val[2] = now
        val[1] = now
    v = VoxelViewer(vm, ViewOptions(show_edges=False))
    v.cam.focus((6.0, 1.0, 0.5), dist=14.0)
    for _ in range(100):
        v.cam.update(0.05)
    fresh = v.render(W, H, Scene(), async_geometry=False).astype(int)
    for val in vm.occ.values():                        # an hour later: nothing is "new" any more
        val[1] = val[2] = now - 3600.0
    v.geo.version = -1
    old = v.render(W, H, Scene(), async_geometry=False).astype(int)
    assert np.abs(fresh - old).sum() > 5e4
    assert v.geo.t_occ.size


def test_rover_and_drone_models_are_3d_not_dots():
    vm = VoxelMap(res=0.5)
    v = VoxelViewer(vm, ViewOptions(model_scale=2.2))
    v.cam.focus((-5.0, -5.0, 0.5), dist=14.0)
    for _ in range(100):
        v.cam.update(0.05)
    sc = _scene_with_everything()
    sc.survivors, sc.drones = [], []
    sc.rovers[0]["route"] = []
    a = v.render(W, H, sc, async_geometry=False)
    sc.rovers = []
    b = v.render(W, H, sc, async_geometry=False)
    diff = (np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 30)
    ys, xs = np.nonzero(diff)
    assert diff.sum() > 400 and (xs.max() - xs.min()) > 25 and (ys.max() - ys.min()) > 12   # extended body, not a point
    # the model is drawn from several shaded faces: at least 6 distinct colours inside the changed area
    assert len({tuple(p) for p in a[diff]}) >= 6


def test_rover_heading_is_visible_front_vs_rear():
    from drone_agent.voxel_viewer import rover_model
    parts = rover_model(0.0, 0.0, 0.0, ROVER_COLORS[0], 1.0)
    # front light bar is at +forward (north when yaw=0), rear lights at -forward
    centers = [(p[1][0][0], tuple(np.round(p[3][0]))) for p in parts]
    front = max(centers)[0]
    rear = min(centers)[0]
    assert front > 0.4 and rear < -0.3


def test_performance_budget_with_many_voxels():
    vm = VoxelMap(res=0.5)
    rng = np.random.default_rng(0)
    for _ in range(40):
        i0, j0 = rng.integers(-100, 80, 2)
        block(vm, int(i0), int(j0), 0, int(rng.integers(6, 16)), int(rng.integers(6, 16)), int(rng.integers(4, 20)), hits=2)
    v = VoxelViewer(vm)
    v.build_now()
    for _ in range(60):
        v.cam.update(0.05)
    v.render(W, H, Scene(), async_geometry=False)
    t0 = time.perf_counter()
    for _ in range(5):
        v.render(W, H, Scene(), async_geometry=False)
    per = (time.perf_counter() - t0) / 5
    assert v.geo.n_vox > 15000
    assert per < 0.6, f"{per * 1000:.0f} ms per frame"


def test_background_geometry_build_swaps_in_without_blocking():
    vm = city()
    v = VoxelViewer(vm)
    assert v.geo.version == -1
    t0 = time.perf_counter()
    v.render(W, H, Scene())                           # starts the build thread and returns immediately
    assert time.perf_counter() - t0 < 0.5
    for _ in range(100):
        time.sleep(0.05)
        v.render(W, H, Scene())
        if v.geo.version == vm.version:
            break
    assert v.geo.version == vm.version and v.geo.n_vox > 0



# ------------------------------------------------------------------------ rover camera (first person)
def _pov_setup():
    vm = city()
    main = VoxelViewer(vm)
    main.build_now()
    pov = VoxelViewer.first_person(main)
    sc = _scene_with_everything()
    return vm, main, pov, sc


def test_rover_camera_shares_the_map_and_never_rebuilds_geometry():
    vm, main, pov, sc = _pov_setup()
    assert pov.geo is main.geo and pov.external_geo
    img = pov.render_pov(main, 320, 200, sc, sc.rovers[0])
    assert img.shape == (200, 320, 3)
    assert pov.geo is main.geo                         # still the very same geometry object


def test_rover_camera_looks_where_the_rover_is_heading():
    vm, main, pov, sc = _pov_setup()
    r = dict(sc.rovers[0], n=14.0, e=-4.0, yaw=0.0)       # facing north toward the first block (n 10..16, e -5..0)
    toward = pov.render_pov(main, 320, 200, sc, r).astype(int)
    away = pov.render_pov(main, 320, 200, sc, dict(r, yaw=math.pi)).astype(int)
    sc2 = Scene(); sc2.rovers = [dict(r)]
    assert np.abs(toward - away).sum() > 2e5            # a different view when it turns around
    # the block is in front: near-wall pixels fill much more of the image than when looking away from it
    bg = main._background(320, 200).astype(int)
    assert (np.abs(toward - bg).sum(axis=2) > 30).mean() > (np.abs(away - bg).sum(axis=2) > 30).mean() - 0.5


def test_rover_camera_sees_a_survivor_marker_ahead_but_not_its_own_body():
    vm, main, pov, sc = _pov_setup()
    sc.survivors = [dict(id=1, name="Standing M1", n=10.0, e=4.0, conf=0.9, drone_id=0, state="detected", radius=3.0)]
    sc.rovers[0].update(n=0.0, e=4.0, yaw=0.0)            # 10 m south of the survivor, looking north
    img = pov.render_pov(main, 320, 200, sc, sc.rovers[0])
    pos, right, up, fwd = pov.cam.pose()
    u, v, z = pov.cam.project(np.array([[10.0, 4.0, 4.0]]), 320, 200)
    assert z[0] > 1 and 0 < u[0] < 320                      # the survivor is inside the field of view
    patch = img[max(0, int(v[0]) - 6):int(v[0]) + 7, max(0, int(u[0]) - 6):int(u[0]) + 7].reshape(-1, 3)
    assert any(np.abs(p.astype(int) - np.array(SURVIVOR_COLOR)).sum() < 80 for p in patch)
    assert pov.cam.eye[:2].tolist() == [0.0, 4.0]


def test_two_rover_cameras_give_different_views():
    vm, main, pov, sc = _pov_setup()
    pov2 = VoxelViewer.first_person(main)
    sc.rovers.append(dict(sc.rovers[0], id=1, n=-20.0, e=10.0, yaw=2.0, name="Rover 2", color=ROVER_COLORS[1]))
    a = pov.render_pov(main, 320, 200, sc, sc.rovers[0]).astype(int)
    b = pov2.render_pov(main, 320, 200, sc, sc.rovers[1]).astype(int)
    assert np.abs(a - b).sum() > 2e5
