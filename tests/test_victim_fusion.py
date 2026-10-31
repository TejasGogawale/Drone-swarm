"""Tests for the ROS-free Gazebo bridge: camera geometry, appearance feature, and the full
detections -> de-duplicated count pipeline (victim_fusion.py).

Run:  python -m pytest tests/test_victim_fusion.py -q
No ROS, Gazebo or YOLO needed: detections are synthesised by FORWARD-projecting known people
through the same camera model the real detector uses, then fed to the pipeline as JSON.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "LLM-controlled-drone" / "src" / "drone_agent"
sys.path.insert(0, str(SRC))

from drone_agent.victim_fusion import DEFAULT_ORIGINS, VictimFusion          # noqa: E402
from drone_agent.victim_geometry import (Camera, Pose, RunningCentering,       # noqa: E402
                                         color_histogram, ground_covariance, pixel_to_ground,
                                         project_ned_to_pixel)

CAM = Camera()


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("seed", range(5))
def test_pixel_to_ground_inverts_the_detector_projection(seed):
    rng = np.random.default_rng(seed)
    for _ in range(40):
        pose = Pose(rng.uniform(-5, 5), rng.uniform(-5, 5), -rng.uniform(8, 14),
                    rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(-math.pi, math.pi))
        target = (rng.uniform(-15, 15), rng.uniform(-15, 15), 0.0)
        proj = project_ned_to_pixel(CAM, pose, *target)
        if proj is None or not (0 <= proj[0] < CAM.width and 0 <= proj[1] < CAM.height):
            continue
        g = pixel_to_ground(CAM, pose, proj[0], proj[1])
        assert g is not None
        assert np.allclose(g, target[:2], atol=1e-6)


def test_rays_at_or_above_the_horizon_have_no_ground_point():
    pose = Pose(0, 0, -10, 0, 0, 0)                  # camera pitched 25 deg down, vfov ~ 1.3 rad
    assert pixel_to_ground(CAM, pose, CAM.width / 2, 0.0) is None     # top row looks above horizon


def test_ground_covariance_grows_with_distance_and_attitude_noise():
    pose = Pose(0, 0, -10, 0, 0, 0)
    near = ground_covariance(CAM, pose, 640, 700, px_sigma=6)
    far = ground_covariance(CAM, pose, 640, 520, px_sigma=6)
    assert np.trace(far) > np.trace(near) > 0
    noisy = ground_covariance(CAM, pose, 640, 700, px_sigma=6, att_sigma=math.radians(5))
    assert np.trace(noisy) > np.trace(near)


# ----------------------------------------------------------------- appearance
def test_colour_histogram_separates_clothing_and_centering_decorrelates():
    rng = np.random.default_rng(0)

    def person(top, bottom):
        img = np.zeros((60, 20, 3), np.uint8)
        img[:30] = top
        img[30:] = bottom
        return np.clip(img + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)

    red_blue = [color_histogram(person((220, 30, 30), (30, 30, 200))) for _ in range(6)]
    green_grey = [color_histogram(person((30, 200, 30), (120, 120, 120))) for _ in range(6)]
    c = RunningCentering()
    out = [c(h) for h in red_blue + green_grey]
    out = [o for o in out if o is not None]
    assert len(out) >= 8                                   # first few crops give no appearance
    same = np.dot(out[-3], out[-2])                         # green_grey vs green_grey
    diff = np.dot(out[-3], out[0])                          # green_grey vs red_blue
    assert same > 0.6 and diff < 0.2


def test_degenerate_crop_is_safe():
    assert color_histogram(np.zeros((1, 1, 3), np.uint8)).sum() == 0


# -------------------------------------------------------------- full pipeline
PEOPLE = {0: (8.0, 6.0), 1: (-7.0, 9.0), 2: (4.0, -8.0), 3: (-9.0, -6.0), 4: (0.0, 11.0)}   # world NED
CLOTHES = {k: np.random.default_rng(100 + k).uniform(0, 255, size=6) for k in PEOPLE}


def _person_crop(k, rng):
    img = np.zeros((50, 18, 3), np.uint8)
    img[:25] = CLOTHES[k][:3]
    img[25:] = CLOTHES[k][3:]
    return np.clip(img + rng.integers(-15, 15, img.shape), 0, 255).astype(np.uint8)


def _flight(fusion, ticks, seed=0, pose_bias=0.25, people=PEOPLE, fp_every=0, orbit=True):
    """4 drones orbit a hover point; each tick every drone 'detects' every visible person."""
    rng = np.random.default_rng(seed)
    bias = {i: rng.normal(scale=pose_bias, size=2) for i in range(fusion.n)}
    sent = 0
    for t in range(ticks):
        for i in range(fusion.n):
            ang = 0.05 * t + i * math.pi / 2 if orbit else i * math.pi / 2
            o = DEFAULT_ORIGINS[i]
            true_world = np.array([5.0 * math.cos(ang), 5.0 * math.sin(ang)])
            true = Pose(true_world[0] - o[0], true_world[1] - o[1], -11.0 - i,
                        rng.normal(scale=0.01), rng.normal(scale=0.01),
                        math.atan2(-true_world[1], -true_world[0]) + rng.normal(scale=0.01))
            reported = Pose(true.n + bias[i][0], true.e + bias[i][1], true.d, true.roll, true.pitch, true.yaw)
            fusion.set_pose(i, reported)
            dets = []
            for k, (pn, pe) in people.items():
                proj = project_ned_to_pixel(CAM, true, pn - o[0], pe - o[1], 0.0)
                if proj is None or not (30 < proj[0] < CAM.width - 30 and 70 < proj[1] < CAM.height - 5):
                    continue
                u, v = proj[0] + rng.normal(scale=4), proj[1] + rng.normal(scale=4)
                if rng.random() < 0.15:
                    continue                                         # missed detection
                emb = color_histogram(_person_crop(k, rng))
                dets.append({"box": [int(u - 14), int(v - 55), int(u + 14), int(v)], "conf": 0.8,
                             "emb": emb.round(3).tolist()})
            if fp_every and t % fp_every == 0 and i == 0:           # one-off ghost detection
                dets.append({"box": [300, 400, 330, 460], "conf": 0.5,
                             "emb": color_histogram(_person_crop(0, rng)).round(3).tolist()})
            if dets:
                payload = {"drone_id": i, "stamp": t * 0.2, "width": CAM.width, "height": CAM.height, "dets": dets}
                sent += fusion.on_detections(json.loads(json.dumps(payload)), t)
        fusion.step(t)
    return sent


def test_pipeline_counts_each_person_once_and_localises_them():
    f = VictimFusion(swarm_root=str(Path(__file__).resolve().parents[1]))
    sent = _flight(f, 120)
    rep = f.report()
    assert sent > 200
    assert rep["count"] == len(PEOPLE)
    for v in rep["victims"]:
        if v["confirmed"]:
            d = min(math.dist(v["n_ned"], p) for p in PEOPLE.values())
            assert d < 2.0, v
    assert set(rep["per_drone_count"].values()) == {len(PEOPLE)}       # every drone agrees


def test_without_sharing_nobody_confirms_when_two_viewpoints_are_required():
    f = VictimFusion(comm_range_m=0.1, swarm_root=str(Path(__file__).resolve().parents[1]))
    _flight(f, 120)
    # every drone sees everyone here, but with no gossip no registry ever holds two viewpoints
    assert f.report()["count"] == 0 and set(f.report()["per_drone_count"].values()) == {0}


def test_one_off_ghost_detection_is_not_counted():
    f = VictimFusion(swarm_root=str(Path(__file__).resolve().parents[1]))
    _flight(f, 120, fp_every=40)
    assert f.report()["count"] == len(PEOPLE)


def test_persistent_single_drone_ghost_is_not_counted_with_two_viewpoint_rule():
    f = VictimFusion(swarm_root=str(Path(__file__).resolve().parents[1]))
    _flight(f, 120, fp_every=1)                 # drone 0 hallucinates a person in every frame
    assert f.report()["count"] == len(PEOPLE)


def test_persistent_ghost_is_counted_if_only_one_viewpoint_is_required():
    """Why min_agents=2 exists: with one viewpoint a repeated hallucination becomes 'victims'."""
    f = VictimFusion(reid_cfg_kw={"min_agents": 1}, swarm_root=str(Path(__file__).resolve().parents[1]))
    _flight(f, 120, fp_every=1)
    assert f.report()["count"] > len(PEOPLE) + 3


def test_known_limit_a_static_mannequin_seen_by_two_drones_is_counted():
    """Documented limitation of an RGB-only setup: YOLO cannot tell a mannequin from a person."""
    people = dict(PEOPLE)
    people[5] = (6.0, -2.0)
    CLOTHES[5] = np.random.default_rng(105).uniform(0, 255, size=6)
    f = VictimFusion(swarm_root=str(Path(__file__).resolve().parents[1]))
    _flight(f, 120, people=people)
    assert f.report()["count"] == len(people)


def test_known_limit_a_person_seen_by_only_one_drone_is_not_counted():
    one = VictimFusion(swarm_root=str(Path(__file__).resolve().parents[1]))
    pose = Pose(0, 0, -11, 0, 0, 0)
    one.set_pose(0, pose)
    rng = np.random.default_rng(1)
    proj = project_ned_to_pixel(CAM, pose, 8.0 - DEFAULT_ORIGINS[0][0], 0.0 - DEFAULT_ORIGINS[0][1], 0.0)
    for t in range(30):
        u, v = proj[0], proj[1]
        one.on_detections({"drone_id": 0, "width": 1280, "height": 960,
                           "dets": [{"box": [int(u - 14), int(v - 55), int(u + 14), int(v)], "conf": 0.9,
                                     "emb": color_histogram(_person_crop(0, rng)).round(3).tolist()}]}, t)
        one.step(t)
    assert one.report()["count"] == 0
    assert len(one.regs[0].tracks) >= 1          # it is tracked, just not confirmed


def test_low_confidence_and_missing_pose_are_ignored():
    f = VictimFusion(swarm_root=str(Path(__file__).resolve().parents[1]))
    payload = {"drone_id": 0, "width": 1280, "height": 960,
               "dets": [{"box": [600, 400, 640, 500], "conf": 0.1, "emb": [0.0] * 128}]}
    assert f.on_detections(payload, 0) == 0 and f.stats["low_conf"] == 1
    payload["dets"][0]["conf"] = 0.9
    assert f.on_detections(payload, 0) == 0 and f.stats["no_pose"] == 1


def test_registry_snapshot_roundtrip_matches_object_merge():
    from sim.reid import VictimRegistry
    f = VictimFusion(swarm_root=str(Path(__file__).resolve().parents[1]))
    _flight(f, 60)
    a = f.regs[0]
    copy = VictimRegistry.from_snapshot(json.loads(json.dumps(a.snapshot())), a.cfg)
    assert sorted(copy.tracks) == sorted(a.tracks)
    assert copy.links == a.links
    assert [v["id"] for v in copy.victims()] == [v["id"] for v in a.victims()]
