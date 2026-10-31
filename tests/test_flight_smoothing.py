"""Tests for the offboard trajectory/yaw smoothing. Run: python -m pytest tests/test_flight_smoothing.py -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LLM-controlled-drone" / "src" / "drone_agent"))

from drone_agent.flight_smoothing import JerkLimitedCarrot, YawSmoother, wrap_pi   # noqa: E402


def run(carrot, targets, dt=0.04):
    out = []
    for t in targets:
        out.append(carrot.step(t, dt))
    return out


def test_respects_speed_acceleration_and_jerk_limits_on_a_huge_target_jump():
    c = JerkLimitedCarrot()
    c.reset([0, 0, -9.0])
    dt = 0.04
    traj = run(c, [[60.0, 40.0, -9.0]] * 3000, dt)
    p = np.array([x[0] for x in traj]); v = np.array([x[1] for x in traj]); a = np.array([x[2] for x in traj])
    assert np.hypot(v[:, 0], v[:, 1]).max() <= c.vmax_xy + 1e-6
    assert np.hypot(a[:, 0], a[:, 1]).max() <= c.amax_xy + 1e-6
    jerk = np.linalg.norm(np.diff(a, axis=0), axis=1) / dt
    assert jerk.max() <= c.jmax + 1e-6
    # it still gets there, and does not overshoot
    assert np.linalg.norm(p[-1] - [60, 40, -9]) < 0.3
    assert np.hypot(p[:, 0] - 60, p[:, 1] - 40).min() < 0.3 and p[:, 0].max() < 60.5 and p[:, 1].max() < 40.5   # overshoot < 0.5 m on a 72 m move


def test_vertical_limits_up_and_down_differ_and_hold():
    c = JerkLimitedCarrot()
    c.reset([0, 0, 0.0])
    up = run(c, [[0, 0, -12.0]] * 2500)
    assert max(-x[1][2] for x in up) <= c.vmax_up + 1e-6
    assert abs(up[-1][0][2] + 12.0) < 0.2
    down = run(c, [[0, 0, 0.0]] * 2500)
    assert max(x[1][2] for x in down) <= c.vmax_down + 1e-6


def test_converts_a_jittery_target_into_a_gentle_path():
    rng = np.random.default_rng(0)
    base = np.array([30.0, 0.0, -10.0])
    jitter = [base + np.array([rng.normal(0, 3.0), rng.normal(0, 3.0), 0.0]) for _ in range(1500)]   # avoidance/allocator noise
    c = JerkLimitedCarrot()
    c.reset([30.0, 0.0, -10.0])
    sm = run(c, jitter)
    raw = np.array(jitter)
    raw_acc = np.diff(raw, 2, axis=0) / 0.04 ** 2
    acc = np.array([x[2] for x in sm])
    assert np.sqrt((acc ** 2).sum(axis=1).mean()) < 0.1 * np.sqrt((raw_acc ** 2).sum(axis=1).mean())
    assert np.hypot(acc[:, 0], acc[:, 1]).max() <= c.amax_xy + 1e-6


def test_stops_smoothly_when_the_target_becomes_the_current_position():
    c = JerkLimitedCarrot()
    c.reset([0, 0, -9.0])
    run(c, [[150.0, 0.0, -9.0]] * 400)                # get up to speed (far target)
    assert np.linalg.norm(c.v) > 1.5
    pos = c.p.copy()
    out = run(c, [pos] * 600)
    assert np.linalg.norm(out[-1][1]) < 0.05           # came to rest
    assert np.linalg.norm(out[-1][0] - pos) < 3.5      # within braking distance of v^2 / (2 a)


def test_yaw_takes_the_short_way_and_is_rate_limited():
    y = YawSmoother(rate_max_deg=30, acc_max_deg=45)
    y.reset(math.radians(170.0))
    dt = 0.04
    seq = [y.step(math.radians(-170.0), dt) for _ in range(300)]     # 20 deg away through +-180, not 340 deg
    yaws = np.array([s[0] for s in seq]); rates = np.array([s[1] for s in seq])
    assert np.abs(rates).max() <= math.radians(30) + 1e-9
    steps = np.abs([wrap_pi(b - a) for a, b in zip(yaws[:-1], yaws[1:])])
    assert steps.max() <= math.radians(30) * dt + 1e-9              # never snaps
    assert abs(wrap_pi(yaws[-1] - math.radians(-170.0))) < math.radians(1)
    assert np.abs(np.diff(rates)).max() <= math.radians(45) * dt + 1e-9


def test_yaw_holds_on_nan_and_does_not_flip_with_noisy_targets():
    rng = np.random.default_rng(1)
    y = YawSmoother()
    y.reset(0.3)
    for _ in range(50):
        y.step(float("nan"), 0.04)
    assert y.yaw == pytest.approx(0.3, abs=1e-6)
    seq = [y.step(0.3 + rng.normal(0, 0.8), 0.04)[0] for _ in range(500)]
    swing = np.abs(np.diff(np.unwrap(seq)))
    assert swing.max() <= math.radians(30) * 0.04 + 1e-9

