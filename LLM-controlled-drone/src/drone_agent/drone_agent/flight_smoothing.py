"""Trajectory and yaw smoothing for the offboard setpoints (ROS-free, unit-tested).

Why: the translator used to send the FINAL target as a raw position setpoint at 10 Hz. PX4's position
controller then accelerates as hard as its limits allow toward a goal tens of metres away and brakes
late, the target (and the yaw setpoint) jump whenever avoidance or the allocator changes them, and yaw
snapped by up to 180 degrees. A fixed camera on the airframe shows every one of those as shake.

Now the translator streams a moving "carrot": a point that chases the target with limits on speed,
acceleration AND jerk, plus matching velocity/acceleration feed-forward. The drone follows the carrot,
so a jump in the target becomes a gentle ramp. Yaw is rate- and acceleration-limited and takes the
shortest way round.

All quantities are NED (x north, y east, z DOWN, so climbing means negative z).
"""
from __future__ import annotations

import math

import numpy as np


class JerkLimitedCarrot:
    def __init__(self, vmax_xy: float = 3.0, vmax_up: float = 1.4, vmax_down: float = 1.0,
                 amax_xy: float = 1.0, amax_z: float = 0.7, jmax: float = 1.6, tau: float = 0.9):
        self.vmax_xy, self.vmax_up, self.vmax_down = vmax_xy, vmax_up, vmax_down
        self.amax_xy, self.amax_z, self.jmax, self.tau = amax_xy, amax_z, jmax, tau
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.ready = False

    def reset(self, pos, vel=None):
        self.p = np.array(pos, float)
        self.v = np.zeros(3) if vel is None else np.array(vel, float)
        self.a = np.zeros(3)
        self.ready = True

    def step(self, target, dt: float):
        """Advance the carrot by dt seconds toward `target`. Returns (position, velocity, acceleration)."""
        dt = float(np.clip(dt, 1e-3, 0.25))
        tgt = np.array(target, float)
        err = tgt - self.p
        # desired velocity: toward the target, never faster than a speed from which we can still stop
        dist_xy = float(np.hypot(err[0], err[1]))
        v_des = np.zeros(3)
        if dist_xy > 1e-6:
            brake = math.sqrt(2.0 * 0.4 * self.amax_xy * dist_xy)       # margin: jerk limiting delays the braking
            v_des[:2] = err[:2] / dist_xy * min(self.vmax_xy, brake, 0.7 * dist_xy)     # linear creep near the goal
        vz_cap = self.vmax_down if err[2] > 0 else self.vmax_up
        v_des[2] = math.copysign(min(vz_cap, math.sqrt(2.0 * 0.4 * self.amax_z * abs(err[2])), 0.7 * abs(err[2])), err[2]) if abs(err[2]) > 1e-6 else 0.0
        # acceleration command, then acceleration limit, then jerk limit
        a_cmd = (v_des - self.v) / self.tau
        n_xy = float(np.hypot(a_cmd[0], a_cmd[1]))
        if n_xy > self.amax_xy:
            a_cmd[:2] *= self.amax_xy / n_xy
        a_cmd[2] = float(np.clip(a_cmd[2], -self.amax_z, self.amax_z))
        da = a_cmd - self.a
        n_da = float(np.linalg.norm(da))
        max_da = self.jmax * dt
        if n_da > max_da:
            da *= max_da / n_da
        self.a = self.a + da
        self.p = self.p + self.v * dt + 0.5 * self.a * dt * dt
        self.v = self.v + self.a * dt
        return self.p.copy(), self.v.copy(), self.a.copy()


def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class YawSmoother:
    """Rate- and acceleration-limited yaw that always turns the short way; NaN target = hold heading."""

    def __init__(self, rate_max_deg: float = 30.0, acc_max_deg: float = 45.0):
        self.rate_max = math.radians(rate_max_deg)
        self.acc_max = math.radians(acc_max_deg)
        self.yaw = 0.0
        self.rate = 0.0
        self.ready = False

    def reset(self, yaw: float):
        self.yaw = wrap_pi(float(yaw))
        self.rate = 0.0
        self.ready = True

    def step(self, desired: float, dt: float):
        dt = float(np.clip(dt, 1e-3, 0.25))
        if desired is None or not math.isfinite(desired):
            want = 0.0
        else:
            err = wrap_pi(desired - self.yaw)
            # proportional turn rate that also brakes in time: rate <= sqrt(2 * acc * |err|)
            cap = min(self.rate_max, math.sqrt(2.0 * 0.6 * self.acc_max * abs(err)))
            want = math.copysign(min(cap, 1.2 * abs(err)), err)
        d = float(np.clip(want - self.rate, -self.acc_max * dt, self.acc_max * dt))
        self.rate += d
        self.yaw = wrap_pi(self.yaw + self.rate * dt)
        return self.yaw, self.rate
