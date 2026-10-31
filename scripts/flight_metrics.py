#!/usr/bin/env python3
"""Measure how steady the 4 drones (and so their cameras) are, from PX4's own odometry.

Run while the swarm is flying (needs ROS 2 + px4_msgs sourced):
    python3 scripts/flight_metrics.py 60          # seconds to record
Prints, per drone, using PX4 timestamps (so results do not depend on the Gazebo real-time factor):
    tilt RMS / max (deg)         how much the airframe leans, i.e. how much the camera pitch/roll wanders
    rate RMS / p95 (deg/s)       body angular rate: the camera shake you actually see in the video
    yaw rate p95 (deg/s)         how violently the camera pans
    accel RMS / jerk p95         translational smoothness
Lower is steadier. A camera looks 'shaky' roughly when p95 body rate is above ~15 deg/s.
"""
import json
import math
import sys
import time

import numpy as np
import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

TOPICS = ['/fmu/out/vehicle_odometry', '/px4_1/fmu/out/vehicle_odometry',
          '/px4_2/fmu/out/vehicle_odometry', '/px4_3/fmu/out/vehicle_odometry']


class Recorder(Node):
    def __init__(self):
        super().__init__('flight_metrics')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.rows = {i: [] for i in range(4)}
        for i, t in enumerate(TOPICS):
            self.create_subscription(VehicleOdometry, t, lambda m, i=i: self._cb(i, m), qos)

    def _cb(self, i, m):
        w, x, y, z = (float(v) for v in m.q[:4])
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        self.rows[i].append((m.timestamp / 1e6, roll, pitch, *map(float, m.angular_velocity[:3]),
                             *map(float, m.velocity[:3]), *map(float, m.position[:3])))


def summarize(rows):
    a = np.array(rows)
    if len(a) < 20:
        return None
    t = a[:, 0]
    keep = np.diff(t, prepend=t[0] - 1) > 0
    a, t = a[keep], t[keep]
    roll, pitch = np.degrees(a[:, 1]), np.degrees(a[:, 2])
    rate = np.degrees(np.linalg.norm(a[:, 3:6], axis=1))
    yaw_rate = np.degrees(np.abs(a[:, 5]))
    vel = a[:, 6:9]
    dt = np.diff(t)
    acc = np.diff(vel, axis=0) / dt[:, None]
    acc_n = np.linalg.norm(acc, axis=1)
    jerk = np.abs(np.diff(acc_n) / dt[1:])
    tilt = np.sqrt(roll ** 2 + pitch ** 2)
    return {"samples": len(a), "sim_s": round(float(t[-1] - t[0]), 1),
            "tilt_rms": round(float(np.sqrt((tilt ** 2).mean())), 2), "tilt_max": round(float(tilt.max()), 1),
            "rate_rms": round(float(np.sqrt((rate ** 2).mean())), 2), "rate_p95": round(float(np.percentile(rate, 95)), 2),
            "yaw_rate_p95": round(float(np.percentile(yaw_rate, 95)), 2),
            "accel_rms": round(float(np.sqrt((acc_n ** 2).mean())), 2),
            "jerk_p95": round(float(np.percentile(jerk, 95)), 1),
            "alt_m": round(float(-a[-1, 11]), 1)}


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    rclpy.init()
    node = Recorder()
    t0 = time.time()
    while time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    out = {f"UAV-{i + 1}": summarize(r) for i, r in node.rows.items()}
    print(json.dumps(out, indent=1))
    ok = [v for v in out.values() if v]
    if ok:
        print("MEAN  tilt_rms %.2f deg | rate_p95 %.2f deg/s | yaw_rate_p95 %.2f deg/s | jerk_p95 %.1f" % (
            np.mean([v["tilt_rms"] for v in ok]), np.mean([v["rate_p95"] for v in ok]),
            np.mean([v["yaw_rate_p95"] for v in ok]), np.mean([v["jerk_p95"] for v in ok])))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
