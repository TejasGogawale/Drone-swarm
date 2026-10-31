#!/usr/bin/env python3
"""Real-ROS check of the victim_registry node (no Gazebo, no PX4).

Publishes synthetic px4_msgs/VehicleOdometry for 4 drones and /yolo/raw_detections JSON for 5 known
people (forward-projected through the detector's camera model), then reads /swarm/victims.

Run in WSL/Ubuntu after `colcon build` (see PROGRESS.md):
  source /opt/ros/jazzy/setup.bash && source ~/swarm_ws/install/setup.bash
  ros2 run drone_agent victim_registry &
  python3 tests/ros_integration_victim_registry.py
Exit code 0 only if the node reports exactly 5 confirmed victims, each near a real person.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_victim_fusion as T                                       # noqa: E402
from drone_agent.victim_fusion import DEFAULT_ORIGINS                 # noqa: E402
from drone_agent.victim_geometry import color_histogram, project_ned_to_pixel, Pose  # noqa: E402

TOPICS = ['/fmu/out/vehicle_odometry', '/px4_1/fmu/out/vehicle_odometry',
          '/px4_2/fmu/out/vehicle_odometry', '/px4_3/fmu/out/vehicle_odometry']


class SyntheticSwarm(Node):
    def __init__(self, seconds):
        super().__init__('synthetic_swarm')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.odom = [self.create_publisher(VehicleOdometry, t, qos) for t in TOPICS]
        self.det = self.create_publisher(String, '/yolo/raw_detections', 20)
        self.create_subscription(String, '/swarm/victims', self._on_report, 10)
        self.rng = np.random.default_rng(0)
        self.bias = {i: self.rng.normal(scale=0.25, size=2) for i in range(4)}
        self.t = 0
        self.ticks = int(seconds * 5)
        self.report = None
        self.create_timer(0.2, self._tick)

    def _on_report(self, msg):
        self.report = json.loads(msg.data)

    def _tick(self):
        t, rng = self.t, self.rng
        self.t += 1
        for i in range(4):
            ang = 0.05 * t + i * math.pi / 2
            o = DEFAULT_ORIGINS[i]
            tw = np.array([5.0 * math.cos(ang), 5.0 * math.sin(ang)])
            yaw = math.atan2(-tw[1], -tw[0]) + rng.normal(scale=0.01)
            true = Pose(tw[0] - o[0], tw[1] - o[1], -11.0 - i, rng.normal(scale=0.01), rng.normal(scale=0.01), yaw)
            m = VehicleOdometry()
            m.position = [float(true.n + self.bias[i][0]), float(true.e + self.bias[i][1]), float(true.d)]
            cr, sr = math.cos(true.roll / 2), math.sin(true.roll / 2)
            cp, sp = math.cos(true.pitch / 2), math.sin(true.pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            m.q = [float(cr * cp * cy + sr * sp * sy), float(sr * cp * cy - cr * sp * sy),
                   float(cr * sp * cy + sr * cp * sy), float(cr * cp * sy - sr * sp * cy)]
            self.odom[i].publish(m)
            dets = []
            for k, (pn, pe) in T.PEOPLE.items():
                proj = project_ned_to_pixel(T.CAM, true, pn - o[0], pe - o[1], 0.0)
                if proj is None or not (30 < proj[0] < T.CAM.width - 30 and 70 < proj[1] < T.CAM.height - 5):
                    continue
                if rng.random() < 0.15:
                    continue
                u, v = proj[0] + rng.normal(scale=4), proj[1] + rng.normal(scale=4)
                dets.append({"box": [int(u - 14), int(v - 55), int(u + 14), int(v)], "conf": 0.8,
                             "emb": color_histogram(T._person_crop(k, rng)).round(3).tolist()})
            if dets:
                s = String()
                s.data = json.dumps({"drone_id": i, "stamp": t * 0.2, "width": 1280, "height": 960, "dets": dets})
                self.det.publish(s)


def main():
    rclpy.init()
    node = SyntheticSwarm(seconds=float(sys.argv[1]) if len(sys.argv) > 1 else 70)
    deadline = time.time() + node.ticks / 5 + 10
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    rep = node.report
    node.destroy_node()
    rclpy.shutdown()
    if rep is None:
        print("FAIL: no /swarm/victims message received (is the victim_registry node running?)")
        return 2
    conf = [v for v in rep["victims"] if v["confirmed"]]
    errs = [min(math.dist(v["n_ned"], p) for p in T.PEOPLE.values()) for v in conf]
    print(f"count={rep['count']} truth={len(T.PEOPLE)} per_drone={rep['per_drone_count']} "
          f"max_pos_err={max(errs) if errs else None} stats={rep['stats']}")
    ok = rep["count"] == len(T.PEOPLE) and errs and max(errs) < 2.0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
