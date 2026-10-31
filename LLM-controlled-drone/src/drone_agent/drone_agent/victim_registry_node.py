#!/usr/bin/env python3
"""ROS 2 node: de-duplicated victim count from the swarm's REAL YOLO detections.

Subscribes
  /yolo/raw_detections   std_msgs/String (JSON)  person boxes + appearance feature (yolo_detector)
  <px4>/fmu/out/vehicle_odometry  px4_msgs/VehicleOdometry  (same topics as yolo_detector)
Publishes
  /swarm/victims         std_msgs/String (JSON)  {count, per_drone_count, victims[], stats}

All logic lives in victim_fusion.VictimFusion (ROS-free, unit-tested). This wrapper is NOT yet
run against Gazebo: see PROGRESS.md for the exact first-run checklist.
"""
import json
import math

import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from drone_agent.victim_fusion import DEFAULT_ORIGINS, VictimFusion
from drone_agent.victim_geometry import Pose

ODOM_TOPICS = ['/fmu/out/vehicle_odometry', '/px4_1/fmu/out/vehicle_odometry',
               '/px4_2/fmu/out/vehicle_odometry', '/px4_3/fmu/out/vehicle_odometry']


class VictimRegistryNode(Node):
    def __init__(self):
        super().__init__('victim_registry')
        self.declare_parameter('n_drones', 4)
        self.declare_parameter('comm_range_m', 25.0)
        self.declare_parameter('pose_var', 0.25)
        self.declare_parameter('min_conf', 0.35)
        self.declare_parameter('swarm_root', '')
        self.declare_parameter('origins_ned', [c for o in DEFAULT_ORIGINS for c in o])
        self.declare_parameter('step_hz', 2.0)

        n = int(self.get_parameter('n_drones').value)
        flat = list(self.get_parameter('origins_ned').value)
        origins = [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]
        self.fusion = VictimFusion(
            n_drones=n, origins_ned=origins,
            comm_range_m=float(self.get_parameter('comm_range_m').value),
            pose_var=float(self.get_parameter('pose_var').value),
            min_conf=float(self.get_parameter('min_conf').value),
            swarm_root=self.get_parameter('swarm_root').value or None)
        self.tick = 0

        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        for i in range(n):
            self.create_subscription(VehicleOdometry, ODOM_TOPICS[i],
                                     lambda m, i=i: self._odom_cb(i, m), px4_qos)
        self.create_subscription(String, '/yolo/raw_detections', self._det_cb, 20)
        self.pub = self.create_publisher(String, '/swarm/victims', 10)
        self.create_timer(1.0 / float(self.get_parameter('step_hz').value), self._step)
        self.get_logger().info(f'victim_registry up: {n} drones, comm range '
                               f'{self.fusion.comm_range} m, mono RGB only (min_sensors=1)')

    def _odom_cb(self, i, msg):
        w, x, y, z = (float(v) for v in msg.q[:4])
        self.fusion.set_pose(i, Pose(
            float(msg.position[0]), float(msg.position[1]), float(msg.position[2]),
            math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))

    def _det_cb(self, msg):
        try:
            self.fusion.on_detections(json.loads(msg.data), self.tick)
        except (ValueError, KeyError) as e:
            self.get_logger().warn(f'bad /yolo/raw_detections payload: {e}')

    def _step(self):
        self.tick += 1
        self.fusion.step(self.tick)
        out = String()
        out.data = json.dumps(self.fusion.report())
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VictimRegistryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
